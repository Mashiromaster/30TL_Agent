# -*- coding: utf-8 -*-
"""Point-in-time macro, funding, and cross-market factor computation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from macro_timing import (
    AVAILABILITY_COLUMN,
    assign_available_ts,
    combine_factor_events,
    factor_asof_column,
)


@dataclass(frozen=True)
class FactorMetadata:
    source: str
    max_age_days: int
    frequency: str


class MacroFactorComputer:
    """Compute macro factors at their native observation frequency."""

    DAILY_TTL_DAYS = 5
    MONTHLY_TTL_DAYS = 50

    def __init__(self, release_calendar: pd.DataFrame | None = None):
        self.release_calendar = release_calendar
        self.factor_names: list[str] = []
        self.factor_metadata: dict[str, FactorMetadata] = {}

    @property
    def factor_ttls(self) -> dict[str, pd.Timedelta]:
        return {
            name: pd.Timedelta(days=meta.max_age_days)
            for name, meta in self.factor_metadata.items()
        }

    @staticmethod
    def _factor_columns(df: pd.DataFrame) -> list[str]:
        return [
            col for col in df.columns
            if col != AVAILABILITY_COLUMN
            and not col.startswith("_asof__")
            and pd.api.types.is_numeric_dtype(df[col])
        ]

    def _register(self, columns, source, max_age_days, frequency):
        for column in columns:
            self.factor_metadata[column] = FactorMetadata(
                source=source,
                max_age_days=max_age_days,
                frequency=frequency,
            )

    def _prepare(self, df, release_kind, series_name):
        if df is None or df.empty or "date" not in df.columns:
            return pd.DataFrame()
        return assign_available_ts(
            df,
            release_kind,
            release_calendar=self.release_calendar,
            series_name=series_name,
        )

    def compute_all(self, raw_data_dict):
        print("[MacroFactors] Computing point-in-time macro factors...")
        frames = []
        shibor = raw_data_dict.get("shibor", pd.DataFrame())
        repo = raw_data_dict.get("repo", pd.DataFrame())

        for frame in (
            self._compute_funding_factors(shibor, repo),
            self._compute_yield_curve_factors(
                raw_data_dict.get("yield_curve", pd.DataFrame())
            ),
            self._compute_liquidity_factors(
                raw_data_dict.get("pboc_omo", pd.DataFrame()), shibor
            ),
            self._compute_cross_market_factors(
                raw_data_dict.get("cross_market", pd.DataFrame()),
                raw_data_dict.get("yield_curve", pd.DataFrame()),
            ),
            self._compute_macro_state_factors(
                raw_data_dict.get("pmi", pd.DataFrame()),
                raw_data_dict.get("cpi", pd.DataFrame()),
                raw_data_dict.get("m2", pd.DataFrame()),
                raw_data_dict.get("social_financing", pd.DataFrame()),
            ),
        ):
            if frame is not None and not frame.empty:
                frames.append(frame)

        combined = combine_factor_events(frames)
        if combined.empty:
            print("[MacroFactors] WARNING: no usable factors computed")
            return combined

        factor_cols = self._factor_columns(combined)
        constant_zero = [
            col for col in factor_cols if combined[col].fillna(0).eq(0).all()
        ]
        if constant_zero:
            combined = combined.drop(
                columns=constant_zero + [factor_asof_column(c) for c in constant_zero],
                errors="ignore",
            )
            for col in constant_zero:
                self.factor_metadata.pop(col, None)

        self.factor_names = self._factor_columns(combined)
        print(
            f"[MacroFactors] Computed {len(self.factor_names)} point-in-time factors: "
            f"{self.factor_names}"
        )
        return combined

    def _compute_funding_factors(self, df_shibor, df_repo):
        frames = []
        shibor = self._prepare(df_shibor, "daily_market", "shibor")
        if not shibor.empty:
            out = shibor[[AVAILABILITY_COLUMN]].copy()
            if {"SHIBOR_3M", "SHIBOR_1M"}.issubset(shibor.columns):
                out["SHIBOR_3M_1M_Spread"] = (
                    shibor["SHIBOR_3M"] - shibor["SHIBOR_1M"]
                )
            if {"SHIBOR_1Y", "SHIBOR_3M"}.issubset(shibor.columns):
                out["SHIBOR_1Y_3M_Spread"] = (
                    shibor["SHIBOR_1Y"] - shibor["SHIBOR_3M"]
                )
            if "SHIBOR_ON" in shibor.columns:
                on = pd.to_numeric(shibor["SHIBOR_ON"], errors="coerce")
                mean20 = on.rolling(20, min_periods=5).mean()
                std20 = on.rolling(20, min_periods=5).std()
                out["SHIBOR_ON_Deviation"] = (
                    (on - mean20) / std20.replace(0, np.nan)
                )
                out["Repo_Volatility"] = (
                    std20 / mean20.abs().replace(0, np.nan)
                )
            factors = self._factor_columns(out)
            if factors:
                self._register(factors, "shibor", self.DAILY_TTL_DAYS, "daily")
                frames.append(out)

        repo = self._prepare(df_repo, "daily_market", "repo")
        if not repo.empty:
            out = repo[[AVAILABILITY_COLUMN]].copy()
            repo_cols = [
                col for col in repo.select_dtypes(include=[np.number]).columns
                if any(token in col.upper() for token in ("DR", "R007", "REPO"))
            ]
            for col in repo_cols:
                series = pd.to_numeric(repo[col], errors="coerce")
                mean20 = series.rolling(20, min_periods=5).mean()
                std20 = series.rolling(20, min_periods=5).std()
                out[f"{col}_Level"] = series
                out[f"{col}_Change_5D"] = series.diff(5)
                out[f"{col}_Deviation"] = (
                    (series - mean20) / std20.replace(0, np.nan)
                )
            if {"Repo_FR007", "Repo_FDR007"}.issubset(repo.columns):
                out["Repo_FR007_FDR007_Spread"] = (
                    pd.to_numeric(repo["Repo_FR007"], errors="coerce")
                    - pd.to_numeric(repo["Repo_FDR007"], errors="coerce")
                )
            factors = self._factor_columns(out)
            if factors:
                self._register(factors, "repo", self.DAILY_TTL_DAYS, "daily")
                frames.append(out)
        return combine_factor_events(frames)

    def _compute_yield_curve_factors(self, df_yield):
        df = self._prepare(df_yield, "daily_market", "yield_curve")
        if df.empty:
            return pd.DataFrame()
        df = df.sort_values(AVAILABILITY_COLUMN)
        out = df[[AVAILABILITY_COLUMN]].copy()

        if {"Yield_10Y", "Yield_1Y"}.issubset(df.columns):
            out["YC_Slope_10Y_1Y"] = df["Yield_10Y"] - df["Yield_1Y"]
        if {"Yield_30Y", "Yield_10Y"}.issubset(df.columns):
            out["YC_Slope_30Y_10Y"] = df["Yield_30Y"] - df["Yield_10Y"]
        if {"Yield_5Y", "Yield_1Y", "Yield_10Y"}.issubset(df.columns):
            out["YC_Curvature"] = (
                2 * df["Yield_5Y"] - df["Yield_1Y"] - df["Yield_10Y"]
            )
        if "Yield_10Y" in df.columns:
            y10 = pd.to_numeric(df["Yield_10Y"], errors="coerce")
            out["YC_Level_10Y"] = y10
            out["YC_10Y_Change_1D"] = y10.diff(1)
            out["YC_10Y_Change_5D"] = y10.diff(5)
            mean60 = y10.rolling(60, min_periods=20).mean()
            std60 = y10.rolling(60, min_periods=20).std()
            out["YC_Level_ZScore"] = (
                (y10 - mean60) / std60.replace(0, np.nan)
            )
        if "Yield_30Y" in df.columns:
            y30 = pd.to_numeric(df["Yield_30Y"], errors="coerce")
            out["YC_Level_30Y"] = y30
            out["YC_30Y_Change_1D"] = y30.diff(1)
            out["YC_30Y_Change_5D"] = y30.diff(5)
        if "YC_Slope_30Y_10Y" in out.columns:
            slope = out["YC_Slope_30Y_10Y"]
            out["YC_Slope_Change_5D"] = slope.diff(5)
            out["YC_Flattening_20D"] = np.sign(
                slope.rolling(5, min_periods=3).mean()
                - slope.rolling(20, min_periods=5).mean()
            )

        factors = self._factor_columns(out)
        self._register(factors, "yield_curve", self.DAILY_TTL_DAYS, "daily")
        return out

    def _compute_liquidity_factors(self, df_omo, df_shibor):
        omo = self._prepare(df_omo, "daily_market", "pboc_omo")
        if not omo.empty:
            out = omo[[AVAILABILITY_COLUMN]].copy()
            zero = pd.Series(0.0, index=omo.index)
            reverse_repo = pd.to_numeric(
                omo.get("reverse_repo_amount", zero), errors="coerce"
            ).fillna(0.0)
            mlf = pd.to_numeric(
                omo.get("mlf_amount", zero), errors="coerce"
            ).fillna(0.0)
            maturity = pd.to_numeric(
                omo.get("maturity_amount", zero), errors="coerce"
            ).fillna(0.0)
            net = reverse_repo + mlf - maturity
            out["Net_Injection_5D"] = net.rolling(5, min_periods=1).sum()
            out["Net_Injection_20D"] = net.rolling(20, min_periods=1).sum()
            out["Injection_Momentum"] = (
                out["Net_Injection_5D"] - out["Net_Injection_20D"]
            )
            factors = self._factor_columns(out)
            self._register(factors, "pboc_omo", self.DAILY_TTL_DAYS, "daily")
            return out

        shibor = self._prepare(df_shibor, "daily_market", "shibor")
        if shibor.empty or not {"SHIBOR_3M", "SHIBOR_1M"}.issubset(shibor.columns):
            return pd.DataFrame()
        spread = shibor["SHIBOR_3M"] - shibor["SHIBOR_1M"]
        threshold = spread.rolling(60, min_periods=20).quantile(0.8)
        out = pd.DataFrame({
            AVAILABILITY_COLUMN: shibor[AVAILABILITY_COLUMN],
            "Liquidity_Stress": (spread > threshold).astype(float).where(
                threshold.notna()
            ),
        })
        self._register(
            ["Liquidity_Stress"], "shibor_proxy", self.DAILY_TTL_DAYS, "daily"
        )
        return out

    def _compute_cross_market_factors(self, df_cross, df_yield):
        cross = self._prepare(df_cross, "daily_market", "cross_market")
        frames = []
        if not cross.empty and "CN_US_10Y_Spread" in cross.columns:
            out = cross[[AVAILABILITY_COLUMN]].copy()
            spread = pd.to_numeric(cross["CN_US_10Y_Spread"], errors="coerce")
            out["CN_US_10Y_Spread"] = spread
            mean60 = spread.rolling(60, min_periods=20).mean()
            std60 = spread.rolling(60, min_periods=20).std()
            out["CN_US_10Y_Spread_Z"] = (
                (spread - mean60) / std60.replace(0, np.nan)
            )
            out["CN_US_Spread_Change_1D"] = spread.diff(1)
            out["CN_US_Spread_Change_5D"] = spread.diff(5)
            out["CN_US_Spread_Change_20D"] = spread.diff(20)
            factors = self._factor_columns(out)
            self._register(
                factors, "cross_market", self.DAILY_TTL_DAYS, "daily"
            )
            frames.append(out)

        lpr_columns = [col for col in ("LPR_1Y", "LPR_5Y") if col in cross.columns]
        if lpr_columns:
            lpr = cross.loc[
                cross[lpr_columns].notna().any(axis=1),
                [AVAILABILITY_COLUMN, *lpr_columns],
            ].copy()
            lpr = lpr.sort_values(AVAILABILITY_COLUMN)
            out_lpr = lpr[[AVAILABILITY_COLUMN]].copy()
            for column in lpr_columns:
                value = pd.to_numeric(lpr[column], errors="coerce")
                out_lpr[f"{column}_Level"] = value
                out_lpr[f"{column}_Change"] = value.diff()
            if set(lpr_columns) == {"LPR_1Y", "LPR_5Y"}:
                out_lpr["LPR_5Y_1Y_Spread"] = (
                    pd.to_numeric(lpr["LPR_5Y"], errors="coerce")
                    - pd.to_numeric(lpr["LPR_1Y"], errors="coerce")
                )
            factors = self._factor_columns(out_lpr)
            self._register(
                factors, "lpr", self.MONTHLY_TTL_DAYS, "monthly"
            )
            frames.append(out_lpr)

        curve = self._prepare(df_yield, "daily_market", "yield_curve")
        if not curve.empty and "Yield_10Y" in curve.columns:
            momentum = pd.to_numeric(
                curve["Yield_10Y"], errors="coerce"
            ).diff(5)
            yc = pd.DataFrame({
                AVAILABILITY_COLUMN: curve[AVAILABILITY_COLUMN],
                "YC_Momentum_5D": momentum,
                "Risk_On_Off": np.sign(momentum),
            })
            factors = self._factor_columns(yc)
            self._register(
                factors, "yield_curve", self.DAILY_TTL_DAYS, "daily"
            )
            frames.append(yc)

        return combine_factor_events(frames)

    def _compute_macro_state_factors(self, df_pmi, df_cpi, df_m2, df_sf):
        frames = []
        pmi = self._prepare(df_pmi, "pmi", "pmi")
        if not pmi.empty and "PMI_Manufacturing" in pmi.columns:
            value = pd.to_numeric(pmi["PMI_Manufacturing"], errors="coerce")
            mean24 = value.rolling(24, min_periods=12).mean()
            std24 = value.rolling(24, min_periods=12).std()
            frame = pd.DataFrame({
                AVAILABILITY_COLUMN: pmi[AVAILABILITY_COLUMN],
                "PMI_State_ZScore": (value - mean24) / std24.replace(0, np.nan),
                "PMI_Change_3M": value.diff(3),
            })
            frames.append(frame)
            self._register(
                self._factor_columns(frame), "pmi", self.MONTHLY_TTL_DAYS, "monthly"
            )

        cpi = self._prepare(df_cpi, "cpi", "cpi")
        if not cpi.empty and "CPI_YoY" in cpi.columns:
            value = pd.to_numeric(cpi["CPI_YoY"], errors="coerce")
            frame = pd.DataFrame({
                AVAILABILITY_COLUMN: cpi[AVAILABILITY_COLUMN],
                "CPI_Acceleration_3M": value - value.shift(3),
            })
            frames.append(frame)
            self._register(
                self._factor_columns(frame), "cpi", self.MONTHLY_TTL_DAYS, "monthly"
            )

        m2 = self._prepare(df_m2, "monetary", "m2")
        if not m2.empty and "M2_YoY" in m2.columns:
            value = pd.to_numeric(m2["M2_YoY"], errors="coerce")
            mean12 = value.rolling(12, min_periods=6).mean()
            std12 = value.rolling(12, min_periods=6).std()
            frame = pd.DataFrame({
                AVAILABILITY_COLUMN: m2[AVAILABILITY_COLUMN],
                "M2_State_ZScore": (value - mean12) / std12.replace(0, np.nan),
                "M2_Change_3M": value.diff(3),
            })
            frames.append(frame)
            self._register(
                self._factor_columns(frame), "m2", self.MONTHLY_TTL_DAYS, "monthly"
            )

        sf = self._prepare(df_sf, "social_financing", "social_financing")
        if not sf.empty and "SocialFin_Flow" in sf.columns:
            value = pd.to_numeric(sf["SocialFin_Flow"], errors="coerce")
            yoy = value.pct_change(12).replace([np.inf, -np.inf], np.nan)
            mean36 = yoy.rolling(36, min_periods=12).mean()
            std36 = yoy.rolling(36, min_periods=12).std()
            frame = pd.DataFrame({
                AVAILABILITY_COLUMN: sf[AVAILABILITY_COLUMN],
                "SocialFin_YoY_ZScore": (
                    (yoy - mean36) / std36.replace(0, np.nan)
                ),
            })
            frames.append(frame)
            self._register(
                self._factor_columns(frame),
                "social_financing",
                self.MONTHLY_TTL_DAYS,
                "monthly",
            )

        if not frames:
            return pd.DataFrame()
        combined = combine_factor_events(frames)
        state_cols = [
            col for col in (
                "PMI_State_ZScore", "CPI_Acceleration_3M",
                "M2_State_ZScore", "SocialFin_YoY_ZScore",
            ) if col in combined.columns
        ]
        if state_cols:
            combined["Macro_State_Composite"] = combined[state_cols].clip(
                -4, 4
            ).mean(axis=1, skipna=True)
            combined[factor_asof_column("Macro_State_Composite")] = combined[
                AVAILABILITY_COLUMN
            ]
            self._register(
                ["Macro_State_Composite"],
                "macro_state",
                self.MONTHLY_TTL_DAYS,
                "monthly",
            )
        return combined


if __name__ == "__main__":
    dates = pd.date_range("2024-01-01", periods=260, freq="B")
    curve = pd.DataFrame({
        "date": dates,
        "Yield_1Y": 1.8 + np.sin(np.arange(len(dates)) / 30) * 0.1,
        "Yield_5Y": 2.0 + np.sin(np.arange(len(dates)) / 30) * 0.1,
        "Yield_10Y": 2.2 + np.sin(np.arange(len(dates)) / 30) * 0.1,
        "Yield_30Y": 2.5 + np.sin(np.arange(len(dates)) / 30) * 0.1,
    })
    print(MacroFactorComputer().compute_all({"yield_curve": curve}).tail())
