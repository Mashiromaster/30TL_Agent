# -*- coding: utf-8 -*-
"""Leakage-aware factor evaluation for pre-built forward-return labels."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class FactorReport:
    factor: str
    fold_ics: tuple[float, ...]
    median_ic: float
    mean_ic: float
    same_sign_ratio: float
    coverage: float
    observations: int


def _spearman(left: pd.Series, right: pd.Series, min_samples: int = 10) -> float:
    pair = pd.concat(
        [pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")],
        axis=1,
    ).dropna()
    if len(pair) < min_samples or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic)


def evaluate_factor(
    df: pd.DataFrame,
    factor_col: str,
    target_col: str = "Target_Ret",
    *,
    fold_col: str | None = None,
) -> dict:
    """Evaluate a factor against an already constructed target.

    No rolling, shifting, or target regeneration occurs here. When ``fold_col``
    is provided, the summary is computed from fold-level out-of-sample ICs.
    """
    if factor_col not in df or target_col not in df:
        raise ValueError(f"missing {factor_col!r} or {target_col!r}")
    groups = [(None, df)] if fold_col is None else list(df.groupby(fold_col, sort=True))
    fold_ics = [
        ic for _, group in groups
        if not np.isnan(ic := _spearman(group[factor_col], group[target_col]))
    ]
    median = float(np.median(fold_ics)) if fold_ics else np.nan
    sign = np.sign(median)
    same_sign = (
        float(np.mean(np.sign(fold_ics) == sign))
        if fold_ics and sign != 0 else 0.0
    )
    report = FactorReport(
        factor=factor_col,
        fold_ics=tuple(round(value, 6) for value in fold_ics),
        median_ic=round(median, 6) if not np.isnan(median) else np.nan,
        mean_ic=round(float(np.mean(fold_ics)), 6) if fold_ics else np.nan,
        same_sign_ratio=round(same_sign, 6),
        coverage=round(float(df[factor_col].notna().mean()), 6),
        observations=int(df[[factor_col, target_col]].dropna().shape[0]),
    )
    return asdict(report)


class FactorEvaluator:
    """Factor evaluator that consumes a frozen label definition."""

    def __init__(self, df_factors, target_col="Target_Ret", date_col="date"):
        self.df = df_factors.copy()
        self.target_col = target_col
        self.date_col = date_col
        if target_col not in self.df:
            raise ValueError(
                f"{target_col} is missing; build labels once with training_data.build_forward_labels"
            )
        if date_col in self.df:
            self.df[date_col] = pd.to_datetime(self.df[date_col], errors="coerce")
            self.df["_trade_date"] = self.df[date_col].dt.normalize()
        self._exclude_cols = {
            "date", "_trade_date", "trade_dt", "ticker", "close", "adj_close",
            "open", "high", "low", "volume", "money", "oi", "time", "Hour",
            "Minute", "Minute_of_Day", "next_close", "future_close", "Target_Ret",
            "label_start_ts", "label_end_ts", "label_valid", "Pred_Ret",
            "Pred_Smooth", "Should_Trade", "Position", "Asset_Ret", "Trades",
            "Cost", "Net_Ret", "Strategy_Ret", "Cum_Ret", "Signal", "Raw_Signal",
            "Confirmed_Signal", "Pred_Rank", "Model_Used", "Trade_Weight",
            "Upper_Q", "Lower_Q", "Leverage", "Target_Vol", "Position_Weight",
            "fold",
        }

    @property
    def factor_cols(self):
        numeric = self.df.select_dtypes(include=[np.number, "bool"]).columns
        return [
            col for col in numeric
            if col not in self._exclude_cols and not col.startswith("_")
        ]

    def compute_ic(self, factor_name, periods=None):
        """Compute total and day-block IC against the provided target."""
        if factor_name not in self.factor_cols:
            raise ValueError(f"not a numeric factor: {factor_name}")
        total_ic = _spearman(self.df[factor_name], self.df[self.target_col])
        daily_ics = []
        if "_trade_date" in self.df:
            for _, group in self.df.groupby("_trade_date", sort=True):
                ic = _spearman(group[factor_name], group[self.target_col])
                if not np.isnan(ic):
                    daily_ics.append(ic)
        daily_mean = float(np.mean(daily_ics)) if daily_ics else np.nan
        daily_std = float(np.std(daily_ics, ddof=1)) if len(daily_ics) > 1 else np.nan
        return {
            "factor": factor_name,
            "ic": round(total_ic, 6) if not np.isnan(total_ic) else np.nan,
            "daily_ic_count": len(daily_ics),
            "daily_ic_mean": round(daily_mean, 6) if not np.isnan(daily_mean) else np.nan,
            "daily_ic_std": round(daily_std, 6) if not np.isnan(daily_std) else np.nan,
            "ic_ir": (
                round(daily_mean / daily_std, 6)
                if not np.isnan(daily_mean) and not np.isnan(daily_std) and daily_std > 0
                else np.nan
            ),
            "ic_positive_ratio": (
                round(float(np.mean(np.asarray(daily_ics) > 0)), 6)
                if daily_ics else np.nan
            ),
        }

    def rank_factors_by_ic(self, top_n=30):
        rows = []
        errors = []
        for factor in self.factor_cols:
            try:
                rows.append(self.compute_ic(factor))
            except (TypeError, ValueError) as exc:
                errors.append({"factor": factor, "error": str(exc)})
        self.last_errors = errors
        if not rows:
            return pd.DataFrame(columns=["factor", "ic"])
        result = pd.DataFrame(rows)
        return result.sort_values("ic", key=lambda value: value.abs(), ascending=False).head(top_n)

    def quantile_returns(self, factor_name, n_quantiles=5, min_history=100):
        """Historical expanding-rank buckets; the target is not transformed."""
        factor = pd.to_numeric(self.df[factor_name], errors="coerce")
        percentile = factor.expanding(min_periods=min_history).rank(pct=True)
        bucket = np.minimum(
            (percentile * n_quantiles).fillna(-1).astype(int), n_quantiles - 1
        )
        work = pd.DataFrame({"bucket": bucket, "target": self.df[self.target_col]})
        work = work.loc[(work["bucket"] >= 0) & work["target"].notna()]
        if work.empty:
            return None
        returns = work.groupby("bucket")["target"].mean()
        if len(returns) < 2:
            return None
        return {
            "quantile_returns": returns.to_dict(),
            "long_short": round(float(returns.iloc[-1] - returns.iloc[0]), 6),
        }

    def factor_turnover(self, factor_name):
        series = pd.to_numeric(self.df[factor_name], errors="coerce").dropna()
        if len(series) < 20:
            return None
        return {
            f"autocorr_{lag}": round(float(series.autocorr(lag=lag)), 6)
            for lag in (1, 5, 20)
        }

    def correlation_matrix(self, top_n=20):
        ranked = self.rank_factors_by_ic(top_n=top_n)
        factors = ranked["factor"].tolist() if "factor" in ranked else []
        if len(factors) < 2:
            return None, factors, []
        sample = self.df[factors].dropna()
        if len(sample) < 100:
            return None, factors, []
        corr = sample.corr(method="spearman")
        redundant = []
        for left_index, left in enumerate(factors):
            for right in factors[left_index + 1:]:
                value = corr.loc[left, right]
                if abs(value) > 0.7:
                    redundant.append({
                        "factor_a": left,
                        "factor_b": right,
                        "correlation": round(float(value), 6),
                    })
        return corr, factors, redundant

    def generate_report(self, output_path):
        report = self.rank_factors_by_ic(top_n=len(self.factor_cols))
        report.to_csv(output_path, index=False, encoding="utf-8-sig")
        return report


def main():
    parser = argparse.ArgumentParser(description="Point-in-time factor evaluation")
    parser.add_argument("--base-dir", default=os.getcwd())
    parser.add_argument("--factor")
    args = parser.parse_args()
    from training_data import LabelSpec, build_forward_labels

    path = os.path.join(args.base_dir, "outputs", "df_factors.pkl")
    df = build_forward_labels(pd.read_pickle(path), LabelSpec())
    evaluator = FactorEvaluator(df)
    if args.factor:
        print(evaluator.compute_ic(args.factor))
    else:
        output = os.path.join(args.base_dir, "outputs", "factor_report.csv")
        evaluator.generate_report(output)
        print(f"report saved: {output}")


if __name__ == "__main__":
    main()
