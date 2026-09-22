# -*- coding: utf-8 -*-
"""Central feature grouping and schema validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


MACRO_PATTERNS = (
    "SHIBOR_", "Repo_", "YC_", "PMI_", "CPI_", "M2_", "SocialFin",
    "Injection_", "OMO_", "Stock_Bond", "CN_US", "Credit_", "Risk_On",
    "Liquidity_", "Macro_", "Policy_Sentiment", "LPR_",
    "JEV_",
)
MICRO_PATTERNS = (
    "Spread", "Depth", "Imbalance", "VPIN", "HF_", "Signed_", "Trade_",
    "Cum_Net_Open", "Close_Pressure", "Open_Price_Push", "Micro_",
)
REGIME_FEATURES = {"Market_Regime", "Is_High_Vol", "Regime_Disruption"}
MACRO_DIAGNOSTICS = {
    "_macro_coverage_count", "_macro_stale_count", "_macro_max_age_days"
}
EXCLUDED = {
    "date", "trade_dt", "ticker", "open", "high", "low", "close", "adj_close",
    "volume", "money", "oi", "time", "Hour", "Minute", "Minute_of_Day",
    "next_close", "future_close", "Target_Ret", "label_start_ts", "label_end_ts",
    "label_valid", "Pred_Ret", "Pred_Smooth", "Should_Trade",
}


@dataclass(frozen=True)
class FeatureGroups:
    micro: tuple[str, ...]
    market: tuple[str, ...]
    macro: tuple[str, ...]
    regime: tuple[str, ...]
    gp: tuple[str, ...]

    @property
    def all_features(self) -> tuple[str, ...]:
        return self.micro + self.market + self.macro + self.regime + self.gp

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "micro": list(self.micro),
            "market": list(self.market),
            "macro": list(self.macro),
            "regime": list(self.regime),
            "gp": list(self.gp),
        }


def discover_feature_groups(
    df: pd.DataFrame,
    *,
    min_coverage: float = 0.02,
) -> FeatureGroups:
    """Discover numeric, non-constant features using deterministic precedence."""
    numeric = list(df.select_dtypes(include=[np.number, "bool"]).columns)
    candidates = []
    for name in numeric:
        if name in EXCLUDED or (name.startswith("_") and name not in MACRO_DIAGNOSTICS):
            continue
        series = pd.to_numeric(df[name], errors="coerce")
        if series.notna().mean() < min_coverage or series.dropna().nunique() <= 1:
            continue
        candidates.append(name)

    groups = {"micro": [], "market": [], "macro": [], "regime": [], "gp": []}
    for name in candidates:
        if name.startswith("GP_"):
            group = "gp"
        elif name in MACRO_DIAGNOSTICS or any(pattern in name for pattern in MACRO_PATTERNS):
            group = "macro"
        elif name in REGIME_FEATURES:
            group = "regime"
        elif any(pattern in name for pattern in MICRO_PATTERNS):
            group = "micro"
        else:
            group = "market"
        groups[group].append(name)
    return FeatureGroups(**{key: tuple(value) for key, value in groups.items()})


def validate_feature_schema(df: pd.DataFrame, features: tuple[str, ...] | list[str]) -> None:
    missing = [name for name in features if name not in df.columns]
    if missing:
        raise ValueError(f"missing model features: {missing}")
    non_numeric = [
        name for name in features
        if not pd.api.types.is_numeric_dtype(df[name])
    ]
    if non_numeric:
        raise ValueError(f"non-numeric model features: {non_numeric}")
