# -*- coding: utf-8 -*-
"""Small, deterministic quality reports for factor artifacts."""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd


def build_factor_artifact_audit(df: pd.DataFrame) -> dict:
    numeric = list(df.select_dtypes(include=[np.number, "bool"]).columns)
    excluded = {"open", "high", "low", "close", "volume", "oi", "money"}
    factors = [name for name in numeric if name not in excluded]
    all_zero = [name for name in factors if df[name].fillna(0).eq(0).all()]
    constant = [name for name in factors if df[name].dropna().nunique() <= 1]
    coverage = {
        name: float(df[name].notna().mean())
        for name in factors
    }
    dates = pd.to_datetime(df.get("date"), errors="coerce")
    macro_features = list(df.attrs.get("macro_features", []))
    return {
        "schema_version": 1,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "date_start": dates.min().isoformat() if dates.notna().any() else None,
        "date_end": dates.max().isoformat() if dates.notna().any() else None,
        "factor_count": len(factors),
        "macro_features": macro_features,
        "macro_feature_count": len(macro_features),
        "all_zero_factors": all_zero,
        "constant_factors": constant,
        "coverage": coverage,
        "macro_coverage_rows": (
            float(df["_macro_coverage_count"].gt(0).mean())
            if "_macro_coverage_count" in df else None
        ),
        "macro_stale_rows": (
            float(df["_macro_stale_count"].gt(0).mean())
            if "_macro_stale_count" in df else None
        ),
    }


def write_factor_artifact_audit(df: pd.DataFrame, output_path: str) -> dict:
    report = build_factor_artifact_audit(df)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_path = f"{output_path}.{os.getpid()}-{time.time_ns()}.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        os.replace(temp_path, output_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return report
