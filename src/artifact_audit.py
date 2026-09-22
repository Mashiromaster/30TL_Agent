# -*- coding: utf-8 -*-
"""Small, deterministic quality reports for factor artifacts."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping

import numpy as np
import pandas as pd


_TEXT_FACTOR_METADATA_KEYS = (
    "jev_model",
    "jev_schema_version",
    "jev_mode",
    "documents_attempted",
    "documents_succeeded",
    "document_coverage",
    "jev_features",
)


def extract_text_factor_metadata(df: pd.DataFrame) -> dict:
    """Return a JSON-safe provenance whitelist for semantic text factors."""
    raw = df.attrs.get("text_factor_metadata", {})
    if not isinstance(raw, Mapping):
        raw = {}
    metadata = {
        key: raw[key]
        for key in _TEXT_FACTOR_METADATA_KEYS
        if key in raw
    }
    actual_features = sorted(
        name for name in df.columns if str(name).startswith("JEV_")
    )
    if actual_features or "jev_features" in metadata:
        metadata["jev_features"] = actual_features
    for key in ("documents_attempted", "documents_succeeded"):
        if key in metadata:
            metadata[key] = int(metadata[key])
    if "document_coverage" in metadata:
        metadata["document_coverage"] = float(metadata["document_coverage"])
    for key in ("jev_model", "jev_schema_version", "jev_mode"):
        if key in metadata:
            metadata[key] = str(metadata[key])
    return metadata


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
    text_factor_metadata = extract_text_factor_metadata(df)
    return {
        "schema_version": 2,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "date_start": dates.min().isoformat() if dates.notna().any() else None,
        "date_end": dates.max().isoformat() if dates.notna().any() else None,
        "factor_count": len(factors),
        "macro_features": macro_features,
        "macro_feature_count": len(macro_features),
        "text_factor_metadata": text_factor_metadata,
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
