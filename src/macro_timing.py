# -*- coding: utf-8 -*-
"""Point-in-time utilities for macroeconomic data.

The central contract is ``available_ts``: the first timestamp at which a value
could have been used by the strategy. Observation periods and data availability
are deliberately kept separate.
"""

from __future__ import annotations

from functools import reduce
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


AVAILABILITY_COLUMN = "available_ts"
_NON_FACTOR_COLUMNS = {
    "date", "period_date", "release_ts", "available_date",
    AVAILABILITY_COLUMN, "source", "series", "vintage",
}


def _local_naive(values):
    """Normalize timestamps to Asia/Shanghai wall time for deterministic joins."""
    parsed = pd.to_datetime(values, errors="coerce")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        return parsed.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    return parsed


def factor_asof_column(factor_name: str) -> str:
    """Return the private timestamp column used to track a factor's age."""
    return f"_asof__{factor_name}"


def _calendar_release_map(release_calendar, series_name):
    if release_calendar is None or release_calendar.empty or not series_name:
        return {}
    required = {"series", "period_date", "release_ts"}
    if not required.issubset(release_calendar.columns):
        raise ValueError(f"release_calendar must contain {sorted(required)}")
    subset = release_calendar.loc[release_calendar["series"] == series_name].copy()
    subset["period_date"] = pd.to_datetime(
        subset["period_date"], errors="coerce"
    ).dt.normalize()
    subset["release_ts"] = _local_naive(subset["release_ts"])
    subset = subset.dropna(subset=["period_date", "release_ts"])
    return dict(zip(subset["period_date"], subset["release_ts"]))


def assign_available_ts(
    df: pd.DataFrame,
    release_kind: str,
    *,
    date_col: str = "date",
    release_calendar: pd.DataFrame | None = None,
    series_name: str | None = None,
) -> pd.DataFrame:
    """Attach a conservative point-in-time availability timestamp.

    Explicit ``available_ts`` or ``release_ts`` values always win. Fallback
    rules are intentionally conservative and should be overridden by a
    project-maintained historical release calendar whenever possible.
    """
    if date_col not in df.columns:
        raise ValueError(f"missing required date column: {date_col}")

    result = df.copy()
    result[date_col] = pd.to_datetime(result[date_col], errors="coerce")
    period = result[date_col].dt.normalize()
    explicit = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns]")
    if AVAILABILITY_COLUMN in result.columns:
        explicit = _local_naive(result[AVAILABILITY_COLUMN])
    if "release_ts" in result.columns:
        explicit = explicit.fillna(_local_naive(result["release_ts"]))

    kind = release_kind.strip().lower()
    if kind in {"daily", "daily_market", "market"}:
        fallback = period + pd.offsets.BDay(1) + pd.Timedelta(hours=9)
    elif kind == "pmi":
        fallback = period + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=9, minutes=30)
    elif kind in {"cpi", "inflation"}:
        fallback = period + pd.offsets.MonthBegin(1) + pd.Timedelta(
            days=14, hours=9, minutes=30
        )
    elif kind in {"monetary", "m2", "social_financing"}:
        fallback = period + pd.offsets.MonthBegin(1) + pd.Timedelta(days=19, hours=17)
    elif kind in {"text", "report"}:
        fallback = period + pd.offsets.BDay(1) + pd.Timedelta(hours=9)
    else:
        raise ValueError(f"unsupported release_kind: {release_kind}")

    result[AVAILABILITY_COLUMN] = explicit.fillna(pd.Series(fallback, index=result.index))
    calendar_map = _calendar_release_map(release_calendar, series_name)
    if calendar_map:
        result[AVAILABILITY_COLUMN] = period.map(calendar_map).fillna(
            result[AVAILABILITY_COLUMN]
        )
    result = result.dropna(subset=[date_col, AVAILABILITY_COLUMN])
    return result.sort_values(AVAILABILITY_COLUMN).reset_index(drop=True)


def _factor_columns(df: pd.DataFrame) -> list[str]:
    return [
        col for col in df.columns
        if col not in _NON_FACTOR_COLUMNS
        and not col.startswith("_")
        and (pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]))
    ]


def combine_factor_events(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Combine independently timed factor frames into one event stream."""
    pieces = []
    for frame in frames:
        if frame is None or frame.empty:
            continue
        if AVAILABILITY_COLUMN not in frame.columns:
            raise ValueError("factor frame is missing available_ts")
        current = frame.copy()
        current[AVAILABILITY_COLUMN] = _local_naive(current[AVAILABILITY_COLUMN])
        current = current.dropna(subset=[AVAILABILITY_COLUMN]).sort_values(
            AVAILABILITY_COLUMN
        )
        for factor in _factor_columns(current):
            asof_col = factor_asof_column(factor)
            source_asof = (
                _local_naive(current[asof_col])
                if asof_col in current.columns else current[AVAILABILITY_COLUMN]
            )
            piece = pd.DataFrame({
                AVAILABILITY_COLUMN: current[AVAILABILITY_COLUMN],
                factor: current[factor],
                asof_col: source_asof,
            }).dropna(subset=[factor])
            if not piece.empty:
                pieces.append(piece.drop_duplicates(AVAILABILITY_COLUMN, keep="last"))

    if not pieces:
        return pd.DataFrame(columns=[AVAILABILITY_COLUMN])
    merged = reduce(
        lambda left, right: pd.merge(left, right, on=AVAILABILITY_COLUMN, how="outer"),
        pieces,
    )
    merged = merged.sort_values(AVAILABILITY_COLUMN).reset_index(drop=True)
    value_cols = _factor_columns(merged)
    asof_cols = [c for c in merged.columns if c.startswith("_asof__")]
    merged[value_cols + asof_cols] = merged[value_cols + asof_cols].ffill()
    return merged


def merge_point_in_time(
    df_main: pd.DataFrame,
    factor_events: pd.DataFrame,
    *,
    timestamp_col: str = "date",
    factor_ttls: Mapping[str, pd.Timedelta] | None = None,
    fill_value: float = 0.0,
    keep_asof_columns: bool = False,
) -> pd.DataFrame:
    """Backward as-of join macro events to market bars with optional TTLs."""
    if factor_events is None or factor_events.empty:
        return df_main.copy()
    if timestamp_col not in df_main.columns:
        raise ValueError(f"missing market timestamp column: {timestamp_col}")
    if AVAILABILITY_COLUMN not in factor_events.columns:
        raise ValueError("factor events must contain available_ts")

    left = df_main.copy()
    original_index = left.index
    prior_coverage = pd.to_numeric(
        left.get("_macro_coverage_count", 0), errors="coerce"
    )
    prior_stale = pd.to_numeric(
        left.get("_macro_stale_count", 0), errors="coerce"
    )
    prior_max_age = pd.to_numeric(
        left.get("_macro_max_age_days", 0.0), errors="coerce"
    )
    left[timestamp_col] = _local_naive(left[timestamp_col])
    left["_macro_order"] = np.arange(len(left))
    right = factor_events.copy()
    right[AVAILABILITY_COLUMN] = _local_naive(right[AVAILABILITY_COLUMN])
    right = right.dropna(subset=[AVAILABILITY_COLUMN]).sort_values(
        AVAILABILITY_COLUMN
    ).drop_duplicates(AVAILABILITY_COLUMN, keep="last")

    merged = pd.merge_asof(
        left.sort_values(timestamp_col), right,
        left_on=timestamp_col, right_on=AVAILABILITY_COLUMN,
        direction="backward", allow_exact_matches=True,
    )
    factor_cols = _factor_columns(right)
    ttl_map = factor_ttls or {}
    coverage = pd.DataFrame(index=merged.index)
    stale = pd.DataFrame(False, index=merged.index, columns=factor_cols)
    ages = pd.DataFrame(index=merged.index)
    for factor in factor_cols:
        asof_col = factor_asof_column(factor)
        if asof_col not in merged.columns:
            merged[asof_col] = merged[AVAILABILITY_COLUMN]
        merged[asof_col] = _local_naive(merged[asof_col])
        coverage[factor] = merged[factor].notna()
        age = merged[timestamp_col] - merged[asof_col]
        ages[factor] = age.dt.total_seconds().div(86400)
        ttl = ttl_map.get(factor)
        if ttl is not None:
            stale[factor] = coverage[factor] & (age > pd.Timedelta(ttl))
            merged.loc[stale[factor], factor] = np.nan

    prior_coverage = pd.Series(prior_coverage, index=left.index).iloc[
        merged["_macro_order"].astype(int).to_numpy()
    ].reset_index(drop=True).fillna(0)
    prior_stale = pd.Series(prior_stale, index=left.index).iloc[
        merged["_macro_order"].astype(int).to_numpy()
    ].reset_index(drop=True).fillna(0)
    prior_max_age = pd.Series(prior_max_age, index=left.index).iloc[
        merged["_macro_order"].astype(int).to_numpy()
    ].reset_index(drop=True).fillna(0.0)
    current_age = ages.where(coverage & ~stale).max(axis=1).fillna(0.0)
    merged["_macro_coverage_count"] = (
        prior_coverage.to_numpy() + coverage.sum(axis=1).to_numpy()
    ).astype(int)
    merged["_macro_stale_count"] = (
        prior_stale.to_numpy() + stale.sum(axis=1).to_numpy()
    ).astype(int)
    merged["_macro_max_age_days"] = np.maximum(
        prior_max_age.to_numpy(), current_age.to_numpy()
    )
    merged[factor_cols] = merged[factor_cols].fillna(fill_value)
    drop_cols = [AVAILABILITY_COLUMN]
    if not keep_asof_columns:
        drop_cols.extend(c for c in merged.columns if c.startswith("_asof__"))
    merged = merged.drop(columns=drop_cols, errors="ignore")
    merged = merged.sort_values("_macro_order").drop(columns=["_macro_order"])
    merged.index = original_index
    return merged
