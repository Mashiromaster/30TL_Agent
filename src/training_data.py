# -*- coding: utf-8 -*-
"""Leakage-safe labels and chronological dataset partitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelSpec:
    horizon_bars: int = 30
    entry_lag_bars: int = 1
    price_col: str = "adj_close"
    ticker_col: str = "ticker"
    timestamp_col: str = "date"
    forbid_contract_crossing: bool = True
    allow_overnight: bool = False


@dataclass(frozen=True)
class SplitSpec:
    validation_days: int = 60
    sealed_holdout_days: int = 60
    min_train_days: int = 120
    train_days: int | None = 270
    step_days: int = 30


def build_forward_labels(df: pd.DataFrame, spec: LabelSpec = LabelSpec()) -> pd.DataFrame:
    """Build forward returns within each contract and record label timestamps."""
    required = {spec.timestamp_col, spec.price_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing label inputs: {sorted(missing)}")
    if spec.horizon_bars <= 0 or spec.entry_lag_bars < 0:
        raise ValueError("horizon_bars must be positive and entry_lag_bars non-negative")
    if spec.forbid_contract_crossing and spec.ticker_col not in df.columns:
        raise ValueError(f"missing contract column: {spec.ticker_col}")

    result = df.copy()
    result[spec.timestamp_col] = pd.to_datetime(
        result[spec.timestamp_col], errors="coerce"
    )
    result["_label_order"] = np.arange(len(result))
    sort_cols = [spec.timestamp_col]
    if spec.ticker_col in result.columns:
        sort_cols.insert(0, spec.ticker_col)
    work = result.sort_values(sort_cols).copy()
    price = pd.to_numeric(work[spec.price_col], errors="coerce")
    if spec.ticker_col in work.columns:
        group_key = work[spec.ticker_col]
        entry_price = price.groupby(group_key, sort=False).shift(-spec.entry_lag_bars)
        exit_price = price.groupby(group_key, sort=False).shift(
            -(spec.entry_lag_bars + spec.horizon_bars)
        )
        entry_ts = work[spec.timestamp_col].groupby(group_key, sort=False).shift(
            -spec.entry_lag_bars
        )
        exit_ts = work[spec.timestamp_col].groupby(group_key, sort=False).shift(
            -(spec.entry_lag_bars + spec.horizon_bars)
        )
    else:
        entry_price = price.shift(-spec.entry_lag_bars)
        exit_price = price.shift(-(spec.entry_lag_bars + spec.horizon_bars))
        entry_ts = work[spec.timestamp_col].shift(-spec.entry_lag_bars)
        exit_ts = work[spec.timestamp_col].shift(
            -(spec.entry_lag_bars + spec.horizon_bars)
        )

    target = (exit_price / entry_price - 1.0) * 100.0
    valid = target.notna() & entry_ts.notna() & exit_ts.notna()
    if not spec.allow_overnight:
        valid &= entry_ts.dt.normalize().eq(exit_ts.dt.normalize())

    work["label_start_ts"] = entry_ts
    work["label_end_ts"] = exit_ts
    work["Target_Ret"] = target.where(valid)
    work["label_valid"] = valid
    work = work.sort_values("_label_order").drop(columns="_label_order")
    work.index = result.index
    return work


def _purge_before(frame: pd.DataFrame, boundary: pd.Timestamp) -> pd.DataFrame:
    if "label_end_ts" not in frame.columns:
        raise ValueError("label_end_ts is required; call build_forward_labels first")
    return frame.loc[frame["label_end_ts"] < boundary].copy()


def split_train_validation_holdout(
    df: pd.DataFrame,
    spec: SplitSpec = SplitSpec(),
    *,
    timestamp_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create date-grouped train/validation/sealed-holdout partitions.

    Labels whose outcome timestamp reaches the next partition are purged. The
    returned holdout is for one final report only and must not drive selection.
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"missing timestamp column: {timestamp_col}")
    work = df.copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
    all_days = pd.Index(work[timestamp_col].dropna().dt.normalize().unique()).sort_values()
    required_days = spec.min_train_days + spec.validation_days + spec.sealed_holdout_days
    if len(all_days) < required_days:
        raise ValueError(
            f"need at least {required_days} trading days, got {len(all_days)}"
        )
    holdout_start = all_days[-spec.sealed_holdout_days]
    validation_start = all_days[-(spec.sealed_holdout_days + spec.validation_days)]
    valid = work.get("label_valid", pd.Series(True, index=work.index)).fillna(False)
    train = work.loc[valid & (work[timestamp_col] < validation_start)].copy()
    validation = work.loc[
        valid
        & (work[timestamp_col] >= validation_start)
        & (work[timestamp_col] < holdout_start)
    ].copy()
    holdout = work.loc[valid & (work[timestamp_col] >= holdout_start)].copy()
    train = _purge_before(train, validation_start)
    validation = _purge_before(validation, holdout_start)
    if min(len(train), len(validation), len(holdout)) == 0:
        raise ValueError("purged split produced an empty partition")
    return train, validation, holdout


def iter_purged_walk_forward(
    df: pd.DataFrame,
    spec: SplitSpec = SplitSpec(),
    *,
    timestamp_col: str = "date",
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield rolling train/validation folds before the sealed holdout."""
    work = df.copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
    days = pd.Index(work[timestamp_col].dropna().dt.normalize().unique()).sort_values()
    discovery_days = days[:-spec.sealed_holdout_days]
    train_days = spec.train_days or spec.min_train_days
    fold_end = train_days + spec.validation_days
    for stop in range(fold_end, len(discovery_days) + 1, spec.step_days):
        validation_days = discovery_days[stop - spec.validation_days:stop]
        train_window = discovery_days[max(0, stop - spec.validation_days - train_days):stop - spec.validation_days]
        if len(train_window) < spec.min_train_days:
            continue
        validation_start = validation_days[0]
        valid = work.get("label_valid", pd.Series(True, index=work.index)).fillna(False)
        train = work.loc[valid & work[timestamp_col].dt.normalize().isin(train_window)]
        validation = work.loc[
            valid & work[timestamp_col].dt.normalize().isin(validation_days)
        ]
        train = _purge_before(train, validation_start)
        if not train.empty and not validation.empty:
            yield train, validation
