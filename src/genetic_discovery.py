# -*- coding: utf-8 -*-
"""Explicit, leakage-safe GP research runs, separate from deployed model data."""

from __future__ import annotations

import copy
import json
import pickle
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from feature_registry import EXCLUDED
from genetic_factor_miner import GeneticFactorMiner, mine_genetic_factors
from training_data import (
    LabelSpec,
    SplitSpec,
    build_forward_labels,
    split_train_validation_holdout,
)


@dataclass
class DiscoveryResult:
    miner: GeneticFactorMiner
    report: dict
    factors: pd.DataFrame


def _validated_input(frame: pd.DataFrame, label_spec: LabelSpec) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("factor input must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("factor input is empty")
    if not frame.columns.is_unique:
        raise ValueError("duplicate factor column names are not supported")
    required = {label_spec.timestamp_col, label_spec.ticker_col, label_spec.price_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing discovery inputs: {sorted(missing)}")

    # Remove stale labels, prior predictions and known look-ahead fields before
    # rebuilding labels. Only the explicit price/time/contract inputs survive.
    excluded = {name.lower() for name in EXCLUDED}
    columns = [name for name in frame.columns if name in required or (
        name.lower() not in excluded
        and not name.lower().startswith(("target_", "label_", "pred_")))]
    work = frame.loc[:, columns].copy().reset_index(drop=True)
    try:
        dates = pd.to_datetime(work[label_spec.timestamp_col], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label_spec.timestamp_col} timestamps") from exc
    if dates.isna().any():
        raise ValueError(f"invalid {label_spec.timestamp_col}: missing timestamps")
    work[label_spec.timestamp_col] = dates
    tickers = work[label_spec.ticker_col]
    if tickers.isna().any() or tickers.astype(str).str.strip().eq("").any():
        raise ValueError(f"invalid {label_spec.ticker_col}: missing contract identities")
    prices = pd.to_numeric(work[label_spec.price_col], errors="coerce")
    if not (np.isfinite(prices) & prices.gt(0)).all():
        raise ValueError(f"invalid {label_spec.price_col}: prices must be finite and positive")
    work[label_spec.price_col] = prices
    if work.duplicated([label_spec.timestamp_col, label_spec.ticker_col]).any():
        raise ValueError("duplicate timestamp/contract observations in discovery inputs")
    return work.sort_values([label_spec.timestamp_col, label_spec.ticker_col]).reset_index(drop=True)


def _prepare(frame, label_spec, split_spec):
    for name in ("validation_days", "sealed_holdout_days", "min_train_days"):
        value = getattr(split_spec, name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    work = _validated_input(frame, label_spec)
    labelled = build_forward_labels(work, label_spec)
    # Match the training pipeline: sparse dates with no usable forward labels
    # cannot occupy a validation/holdout slot. Report them, never fill them.
    valid = labelled["label_valid"] & np.isfinite(labelled["Target_Ret"])
    labelled = labelled.loc[valid].copy()
    input_days = work[label_spec.timestamp_col].dt.normalize().nunique()
    labeled_days = labelled[label_spec.timestamp_col].dt.normalize().nunique()
    quality = {
        "input_rows": int(len(work)),
        "labeled_rows": int(len(labelled)),
        "dropped_label_rows": int(len(work) - len(labelled)),
        "input_days": int(input_days),
        "labeled_days": int(labeled_days),
        "days_without_valid_labels": int(input_days - labeled_days),
    }
    partitions = split_train_validation_holdout(
        labelled, split_spec, timestamp_col=label_spec.timestamp_col)
    return partitions, quality


def prepare_discovery(
    frame: pd.DataFrame,
    *,
    label_spec: LabelSpec = LabelSpec(),
    split_spec: SplitSpec = SplitSpec(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rebuild targets and return disjoint purged train/validation/holdout frames."""
    partitions, _ = _prepare(frame, label_spec, split_spec)
    return partitions


def _partition_summary(frame: pd.DataFrame, timestamp_col: str) -> dict:
    dates = frame[timestamp_col]
    return {"rows": int(len(frame)), "start": dates.min().isoformat(),
            "end": dates.max().isoformat(), "days": int(dates.dt.normalize().nunique())}


def run_discovery(
    frame: pd.DataFrame,
    *,
    population_size=1500,
    generations=15,
    max_new_factors=20,
    min_ic_threshold=0.01,
    label_spec: LabelSpec = LabelSpec(),
    split_spec: SplitSpec = SplitSpec(),
    n_jobs=1,
) -> DiscoveryResult:
    """Discover on train, select on validation, and leave holdout unevaluated.

    This operation returns research artifacts in memory. It does not persist or
    activate factors in any training, model, or execution pipeline.
    """
    (train, validation, holdout), quality = _prepare(frame, label_spec, split_spec)
    # The miner excludes canonical metadata names from feature selection. Honor
    # configurable LabelSpec names without accidentally treating price as alpha.
    metadata_names = {label_spec.timestamp_col: "date", label_spec.ticker_col: "ticker",
                      label_spec.price_col: "adj_close"}
    miner, miner_report = mine_genetic_factors(
        train.rename(columns=metadata_names), validation.rename(columns=metadata_names),
        target_col="Target_Ret",
        population_size=population_size, generations=generations,
        max_new_factors=max_new_factors, min_ic_threshold=min_ic_threshold,
        n_jobs=n_jobs,
    )
    report = copy.deepcopy(miner_report)
    discovery = pd.concat([train, validation], ignore_index=True)
    factors = discovery[[label_spec.timestamp_col, label_spec.ticker_col]].rename(
        columns={label_spec.timestamp_col: "date", label_spec.ticker_col: "ticker"}).copy()
    factors["partition"] = ["train"] * len(train) + ["validation"] * len(validation)
    programs = report["programs"]
    if programs:
        # Never transform (or calculate IC on) the sealed holdout here.
        generated = miner.transform(discovery[report["feature_names"]])
        factors = pd.concat([factors, generated], axis=1)
    report.update({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label_spec": asdict(label_spec),
        "split_spec": asdict(split_spec),
        "mining_config": {"population_size": population_size,
                          "generations": generations,
                          "max_new_factors": max_new_factors,
                          "min_ic_threshold": min_ic_threshold, "n_jobs": n_jobs},
        "partitions": {name: _partition_summary(part, label_spec.timestamp_col)
                       for name, part in (("train", train), ("validation", validation),
                                          ("sealed_holdout", holdout))},
        "data_quality": quality,
        "n_final": len(programs),
        "holdout_evaluated": False,
    })
    return DiscoveryResult(miner=miner, report=report, factors=factors)


def save_discovery(result: DiscoveryResult, base_dir: str | Path) -> Path:
    """Publish a unique research-only directory after all three files succeed."""
    base = Path(base_dir).resolve()
    root = base / "outputs" / "genetic_factors"
    if not root.resolve().is_relative_to(base):
        raise ValueError("research output directory resolves outside project base")
    root.mkdir(parents=True, exist_ok=True)
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex
    destination = root / name
    # Staging is created beside the final directory, so rename is atomic. Failed
    # serialization is cleaned by TemporaryDirectory and is never published.
    with tempfile.TemporaryDirectory(prefix=".pending-", dir=root) as directory:
        staging = Path(directory)
        result.factors.to_pickle(staging / "factors.pkl")
        with (staging / "miner.pkl").open("wb") as handle:
            pickle.dump(result.miner, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with (staging / "report.json").open("w", encoding="utf-8") as handle:
            json.dump(result.report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        staging.rename(destination)
    return destination
