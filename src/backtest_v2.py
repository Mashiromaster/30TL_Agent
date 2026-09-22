# -*- coding: utf-8 -*-
"""Compatibility entry point for the versioned legacy rank strategy.

The former module carried a second, independently configurable strategy
implementation. That made command-line backtests disagree with inference and
the dashboard. It now delegates to the shared strategy runtime and refuses
ad-hoc parameter overrides; changing strategy behaviour requires a new
``StrategySpec`` and fingerprint.
"""

from __future__ import annotations

import os

import pandas as pd

from backtest import evaluate_predictions
from config import BASE_DIR
from model_release import ReleaseManager


LEGACY_RANK_STRATEGY_ID = "tl_rank_sweetspot_legacy_v1"


def _prediction_path(pred_file=None):
    if pred_file is not None:
        return os.fspath(pred_file)

    manager = ReleaseManager(
        os.path.join(BASE_DIR, "artifacts", "models"),
        legacy_model_path=os.path.join(BASE_DIR, "models", "trained_model.pkl"),
    )
    managed = manager.active_artifact_path("predictions")
    return os.fspath(managed) if managed is not None else None


def run_rank_backtest(pred_file=None, rank_pct=None, smooth_span=None):
    """Evaluate the registered legacy rank policy using managed predictions."""

    if rank_pct is not None or smooth_span is not None:
        raise ValueError(
            "ad-hoc rank/smoothing overrides are disabled; register a new "
            "versioned StrategySpec instead"
        )

    resolved = _prediction_path(pred_file)
    if resolved is None or not os.path.exists(resolved):
        print("[ERROR] 未找到托管 release 的预测 artifact")
        return None

    predictions = pd.read_pickle(resolved)
    tester, _, daily = evaluate_predictions(
        predictions,
        strategy_id=LEGACY_RANK_STRATEGY_ID,
    )
    display, _ = tester.compute_metrics()
    return daily, display


def sweep_params():
    """Prevent unversioned parameter search from silently changing policy."""

    raise RuntimeError(
        "inline parameter sweep is disabled; evaluate candidate StrategySpec "
        "versions through the sealed training/release pipeline"
    )


if __name__ == "__main__":
    result = run_rank_backtest()
    raise SystemExit(0 if result is not None else 1)
