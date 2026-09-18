import numpy as np
import pandas as pd

from factor_evaluator import FactorEvaluator, evaluate_factor


def _sample():
    rows = 90
    dates = pd.date_range("2025-01-01 09:30", periods=rows, freq="min")
    dates = dates.normalize() + pd.to_timedelta(np.arange(rows) % 30, unit="min")
    dates += pd.to_timedelta(np.arange(rows) // 30, unit="D")
    factor = np.tile([1.0, -1.0], rows // 2)
    return pd.DataFrame(
        {
            "date": dates,
            "factor": factor,
            "text": "x",
            "Target_Ret": factor,
            "fold": np.repeat([0, 1, 2], 30),
        }
    )


def test_evaluator_uses_prebuilt_target_without_extra_shift_or_roll():
    report = FactorEvaluator(_sample()).compute_ic("factor")

    assert report["ic"] == 1.0
    assert report["daily_ic_count"] == 3
    assert report["daily_ic_mean"] == 1.0


def test_factor_columns_are_numeric_only():
    evaluator = FactorEvaluator(_sample())
    assert "factor" in evaluator.factor_cols
    assert "text" not in evaluator.factor_cols


def test_fold_report_is_out_of_sample_aggregation():
    report = evaluate_factor(
        _sample(), "factor", "Target_Ret", fold_col="fold"
    )
    assert report["median_ic"] == 1.0
    assert report["same_sign_ratio"] == 1.0
    assert len(report["fold_ics"]) == 3


def test_correlation_matrix_always_returns_three_values():
    corr, factors, redundant = FactorEvaluator(_sample()).correlation_matrix(top_n=2)
    assert factors == ["factor"]
    assert redundant == []
    assert corr is None
