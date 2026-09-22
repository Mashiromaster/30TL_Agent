import numpy as np
import pandas as pd

from LightGBM_model import (
    _apply_jev_semantic_gate,
    _date_block_stability,
    _release_gate_passed,
    _validation_promotion_eligible,
    prepare_training_data,
)
from training_data import LabelSpec
from jev_evaluation import write_semantic_gate_report


def test_prepare_training_data_uses_registry_and_adjusted_price_labels():
    rows = 80
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=rows, freq="min"),
            "ticker": "TL",
            "close": np.r_[np.arange(40.0), np.arange(1000.0, 1040.0)],
            "adj_close": np.arange(100.0, 180.0),
            "Spread_Mean": np.linspace(1, 2, rows),
            "YC_Slope_10Y_1Y": np.linspace(0.1, 0.2, rows),
            "GP_001_alpha": np.linspace(-1, 1, rows),
            "constant": 1.0,
        }
    )
    frame.attrs["text_factor_metadata"] = {
        "jev_model": "jev-1.13.0",
        "jev_mode": "cache",
    }

    model, features = prepare_training_data(
        frame,
        label_spec=LabelSpec(
            horizon_bars=2, entry_lag_bars=0, allow_overnight=True
        ),
    )

    expected = (102.0 / 100.0 - 1) * 100
    assert model.iloc[0]["Target_Ret"] == expected
    assert "Spread_Mean" in features
    assert "YC_Slope_10Y_1Y" in features
    assert "GP_001_alpha" in features
    assert "constant" not in features
    assert model.attrs["text_factor_metadata"] == frame.attrs["text_factor_metadata"]


def test_validation_stability_uses_five_contiguous_date_blocks():
    days = pd.date_range("2025-01-01", periods=10, freq="D")
    frame = pd.DataFrame(
        {
            "date": np.repeat(days, 12),
            "Target_Ret": np.tile(np.arange(12.0), 10),
        }
    )
    prediction = frame["Target_Ret"].to_numpy()

    report = _date_block_stability(frame, prediction, blocks=5)

    assert report["fold_count"] == 5
    assert report["positive_fold_count"] == 5
    assert report["same_sign_ratio"] == 1.0


def test_sealed_holdout_can_only_veto_a_validation_qualified_candidate():
    stable = {
        "fold_count": 5,
        "positive_fold_count": 3,
        "median_ic": 0.01,
    }
    eligible = _validation_promotion_eligible(stable)

    assert eligible is True
    assert _release_gate_passed(eligible, 0.005) is True
    assert _release_gate_passed(eligible, -0.001) is False
    assert _release_gate_passed(False, 0.10) is False


def test_training_drops_jev_features_without_matching_semantic_gate(tmp_path):
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=3),
            "JEV_Bond_Direction": [0.1, 0.2, 0.3],
            "Policy_Sentiment": [0.0, 0.1, 0.0],
        }
    )
    frame.attrs["text_factor_metadata"] = {
        "jev_model": "jev-1.13.0",
        "jev_schema_version": "macro-document-v1",
        "jev_features": ["JEV_Bond_Direction"],
    }

    gated, provenance = _apply_jev_semantic_gate(frame, tmp_path)

    assert "JEV_Bond_Direction" not in gated.columns
    assert "Policy_Sentiment" in gated.columns
    assert provenance["semantic_gate_passed"] is False
    assert provenance["semantic_gate_reason"] == "missing_report"
    assert provenance["training_jev_features"] == []


def test_training_keeps_jev_features_after_matching_semantic_gate(tmp_path):
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=3),
            "JEV_Bond_Direction": [0.1, 0.2, 0.3],
        }
    )
    frame.attrs["text_factor_metadata"] = {
        "jev_model": "jev-1.13.0",
        "jev_schema_version": "macro-document-v1",
        "jev_features": ["JEV_Bond_Direction"],
    }
    write_semantic_gate_report(
        tmp_path / "outputs" / "jev_semantic_gate.json",
        {
            "rows": 300,
            "coverage": 0.98,
            "macro_f1": 0.84,
            "brier": 0.18,
            "ece": 0.08,
        },
        {"macro_f1": 0.81, "brier": 0.21},
        model="jev-1.13.0",
        question_schema_version="macro-document-v1",
        dataset_id="sealed-cn-macro-v1-sha256:abc123",
    )

    gated, provenance = _apply_jev_semantic_gate(frame, tmp_path)

    assert "JEV_Bond_Direction" in gated.columns
    assert provenance["semantic_gate_passed"] is True
    assert provenance["semantic_gate_reason"] == "passed"
    assert provenance["training_jev_features"] == ["JEV_Bond_Direction"]
