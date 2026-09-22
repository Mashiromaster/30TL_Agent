import pandas as pd
import pytest

from jev_evaluation import (
    SemanticGateError,
    assert_semantic_promotion_ready,
    evaluate_choice_predictions,
    load_semantic_gate_status,
    write_semantic_gate_report,
)


def test_choice_evaluation_reports_calibration_and_classification_metrics():
    frame = pd.DataFrame(
        {
            "available_ts": pd.date_range("2025-01-01", periods=4),
            "label": ["bullish", "neutral", "bearish", "unclear"],
            "p_bullish": [0.9, 0.05, 0.05, 0.05],
            "p_neutral": [0.03, 0.85, 0.05, 0.05],
            "p_bearish": [0.02, 0.05, 0.85, 0.05],
            "p_unclear": [0.05, 0.05, 0.05, 0.85],
        }
    )

    metrics = evaluate_choice_predictions(
        frame,
        label_col="label",
        probability_columns={
            "bullish": "p_bullish",
            "neutral": "p_neutral",
            "bearish": "p_bearish",
            "unclear": "p_unclear",
        },
    )

    assert metrics["rows"] == 4
    assert metrics["coverage"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert 0.0 <= metrics["ece"] <= 0.15
    assert metrics["brier"] < 0.05


def test_choice_evaluation_rejects_invalid_probability_rows():
    frame = pd.DataFrame(
        {
            "available_ts": ["2025-01-01"],
            "label": ["bullish"],
            "p_bullish": [0.8],
            "p_bearish": [0.8],
        }
    )

    with pytest.raises(ValueError, match="sum to one"):
        evaluate_choice_predictions(
            frame,
            label_col="label",
            probability_columns={"bullish": "p_bullish", "bearish": "p_bearish"},
        )


def test_choice_evaluation_counts_invalid_availability_against_coverage():
    frame = pd.DataFrame(
        {
            "available_ts": ["2025-01-01", "not-a-timestamp"],
            "label": ["bullish", "bearish"],
            "p_bullish": [0.9, 0.1],
            "p_bearish": [0.1, 0.9],
        }
    )

    metrics = evaluate_choice_predictions(
        frame,
        label_col="label",
        probability_columns={"bullish": "p_bullish", "bearish": "p_bearish"},
    )

    assert metrics["rows"] == 1
    assert metrics["coverage"] == 0.5


def test_semantic_gate_fails_closed_without_enough_rows_or_baseline_gain():
    candidate = {
        "rows": 299,
        "coverage": 1.0,
        "macro_f1": 0.82,
        "brier": 0.20,
        "ece": 0.05,
    }
    baseline = {"macro_f1": 0.81, "brier": 0.21}

    with pytest.raises(SemanticGateError, match="rows"):
        assert_semantic_promotion_ready(candidate, baseline)

    candidate["rows"] = 300
    with pytest.raises(SemanticGateError, match="macro_f1_gain"):
        assert_semantic_promotion_ready(candidate, baseline)


def test_semantic_gate_accepts_calibrated_candidate_that_beats_baseline():
    candidate = {
        "rows": 300,
        "coverage": 0.98,
        "macro_f1": 0.84,
        "brier": 0.18,
        "ece": 0.08,
    }
    baseline = {"macro_f1": 0.81, "brier": 0.21}

    assert_semantic_promotion_ready(candidate, baseline)


def test_semantic_gate_rejects_non_finite_or_out_of_range_metrics():
    baseline = {"macro_f1": 0.81, "brier": 0.21}
    candidate = {
        "rows": 300,
        "coverage": 0.98,
        "macro_f1": float("nan"),
        "brier": 0.18,
        "ece": 0.08,
    }

    with pytest.raises(SemanticGateError, match="finite"):
        assert_semantic_promotion_ready(candidate, baseline)

    candidate["macro_f1"] = 0.84
    candidate["coverage"] = 1.2
    with pytest.raises(SemanticGateError, match="coverage"):
        assert_semantic_promotion_ready(candidate, baseline)


def test_semantic_gate_report_is_revalidated_and_version_matched(tmp_path):
    candidate = {
        "rows": 300,
        "coverage": 0.98,
        "macro_f1": 0.84,
        "brier": 0.18,
        "ece": 0.08,
    }
    baseline = {"macro_f1": 0.81, "brier": 0.21}
    path = tmp_path / "outputs" / "jev_semantic_gate.json"

    report = write_semantic_gate_report(
        path,
        candidate,
        baseline,
        model="jev-1.13.0",
        question_schema_version="macro-document-v1",
        dataset_id="sealed-cn-macro-v1-sha256:abc123",
    )
    passed, reason, loaded = load_semantic_gate_status(
        tmp_path,
        {
            "jev_model": "jev-1.13.0",
            "jev_schema_version": "macro-document-v1",
        },
    )

    assert report["passed"] is True
    assert passed is True
    assert reason == "passed"
    assert loaded["dataset_id"] == "sealed-cn-macro-v1-sha256:abc123"

    passed, reason, _ = load_semantic_gate_status(
        tmp_path,
        {
            "jev_model": "jev-1.14.0",
            "jev_schema_version": "macro-document-v1",
        },
    )
    assert passed is False
    assert reason == "model_mismatch"
