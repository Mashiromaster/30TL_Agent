# -*- coding: utf-8 -*-
"""Offline evaluation and fail-closed promotion gate for Jev semantics."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


class SemanticGateError(ValueError):
    """Raised when Jev semantic features are not ready for model experiments."""


def _macro_f1(actual: np.ndarray, predicted: np.ndarray, classes: list[str]) -> float:
    scores = []
    for label in classes:
        true_positive = np.sum((actual == label) & (predicted == label))
        false_positive = np.sum((actual != label) & (predicted == label))
        false_negative = np.sum((actual == label) & (predicted != label))
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = (
            true_positive / precision_denominator if precision_denominator else 0.0
        )
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        scores.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return float(np.mean(scores))


def _expected_calibration_error(
    confidence: np.ndarray,
    correct: np.ndarray,
    *,
    bins: int,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        if index == bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if not mask.any():
            continue
        error += float(mask.mean()) * abs(
            float(correct[mask].mean()) - float(confidence[mask].mean())
        )
    return error


def evaluate_choice_predictions(
    frame: pd.DataFrame,
    *,
    label_col: str,
    probability_columns: Mapping[str, str],
    ece_bins: int = 10,
) -> dict[str, float | int]:
    """Evaluate one held-out Choice prediction table.

    ``available_ts`` is required so every scored row retains its point-in-time
    identity. Rows with missing probabilities count against coverage and are
    excluded from quality metrics. Invalid distributions fail the evaluation.
    """
    if "available_ts" not in frame.columns:
        raise ValueError("evaluation frame must contain available_ts")
    if label_col not in frame.columns:
        raise ValueError(f"evaluation frame is missing label column: {label_col}")
    if not probability_columns:
        raise ValueError("probability_columns must not be empty")
    missing = [column for column in probability_columns.values() if column not in frame]
    if missing:
        raise ValueError(f"evaluation frame is missing probability columns: {missing}")
    if ece_bins < 2:
        raise ValueError("ece_bins must be at least 2")

    classes = list(probability_columns)
    probability_frame = frame[list(probability_columns.values())].apply(
        pd.to_numeric, errors="coerce"
    )
    availability = pd.to_datetime(frame["available_ts"], errors="coerce")
    complete = (
        probability_frame.notna().all(axis=1)
        & frame[label_col].isin(classes)
        & availability.notna()
    )
    coverage = float(complete.mean()) if len(frame) else 0.0
    if not complete.any():
        raise ValueError("evaluation frame has no complete labelled rows")

    probabilities = probability_frame.loc[complete].to_numpy(dtype=float)
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("probabilities must be in [0, 1]")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=0.02):
        raise ValueError("probabilities must sum to one")

    actual = frame.loc[complete, label_col].astype(str).to_numpy()
    predicted_indices = probabilities.argmax(axis=1)
    predicted = np.array(classes, dtype=object)[predicted_indices]
    correct = predicted == actual
    confidence = probabilities.max(axis=1)
    class_indices = {label: index for index, label in enumerate(classes)}
    actual_indices = np.array([class_indices[label] for label in actual])
    one_hot = np.eye(len(classes), dtype=float)[actual_indices]
    true_probabilities = probabilities[np.arange(len(probabilities)), actual_indices]

    return {
        "rows": int(len(probabilities)),
        "coverage": coverage,
        "accuracy": float(correct.mean()),
        "macro_f1": _macro_f1(actual, predicted, classes),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "log_loss": float(-np.log(np.clip(true_probabilities, 1e-12, 1.0)).mean()),
        "ece": _expected_calibration_error(confidence, correct, bins=ece_bins),
    }


def assert_semantic_promotion_ready(
    candidate: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    min_rows: int = 300,
    min_coverage: float = 0.95,
    min_f1_gain: float = 0.02,
    max_ece: float = 0.10,
) -> None:
    """Require Jev to beat the keyword baseline on a sealed semantic set."""
    required_candidate = {"rows", "coverage", "macro_f1", "brier", "ece"}
    required_baseline = {"macro_f1", "brier"}
    missing_candidate = sorted(required_candidate - set(candidate))
    missing_baseline = sorted(required_baseline - set(baseline))
    if missing_candidate or missing_baseline:
        raise SemanticGateError(
            f"missing metrics: candidate={missing_candidate}, baseline={missing_baseline}"
        )

    try:
        candidate_values = {
            name: float(candidate[name]) for name in required_candidate
        }
        baseline_values = {
            name: float(baseline[name]) for name in required_baseline
        }
    except (TypeError, ValueError) as exc:
        raise SemanticGateError("semantic metrics must be numeric") from exc
    if not all(
        np.isfinite(value)
        for value in (*candidate_values.values(), *baseline_values.values())
    ):
        raise SemanticGateError("semantic metrics must be finite")
    if not candidate_values["rows"].is_integer() or candidate_values["rows"] < 0:
        raise SemanticGateError("rows must be a non-negative integer")
    for name in ("coverage", "macro_f1", "ece"):
        if not 0.0 <= candidate_values[name] <= 1.0:
            raise SemanticGateError(f"{name} must be in [0, 1]")
    if not 0.0 <= baseline_values["macro_f1"] <= 1.0:
        raise SemanticGateError("baseline macro_f1 must be in [0, 1]")
    if candidate_values["brier"] < 0.0 or baseline_values["brier"] < 0.0:
        raise SemanticGateError("brier metrics must be non-negative")

    failures = {}
    if candidate_values["rows"] < min_rows:
        failures["rows"] = candidate_values["rows"]
    if candidate_values["coverage"] < min_coverage:
        failures["coverage"] = candidate_values["coverage"]
    f1_gain = candidate_values["macro_f1"] - baseline_values["macro_f1"]
    if f1_gain < min_f1_gain:
        failures["macro_f1_gain"] = f1_gain
    if candidate_values["brier"] >= baseline_values["brier"]:
        failures["brier"] = candidate_values["brier"]
    if candidate_values["ece"] > max_ece:
        failures["ece"] = candidate_values["ece"]
    if failures:
        raise SemanticGateError(f"Jev semantic promotion gate failed: {failures}")


def write_semantic_gate_report(
    output_path,
    candidate: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    model: str,
    question_schema_version: str,
    dataset_id: str,
) -> dict:
    """Atomically persist a version-bound semantic promotion decision."""
    if not model or not question_schema_version or not dataset_id:
        raise ValueError("model, question_schema_version, and dataset_id are required")
    failure = ""
    try:
        assert_semantic_promotion_ready(candidate, baseline)
        passed = True
    except SemanticGateError as exc:
        passed = False
        failure = str(exc)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(model),
        "question_schema_version": str(question_schema_version),
        "dataset_id": str(dataset_id),
        "candidate": dict(candidate),
        "baseline": dict(baseline),
        "thresholds": {
            "min_rows": 300,
            "min_coverage": 0.95,
            "min_f1_gain": 0.02,
            "max_ece": 0.10,
            "brier_must_improve": True,
        },
        "passed": passed,
        "failure": failure,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise
    return report


def load_semantic_gate_status(
    base_dir,
    text_factor_metadata: Mapping[str, object],
) -> tuple[bool, str, dict]:
    """Revalidate the local gate report against exact factor provenance."""
    path = Path(base_dir) / "outputs" / "jev_semantic_gate.json"
    if not path.exists():
        return False, "missing_report", {}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return False, "invalid_report", {}
    if report.get("schema_version") != 1:
        return False, "schema_mismatch", report
    if report.get("model") != text_factor_metadata.get("jev_model"):
        return False, "model_mismatch", report
    if report.get("question_schema_version") != text_factor_metadata.get(
        "jev_schema_version"
    ):
        return False, "question_schema_mismatch", report
    if not str(report.get("dataset_id", "")).strip():
        return False, "missing_dataset_id", report
    if report.get("passed") is not True:
        return False, "reported_failure", report
    try:
        assert_semantic_promotion_ready(report["candidate"], report["baseline"])
    except (KeyError, TypeError, SemanticGateError):
        return False, "metrics_failed", report
    return True, "passed", report
