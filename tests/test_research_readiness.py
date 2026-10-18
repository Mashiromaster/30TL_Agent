import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research_readiness import inspect_research_readiness, resolve_factor_path, verify_factor_provenance


NOW = pd.Timestamp("2026-10-04T00:00:00+08:00")


def factors(days=250):
    dates = [day + pd.Timedelta(hours=10, minutes=i)
             for day in pd.date_range("2025-01-01", periods=days, freq="B") for i in range(40)]
    return pd.DataFrame({"date": dates, "ticker": "TL2512", "close": 100 + np.arange(len(dates)) * .001,
                         "adj_close": 100 + np.arange(len(dates)) * .001,
                         "momentum": np.sin(np.arange(len(dates))), "Market_Regime": 0})


def candidate(base, frame):
    directory = base / "artifacts" / "factor_candidates" / "factor_test"
    directory.mkdir(parents=True)
    provenance = {"schema_version": 1, "kind": "local_causal_rebuild_v1",
                  "pit_contract": "previous_completed_observed_day_oi_v1",
                  "source_sha256": "a" * 64, "candidate_id": "factor_test", "limitations": []}
    frame = frame.copy()
    frame.attrs["factor_provenance"] = provenance
    path = directory / "factors.pkl"
    frame.to_pickle(path)
    manifest = {"schema_version": 1, "status": "built", "candidate_id": "factor_test",
                "source_file": "data/TL分钟级量价数据.pkl", "source_sha256": "a" * 64,
                "factor_file": "factors.pkl", "factor_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "provenance": provenance, "coverage": {}, "sources": {}, "limitations": []}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path, frame


def test_empty_project_check_is_read_only_and_actionable(tmp_path):
    result = inspect_research_readiness(tmp_path, now=NOW)
    assert result["status"] == "blocked"
    assert result["training"]["eligible"] is False
    assert result["model"]["status"] == "missing"
    assert result["next_actions"]
    assert list(tmp_path.iterdir()) == []


def test_legacy_factors_are_diagnosed_not_laundered_into_training(tmp_path):
    directory = tmp_path / "outputs"
    directory.mkdir()
    frame = factors()
    frame["CNN_0"] = 1.0
    path = directory / "df_factors.pkl"
    frame.to_pickle(path)
    before = path.read_bytes()
    result = inspect_research_readiness(tmp_path, now=NOW)
    assert result["factors"]["pit_verified"] is False
    assert result["training"]["valid_days"] == 250
    assert result["training"]["eligible"] is False
    assert any(item["id"] == "factor_provenance" and item["status"] == "blocked" for item in result["checks"])
    assert path.read_bytes() == before


def test_verified_historical_candidate_can_train_but_is_not_current_research(tmp_path):
    path, frame = candidate(tmp_path, factors())
    result = inspect_research_readiness(tmp_path, factor_path=path, now=NOW)
    assert result["factors"]["pit_verified"] is True
    assert result["training"]["eligible"] is True
    assert result["training"]["partitions"]["sealed_holdout"]["days"] == 60
    assert result["status"] == "blocked"
    assert result["model"]["status"] == "missing"
    assert result["factors"]["age_hours"] > 24
    assert verify_factor_provenance(tmp_path, path, frame)["verified"] is True


def test_sparse_days_are_not_counted_as_labelled_training_days(tmp_path):
    frame = factors()
    sparse = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=60, freq="B") + pd.Timedelta(hours=15),
                           "ticker": "TL2603", "close": 110., "adj_close": 110., "momentum": 0., "Market_Regime": 0})
    path, _ = candidate(tmp_path, pd.concat([frame, sparse], ignore_index=True))
    result = inspect_research_readiness(tmp_path, factor_path=path, now=NOW)
    assert result["factors"]["observed_days"] == 310
    assert result["training"]["valid_days"] == 250
    assert result["training"]["days_without_valid_labels"] == 60


@pytest.mark.parametrize("change", ["hash", "contract", "identity", "status", "attrs", "manifest_missing"])
def test_provenance_fails_closed(tmp_path, change):
    path, frame = candidate(tmp_path, factors(3))
    manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if change == "hash":
        manifest["factor_sha256"] = "b" * 64
    elif change == "contract":
        manifest["provenance"]["pit_contract"] = "trust_me"
    elif change == "identity":
        manifest["candidate_id"] = "other"
    elif change == "status":
        manifest["status"] = "failed"
    elif change == "attrs":
        frame.attrs = {}
    else:
        manifest_path.unlink()
    if change != "manifest_missing":
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_factor_provenance(tmp_path, path, frame)["verified"] is False


def test_factor_paths_cannot_escape_project(tmp_path):
    with pytest.raises(ValueError):
        resolve_factor_path(tmp_path, "../outside.pkl")
    with pytest.raises(ValueError):
        resolve_factor_path(tmp_path, "models/trained_model.pkl")
    outside = tmp_path / "outside"
    outside.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    (project / "outputs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        resolve_factor_path(project, "outputs/df_factors.pkl")


def test_future_rows_are_reported_and_cannot_make_training_ready(tmp_path):
    path, _ = candidate(tmp_path, factors())
    result = inspect_research_readiness(tmp_path, factor_path=path, now="2025-02-01")
    assert result["training"]["eligible"] is False
    assert any(item["id"] == "factor_time" and item["status"] == "blocked" for item in result["checks"])


def test_verified_managed_model_schema_allows_ready_state(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import research_readiness as readiness
    from strategy_runtime import ModelRuntime, get_strategy_spec
    from sklearn.preprocessing import StandardScaler
    path, frame = candidate(tmp_path, factors())
    runtime = ModelRuntime({"model_base": object(), "scaler": StandardScaler(), "features": ["momentum"],
                            "artifact_schema_version": 3, "model_release_id": "verified-release"})
    signature = SimpleNamespace(source="release", release_id="verified-release", manifest_path=tmp_path / "manifest.json",
                                artifact_path=tmp_path / "model.pkl")
    monkeypatch.setattr(readiness.ReleaseManager, "current_signature", lambda self: signature)
    monkeypatch.setattr(readiness, "load_manifest", lambda *a, **kw: {"metadata": {"strategy_fingerprint": get_strategy_spec().fingerprint()}})
    monkeypatch.setattr(readiness.ModelRuntime, "from_path", lambda *a: runtime)
    result = inspect_research_readiness(tmp_path, factor_path=path, now=frame.date.max() + pd.Timedelta(minutes=5))
    assert result["status"] == "ready"
    assert result["model"]["status"] == "verified"
    assert result["execution_authorized"] is False


def test_managed_model_strategy_mismatch_is_not_loaded(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import research_readiness as readiness
    signature = SimpleNamespace(source="release", release_id="wrong", manifest_path=tmp_path / "manifest.json")
    monkeypatch.setattr(readiness.ReleaseManager, "current_signature", lambda self: signature)
    monkeypatch.setattr(readiness, "load_manifest", lambda *a, **kw: {"metadata": {"strategy_fingerprint": "wrong"}})
    loader = Mock()
    monkeypatch.setattr(readiness.ModelRuntime, "from_path", loader)
    with pytest.raises(ValueError, match="strategy fingerprint"):
        readiness.resolve_managed_runtime(tmp_path)
    loader.assert_not_called()


def test_loaded_snapshot_digest_must_match_manifest_even_if_disk_replaced(tmp_path):
    path, frame = candidate(tmp_path, factors(3))
    original_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    replacement = frame.copy()
    replacement["momentum"] = 42.
    replacement.to_pickle(path)
    manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["factor_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    result = verify_factor_provenance(tmp_path, path, frame, expected_sha256=original_digest)
    assert result["verified"] is False


def test_readiness_surfaces_unverified_calendar_and_large_coverage_gap(tmp_path):
    path, _ = candidate(tmp_path, factors())
    manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["coverage"] = {"largest_observed_day_gap_days": 207, "quarantined_days": 135,
                            "calendar_verified": False}
    manifest_path.write_text(json.dumps(manifest))
    result = inspect_research_readiness(tmp_path, factor_path=path, now=NOW)
    assert result["factors"]["provenance"]["coverage"]["quarantined_days"] == 135
    assert any(item["id"] == "coverage_gap" and item["status"] == "warning" for item in result["checks"])
    assert result["training"]["eligible"] is True  # Historical candidate only, not production approval.
