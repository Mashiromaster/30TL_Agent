import hashlib
import json
import os
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor

import pytest

from model_release import (
    ArtifactIntegrityError,
    GateRejectedError,
    ManifestValidationError,
    ReleaseManager,
    load_manifest,
    write_candidate_manifest,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _candidate(
    tmp_path: Path,
    release_id: str,
    model_bytes: bytes,
    *,
    gate_passed: bool = True,
    strategy_fingerprint: str = "sha256:test-strategy",
) -> Path:
    candidate_dir = tmp_path / "experiments" / release_id
    model_path = candidate_dir / "model" / "trained_model.pkl"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(model_bytes)
    model_sha = _sha256(model_bytes)
    run_id = f"run-{release_id}"
    (candidate_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "release_id": release_id,
                "run_id": run_id,
                "strategy_id": "tl_regime_breakout_v1",
                "strategy_fingerprint": strategy_fingerprint,
                "min_total_return": 0.0,
                "min_sharpe": 0.0,
                "max_drawdown_limit": 0.35,
                "promotion_eligible": gate_passed,
                "release_gate_passed": gate_passed,
                "strategy_gate_passed": gate_passed,
                "sealed_holdout_ic": 0.03 if gate_passed else -0.01,
                "strategy_backtest": {
                    "strategy_id": "tl_regime_breakout_v1",
                    "strategy_fingerprint": strategy_fingerprint,
                    "model_release_id": release_id,
                    "observation_days": 60,
                    "total_return": 0.05,
                    "sharpe": 0.8,
                    "annual_volatility": 0.1,
                    "max_drawdown": -0.1,
                    "active_ratio": 0.5,
                },
                "gate_attestation": {
                    "schema_version": 1,
                    "release_id": release_id,
                    "run_id": run_id,
                    "strategy_id": "tl_regime_breakout_v1",
                    "strategy_fingerprint": strategy_fingerprint,
                    "model_sha256": model_sha,
                    "gate_passed": gate_passed,
                    "policy": {
                        "min_observation_days": 20,
                        "min_total_return": 0.0,
                        "min_sharpe": 0.0,
                        "max_drawdown_limit": 0.35,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return write_candidate_manifest(
        candidate_dir,
        release_id=release_id,
        run_id=run_id,
        artifacts={
            "model": "model/trained_model.pkl",
            "evaluation": "evaluation.json",
        },
        metadata={
            "strategy_id": "tl_regime_breakout_v1",
            "strategy_version": "tl-v1",
            "strategy_fingerprint": strategy_fingerprint,
            "release_gate_passed": gate_passed,
        },
    )


def test_candidate_manifest_records_relative_artifacts_and_hashes(tmp_path):
    manifest_path = _candidate(tmp_path, "rel-001", b"candidate-model-v1")

    manifest = load_manifest(manifest_path, verify_artifacts=True)

    assert manifest["manifest_type"] == "candidate"
    assert manifest["release_id"] == "rel-001"
    assert manifest["run_id"] == "run-rel-001"
    assert manifest["metadata"] == {
        "strategy_id": "tl_regime_breakout_v1",
        "strategy_version": "tl-v1",
        "strategy_fingerprint": "sha256:test-strategy",
        "release_gate_passed": True,
    }
    assert manifest["artifacts"]["model"] == {
        "path": "model/trained_model.pkl",
        "sha256": _sha256(b"candidate-model-v1"),
        "size_bytes": len(b"candidate-model-v1"),
    }


def test_promote_requires_passing_hashed_gate_attestation(tmp_path):
    manifest_path = _candidate(
        tmp_path, "rel-001", b"candidate-model-v1", gate_passed=False
    )
    manager = ReleaseManager(tmp_path / "artifacts")

    with pytest.raises(GateRejectedError, match="metadata"):
        manager.promote(manifest_path)

    assert not manager.current_path.exists()
    assert not (manager.releases_dir / "rel-001").exists()


def test_caller_cannot_override_gate_with_a_boolean(tmp_path):
    manifest_path = _candidate(tmp_path, "rel-001", b"candidate-model-v1")
    manager = ReleaseManager(tmp_path / "artifacts")

    with pytest.raises(TypeError):
        manager.promote(manifest_path, gate_passed=True)


def test_gate_attestation_must_match_strategy_fingerprint(tmp_path):
    manifest_path = _candidate(tmp_path, "rel-001", b"candidate-model-v1")
    evaluation_path = manifest_path.parent / "evaluation.json"
    report = json.loads(evaluation_path.read_text(encoding="utf-8"))
    report["gate_attestation"]["strategy_fingerprint"] = "sha256:wrong"
    evaluation_path.write_text(json.dumps(report), encoding="utf-8")
    manifest_path = write_candidate_manifest(
        manifest_path.parent,
        release_id="rel-001",
        run_id="run-rel-001",
        artifacts={
            "model": "model/trained_model.pkl",
            "evaluation": "evaluation.json",
        },
        metadata={
            "strategy_id": "tl_regime_breakout_v1",
            "strategy_version": "tl-v1",
            "strategy_fingerprint": "sha256:test-strategy",
            "release_gate_passed": True,
        },
    )

    with pytest.raises(GateRejectedError, match="fingerprint mismatch"):
        ReleaseManager(tmp_path / "artifacts").promote(manifest_path)


def test_gate_attestation_cannot_weaken_baseline_policy(tmp_path):
    manifest_path = _candidate(tmp_path, "rel-001", b"candidate-model-v1")
    evaluation_path = manifest_path.parent / "evaluation.json"
    report = json.loads(evaluation_path.read_text(encoding="utf-8"))
    report["min_total_return"] = -0.01
    report["gate_attestation"]["policy"]["min_total_return"] = -0.01
    evaluation_path.write_text(json.dumps(report), encoding="utf-8")
    manifest_path = write_candidate_manifest(
        manifest_path.parent,
        release_id="rel-001",
        run_id="run-rel-001",
        artifacts={
            "model": "model/trained_model.pkl",
            "evaluation": "evaluation.json",
        },
        metadata={
            "strategy_id": "tl_regime_breakout_v1",
            "strategy_version": "tl-v1",
            "strategy_fingerprint": "sha256:test-strategy",
            "release_gate_passed": True,
        },
    )

    with pytest.raises(GateRejectedError, match="weakens total return"):
        ReleaseManager(tmp_path / "artifacts").promote(manifest_path)


def test_backtest_metrics_must_bind_to_release_id(tmp_path):
    manifest_path = _candidate(tmp_path, "rel-001", b"candidate-model-v1")
    evaluation_path = manifest_path.parent / "evaluation.json"
    report = json.loads(evaluation_path.read_text(encoding="utf-8"))
    report["strategy_backtest"]["model_release_id"] = "rel-other"
    evaluation_path.write_text(json.dumps(report), encoding="utf-8")
    manifest_path = write_candidate_manifest(
        manifest_path.parent,
        release_id="rel-001",
        run_id="run-rel-001",
        artifacts={
            "model": "model/trained_model.pkl",
            "evaluation": "evaluation.json",
        },
        metadata={
            "strategy_id": "tl_regime_breakout_v1",
            "strategy_version": "tl-v1",
            "strategy_fingerprint": "sha256:test-strategy",
            "release_gate_passed": True,
        },
    )

    with pytest.raises(GateRejectedError, match="model release mismatch"):
        ReleaseManager(tmp_path / "artifacts").promote(manifest_path)


def test_candidate_outside_configured_experiments_root_is_rejected(tmp_path):
    manifest_path = _candidate(tmp_path, "rel-001", b"candidate-model-v1")
    rogue_dir = tmp_path / "rogue" / "rel-001"
    rogue_dir.parent.mkdir()
    shutil.move(str(manifest_path.parent), rogue_dir)

    with pytest.raises(
        ManifestValidationError,
        match="outside the configured experiments root",
    ):
        ReleaseManager(tmp_path / "artifacts").promote(
            rogue_dir / "manifest.json"
        )


def test_promote_writes_release_manifest_and_switches_current(tmp_path):
    manifest_path = _candidate(tmp_path, "rel-001", b"candidate-model-v1")
    manager = ReleaseManager(tmp_path / "artifacts")

    signature = manager.promote(manifest_path)

    release_dir = manager.releases_dir / "rel-001"
    release_manifest = load_manifest(
        release_dir / "manifest.json", verify_artifacts=True
    )
    pointer = json.loads(manager.current_path.read_text(encoding="utf-8"))
    assert release_manifest["manifest_type"] == "release"
    assert release_manifest["gate_passed"] is True
    assert release_manifest["source_candidate_sha256"] == pointer[
        "source_candidate_sha256"
    ]
    assert pointer["release_id"] == "rel-001"
    assert signature == manager.current_signature()
    assert signature.source == "release"
    assert signature.read_only is False
    assert signature.artifact_sha256 == _sha256(b"candidate-model-v1")


def test_tampered_candidate_is_rejected_without_switching_current(tmp_path):
    manifest_path = _candidate(tmp_path, "rel-001", b"candidate-model-v1")
    (manifest_path.parent / "model" / "trained_model.pkl").write_bytes(
        b"candidate-model-v2"
    )
    manager = ReleaseManager(tmp_path / "artifacts")

    with pytest.raises(ArtifactIntegrityError, match="sha256"):
        manager.promote(manifest_path)

    assert not manager.current_path.exists()
    assert not (manager.releases_dir / "rel-001").exists()


@pytest.mark.parametrize("interruption", [OSError("disk error"), KeyboardInterrupt()])
def test_failed_or_interrupted_switch_keeps_previous_release_and_legacy_projection(
    tmp_path, interruption
):
    legacy_path = tmp_path / "models" / "trained_model.pkl"
    manager = ReleaseManager(
        tmp_path / "artifacts", legacy_model_path=legacy_path
    )
    first = _candidate(tmp_path, "rel-001", b"candidate-model-v1")
    second = _candidate(tmp_path, "rel-002", b"candidate-model-v2")
    manager.promote(first, project_legacy=True)

    real_replace = os.replace
    failed = False

    def interrupt_current_switch(source, destination):
        nonlocal failed
        if Path(destination) == manager.current_path and not failed:
            failed = True
            raise interruption
        return real_replace(source, destination)

    manager._replace = interrupt_current_switch
    with pytest.raises(type(interruption)):
        manager.promote(second, project_legacy=True)

    assert manager.current_signature().release_id == "rel-001"
    assert legacy_path.read_bytes() == b"candidate-model-v1"
    assert load_manifest(
        manager.releases_dir / "rel-001" / "manifest.json",
        verify_artifacts=True,
    )["release_id"] == "rel-001"


def test_legacy_projection_is_optional(tmp_path):
    legacy_path = tmp_path / "models" / "trained_model.pkl"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_bytes(b"existing-legacy")
    manager = ReleaseManager(
        tmp_path / "artifacts", legacy_model_path=legacy_path
    )

    manager.promote(
        _candidate(tmp_path, "rel-001", b"candidate-model-v1"),
        project_legacy=False,
    )

    assert legacy_path.read_bytes() == b"existing-legacy"


def test_missing_release_metadata_exposes_read_only_legacy_signature(tmp_path):
    legacy_path = tmp_path / "models" / "trained_model.pkl"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_bytes(b"legacy-without-release-metadata")
    manager = ReleaseManager(
        tmp_path / "artifacts", legacy_model_path=legacy_path
    )

    signature = manager.current_signature()

    assert signature.source == "legacy"
    assert signature.read_only is True
    assert signature.release_id is None
    assert signature.artifact_path == legacy_path.resolve()
    assert signature.artifact_sha256 == _sha256(
        b"legacy-without-release-metadata"
    )
    assert not manager.current_path.exists()
    assert not manager.artifacts_root.exists()


def test_corrupt_active_release_does_not_silently_fall_back_to_legacy(tmp_path):
    legacy_path = tmp_path / "models" / "trained_model.pkl"
    manager = ReleaseManager(
        tmp_path / "artifacts", legacy_model_path=legacy_path
    )
    manager.promote(
        _candidate(tmp_path, "rel-001", b"candidate-model-v1"),
        project_legacy=True,
    )
    active_model = manager.releases_dir / "rel-001" / "model" / "trained_model.pkl"
    active_model.write_bytes(b"corrupt-active-release")

    with pytest.raises(ArtifactIntegrityError):
        manager.current_signature()


def test_active_artifact_resolution_and_projection_are_verified(tmp_path):
    manager = ReleaseManager(tmp_path / "artifacts")
    manifest_path = _candidate(tmp_path, "rel-001", b"candidate-model-v1")
    manager.promote(manifest_path)

    evaluation = manager.active_artifact_path("evaluation")
    evaluation_data = json.loads(evaluation.read_text(encoding="utf-8"))
    assert evaluation_data["release_gate_passed"] is True
    assert evaluation_data["gate_attestation"]["model_sha256"] == _sha256(
        b"candidate-model-v1"
    )

    destination = tmp_path / "outputs" / "evaluation.json"
    manager.project_active_artifact("evaluation", destination)
    assert destination.read_bytes() == evaluation.read_bytes()
    assert manager.active_artifact_path("missing") is None


def test_previous_immutable_release_can_be_reactivated_atomically(tmp_path):
    legacy_path = tmp_path / "models" / "trained_model.pkl"
    manager = ReleaseManager(
        tmp_path / "artifacts", legacy_model_path=legacy_path
    )
    manager.promote(
        _candidate(tmp_path, "rel-001", b"candidate-model-v1"),
        project_legacy=True,
    )
    manager.promote(
        _candidate(tmp_path, "rel-002", b"candidate-model-v2"),
        project_legacy=True,
    )

    signature = manager.activate_release("rel-001", project_legacy=True)

    assert signature.release_id == "rel-001"
    assert legacy_path.read_bytes() == b"candidate-model-v1"


def test_legacy_projection_is_not_authoritative_for_managed_runtime(tmp_path):
    legacy_path = tmp_path / "models" / "trained_model.pkl"
    manager = ReleaseManager(
        tmp_path / "artifacts", legacy_model_path=legacy_path
    )
    manager.promote(
        _candidate(tmp_path, "rel-001", b"candidate-model-v1"),
        project_legacy=True,
    )
    legacy_path.write_bytes(b"stale-compatibility-view")

    signature = manager.current_signature()

    assert signature.release_id == "rel-001"
    assert signature.artifact_path.read_bytes() == b"candidate-model-v1"


def test_legacy_projection_failure_does_not_rollback_authoritative_pointer(
    tmp_path, monkeypatch
):
    legacy_path = tmp_path / "models" / "trained_model.pkl"
    manager = ReleaseManager(
        tmp_path / "artifacts", legacy_model_path=legacy_path
    )
    manager.promote(
        _candidate(tmp_path, "rel-001", b"candidate-model-v1"),
        project_legacy=True,
    )

    def fail_projection(*_args, **_kwargs):
        raise OSError("compatibility disk unavailable")

    monkeypatch.setattr(manager, "project_active_artifact", fail_projection)
    with pytest.warns(RuntimeWarning, match="non-authoritative legacy"):
        signature = manager.promote(
            _candidate(tmp_path, "rel-002", b"candidate-model-v2"),
            project_legacy=True,
        )

    assert signature.release_id == "rel-002"
    assert manager.current_signature().artifact_path.read_bytes() == (
        b"candidate-model-v2"
    )
    assert legacy_path.read_bytes() == b"candidate-model-v1"


def test_concurrent_promotions_are_serialized_and_both_releases_remain_valid(
    tmp_path,
):
    manager = ReleaseManager(tmp_path / "artifacts")
    manifests = [
        _candidate(tmp_path, "rel-001", b"candidate-model-v1"),
        _candidate(tmp_path, "rel-002", b"candidate-model-v2"),
    ]

    with ThreadPoolExecutor(max_workers=2) as pool:
        signatures = list(pool.map(manager.promote, manifests))

    assert {signature.release_id for signature in signatures} == {
        "rel-001",
        "rel-002",
    }
    for release_id in ("rel-001", "rel-002"):
        assert load_manifest(
            manager.releases_dir / release_id / "manifest.json",
            verify_artifacts=True,
        )["release_id"] == release_id
    assert manager.current_signature().release_id in {"rel-001", "rel-002"}
