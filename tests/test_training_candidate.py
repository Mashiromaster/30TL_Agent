import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import LightGBM_model as training
from model_release import GateRejectedError, _load_gate_attestation, load_manifest


class _FakeModel:
    def set_params(self, **parameters):
        self.parameters = parameters
        return self

    def predict(self, frame):
        return np.linspace(-0.01, 0.01, len(frame))


class _FakeHierarchy(_FakeModel):
    def __init__(self):
        self.micro_model = _FakeModel()

    def fit(self, train, validation, *, feature_groups):
        return self

    def components(self, frame):
        return pd.DataFrame({name: np.zeros(len(frame)) for name in
                             ("prediction", "micro_alpha", "macro_state", "gate_weight", "event_alpha")}, index=frame.index)


@pytest.fixture
def candidate_env(tmp_path, monkeypatch):
    base = tmp_path / "project"
    (base / "outputs").mkdir(parents=True)
    (base / "models").mkdir()
    source = base / "artifacts/factor_candidates/synthetic/factors.pkl"
    source.parent.mkdir(parents=True)
    days = pd.bdate_range("2024-01-02", periods=260)
    dates = pd.DatetimeIndex([day + pd.Timedelta(hours=10, minutes=minute) for day in days for minute in range(60)])
    positions = np.arange(len(dates))
    frame = pd.DataFrame({
        "date": dates, "ticker": "TL2612", "close": 100 + positions * 0.0001,
        "adj_close": 100 + positions * 0.0001,
        "Spread_Mean": np.sin(positions / 20), "YC_Slope": np.cos(positions / 500),
        "Market_Regime": positions % 3,
    })
    frame.to_pickle(source)
    protected = {
        base / "models/trained_model.pkl": b"old model",
        base / "outputs/df_predictions.pkl": b"old predictions",
        base / "outputs/feature_importance.csv": b"old importance",
        base / "outputs/model_ablation_report.json": b"old report",
    }
    for path, content in protected.items():
        path.write_bytes(content)
    monkeypatch.setenv("F_AGENT_GENETIC_MINING", "0")
    monkeypatch.setattr(training, "_verify_factor_provenance", lambda *_, **__: {"verified": True, "reason": "synthetic_test_contract"})
    monkeypatch.setattr(training, "_fit_flat", lambda *_, **__: _FakeModel())
    monkeypatch.setattr(training, "HierarchicalMacroMicroModel", _FakeHierarchy)
    monkeypatch.setattr(training, "_ic", lambda *_: 0.05)
    monkeypatch.setattr(training, "_date_block_stability", lambda *_: {
        "fold_ics": [0.05] * 5, "median_ic": 0.05, "same_sign_ratio": 1.0,
        "positive_fold_count": 5, "fold_count": 5,
    })
    monkeypatch.setattr(training, "_macro_block_permutation", lambda *_, **__: {})
    monkeypatch.setattr(training, "_feature_importance", lambda _, features: pd.DataFrame({"feature": features, "importance": 1.0}))

    def evaluate(frame, *, strategy_spec, model_release_id):
        return None, {
            "total_return": 0.02, "sharpe": 0.5, "annual_volatility": 0.08,
            "max_drawdown": -0.1, "active_ratio": 0.3, "observation_days": 60,
            "strategy_id": strategy_spec.strategy_id, "strategy_fingerprint": strategy_spec.fingerprint(),
            "model_release_id": model_release_id,
        }, None

    monkeypatch.setattr("backtest.evaluate_predictions", evaluate)
    promote = Mock(side_effect=AssertionError("candidate training must not publish"))
    project = Mock(side_effect=AssertionError("candidate training must not project artifacts"))
    monkeypatch.setattr(training.ReleaseManager, "promote", promote)
    monkeypatch.setattr(training.ReleaseManager, "project_active_artifact", project)
    return SimpleNamespace(base=base, source=source, frame=frame, protected=protected, promote=promote, project=project)


def test_candidate_only_freezes_input_and_never_publishes(candidate_env):
    env = candidate_env
    before = env.source.read_bytes()
    result = training.train_candidate(env.base, factor_path=env.source)
    assert result["status"] == "candidate_ready"
    assert result["published"] is False
    assert result["gates"]["release_gate_passed"] is True
    assert result["gates"]["input_unchanged"] is True
    snapshot = result["input_snapshot"]
    assert snapshot["sha256"] == hashlib.sha256(before).hexdigest()
    assert Path(snapshot["path"]).read_bytes() == before
    assert snapshot["rows"] == len(env.frame)
    assert snapshot["observed_days"] == 260
    assert env.source.read_bytes() == before
    manifest = load_manifest(result["candidate_manifest"], verify_artifacts=True)
    assert manifest["manifest_type"] == "candidate"
    assert manifest["artifacts"]["training_input"]["sha256"] == snapshot["sha256"]
    assert manifest["metadata"]["input_sha256"] == snapshot["sha256"]
    attestation = _load_gate_attestation(Path(result["candidate_manifest"]), manifest)
    assert attestation["gate_passed"] is True
    assert result["partitions"]["validation"]["observed_days"] == 60
    assert result["partitions"]["sealed_holdout"]["observed_days"] == 60
    assert result["partitions"]["train"]["end"] < result["partitions"]["validation"]["start"]
    assert result["partitions"]["validation"]["end"] < result["partitions"]["sealed_holdout"]["start"]
    env.promote.assert_not_called()
    env.project.assert_not_called()
    assert not (env.base / "artifacts/models/current.json").exists()
    for path, content in env.protected.items():
        assert path.read_bytes() == content


@pytest.mark.parametrize("gate", ["validation", "holdout", "strategy"])
def test_failed_candidates_remain_inspectable_and_keep_strict_gates(candidate_env, monkeypatch, gate):
    if gate == "validation":
        monkeypatch.setattr(training, "_date_block_stability", lambda *_: {"fold_count": 5, "positive_fold_count": 2, "median_ic": -0.01})
    elif gate == "holdout":
        monkeypatch.setattr(training, "_ic", lambda *_: -0.05)
    else:
        monkeypatch.setattr("backtest.evaluate_predictions", lambda *_, **__: (None, {}, None))
    result = training.train_candidate(candidate_env.base, factor_path=candidate_env.source)
    assert result["status"] == "candidate_rejected"
    assert result["gates"]["release_gate_passed"] is False
    assert result["gates"][gate + "_passed"] is False
    assert Path(result["candidate_manifest"]).exists()
    candidate_env.promote.assert_not_called()


def test_source_changed_during_training_blocks_candidate_gate(candidate_env, monkeypatch):
    env = candidate_env
    original = env.source.read_bytes()

    def mutate_source(*_, **__):
        env.source.write_bytes(b"changed by another writer")
        return _FakeModel()

    monkeypatch.setattr(training, "_fit_flat", mutate_source)
    result = training.train_candidate(env.base, factor_path=env.source)
    assert result["status"] == "candidate_rejected"
    assert result["gates"]["input_unchanged"] is False
    assert result["gates"]["release_gate_passed"] is False
    assert Path(result["input_snapshot"]["path"]).read_bytes() == original
    manifest = load_manifest(result["candidate_manifest"], verify_artifacts=True)
    report = json.loads((Path(result["candidate_manifest"]).parent / manifest["artifacts"]["evaluation"]["path"]).read_text())
    assert report["release_gate_passed"] is False
    assert report["input_snapshot"]["source_unchanged"] is False
    env.promote.assert_not_called()


def test_candidate_api_reports_missing_and_invalid_data_without_production_writes(candidate_env):
    env = candidate_env
    missing = training.train_candidate(env.base, factor_path=env.source.parent / "missing.pkl")
    assert missing["status"] == "failed"
    assert missing["candidate_manifest"] is None
    short = env.frame.iloc[:60]
    short.to_pickle(env.source)
    invalid = training.train_candidate(env.base, factor_path=env.source)
    assert invalid["status"] == "failed"
    assert "240" in " ".join(invalid["errors"])
    for path, content in env.protected.items():
        assert path.read_bytes() == content


def test_run_process_candidate_only_uses_same_public_result(candidate_env):
    result = training.run_process(candidate_env.base, factor_path=candidate_env.source, candidate_only=True)
    assert result["status"] == "candidate_ready"
    assert result["published"] is False
    candidate_env.promote.assert_not_called()


def test_default_run_process_still_promotes_only_after_valid_gate(candidate_env):
    env = candidate_env
    env.promote.side_effect = None
    env.promote.return_value = SimpleNamespace(release_id="verified-test-release")
    env.project.side_effect = None
    result = training.run_process(env.base, factor_path=env.source)
    assert result is True
    env.promote.assert_called_once()
    assert env.promote.call_args.kwargs["project_legacy"] is True
    assert env.project.call_count == 3


def test_candidate_creation_does_not_create_compatibility_directories(candidate_env, tmp_path):
    base = tmp_path / "empty-project"
    source = base / "artifacts/factor_candidates/synthetic/factors.pkl"
    source.parent.mkdir(parents=True)
    source.write_bytes(candidate_env.source.read_bytes())
    result = training.train_candidate(base, factor_path=source)
    assert result["status"] == "candidate_ready"
    assert not (base / "models").exists()
    assert not (base / "outputs").exists()


def test_candidate_errors_are_redacted(candidate_env, monkeypatch):
    def fail(*_, **__):
        raise ValueError("api_key=sk-sensitive; Authorization: Bearer credential")
    monkeypatch.setattr(training, "_fit_flat", fail)
    result = training.train_candidate(candidate_env.base, factor_path=candidate_env.source)
    assert result["status"] == "failed"
    assert "sk-sensitive" not in json.dumps(result)
    assert "credential" not in json.dumps(result)


def test_unverified_factor_origin_cannot_become_release_eligible(candidate_env, monkeypatch):
    monkeypatch.setattr(training, "_verify_factor_provenance", lambda *_, **__: {"verified": False, "reason": "unverified_factor_provenance"})
    result = training.train_candidate(candidate_env.base, factor_path=candidate_env.source)
    assert result["status"] == "candidate_rejected"
    assert result["gates"]["validation_passed"] is True
    assert result["gates"]["holdout_passed"] is True
    assert result["gates"]["strategy_passed"] is True
    assert result["gates"]["factor_provenance_verified"] is False
    assert result["gates"]["release_gate_passed"] is False
    assert "unverified_factor_provenance" in result["gates"]["factor_provenance_reason"]
    manifest = load_manifest(result["candidate_manifest"], verify_artifacts=True)
    assert manifest["metadata"]["release_gate_passed"] is False
    with pytest.raises(GateRejectedError, match="does not record a passing gate"):
        _load_gate_attestation(Path(result["candidate_manifest"]), manifest)
    candidate_env.promote.assert_not_called()


def test_factor_file_mutated_during_freeze_aborts_before_fitting(candidate_env, monkeypatch):
    original_copy = training.shutil.copyfile

    def changing_copy(source, destination):
        original_copy(source, destination)
        Path(source).write_bytes(b"concurrent replacement")

    fitter = Mock(side_effect=AssertionError("must not fit an unstable snapshot"))
    monkeypatch.setattr(training.shutil, "copyfile", changing_copy)
    monkeypatch.setattr(training, "_fit_flat", fitter)
    result = training.train_candidate(candidate_env.base, factor_path=candidate_env.source)
    assert result["status"] == "failed"
    assert "changed while freezing" in " ".join(result["errors"])
    fitter.assert_not_called()
    candidate_env.promote.assert_not_called()


def test_source_mutation_while_serializing_rewrites_a_rejected_bundle(candidate_env, monkeypatch):
    original_write = training._write_training_bundle
    writes = []

    def changing_write(*args):
        path = original_write(*args)
        writes.append(path)
        if len(writes) == 1:
            candidate_env.source.write_bytes(b"concurrent replacement after fitting")
        return path

    monkeypatch.setattr(training, "_write_training_bundle", changing_write)
    result = training.train_candidate(candidate_env.base, factor_path=candidate_env.source)
    assert result["status"] == "candidate_rejected"
    assert len(writes) == 2
    manifest = load_manifest(result["candidate_manifest"], verify_artifacts=True)
    assert manifest["metadata"]["release_gate_passed"] is False
    report = json.loads((Path(result["candidate_manifest"]).parent / "evaluation.json").read_text())
    assert report["gate_attestation"]["gate_passed"] is False
    assert report["gates"]["input_unchanged"] is False
    candidate_env.promote.assert_not_called()


def test_default_input_path_and_multiple_candidates_remain_isolated(candidate_env):
    env = candidate_env
    env.frame.to_pickle(env.base / "outputs/df_factors.pkl")
    first = training.train_candidate(env.base)
    second = training.train_candidate(env.base)
    assert first["status"] == second["status"] == "candidate_ready"
    assert first["candidate_manifest"] != second["candidate_manifest"]
    assert first["input_snapshot"]["sha256"] == second["input_snapshot"]["sha256"]
    assert first["run_id"] != second["run_id"]


def test_default_publication_path_rejects_changed_input(candidate_env, monkeypatch):
    def changing_fit(*_, **__):
        candidate_env.source.write_bytes(b"modified")
        return _FakeModel()

    monkeypatch.setattr(training, "_fit_flat", changing_fit)
    assert training.run_process(candidate_env.base, factor_path=candidate_env.source) is False
    candidate_env.promote.assert_not_called()


def test_provenance_verifier_delegates_and_fails_closed(tmp_path, monkeypatch):
    import sys
    captured = Mock(return_value={"verified": True, "reason": "verified"})
    monkeypatch.setitem(sys.modules, "research_readiness", SimpleNamespace(verify_factor_provenance=captured))
    frame = pd.DataFrame()
    assert training._verify_factor_provenance(tmp_path, tmp_path / "factors.pkl", frame, expected_sha256="a" * 64)["verified"] is True
    captured.assert_called_once_with(tmp_path, tmp_path / "factors.pkl", frame, expected_sha256="a" * 64)
    captured.side_effect = ValueError("api_key=private-key")
    result = training._verify_factor_provenance(tmp_path, tmp_path / "factors.pkl", frame)
    assert result["verified"] is False
    assert "private-key" not in json.dumps(result)


def test_legacy_missing_default_input_remains_false(tmp_path):
    assert training.run_process(tmp_path) is False


def test_relative_factor_path_is_resolved_against_project(candidate_env, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    relative = candidate_env.source.relative_to(candidate_env.base)
    result = training.train_candidate(candidate_env.base, factor_path=relative)
    assert result["status"] == "candidate_ready"
    assert result["input_snapshot"]["source_path"] == str(candidate_env.source)


@pytest.mark.parametrize("variant", ["absolute_escape", "relative_escape", "wrong_directory", "symlink_escape"])
def test_factor_path_escape_is_rejected_before_deserializing(candidate_env, monkeypatch, tmp_path, variant):
    outside = tmp_path / "outside.pkl"
    outside.write_bytes(candidate_env.source.read_bytes())
    if variant == "absolute_escape":
        selected = outside
    elif variant == "relative_escape":
        selected = Path("../outside.pkl")
    elif variant == "wrong_directory":
        selected = candidate_env.base / "other.pkl"
        selected.write_bytes(outside.read_bytes())
    else:
        selected = candidate_env.source.parent / "link.pkl"
        selected.symlink_to(outside)
    reader = Mock(side_effect=AssertionError("must reject path before loading"))
    monkeypatch.setattr(training.pd, "read_pickle", reader)
    result = training.train_candidate(candidate_env.base, factor_path=selected)
    assert result["status"] == "failed"
    assert result["candidate_manifest"] is None
    reader.assert_not_called()
    assert not (candidate_env.base / "artifacts/experiments").exists()


def test_provenance_is_bound_to_actual_frozen_bytes(candidate_env, monkeypatch):
    verifier = Mock(return_value={"verified": True, "reason": "synthetic_test_contract"})
    monkeypatch.setattr(training, "_verify_factor_provenance", verifier)
    result = training.train_candidate(candidate_env.base, factor_path=candidate_env.source)
    assert result["status"] == "candidate_ready"
    assert verifier.call_args.kwargs["expected_sha256"] == result["input_snapshot"]["sha256"]


def test_mutated_frozen_snapshot_cannot_pass_release_gate(candidate_env, monkeypatch):
    def changing_fit(*_, **__):
        for frozen in (candidate_env.base / "artifacts/experiments").glob("*/input_snapshot.pkl"):
            frozen.write_bytes(b"modified frozen copy")
        return _FakeModel()
    monkeypatch.setattr(training, "_fit_flat", changing_fit)
    result = training.train_candidate(candidate_env.base, factor_path=candidate_env.source)
    assert result["status"] == "candidate_rejected"
    assert result["gates"]["input_unchanged"] is False
    assert result["gates"]["release_gate_passed"] is False


def test_candidate_thread_limit_reaches_all_flat_and_hierarchical_models(candidate_env, monkeypatch):
    fits = Mock(return_value=_FakeModel())
    hierarchy = _FakeHierarchy()
    monkeypatch.setattr(training, "_fit_flat", fits)
    monkeypatch.setattr(training, "HierarchicalMacroMicroModel", lambda: hierarchy)
    result = training.train_candidate(candidate_env.base, factor_path=candidate_env.source, thread_limit=2)
    assert result["status"] == "candidate_ready"
    assert len(fits.call_args_list) == 3
    assert all(call.kwargs["thread_limit"] == 2 for call in fits.call_args_list)
    assert hierarchy.micro_model.parameters["n_jobs"] == 2
    assert result["thread_limit"] == 2


@pytest.mark.parametrize("limit", [True, 0, -1, 1.5, "2", None])
def test_candidate_rejects_invalid_thread_limit_before_writes(candidate_env, limit):
    result = training.train_candidate(candidate_env.base, factor_path=candidate_env.source, thread_limit=limit)
    assert result["status"] == "failed"
    assert "thread_limit" in " ".join(result["errors"])
    assert not (candidate_env.base / "artifacts/experiments").exists()


def test_flat_model_passes_explicit_thread_limit_to_lightgbm(monkeypatch):
    regressor = Mock()
    constructor = Mock(return_value=regressor)
    monkeypatch.setattr(training, "LGBMRegressor", constructor)
    frame = pd.DataFrame({"Spread_Mean": [0.1, 0.2], "Target_Ret": [0.1, 0.2]})
    training._fit_flat(frame, frame, ["Spread_Mean"], thread_limit=2)
    assert constructor.call_args.kwargs["n_jobs"] == 2
    training._fit_flat(frame, frame, ["Spread_Mean"])
    assert constructor.call_args.kwargs["n_jobs"] == -1


def test_candidate_limits_optional_gp_and_native_threadpools(candidate_env, monkeypatch):
    from threadpoolctl import threadpool_info
    miner = SimpleNamespace(selected_programs_=[])
    mine = Mock(return_value=(miner, {"selected_count": 0}))
    monkeypatch.setenv("F_AGENT_GENETIC_MINING", "1")
    monkeypatch.setattr("genetic_factor_miner.mine_genetic_factors", mine)
    observed = []

    def inspect_fit(*_, **__):
        observed.extend(item["num_threads"] for item in threadpool_info())
        return _FakeModel()

    monkeypatch.setattr(training, "_fit_flat", inspect_fit)
    result = training.train_candidate(candidate_env.base, factor_path=candidate_env.source)
    assert result["status"] == "candidate_ready"
    assert mine.call_args.kwargs["n_jobs"] == 2
    assert observed and max(observed) <= 2


def test_legacy_entry_rejects_candidate_thread_option(candidate_env):
    with pytest.raises(ValueError, match="candidate_only"):
        training.run_process(candidate_env.base, thread_limit=2)


def test_real_provenance_verification_matches_candidate_snapshot(candidate_env, monkeypatch):
    from research_readiness import verify_factor_provenance
    env = candidate_env
    provenance = {"schema_version": 1, "kind": "local_causal_rebuild_v1",
                  "pit_contract": "previous_completed_observed_day_oi_v1",
                  "source_sha256": "a" * 64, "candidate_id": "synthetic", "limitations": []}
    frame = env.frame.copy()
    frame.attrs["factor_provenance"] = provenance
    frame.to_pickle(env.source)
    digest = hashlib.sha256(env.source.read_bytes()).hexdigest()
    manifest = {"schema_version": 1, "status": "built", "candidate_id": "synthetic",
                "factor_file": "factors.pkl", "factor_sha256": digest,
                "source_sha256": "a" * 64, "provenance": provenance}
    (env.source.parent / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(training, "_verify_factor_provenance", verify_factor_provenance)
    result = training.train_candidate(env.base, factor_path=env.source)
    assert result["status"] == "candidate_ready"
    assert result["gates"]["factor_provenance_verified"] is True
    assert result["input_snapshot"]["sha256"] == digest


@pytest.mark.parametrize("ancestor", ["artifacts", "experiments", "run"])
@pytest.mark.parametrize("destination", ["outside", "production"])
def test_candidate_output_symlink_ancestors_are_rejected_before_writes(candidate_env, monkeypatch, tmp_path, ancestor, destination):
    env = candidate_env
    env.frame.to_pickle(env.base / "outputs/df_factors.pkl")
    monkeypatch.setattr(training, "_release_identifiers", lambda *_: ("forced-run", "forced-release"))
    target = tmp_path / "external-output" if destination == "outside" else env.base / "models"
    target.mkdir(exist_ok=True)
    before = {str(path.relative_to(target)): path.read_bytes() for path in target.rglob("*") if path.is_file()}
    if ancestor == "artifacts":
        selected = env.base / "artifacts"
        selected.rename(env.base / "saved-artifacts")
    elif ancestor == "experiments":
        selected = env.base / "artifacts/experiments"
    else:
        selected = env.base / "artifacts/experiments/forced-run"
        selected.parent.mkdir()
    selected.symlink_to(target, target_is_directory=True)
    result = training.train_candidate(env.base)
    assert result["status"] == "failed"
    assert "symbolic" in " ".join(result["errors"])
    assert {str(path.relative_to(target)): path.read_bytes() for path in target.rglob("*") if path.is_file()} == before
    assert not any(path.is_dir() for path in target.iterdir())
    env.promote.assert_not_called()


@pytest.mark.parametrize("ancestor", ["experiments", "run"])
@pytest.mark.parametrize("destination", ["outside", "production"])
def test_candidate_output_redirected_during_fitting_never_writes_target(candidate_env, monkeypatch, tmp_path, ancestor, destination):
    env = candidate_env
    env.frame.to_pickle(env.base / "outputs/df_factors.pkl")
    target = tmp_path / "external-output" if destination == "outside" else env.base / "models"
    target.mkdir(exist_ok=True)
    before = {str(path.relative_to(target)): path.read_bytes() for path in target.rglob("*") if path.is_file()}
    redirected = []

    def redirect_once(*_, **__):
        if not redirected:
            experiments = env.base / "artifacts/experiments"
            selected = experiments if ancestor == "experiments" else next(experiments.iterdir())
            selected.rename(selected.with_name(selected.name + "-saved"))
            selected.symlink_to(target, target_is_directory=True)
            redirected.append(selected)
        return _FakeModel()

    monkeypatch.setattr(training, "_fit_flat", redirect_once)
    result = training.train_candidate(env.base)
    assert result["status"] == "failed"
    assert "symbolic" in " ".join(result["errors"])
    assert {str(path.relative_to(target)): path.read_bytes() for path in target.rglob("*") if path.is_file()} == before
    assert not any(path.is_dir() for path in target.iterdir())
    env.promote.assert_not_called()


@pytest.mark.parametrize("artifact", ["df_predictions.pkl", "feature_importance.csv", "trained_model.pkl", "evaluation.json", "evaluation.json.tmp", "manifest.json"])
def test_candidate_artifact_symlink_never_overwrites_production(candidate_env, monkeypatch, artifact):
    env = candidate_env
    protected = env.base / "models/trained_model.pkl"
    original = protected.read_bytes()

    def redirect_once(*_, **__):
        directory = next((env.base / "artifacts/experiments").iterdir())
        path = directory / artifact
        if not path.is_symlink():
            path.symlink_to(protected)
        return _FakeModel()

    monkeypatch.setattr(training, "_fit_flat", redirect_once)
    result = training.train_candidate(env.base, factor_path=env.source)
    assert result["status"] == "failed"
    assert "symbolic" in " ".join(result["errors"])
    assert protected.read_bytes() == original
    env.promote.assert_not_called()


@pytest.mark.parametrize("stage", ["importance", "model", "evaluation"])
def test_output_redirected_between_writes_never_overwrites_or_cleans_production(candidate_env, monkeypatch, stage):
    env = candidate_env
    target = env.base / "models"
    (target / "evaluation.json.tmp").write_bytes(b"production temporary file")
    before = {path.name: path.read_bytes() for path in target.iterdir()}

    def redirect():
        directory = next((env.base / "artifacts/experiments").iterdir())
        directory.rename(directory.with_name(directory.name + "-saved"))
        directory.symlink_to(target, target_is_directory=True)

    if stage == "importance":
        def importance(_, features):
            redirect()
            return pd.DataFrame({"feature": features, "importance": 1.0})
        monkeypatch.setattr(training, "_feature_importance", importance)
    elif stage == "model":
        original_dump = training.pickle.dump
        def model_dump(value, handle, *args, **kwargs):
            original_dump(value, handle, *args, **kwargs)
            redirect()
        monkeypatch.setattr(training.pickle, "dump", model_dump)
    else:
        original_dump = training.json.dump
        def evaluation_dump(value, handle, *args, **kwargs):
            original_dump(value, handle, *args, **kwargs)
            redirect()
        monkeypatch.setattr(training.json, "dump", evaluation_dump)
    result = training.train_candidate(env.base, factor_path=env.source)
    assert result["status"] == "failed"
    assert "symbolic" in " ".join(result["errors"])
    assert {path.name: path.read_bytes() for path in target.iterdir()} == before
    env.promote.assert_not_called()


@pytest.mark.parametrize("threshold", ["max_drawdown_limit", "min_total_return", "min_sharpe"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True, "0.1", None])
def test_release_thresholds_require_finite_real_numbers(threshold, value):
    metrics = {"total_return": -.1, "sharpe": -.2, "annual_volatility": .08,
               "max_drawdown": -.1, "active_ratio": .3, "observation_days": 60}
    with pytest.raises(ValueError, match=threshold):
        training._strategy_release_gate(metrics, **{threshold: value})


@pytest.mark.parametrize("variable", ["F_AGENT_RELEASE_MAX_DRAWDOWN", "F_AGENT_RELEASE_MIN_TOTAL_RETURN", "F_AGENT_RELEASE_MIN_SHARPE"])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "invalid", "-0.5"])
def test_invalid_release_environment_fails_before_fitting_or_writes(candidate_env, monkeypatch, variable, value):
    monkeypatch.setenv(variable, value)
    fitter = Mock(side_effect=AssertionError("invalid release policy must not train"))
    monkeypatch.setattr(training, "_fit_flat", fitter)
    result = training.train_candidate(candidate_env.base, factor_path=candidate_env.source)
    assert result["status"] == "failed"
    assert result["candidate_manifest"] is None
    fitter.assert_not_called()
    assert not (candidate_env.base / "artifacts/experiments").exists()


def test_release_policy_is_frozen_before_training(candidate_env, monkeypatch):
    monkeypatch.setenv("F_AGENT_RELEASE_MIN_SHARPE", "0.0")

    def change_environment(*_, **__):
        monkeypatch.setenv("F_AGENT_RELEASE_MIN_SHARPE", "nan")
        return _FakeModel()

    monkeypatch.setattr(training, "_fit_flat", change_environment)
    result = training.train_candidate(candidate_env.base, factor_path=candidate_env.source)
    assert result["status"] == "candidate_ready"
    report = json.loads((Path(result["candidate_manifest"]).parent / "evaluation.json").read_text())
    assert report["min_sharpe"] == 0.0


def test_evaluation_never_serializes_non_json_nan(candidate_env, monkeypatch):
    def evaluate(*_, **__):
        return None, {"total_return": float("nan"), "sharpe": .5, "annual_volatility": .08,
                      "max_drawdown": -.1, "active_ratio": .3, "observation_days": 60}, None
    monkeypatch.setattr("backtest.evaluate_predictions", evaluate)
    result = training.train_candidate(candidate_env.base, factor_path=candidate_env.source)
    assert result["status"] == "failed"
    assert result["candidate_manifest"] is None
    assert not list((candidate_env.base / "artifacts/experiments").glob("*/evaluation.json"))
    assert not list((candidate_env.base / "artifacts/experiments").glob("*/evaluation.json.tmp"))
