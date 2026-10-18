import json
import sys
from unittest.mock import Mock

import main
import research_agent
import pytest


def test_hybrid_cli_is_explicit_and_preserves_offline_default(tmp_path, monkeypatch, capsys):
    run = Mock(return_value={"run_id": "test", "decision": {"status": "blocked", "execution_authorized": False}})
    monkeypatch.setattr(research_agent.ResearchAgent, "run_hybrid_cycle", run)
    monkeypatch.setattr(main, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["main.py", "--mode", "hybrid", "--question", "研究TL", "--no-persist"])
    main.main()
    assert run.call_args.kwargs["config"].network_mode == "offline"
    assert run.call_args.kwargs["persist"] is False
    assert run.call_args.kwargs["question"] == "研究TL"
    output = capsys.readouterr()
    assert json.loads(output.out)["decision"]["execution_authorized"] is False
    assert "HYBRID" in output.err


def test_hybrid_cli_explicit_network_and_asof(tmp_path, monkeypatch):
    run = Mock(return_value={"run_id": "test"})
    monkeypatch.setattr(research_agent.ResearchAgent, "run_hybrid_cycle", run)
    monkeypatch.setattr(main, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["main.py", "--mode", "hybrid", "--network-mode", "cache", "--as-of", "2026-06-02 15:15"])
    main.main()
    assert run.call_args.kwargs["config"].network_mode == "cache"
    assert run.call_args.kwargs["as_of"] == "2026-06-02 15:15"


@pytest.mark.parametrize("mode,module,method", [
    ("readiness", "research_readiness", "inspect_research_readiness"),
    ("prepare-factors", "factor_candidates", "build_local_factor_candidate"),
])
def test_preparation_cli_has_explicit_local_json_entrypoints(tmp_path, monkeypatch, capsys, mode, module, method):
    import importlib
    worker = Mock(return_value={"status": "checked", "network": False})
    monkeypatch.setattr(importlib.import_module(module), method, worker)
    monkeypatch.setattr(main, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["main.py", "--mode", mode])
    main.main()
    assert json.loads(capsys.readouterr().out)["status"] == "checked"
    worker.assert_called_once()


def test_candidate_training_cli_requires_eligible_input_and_never_calls_old_train(tmp_path, monkeypatch, capsys):
    import research_readiness
    import LightGBM_model
    readiness = Mock(return_value={"training": {"eligible": True}})
    trainer = Mock(return_value={"status": "candidate_rejected", "published": False})
    monkeypatch.setattr(research_readiness, "inspect_research_readiness", readiness)
    monkeypatch.setattr(LightGBM_model, "train_candidate", trainer)
    old = Mock(side_effect=AssertionError("old auto-release path"))
    monkeypatch.setattr(main, "run_train", old)
    monkeypatch.setattr(main, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["main.py", "--mode", "train-candidate", "--factor-path", "artifacts/factor_candidates/test/factors.pkl"])
    main.main()
    assert json.loads(capsys.readouterr().out)["published"] is False
    trainer.assert_called_once()
    old.assert_not_called()


def test_candidate_training_cli_refuses_unready_input(tmp_path, monkeypatch, capsys):
    import research_readiness
    import LightGBM_model
    monkeypatch.setattr(research_readiness, "inspect_research_readiness", lambda *a, **kw: {"training": {"eligible": False}})
    trainer = Mock()
    monkeypatch.setattr(LightGBM_model, "train_candidate", trainer)
    monkeypatch.setattr(main, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["main.py", "--mode", "train-candidate"])
    main.main()
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "blocked"
    trainer.assert_not_called()


@pytest.mark.parametrize("args", [
    ["--mode", "prepare-factors", "--network-mode", "online"],
    ["--mode", "train-candidate", "--no-persist"],
    ["--mode", "prepare-factors", "--factor-path", "wrong.pkl"],
    ["--mode", "readiness", "--rebuild-factors"],
])
def test_preparation_cli_rejects_conflicting_side_effect_flags(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["main.py", *args])
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 2


def test_factor_build_can_explicitly_select_isolated_minute_snapshot(tmp_path, monkeypatch, capsys):
    import factor_candidates
    build = Mock(return_value={"status": "built"})
    monkeypatch.setattr(factor_candidates, "build_local_factor_candidate", build)
    monkeypatch.setattr(main, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["main.py", "--mode", "prepare-factors", "--minute-snapshot", "minset-test"])
    main.main()
    assert build.call_args.kwargs["minute_snapshot_id"] == "minset-test"
    assert json.loads(capsys.readouterr().out)["status"] == "built"


def test_minute_snapshot_cannot_be_ignored_in_other_modes(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "--mode", "hybrid", "--minute-snapshot", "minset-test"])
    with pytest.raises(SystemExit):
        main.main()
