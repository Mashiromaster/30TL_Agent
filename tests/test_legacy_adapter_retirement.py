from pathlib import Path

import numpy as np
import pytest

from cron_scheduler import CronScheduler
from self_evolution import (
    AdapterStack,
    LegacyAdapterDisabledError,
    SelfEvolutionEngine,
    run_bimonthly_retrain,
    run_combine_predict,
    run_weekly_adapt,
)


def _write_legacy_stack(base_dir: Path) -> Path:
    models = base_dir / "models"
    models.mkdir(parents=True)
    path = models / "adapter_stack.pkl"
    path.write_bytes(b"untrusted-or-corrupt-pickle")
    return path


def test_adapter_stack_never_reads_moves_or_deletes_legacy_pickle(tmp_path):
    stack_path = _write_legacy_stack(tmp_path)
    original = stack_path.read_bytes()

    stack = AdapterStack(str(tmp_path))

    assert stack.enabled is False
    assert stack.legacy_artifact_present is True
    assert stack.size == 0
    assert stack.get_active_adapters() == []
    assert stack_path.read_bytes() == original
    assert list(stack_path.parent.glob("adapter_stack.pkl.corrupted*")) == []

    with pytest.raises(LegacyAdapterDisabledError):
        stack._load()
    with pytest.raises(LegacyAdapterDisabledError):
        stack._save()
    with pytest.raises(LegacyAdapterDisabledError):
        stack.push(object())
    with pytest.raises(LegacyAdapterDisabledError):
        stack.predict(np.zeros((1, 1)))
    with pytest.raises(LegacyAdapterDisabledError):
        stack.clear()

    assert stack_path.read_bytes() == original


def test_legacy_adapter_cli_entrypoints_fail_closed_before_training(tmp_path):
    stack_path = _write_legacy_stack(tmp_path)
    original = stack_path.read_bytes()

    assert run_weekly_adapt(str(tmp_path)) == 1
    assert run_bimonthly_retrain(str(tmp_path)) == 1
    assert run_combine_predict(str(tmp_path)) == 1

    status = SelfEvolutionEngine(str(tmp_path)).get_status()
    assert status["legacy_adapter_enabled"] is False
    assert status["legacy_artifact_present"] is True
    assert stack_path.read_bytes() == original


def test_scheduler_refuses_manual_legacy_adapter_job(tmp_path, monkeypatch):
    scheduler = CronScheduler(tmp_path)
    legacy_job = next(job for job in scheduler.jobs if job["id"] == "weekly_adapt")

    def unexpected_subprocess(*_args, **_kwargs):
        raise AssertionError("legacy adapter command must not spawn a process")

    monkeypatch.setattr("cron_scheduler.subprocess.run", unexpected_subprocess)

    assert scheduler._run_command(legacy_job) is False
    persisted = next(
        job for job in scheduler.jobs if job["id"] == "weekly_adapt"
    )
    assert "已停用" in persisted["last_error"]
