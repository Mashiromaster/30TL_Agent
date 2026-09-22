from datetime import datetime
import json
import subprocess
import sys

import cron_scheduler
from cron_scheduler import CronScheduler, cron_matches


def test_cron_matches_all_five_fields():
    tuesday = datetime(2026, 9, 22, 8, 0)

    assert not cron_matches("0 8 * * 6", tuesday)
    assert not cron_matches("0 9 1 * *", tuesday.replace(hour=9))
    assert not cron_matches("0 9 1 */2 *", tuesday.replace(hour=9))

    saturday = datetime(2026, 9, 26, 8, 0)
    assert cron_matches("0 8 * * 6", saturday)
    assert cron_matches("*/10 8 * * 1-6", saturday)
    assert not cron_matches("*/10 8 * * 1-5", saturday)


def test_cron_month_step_and_lists():
    assert cron_matches("15 9 1 1,3,5 *", datetime(2027, 3, 1, 9, 15))
    assert cron_matches("0 9 1 */2 *", datetime(2027, 3, 1, 9, 0))
    assert not cron_matches("0 9 1 */2 *", datetime(2027, 2, 1, 9, 0))


def test_scheduler_store_is_scoped_to_requested_base_dir(tmp_path):
    scheduler = CronScheduler(tmp_path)

    assert scheduler.store_path == str(tmp_path / "outputs" / "cron_jobs.json")
    assert (tmp_path / "outputs" / "cron_jobs.json").exists()


def test_legacy_hour_match_delegates_to_complete_matcher():
    tuesday = datetime(2026, 9, 22, 8, 0)

    assert not CronScheduler._cron_hour_match("0 8 * * 6", tuesday)


def test_existing_store_is_migrated_without_enabling_legacy_writers(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    store = output_dir / "cron_jobs.json"
    store.write_text(
        json.dumps([
            {
                "id": "weekly_adapt",
                "name": "legacy",
                "kind": "cron",
                "expr": "0 8 * * 6",
                "command": "weekly_adapt",
                "enabled": True,
            }
        ]),
        encoding="utf-8",
    )

    scheduler = CronScheduler(tmp_path)
    by_id = {job["id"]: job for job in scheduler.jobs}

    assert by_id["weekly_adapt"]["enabled"] is False
    assert by_id["sentiment_monitor"]["command"] == "monitor"


def test_child_failure_and_timeout_propagate_to_scheduler_status(
    tmp_path, monkeypatch
):
    scheduler = CronScheduler(tmp_path)
    monitor_job = next(
        job for job in scheduler.jobs if job["id"] == "sentiment_monitor"
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=7,
            stdout="",
            stderr=(
                "provider failed Authorization: Bearer secret-token; "
                "api_key=sk-sensitive"
            ),
        ),
    )

    assert scheduler._run_command(monitor_job) is False
    assert scheduler.run_job("unknown-task") is False
    persisted = json.loads(
        (tmp_path / "outputs" / "cron_jobs.json").read_text(encoding="utf-8")
    )
    failed = next(job for job in persisted if job["id"] == "sentiment_monitor")
    assert "provider failed" in failed["last_error"]
    assert "secret-token" not in failed["last_error"]
    assert "sk-sensitive" not in failed["last_error"]

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="monitor", timeout=1)

    monkeypatch.setattr(subprocess, "run", timeout)
    assert scheduler._run_command(monitor_job) is False


def test_cli_returns_nonzero_when_child_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(cron_scheduler, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="failed"
        ),
    )
    monkeypatch.setattr(
        sys, "argv", ["cron_scheduler.py", "run", "sentiment_monitor"]
    )

    assert cron_scheduler.main() == 1
