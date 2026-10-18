from types import SimpleNamespace

from main import run_monitor_mode


class _Monitor:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def run_once(self):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        status = outcome if isinstance(outcome, str) else "ok"
        return SimpleNamespace(
            status=status,
            documents_inserted=0 if isinstance(outcome, str) else outcome,
            assessments_created=0 if isinstance(outcome, str) else outcome,
            alerts_opened=0,
        )


def test_monitor_once_reports_failure_without_exiting_process(tmp_path):
    monitor = _Monitor([RuntimeError("offline")])

    assert run_monitor_mode(tmp_path, monitor=monitor) is False
    assert monitor.calls == 1


def test_monitor_console_redacts_credentials(tmp_path, capsys):
    monitor = _Monitor(
        [RuntimeError("Authorization: Bearer secret-token; api_key=sk-sensitive")]
    )

    assert run_monitor_mode(tmp_path, monitor=monitor) is False

    output = capsys.readouterr().out
    assert "secret-token" not in output
    assert "sk-sensitive" not in output
    assert "[REDACTED]" in output


def test_watch_recovers_after_one_source_failure(tmp_path):
    monitor = _Monitor([RuntimeError("offline"), 2])
    sleeps = []

    result = run_monitor_mode(
        tmp_path,
        watch=True,
        interval_seconds=1,
        monitor=monitor,
        sleep_fn=sleeps.append,
        max_iterations=2,
    )

    assert result is True
    assert monitor.calls == 2
    assert sleeps == [30]


def test_monitor_once_propagates_partial_failure(tmp_path):
    monitor = _Monitor(["partial_failure"])

    assert run_monitor_mode(tmp_path, monitor=monitor) is False


def test_watch_backs_off_after_partial_failure(tmp_path):
    monitor = _Monitor(["partial_failure", 1])
    sleeps = []

    result = run_monitor_mode(
        tmp_path,
        watch=True,
        interval_seconds=30,
        monitor=monitor,
        sleep_fn=sleeps.append,
        max_iterations=2,
    )

    assert result is True
    assert sleeps == [30]


def test_monitor_respects_configured_polling_interval(tmp_path):
    monitor = _Monitor([1, 1])
    monitor.config = SimpleNamespace(interval_seconds=123)
    sleeps = []
    assert run_monitor_mode(tmp_path, monitor=monitor, watch=True,
                            sleep_fn=sleeps.append, max_iterations=2)
    assert sleeps == [123]


def test_bad_monitor_config_fails_before_network_calls(tmp_path, capsys):
    assert run_monitor_mode(tmp_path, config_path="missing.json") is False
    assert "CONFIG ERROR" in capsys.readouterr().out
