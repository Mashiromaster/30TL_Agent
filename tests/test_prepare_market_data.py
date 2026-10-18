import json
from unittest.mock import Mock

import pandas as pd
import pytest

import prepare_market_data as cli
from types import SimpleNamespace


def test_recent_fetch_requires_explicit_network_before_call(tmp_path, monkeypatch, capsys):
    worker = Mock()
    monkeypatch.setattr(cli, "fetch_recent_minutes", worker, raising=False)
    assert cli.main(["--base-dir", str(tmp_path), "fetch-recent", "--symbol", "TL2612"]) == 1
    worker.assert_not_called()
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


def test_recent_fetch_defaults_to_preview_and_never_commits(tmp_path, monkeypatch, capsys):
    item = SimpleNamespace(report={"status": "review_required", "history_backfilled": False})
    fetch = Mock(return_value=item)
    commit = Mock()
    monkeypatch.setattr(cli, "fetch_recent_minutes", fetch, raising=False)
    monkeypatch.setattr(cli, "commit_recent_capture", commit, raising=False)
    assert cli.main(["--base-dir", str(tmp_path), "fetch-recent", "--symbol", "TL2612", "--allow-network"]) == 0
    assert fetch.call_args.kwargs["bar_label"] == "unverified"
    commit.assert_not_called()
    assert json.loads(capsys.readouterr().out)["capture"]["history_backfilled"] is False


@pytest.mark.parametrize("extra", [[], ["--confirm-semantics"], ["--confirm-semantics", "--license-note", "research"]])
def test_recent_commit_missing_declarations_never_fetches(tmp_path, monkeypatch, capsys, extra):
    fetch = Mock()
    monkeypatch.setattr(cli, "fetch_recent_minutes", fetch, raising=False)
    assert cli.main(["--base-dir", str(tmp_path), "fetch-recent", "--symbol", "TL2612", "--allow-network", "--commit", *extra]) == 1
    fetch.assert_not_called()


def test_recent_commit_uses_the_inspected_capture(tmp_path, monkeypatch, capsys):
    item = SimpleNamespace(report={"status": "inspected"})
    monkeypatch.setattr(cli, "fetch_recent_minutes", Mock(return_value=item), raising=False)
    commit = Mock(return_value={"status": "imported", "batch_id": "minute-test"})
    monkeypatch.setattr(cli, "commit_recent_capture", commit, raising=False)
    args = ["--base-dir", str(tmp_path), "fetch-recent", "--symbol", "TL2612", "--allow-network", "--bar-label", "end",
            "--commit", "--confirm-semantics", "--license-note", "research"]
    assert cli.main(args) == 0
    assert commit.call_args.args[1] is item
    assert commit.call_args.kwargs["confirmed"] is True
    assert json.loads(capsys.readouterr().out)["batch"]["batch_id"] == "minute-test"


def test_tq_export_options_reach_importer(tmp_path, monkeypatch, capsys):
    worker = Mock(return_value={"status": "inspected"})
    monkeypatch.setattr(cli, "import_minute_csv", worker)
    assert cli.main(["--base-dir", str(tmp_path), "import", "export.csv", "--provider", "my-export", "--license-note", "research",
        "--input-format", "tqsdk", "--bar-seconds", "60"]) == 0
    assert worker.call_args.kwargs["input_format"] == "tqsdk"
    assert worker.call_args.kwargs["bar_seconds"] == 60


def test_import_defaults_to_inspection_without_writes(tmp_path, monkeypatch, capsys):
    worker = Mock(return_value={"status": "inspected"})
    monkeypatch.setattr(cli, "import_minute_csv", worker)
    code = cli.main(["--base-dir", str(tmp_path), "import", "sample.csv", "--provider", "export", "--license-note", "local research"])
    assert code == 0
    assert worker.call_args.kwargs["persist"] is False
    assert json.loads(capsys.readouterr().out)["status"] == "inspected"


def test_import_commit_is_explicit_and_json_stdout_survives_logs(tmp_path, monkeypatch, capsys):
    def worker(*args, **kwargs):
        assert kwargs["persist"] is True
        print("internal diagnostic")
        return {"status": "imported", "batch_id": "minute-test"}
    monkeypatch.setattr(cli, "import_minute_csv", worker)
    assert cli.main(["--base-dir", str(tmp_path), "import", "sample.csv", "--provider", "export", "--license-note", "research", "--commit"]) == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["batch_id"] == "minute-test"
    assert "internal diagnostic" in output.err


def test_snapshot_uses_only_explicit_batches(tmp_path, monkeypatch, capsys):
    build = Mock(return_value={"status": "built"})
    monkeypatch.setattr(cli, "build_minute_snapshot", build)
    assert cli.main(["--base-dir", str(tmp_path), "snapshot", "--batch-id", "minute-a", "--batch-id", "minute-b"]) == 0
    assert build.call_args.args[1] == ["minute-a", "minute-b"]
    assert build.call_args.kwargs["include_legacy"] is False
    json.loads(capsys.readouterr().out)


def test_snapshot_empty_selection_is_not_implicit_legacy(tmp_path, monkeypatch, capsys):
    build = Mock()
    monkeypatch.setattr(cli, "build_minute_snapshot", build)
    assert cli.main(["--base-dir", str(tmp_path), "snapshot"]) == 1
    build.assert_not_called()
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


def test_failure_returns_redacted_json_and_nonzero_exit(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "list_minute_batches", Mock(side_effect=ValueError("bad batch")))
    assert cli.main(["--base-dir", str(tmp_path), "list"]) == 1
    assert json.loads(capsys.readouterr().out) == {"status": "failed", "error": "bad batch"}


def test_audit_defaults_to_fixed_legacy_input_and_never_writes(tmp_path, monkeypatch, capsys):
    audit = Mock(return_value={"status": "incomplete"})
    monkeypatch.setattr(cli, "audit_local_minutes", audit)
    assert cli.main(["--base-dir", str(tmp_path), "audit"]) == 0
    assert audit.call_args.kwargs["snapshot_id"] is None
    assert not list(tmp_path.iterdir())
    json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("arguments", [["import", "x.pkl"], ["import", "x.csv", "--commit"], ["snapshot", "--network-mode", "online"]])
def test_cli_rejects_missing_metadata_or_unknown_network_switch(arguments):
    with pytest.raises(SystemExit):
        cli.main(arguments)
