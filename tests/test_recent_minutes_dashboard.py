from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

import recent_minutes_dashboard as ui


def app(base):
    ui.render_recent_minutes(base)


@pytest.fixture
def view(tmp_path, monkeypatch):
    def script(base):
        from recent_minutes_dashboard import render_recent_minutes
        render_recent_minutes(base)
    return AppTest.from_function(script, args=(str(tmp_path),))


def test_no_implicit_fetch_or_admission(view, monkeypatch):
    fetch, commit = Mock(), Mock()
    monkeypatch.setattr(ui, "fetch_recent_minutes", fetch)
    monkeypatch.setattr(ui, "commit_recent_capture", commit)
    view.run()
    fetch.assert_not_called()
    commit.assert_not_called()
    view.button(key="minute_recent_fetch").click().run()
    fetch.assert_not_called()
    assert view.error


def test_fetch_unknown_semantics_only_previews_and_rerun_does_not_fetch(view, monkeypatch):
    fetch = Mock(return_value=SimpleNamespace(report={"status": "review_required", "received_rows": 3,
        "symbol": "TL2612", "bar_label": "unverified"}, csv_bytes=None))
    monkeypatch.setattr(ui, "fetch_recent_minutes", fetch)
    view.run().text_input(key="minute_recent_symbol").set_value("TL2612").run()
    view.checkbox(key="minute_recent_network").check().run()
    view.button(key="minute_recent_fetch").click().run()
    assert not view.exception
    assert fetch.call_args.kwargs["bar_label"] == "unverified"
    view.run()
    fetch.assert_called_once()


def test_capture_change_or_missing_consent_blocks_commit(view, monkeypatch):
    fetch = Mock(return_value=SimpleNamespace(report={"status": "inspected", "received_rows": 3,
        "symbol": "TL2612", "bar_label": "end"}, csv_bytes=b"csv"))
    commit = Mock(return_value={"status": "imported", "batch_id": "minute-one"})
    monkeypatch.setattr(ui, "fetch_recent_minutes", fetch)
    monkeypatch.setattr(ui, "commit_recent_capture", commit)
    view.run().selectbox(key="minute_recent_label").set_value("end").run()
    view.text_input(key="minute_recent_symbol").set_value("TL2612").run()
    view.checkbox(key="minute_recent_network").check().run()
    view.button(key="minute_recent_fetch").click().run()
    view.button(key="minute_recent_commit").click().run()
    commit.assert_not_called()
    view.checkbox(key="minute_recent_consent").check().run()
    view.text_input(key="minute_recent_license").set_value("authorized research").run()
    view.text_input(key="minute_recent_symbol").set_value("TL2703").run()
    view.button(key="minute_recent_commit").click().run()
    commit.assert_not_called()
    view.text_input(key="minute_recent_symbol").set_value("TL2612").run()
    assert view.checkbox(key="minute_recent_consent").value is False
    view.checkbox(key="minute_recent_consent").check().run()
    view.button(key="minute_recent_commit").click().run()
    assert not view.exception
    commit.assert_called_once()
    fetch.assert_called_once()
    assert "minute-one" in view.session_state["minute_batch_options"]
    assert "minute_recent_capture" not in view.session_state


def test_failed_refetch_clears_old_preview(view, monkeypatch):
    fetch = Mock(side_effect=[SimpleNamespace(report={"status": "review_required"}, csv_bytes=None), ValueError("provider failed")])
    monkeypatch.setattr(ui, "fetch_recent_minutes", fetch)
    view.run().checkbox(key="minute_recent_network").check().run()
    view.button(key="minute_recent_fetch").click().run()
    view.button(key="minute_recent_fetch").click().run()
    assert view.error
    assert "minute_recent_capture" not in view.session_state


def test_component_project_change_clears_previous_capture(view):
    view.run()
    view.session_state["minute_recent_base"] = "another-project"
    view.session_state["minute_recent_capture"] = {"old": True}
    view.run()
    assert not view.exception
    assert "minute_recent_capture" not in view.session_state


def test_stale_request_with_fresh_consent_still_cannot_save(view, monkeypatch):
    commit = Mock()
    monkeypatch.setattr(ui, "commit_recent_capture", commit)
    view.run().checkbox(key="minute_recent_consent").check().run()
    view.text_input(key="minute_recent_license").set_value("research").run()
    view.button(key="minute_recent_commit").click().run()
    commit.assert_not_called()
    assert view.error


def test_invalid_save_result_is_not_presented_as_success(view, monkeypatch):
    monkeypatch.setattr(ui, "fetch_recent_minutes", Mock(return_value=SimpleNamespace(
        report={"status": "inspected"}, csv_bytes=b"csv")))
    monkeypatch.setattr(ui, "commit_recent_capture", Mock(return_value={"status": "failed"}))
    view.run().checkbox(key="minute_recent_network").check().run()
    view.button(key="minute_recent_fetch").click().run()
    view.text_input(key="minute_recent_license").set_value("research").run()
    view.checkbox(key="minute_recent_consent").check().run()
    view.button(key="minute_recent_commit").click().run()
    assert view.error
    assert not view.success
