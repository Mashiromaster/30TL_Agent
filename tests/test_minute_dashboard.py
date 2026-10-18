from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

import minute_dashboard as ui


def app(base):
    import streamlit as st
    from minute_dashboard import render_minute_preparation
    st.text("selected=" + (render_minute_preparation(base) or "legacy"))


@pytest.fixture
def view(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "_upload", lambda: None)
    return AppTest.from_function(app, args=(str(tmp_path),))


def test_initial_render_does_not_read_import_or_build(view, monkeypatch):
    workers = [Mock(side_effect=AssertionError("implicit action")) for _ in range(4)]
    for name, worker in zip(["import_minute_csv", "list_sources", "build_minute_snapshot", "audit_local_minutes"], workers):
        monkeypatch.setattr(ui, name, worker)
    view.run()
    assert not view.exception
    assert all(worker.call_count == 0 for worker in workers)


def test_import_needs_file_metadata_and_consent(view, monkeypatch):
    worker = Mock()
    monkeypatch.setattr(ui, "import_minute_csv", worker)
    view.run().button(key="minute_import").click().run()
    worker.assert_not_called()
    assert view.error


def test_csv_inspection_does_not_persist(view, monkeypatch):
    monkeypatch.setattr(ui, "_upload", lambda: b"CSV bytes")
    worker = Mock(return_value={"status": "inspected", "quality": {"status": "incomplete"}})
    monkeypatch.setattr(ui, "import_minute_csv", worker)
    view.run()
    view.text_input(key="minute_license").set_value("local research").run()
    view.button(key="minute_inspect").click().run()
    assert not view.exception
    assert worker.call_args.args[1] == b"CSV bytes"
    assert worker.call_args.kwargs["persist"] is False


def test_imported_batch_is_not_automatically_selected(view, monkeypatch):
    monkeypatch.setattr(ui, "_upload", lambda: b"CSV bytes")
    monkeypatch.setattr(ui, "import_minute_csv", Mock(return_value={"status": "imported", "batch_id": "minute-test", "quality": {}}))
    view.run()
    view.text_input(key="minute_license").set_value("local research").run()
    view.checkbox(key="minute_import_consent").check().run()
    view.button(key="minute_import").click().run()
    assert not view.exception
    assert view.multiselect(key="minute_batches").value == []
    assert view.selectbox(key="minute_source").value == ""


def test_tq_export_format_requires_declared_interval_and_passes_options(view, monkeypatch):
    monkeypatch.setattr(ui, "_upload", lambda: b"export bytes")
    worker = Mock(return_value={"status": "inspected"})
    monkeypatch.setattr(ui, "import_minute_csv", worker)
    view.run().selectbox(key="minute_input_format").set_value("tqsdk").run()
    assert view.code[0].value.startswith("datetime,datetime_nano,CFFEX.TL2606.open,")
    assert not any(element.label == "下载 CSV 表头模板" for element in view.get("download_button"))
    view.text_input(key="minute_license").set_value("authorized export").run()
    view.button(key="minute_inspect").click().run()
    worker.assert_not_called()
    view.checkbox(key="minute_tq_interval").check().run()
    view.button(key="minute_inspect").click().run()
    assert not view.exception
    assert worker.call_args.kwargs["input_format"] == "tqsdk"
    assert worker.call_args.kwargs["bar_seconds"] == 60
    assert worker.call_args.kwargs["persist"] is False


def test_empty_snapshot_does_not_implicitly_include_legacy(view, monkeypatch):
    build = Mock()
    monkeypatch.setattr(ui, "build_minute_snapshot", build)
    view.run().button(key="minute_assemble").click().run()
    build.assert_not_called()


def test_reload_select_and_audit_are_explicit_read_only(view, monkeypatch):
    listing = Mock(return_value={"batches": [], "snapshots": [{"snapshot_id": "minset-test", "coverage": {"rows": 99}}]})
    audit = Mock(return_value={"status": "incomplete", "coverage": {"rows": 99}, "repair_ranges": []})
    monkeypatch.setattr(ui, "list_sources", listing)
    monkeypatch.setattr(ui, "audit_local_minutes", audit)
    view.run().button(key="minute_list").click().run()
    assert view.selectbox(key="minute_source").value == ""
    view.selectbox(key="minute_source").set_value("minset-test").run()
    audit.assert_not_called()
    view.button(key="minute_audit").click().run()
    assert audit.call_args.kwargs["snapshot_id"] == "minset-test"
    assert not view.exception


def test_failed_inspection_clears_old_success(view, monkeypatch):
    monkeypatch.setattr(ui, "_upload", lambda: b"CSV bytes")
    monkeypatch.setattr(ui, "import_minute_csv", Mock(side_effect=[{"status": "inspected"}, ValueError("invalid minute data")]))
    view.run()
    view.text_input(key="minute_license").set_value("research").run()
    view.button(key="minute_inspect").click().run()
    view.button(key="minute_inspect").click().run()
    assert view.error
    assert "minute_import_result" not in view.session_state


def test_legacy_inclusion_and_snapshot_selection_stay_separate(view, monkeypatch):
    build = Mock(return_value={"status": "built", "snapshot_id": "minset-built"})
    monkeypatch.setattr(ui, "build_minute_snapshot", build)
    view.run().checkbox(key="minute_include_legacy").check().run()
    view.button(key="minute_assemble").click().run()
    assert build.call_args.kwargs["include_legacy"] is True
    assert view.selectbox(key="minute_source").value == ""
    view.selectbox(key="minute_source").set_value("minset-built").run()
    assert not view.exception
    assert "selected=minset-built" in [element.value for element in view.text]


@pytest.mark.parametrize("bad", [None, {"status": "failed"}, {"status": "built"}])
def test_invalid_snapshot_result_cannot_become_a_source(view, monkeypatch, bad):
    monkeypatch.setattr(ui, "build_minute_snapshot", Mock(return_value=bad))
    view.run().checkbox(key="minute_include_legacy").check().run()
    view.button(key="minute_assemble").click().run()
    assert view.error
    assert view.selectbox(key="minute_source").value == ""


def test_import_does_not_accept_implicit_consent(view, monkeypatch):
    monkeypatch.setattr(ui, "_upload", lambda: b"CSV bytes")
    worker = Mock()
    monkeypatch.setattr(ui, "import_minute_csv", worker)
    view.run().text_input(key="minute_license").set_value("research").run()
    view.button(key="minute_import").click().run()
    worker.assert_not_called()
    assert view.error


def test_rejected_csv_result_is_not_success(view, monkeypatch):
    monkeypatch.setattr(ui, "_upload", lambda: b"CSV bytes")
    monkeypatch.setattr(ui, "import_minute_csv", Mock(return_value={"status": "rejected"}))
    view.run().text_input(key="minute_license").set_value("research").run()
    view.button(key="minute_inspect").click().run()
    assert view.error
    assert "minute_import_result" not in view.session_state


def test_audit_failure_clears_previous_report(view, monkeypatch):
    monkeypatch.setattr(ui, "audit_local_minutes", Mock(side_effect=[
        {"status": "incomplete", "coverage": {"rows": 1}, "repair_ranges": [{"ticker": "TL2612", "contract_days": 1}]},
        ValueError("bad source"),
    ]))
    view.run().button(key="minute_audit").click().run()
    assert not view.exception
    assert view.dataframe
    view.button(key="minute_audit").click().run()
    assert view.error
    assert "minute_quality" not in view.session_state


def test_coverage_summary_shows_counts_without_certifying_completeness(view, monkeypatch):
    monkeypatch.setattr(ui, "audit_local_minutes", Mock(return_value={
        "status": "incomplete", "coverage": {"rows": 477873, "incomplete_completed_contract_days": 335},
        "repair_summary": {"range_count": 5}, "repair_ranges": [],
    }))
    view.run().button(key="minute_audit").click().run()
    assert not view.exception
    assert [metric.value for metric in view.metric] == ["477,873", "335", "5"]
    assert any("不认证" in item.value for item in view.caption)


def test_listing_failure_is_not_presented_as_success(view, monkeypatch):
    monkeypatch.setattr(ui, "list_sources", Mock(side_effect=ValueError("invalid manifests")))
    view.run().button(key="minute_list").click().run()
    assert view.error
    assert not view.exception


def test_new_project_clears_previous_minute_choices(view):
    view.run()
    view.session_state["minute_base"] = "another-project"
    view.session_state["minute_quality"] = {"stale": True}
    view.run()
    assert not view.exception
    assert "minute_quality" not in view.session_state


def test_upload_helper_returns_bytes(monkeypatch):
    class Upload:
        def getvalue(self):
            return b"data"
    monkeypatch.setattr(ui.st, "file_uploader", Mock(side_effect=[None, Upload()]))
    assert ui._upload() is None
    assert ui._upload() == b"data"
    assert ui.st.file_uploader.call_args.kwargs["max_upload_size"] == 25
    assert "格式以上方选择为准" in ui.st.file_uploader.call_args.args[0]


def test_listing_wrappers_do_not_load_data(tmp_path, monkeypatch):
    import minute_store
    monkeypatch.setattr(minute_store, "list_minute_batches", lambda *a: [{"batch_id": "b"}])
    monkeypatch.setattr(minute_store, "list_minute_snapshots", lambda *a: [{"snapshot_id": "s"}])
    assert ui.list_sources(tmp_path) == {"batches": [{"batch_id": "b"}], "snapshots": [{"snapshot_id": "s"}]}


def paged_app(base):
    import streamlit as st
    from minute_dashboard import render_minute_preparation
    if st.radio("Page", ["data", "other"], key="test_page") == "data":
        render_minute_preparation(base)


def test_source_choices_survive_navigation_but_import_permissions_do_not(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "_upload", lambda: None)
    monkeypatch.setattr(ui, "list_sources", lambda *a: {"batches": [{"batch_id": "minute-test"}],
                                                       "snapshots": [{"snapshot_id": "minset-test"}]})
    app = AppTest.from_function(paged_app, args=(str(tmp_path),)).run()
    app.button(key="minute_list").click().run()
    app.multiselect(key="minute_batches").set_value(["minute-test"]).run()
    app.selectbox(key="minute_source").set_value("minset-test").run()
    app.checkbox(key="minute_import_consent").check().run()
    app.radio(key="test_page").set_value("other").run()
    app.radio(key="test_page").set_value("data").run()
    assert not app.exception
    assert app.selectbox(key="minute_source").value == "minset-test"
    assert app.multiselect(key="minute_batches").value == ["minute-test"]
    assert not app.checkbox(key="minute_import_consent").value
