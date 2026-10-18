import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import dashboard_workspace as ui


def workspace_app():
    import pandas as pd
    import streamlit as st
    from dashboard_workspace import SECTIONS, render_workspace
    class Context:
        df_factors = pd.DataFrame({"date": ["2026-06-02 15:15:00"]})
        @property
        def signal(self):
            raise AssertionError("overview must not infer signals")
    def renderer(name):
        return lambda: st.text("rendered=" + name)
    renderers = {view: renderer(view) for section in SECTIONS.values() for view, _ in section["views"]}
    render_workspace(Context(), renderers)


def test_home_does_not_render_hidden_tools():
    app = AppTest.from_function(workspace_app).run()
    assert not app.exception
    assert not any(item.value.startswith("rendered=") for item in app.text)
    assert app.radio(key="workspace_section").value == "overview"
    assert not app.tabs
    assert len(app.metric) == 4


@pytest.mark.parametrize("section", ["monitor", "research", "factors", "data", "review", "settings"])
def test_only_selected_destination_renders(section):
    app = AppTest.from_function(workspace_app).run()
    app.radio(key="workspace_section").set_value(section).run()
    assert not app.exception
    expected = ui.SECTIONS[section]["views"][0][0]
    assert [item.value for item in app.text if item.value.startswith("rendered=")] == ["rendered=" + expected]


def test_secondary_selection_survives_navigation():
    app = AppTest.from_function(workspace_app).run()
    app.radio(key="workspace_section").set_value("monitor").run()
    app.radio(key="workspace_view_monitor").set_value("sentiment").run()
    app.radio(key="workspace_section").set_value("overview").run()
    app.radio(key="workspace_section").set_value("monitor").run()
    assert not app.exception
    assert app.radio(key="workspace_view_monitor").value == "sentiment"
    assert any(item.value == "rendered=sentiment" for item in app.text)


def test_home_shortcut_navigates_without_running_research():
    app = AppTest.from_function(workspace_app).run()
    app.button(key="workspace_start_research").click().run()
    assert not app.exception
    assert app.radio(key="workspace_section").value == "research"
    assert [item.value for item in app.text if item.value.startswith("rendered=")] == ["rendered=hybrid"]


@pytest.mark.parametrize("dates,status", [
    ([], "missing"), (["bad"], "invalid"), (["2026-01-01"], "stale"),
    (["2026-10-05"], "future"), (["2026-10-04 10:00"], "recent"),
])
def test_factor_summary_is_honest_about_freshness(dates, status):
    result = ui.summarize_factors(pd.DataFrame({"date": dates}), now="2026-10-04 12:00+08:00")
    assert result["status"] == status


def test_missing_date_and_invalid_rows_are_not_certified():
    assert ui.summarize_factors(pd.DataFrame({"x": [1]}))["status"] == "invalid"
    result = ui.summarize_factors(pd.DataFrame({"date": ["2026-10-04", "invalid"]}), now="2026-10-04 12:00+08:00")
    assert result["invalid_rows"] == 1


def broken_snapshot_app():
    from dashboard_workspace import render_workspace
    class Broken:
        @property
        def df_factors(self):
            raise ValueError("snapshot unreadable")
    render_workspace(Broken(), {})


def test_home_handles_broken_local_snapshot_without_losing_navigation():
    app = AppTest.from_function(broken_snapshot_app).run()
    assert not app.exception
    assert app.radio(key="workspace_section").value == "overview"
    assert "snapshot unreadable" in app.error[0].value


def test_all_existing_destinations_are_reachable():
    routes = {view for section in ui.SECTIONS.values() for view, _ in section["views"]}
    assert {"market", "signal", "macro", "sentiment", "hybrid", "factor_analysis", "factor_eval",
            "genetic", "preparation", "model_eval", "evolution", "backtest", "memory", "api", "scheduler"} <= routes
