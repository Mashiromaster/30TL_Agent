"""Main entrypoint routing, without models, providers, training or real data."""
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import dashboard_v2 as dashboard
from dashboard_workspace import SECTIONS


RENDERERS = {
    "market": "render_market_from_v1", "signal": "render_signal_v2", "sentiment": "render_sentiment_monitor",
    "macro": "render_macro_from_v1", "hybrid": "render_ai_fusion", "factor_analysis": "render_factor_from_v1",
    "factor_eval": "render_factor_eval", "genetic": "render_genetic_mining", "preparation": "render_data_preparation",
    "model_eval": "render_eval_tab", "evolution": "render_evolution_combined", "backtest": "render_backtest_from_v1",
    "memory": "render_memory_v2", "api": "render_connection_settings", "scheduler": "render_cron_tab",
}


def main_app():
    from dashboard_v2 import main
    main()


@pytest.fixture
def main_view(monkeypatch):
    monkeypatch.setattr(dashboard, "load_ctx", lambda: SimpleNamespace(df_factors=pd.DataFrame({"date": ["2026-06-02"]})))
    signal = Mock(side_effect=AssertionError("navigation must not load signal models"))
    monkeypatch.setattr(dashboard, "load_signal_dash", signal)
    workers = {route: Mock() for route in RENDERERS}
    for route, name in RENDERERS.items():
        monkeypatch.setattr(dashboard, name, workers[route])
    return AppTest.from_function(main_app), workers, signal


def test_main_home_never_loads_hidden_renderers_or_signal(main_view):
    app, workers, signal = main_view
    app.run()
    assert not app.exception
    assert not app.tabs
    assert all(worker.call_count == 0 for worker in workers.values())
    signal.assert_not_called()


@pytest.mark.parametrize("section,view", [(section, route) for section, config in SECTIONS.items() for route, _ in config["views"]])
def test_main_dispatches_exactly_one_renderer(main_view, section, view):
    app, workers, signal = main_view
    app.run().radio(key="workspace_section").set_value(section).run()
    if len(SECTIONS[section]["views"]) > 1:
        for worker in workers.values():
            worker.reset_mock()
        app.radio(key="workspace_view_" + section).set_value(view).run()
    assert not app.exception
    assert {route: worker.call_count for route, worker in workers.items()} == {route: int(route == view) for route in workers}
    signal.assert_not_called()


def settings_app():
    import streamlit as st
    from dashboard_v2 import render_connection_settings, _effective_api_key
    route = st.radio("测试页面", ["settings", "elsewhere"], key="test_route")
    if route == "settings":
        render_connection_settings()
    st.text("session_key_present=" + str(bool(_effective_api_key())))


def test_session_connection_survives_page_cleanup_without_export(monkeypatch):
    monkeypatch.setattr(dashboard, "_SERVER_DEEPSEEK_API_KEY", "")
    app = AppTest.from_function(settings_app).run()
    app.text_input(key="sidebar_api_key").set_value("test-session-only").run()
    app.radio(key="test_route").set_value("elsewhere").run()
    assert app.session_state["user_deepseek_api_key"] == "test-session-only"
    app.radio(key="test_route").set_value("settings").run()
    assert app.text_input(key="sidebar_api_key").value == "test-session-only"
    app.text_input(key="sidebar_api_key").set_value("").run()
    assert app.session_state["user_deepseek_api_key"] == ""
    assert not app.exception


def test_research_wrapper_does_not_render_preparation_or_monitor(monkeypatch):
    import hybrid_dashboard
    research = Mock()
    monitor = Mock(side_effect=AssertionError("hidden monitor rendered"))
    monkeypatch.setattr(hybrid_dashboard, "render_hybrid_workbench", research)
    monkeypatch.setattr(dashboard, "render_sentiment_monitor", monitor)
    ctx = object()
    dashboard.render_ai_fusion(ctx)
    assert research.call_args.kwargs["include_preparation"] is False
    monitor.assert_not_called()


def test_sentiment_snapshot_opens_existing_database_read_only(tmp_path, monkeypatch):
    import intelligence_events
    db = tmp_path / "data" / "intelligence" / "events.sqlite"
    db.parent.mkdir(parents=True)
    db.touch()
    store = Mock()
    store.list_source_health.return_value = []
    store.recent_event_records.return_value = []
    store.list_alerts.return_value = []
    store.list_deliveries.return_value = []
    factory = Mock(return_value=store)
    monkeypatch.setattr(intelligence_events, "IntelligenceEventStore", factory)
    monkeypatch.setattr(dashboard, "BASE_DIR", str(tmp_path))
    dashboard.load_sentiment_snapshot.clear()
    try:
        assert dashboard.load_sentiment_snapshot()["alerts"] == []
        factory.assert_called_once_with(str(tmp_path), read_only=True)
    finally:
        dashboard.load_sentiment_snapshot.clear()
