from types import SimpleNamespace

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import market_dashboard


def bars(dates, ticker="TL0", volume=100):
    return pd.DataFrame({
        "date": pd.to_datetime(dates), "ticker": ticker,
        "open": 100.0, "high": 103.0, "low": 99.0, "close": 102.0,
        "volume": volume, "oi": 200000,
    })


def test_quality_reports_missing_days_and_partial_volume():
    daily = bars(["2026-09-14", "2026-09-23", "2026-09-24"])
    minute = bars(["2026-09-23 13:55", "2026-09-24 09:31"], volume=50)
    warnings = market_dashboard.minute_quality_warnings(minute, daily)
    assert any("2026-09-14" in text for text in warnings)
    assert any("成交量" in text and "2" in text for text in warnings)


def test_matching_daily_reference_has_no_gap_or_volume_warning():
    daily = bars(["2026-09-24"])
    minute = bars(["2026-09-24 09:31"])
    assert market_dashboard.minute_quality_warnings(minute, daily) == []


def test_empty_daily_reference_does_not_claim_completeness():
    warning = market_dashboard.minute_quality_warnings(bars(["2026-09-24 09:31"]), pd.DataFrame())
    assert any("无法核对" in text for text in warning)


def _market_app():
    import os
    from types import SimpleNamespace
    from market_dashboard import render_market_tab
    render_market_tab(SimpleNamespace(base_dir=os.environ["TEST_MARKET_BASE"], df_factors=None))


def test_daily_chart_uses_daily_source_not_minute_history(tmp_path, monkeypatch):
    calls = []
    frame = bars(["2026-09-03", "2026-09-14", "2026-09-30"])

    def fake_load(base_dir, symbol, frequency, refresh=True):
        calls.append((symbol, frequency, refresh))
        return SimpleNamespace(bars=frame, error=None, from_cache=False, fetched_at="2026-09-30T16:00:00+08:00")

    monkeypatch.setenv("TEST_MARKET_BASE", str(tmp_path))
    monkeypatch.setattr(market_dashboard, "_load_snapshot", fake_load)
    app = AppTest.from_function(_market_app).run()
    assert not app.exception
    assert calls == [("TL0", "1D", False)]
    assert app.selectbox(key="market_symbol_TL").value == "TL0"
    assert any("主力连续" in info.value for info in app.info)
    assert len(app.get("plotly_chart")) == 1


def test_failure_is_visible_and_does_not_fabricate_daily_bars(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MARKET_BASE", str(tmp_path))
    monkeypatch.setattr(market_dashboard, "_load_snapshot", lambda *args, **kwargs: SimpleNamespace(
        bars=pd.DataFrame(), error="上游超时", from_cache=False, fetched_at=None))
    app = AppTest.from_function(_market_app).run()
    assert not app.exception
    assert any("上游超时" in message.value for message in app.warning)
    assert not app.get("plotly_chart")


def test_cached_network_failure_remains_visible(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MARKET_BASE", str(tmp_path))
    monkeypatch.setattr(market_dashboard, "_load_snapshot", lambda *args, **kwargs: SimpleNamespace(
        bars=bars(["2026-09-03"]), error="上游超时", from_cache=True,
        fetched_at="2026-09-03T16:00:00+08:00"))
    app = AppTest.from_function(_market_app).run()
    assert not app.exception
    assert any("上游超时" in message.value for message in app.warning)
    assert any("缓存" in message.value for message in app.caption)


def test_minute_chart_does_not_mix_in_old_contracts(tmp_path, monkeypatch):
    calls = []
    def fake_load(base_dir, symbol, frequency, refresh=True):
        calls.append((symbol, frequency))
        frame = bars(["2026-09-23 13:55", "2026-09-30 15:15"]) if frequency == "1min" else bars(["2026-09-14", "2026-09-23", "2026-09-30"])
        return SimpleNamespace(bars=frame, error=None, from_cache=False, fetched_at="2026-09-30T16:00:00+08:00")
    monkeypatch.setenv("TEST_MARKET_BASE", str(tmp_path))
    monkeypatch.setattr(market_dashboard, "_load_snapshot", fake_load)
    app = AppTest.from_function(_market_app).run()
    app.selectbox(key="market_resolution").select("5分钟").run()
    assert not app.exception
    assert ("TL0", "1min") in calls
    assert all(symbol == "TL0" for symbol, _ in calls)
    assert any("2026-09-14" in warning.value for warning in app.warning)


def test_missing_date_not_mislabelled_non_trading_day(tmp_path, monkeypatch):
    from datetime import date
    monkeypatch.setenv("TEST_MARKET_BASE", str(tmp_path))
    monkeypatch.setattr(market_dashboard, "_load_snapshot", lambda *args, **kwargs: SimpleNamespace(
        bars=bars(["2026-09-03", "2026-09-30"]), error=None, from_cache=False, fetched_at=None))
    app = AppTest.from_function(_market_app).run()
    app.date_input(key="market_specific_date").set_value(date(2026, 9, 14)).run()
    assert not app.exception
    assert any("无返回记录" in message.value for message in app.info)
    assert not any("非交易日" in message.value for message in app.info)


def test_invalid_contract_month_is_rejected_without_page_exception(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MARKET_BASE", str(tmp_path))
    monkeypatch.setattr(market_dashboard, "_load_snapshot", lambda *args, **kwargs: SimpleNamespace(
        bars=bars(["2026-09-03"]), error=None, from_cache=False, fetched_at=None))
    app = AppTest.from_function(_market_app).run()
    monkeypatch.setattr(market_dashboard, "_load_snapshot", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("invalid symbol")))
    app.text_input(key="market_custom_TL").input("TL2613").run()
    assert not app.exception
    assert any("合约代码" in message.value for message in app.error)


def test_stale_daily_reference_does_not_hide_newer_minutes(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MARKET_BASE", str(tmp_path))
    def fake_load(base_dir, symbol, frequency, refresh=True):
        return SimpleNamespace(
            bars=bars(["2026-09-30 09:31"] if frequency == "1min" else ["2026-09-29"]),
            error=None if frequency == "1min" else "daily timeout",
            from_cache=frequency == "1D", fetched_at=None,
        )
    monkeypatch.setattr(market_dashboard, "_load_snapshot", fake_load)
    app = AppTest.from_function(_market_app).run()
    app.selectbox(key="market_resolution").select("1分钟").run()
    assert not app.exception
    assert len(app.get("plotly_chart")) == 1
    assert any("日线参考" in warning.value for warning in app.warning)


def _recording_app(tmp_path, monkeypatch):
    calls = []

    def fake_load(base_dir, symbol, frequency, refresh=False):
        calls.append((str(base_dir), symbol, frequency, refresh))
        return SimpleNamespace(
            bars=bars(["2026-09-30 09:31"] if frequency == "1min" else ["2026-09-30"], ticker=symbol),
            error=None, from_cache=not refresh, fetched_at="2026-09-30T16:00:00+08:00",
        )

    monkeypatch.setenv("TEST_MARKET_BASE", str(tmp_path))
    monkeypatch.setattr(market_dashboard, "_load_snapshot", fake_load)
    return AppTest.from_function(_market_app).run(), calls


def test_navigation_and_filter_changes_are_cached_only(tmp_path, monkeypatch):
    from datetime import date

    app, calls = _recording_app(tmp_path, monkeypatch)
    assert app.toggle(key="market_network").value is False
    assert app.button(key="market_refresh").disabled
    app.run()
    app.selectbox(key="market_lookback").select("全部").run()
    app.selectbox(key="market_resolution").select("1分钟").run()
    contract = app.selectbox(key="market_symbol_TL").options[1]
    app.selectbox(key="market_symbol_TL").select(contract).run()
    app.date_input(key="market_specific_date").set_value(date(2026, 9, 30)).run()
    assert not app.exception
    assert all(refresh is False for _, _, _, refresh in calls)
    assert calls[-2:] == [(str(tmp_path), contract, "1min", False), (str(tmp_path), contract, "1D", False)]
    assert len(app.get("plotly_chart")) == 1


def test_permission_alone_and_unconfirmed_click_never_fetch(tmp_path, monkeypatch):
    app, calls = _recording_app(tmp_path, monkeypatch)
    app.button(key="market_refresh").click().run()
    app.toggle(key="market_network").set_value(True).run()
    assert not app.exception
    assert not app.button(key="market_refresh").disabled
    assert all(refresh is False for _, _, _, refresh in calls)


def test_confirmed_refresh_is_one_shot_not_repeated_by_filters(tmp_path, monkeypatch):
    app, calls = _recording_app(tmp_path, monkeypatch)
    app.toggle(key="market_network").set_value(True).run()
    app.button(key="market_refresh").click().run()
    assert not app.exception
    assert [(symbol, frequency) for _, symbol, frequency, refresh in calls if refresh] == [("TL0", "1D")]
    assert any("新浪接口快照" in item.value for item in app.caption)
    app.run()
    app.selectbox(key="market_lookback").select("全部").run()
    app.selectbox(key="market_resolution").select("1分钟").run()
    app.text_input(key="market_custom_TL").input("TL2609").run()
    assert not app.exception
    assert len([call for call in calls if call[3]]) == 1
    assert any("本地缓存" in item.value for item in app.caption)
    calls.clear()
    app.button(key="market_refresh").click().run()
    assert calls == [(str(tmp_path), "TL2609", "1min", True), (str(tmp_path), "TL2609", "1D", True)]
    app.run()
    assert calls[-2:] == [(str(tmp_path), "TL2609", "1min", False), (str(tmp_path), "TL2609", "1D", False)]


def test_explicit_refresh_bypasses_cache_and_invalidates_cached_only_key(tmp_path, monkeypatch):
    calls = []
    stored = {}

    def fake_load(base_dir, symbol, frequency, refresh=False):
        key = (str(base_dir), symbol, frequency)
        calls.append((*key, refresh))
        if refresh:
            stored[key] = stored.get(key, 0) + 1
        return SimpleNamespace(revision=stored.get(key, 0), from_cache=not refresh)

    monkeypatch.setattr(market_dashboard, "load_quote_series", fake_load)
    root = str(tmp_path)
    other_root = str(tmp_path / "other")
    assert market_dashboard._load_snapshot(root, "TL2606", "1D").revision == 0
    assert market_dashboard._load_snapshot(root, "TL2606", "1D").revision == 0
    market_dashboard._load_snapshot(root, "TL2609", "1D", False)
    market_dashboard._load_snapshot(root, "TL2606", "1min", False)
    market_dashboard._load_snapshot(other_root, "TL2606", "1D", False)
    assert len(calls) == 4
    assert market_dashboard._load_snapshot(root, "TL2606", "1D", True).revision == 1
    assert market_dashboard._load_snapshot(root, "TL2606", "1D", False).revision == 1
    assert market_dashboard._load_snapshot(root, "TL2606", "1D", True).revision == 2
    assert market_dashboard._load_snapshot(root, "TL2606", "1D", False).revision == 2
    market_dashboard._load_snapshot(root, "TL2609", "1D", False)
    market_dashboard._load_snapshot(root, "TL2606", "1min", False)
    market_dashboard._load_snapshot(other_root, "TL2606", "1D", False)
    assert len(calls) == 8


def test_refreshed_data_remains_visible_on_cached_only_rerender(tmp_path, monkeypatch):
    calls = []
    cache = {}

    def fake_load(base_dir, symbol, frequency, refresh=False):
        key = (str(base_dir), symbol, frequency)
        calls.append((*key, refresh))
        if refresh:
            cache[key] = bars(["2026-09-30"], ticker=symbol)
        return SimpleNamespace(
            bars=cache.get(key, pd.DataFrame()), error=None, from_cache=not refresh,
            fetched_at="2026-09-30T16:00:00+08:00" if key in cache else None,
        )

    monkeypatch.setattr(market_dashboard, "load_quote_series", fake_load)
    monkeypatch.setenv("TEST_MARKET_BASE", str(tmp_path))
    app = AppTest.from_function(_market_app).run()
    assert not app.get("plotly_chart")
    app.toggle(key="market_network").set_value(True).run()
    app.button(key="market_refresh").click().run()
    assert not app.exception
    assert len(app.get("plotly_chart")) == 1
    app.selectbox(key="market_lookback").select("全部").run()
    assert not app.exception
    assert len(app.get("plotly_chart")) == 1
    assert any("本地缓存（非本轮实时获取）" in item.value for item in app.caption)
    assert [call[3] for call in calls] == [False, True, False]
    app.text_input(key="market_custom_TL").input("TL2609").run()
    assert not app.get("plotly_chart")
    assert calls[-1][1:] == ("TL2609", "1D", False)
    app.text_input(key="market_custom_TL").input("").run()
    assert len(app.get("plotly_chart")) == 1
    assert len([call for call in calls if call[3]]) == 1


def test_failed_explicit_refresh_still_invalidates_only_local_view(tmp_path, monkeypatch):
    calls = []

    def fake_load(base_dir, symbol, frequency, refresh=False):
        calls.append(refresh)
        if refresh:
            raise ValueError("unexpected cache write failure")
        return SimpleNamespace(from_cache=True)

    monkeypatch.setattr(market_dashboard, "load_quote_series", fake_load)
    args = (str(tmp_path), "TL2606", "1D")
    market_dashboard._load_snapshot(*args)
    with pytest.raises(ValueError, match="cache write"):
        market_dashboard._load_snapshot(*args, refresh=True)
    assert market_dashboard._load_snapshot(*args).from_cache
    assert calls == [False, True, False]
