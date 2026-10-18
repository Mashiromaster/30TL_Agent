"""Market bars must preserve identity, price extrema, and cache provenance."""

import hashlib
import json
from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from market_data import (
    VARIETY_CONFIG,
    contract_candidates,
    fetch_bars,
    load_quote_series,
    resample_ohlc,
)


def response_session(rows):
    response = Mock()
    response.text = "/* vendor comment */\n=(" + json.dumps(rows) + ");"
    session = Mock()
    session.get.return_value = response
    return session


def quote_row(date="2026-09-30 09:30:00", **changes):
    row = {"d": date, "o": "115.5", "h": "116.5", "l": "115.0",
           "c": "116.0", "v": "12", "p": "237610"}
    row.update(changes)
    return row


def bars(symbol="TL0", frequency="1min", **changes):
    date = "2026-09-30" if frequency == "1D" else "2026-09-30 09:30:00"
    return fetch_bars(symbol, frequency, session=response_session([quote_row(date, **changes)]))


@pytest.mark.parametrize("oi", ["0", "50000", "237610"])
def test_continuous_identity_never_depends_on_open_interest(oi):
    frame = bars(p=oi)
    assert set(frame.ticker) == set(frame.source_symbol) == {"TL0"}
    assert set(frame.series_kind) == {"continuous"}
    assert frame.oi.iloc[0] == float(oi)
    assert frame.source.iloc[0] == "sina"
    assert frame.date.iloc[0] == pd.Timestamp("2026-09-30 09:30")
    assert frame.trade_dt.iloc[0] == pd.Timestamp("2026-09-30")
    assert frame.time.iloc[0] == "09:30"


def test_contract_identity_is_explicit():
    frame = bars("TL2612")
    assert frame.ticker.tolist() == ["TL2612"]
    assert frame.series_kind.tolist() == ["contract"]


@pytest.mark.parametrize("frequency,endpoint,period", [
    ("1min", "getFewMinLine", "1"), ("1D", "getDailyKLine", None),
])
def test_timeout_and_endpoint_are_explicit(frequency, endpoint, period):
    date = "2026-09-30" if frequency == "1D" else "2026-09-30 09:30:00"
    session = response_session([quote_row(date)])
    fetch_bars("TL0", frequency, timeout=7, session=session)
    args, kwargs = session.get.call_args
    assert args[0].endswith(endpoint)
    assert kwargs["timeout"] == 7
    assert kwargs["params"]["symbol"] == "TL0"
    if period:
        assert kwargs["params"]["type"] == period
    session.get.return_value.raise_for_status.assert_called_once()


def test_rows_are_sorted_deduplicated_and_keys_not_positional():
    late = quote_row("2026-09-30 09:31:00")
    early = dict(reversed(list(quote_row().items())))
    revised = quote_row(c="116.2")
    frame = fetch_bars("TL0", session=response_session([late, early, revised]))
    assert len(frame) == 2
    assert frame.date.is_monotonic_increasing
    assert frame.close.tolist() == [116.2, 116.0]


@pytest.mark.parametrize("rows", [[], None, {}, {"error": "not found"}, [["2026-09-30", "1"]]])
def test_empty_or_unexpected_response_is_not_accepted(rows):
    with pytest.raises(ValueError):
        fetch_bars("TL0", session=response_session(rows))


@pytest.mark.parametrize("changes", [
    {"h": "115.2"}, {"l": "117"}, {"o": "nan"}, {"c": "Infinity"},
    {"v": "-1"}, {"p": "-1"}, {"d": "not a date"}, {"c": "0"},
])
def test_bad_prices_dates_and_volumes_fail_closed(changes):
    with pytest.raises(ValueError):
        bars(**changes)


def test_missing_column_or_malformed_json_is_reported():
    row = quote_row()
    del row["h"]
    with pytest.raises(ValueError, match="high|字段|schema"):
        fetch_bars("TL0", session=response_session([row]))
    session = response_session([])
    session.get.return_value.text = "<html>error</html>"
    with pytest.raises(ValueError):
        fetch_bars("TL0", session=session)


def test_network_failure_is_reported():
    session = Mock()
    session.get.side_effect = requests.Timeout("provider timeout")
    with pytest.raises((ValueError, requests.RequestException), match="timeout"):
        fetch_bars("TL0", session=session)


@pytest.mark.parametrize("symbol,frequency", [("../../escape", "1D"), ("TL0", "1h"), ("BAD0", "1min"), ("TL2613", "1D")])
def test_invalid_requests_do_not_access_network(symbol, frequency):
    session = Mock()
    with pytest.raises(ValueError):
        fetch_bars(symbol, frequency, session=session)
    session.get.assert_not_called()


def test_contract_candidates_roll_calendar_without_open_interest_guesses():
    assert contract_candidates("TL", "2026-09-30") == ["TL2609", "TL2612", "TL2703"]
    assert contract_candidates("T", "2026-12-31") == ["T2612", "T2703", "T2706"]
    assert contract_candidates("TF", "2026-01-01") == ["TF2603", "TF2606", "TF2609"]
    assert contract_candidates("TS", "2026-03-31T20:00:00Z")[0] == "TS2606"
    assert all("oi_thresholds" not in cfg and "oi_fallback" not in cfg for cfg in VARIETY_CONFIG.values())
    with pytest.raises(ValueError):
        contract_candidates("BAD")
    with pytest.raises(ValueError):
        contract_candidates("TL", pd.NaT)
    assert len(contract_candidates("TL")) == 3


def test_ohlc_aggregation_uses_each_real_column_and_never_fills_holes():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2026-09-28 09:30", "2026-09-28 09:31", "2026-09-30 09:30"]),
        "open": [115.0, 115.7, 117.0], "high": [116.0, 116.8, 117.5],
        "low": [114.5, 115.5, 116.5], "close": [115.7, 116.0, 117.2],
        "volume": [3, 5, 7], "ticker": ["TL0"] * 3, "oi": [10, 11, 12],
        "Market_Regime": [1, 2, 3],
    })
    original = frame.copy(deep=True)
    daily = resample_ohlc(frame, "1D")
    assert len(daily) == 2
    first = daily.iloc[0]
    assert (first.open, first.high, first.low, first.close, first.volume) == (115.0, 116.8, 114.5, 116.0, 8)
    assert first.oi == 11 and first.Market_Regime == 2
    assert daily.ticker.tolist() == ["TL0", "TL0"]
    pd.testing.assert_frame_equal(frame, original)


def test_aggregation_rejects_mixed_tickers_and_missing_prices():
    frame = pd.concat([bars("TL0"), bars("TL2612")], ignore_index=True)
    with pytest.raises(ValueError, match="ticker|合约|series"):
        resample_ohlc(frame, "1D")
    with pytest.raises(ValueError, match="open|字段|schema"):
        resample_ohlc(bars().drop(columns=["open"]), "1D")
    assert resample_ohlc(pd.DataFrame(), "1D").empty


def test_hourly_bars_keep_session_boundaries_and_actual_timestamp_labels():
    timestamps = ["2026-09-30 " + time for time in ("09:31", "10:31", "11:30", "13:01", "15:15")]
    frame = pd.concat([bars() for _ in timestamps], ignore_index=True)
    frame["date"] = pd.to_datetime(timestamps)
    hourly = resample_ohlc(frame, "1h")
    assert hourly.date.dt.strftime("%H:%M").tolist() == ["09:31", "10:31", "13:01", "15:15"]
    assert hourly.volume.tolist() == [12, 24, 12, 12]
    assert set(hourly.date).issubset(set(frame.date))


def test_wide_intraday_bars_never_join_across_lunch():
    frame = pd.concat([bars(), bars()], ignore_index=True)
    frame["date"] = pd.to_datetime(["2026-09-30 11:30", "2026-09-30 13:01"])
    assert len(resample_ohlc(frame, "8h")) == 2


def test_cache_is_isolated_and_cache_only_does_not_fetch(tmp_path):
    fetcher = Mock(return_value=bars())
    first = load_quote_series(tmp_path, "TL0", "1min", fetcher=fetcher)
    assert first.error is None and not first.from_cache and first.fetched_at
    assert list((tmp_path / "data" / "market_quotes").glob("*.json"))
    fetcher.reset_mock()
    cached = load_quote_series(tmp_path, "TL0", "1min", refresh=False, fetcher=fetcher)
    fetcher.assert_not_called()
    assert cached.error is None and cached.from_cache and cached.fetched_at == first.fetched_at
    pd.testing.assert_frame_equal(cached.bars, first.bars)
    assert load_quote_series(tmp_path, "TL2612", "1min", refresh=False).bars.empty
    assert load_quote_series(tmp_path, "TL0", "1D", refresh=False).bars.empty


def test_network_failure_falls_back_with_visible_error_and_original_timestamp(tmp_path):
    cached = load_quote_series(tmp_path, "TL0", "1D", fetcher=lambda *args: bars(frequency="1D"))
    failed = load_quote_series(tmp_path, "TL0", "1D", fetcher=Mock(side_effect=RuntimeError("offline")))
    assert failed.from_cache and not failed.bars.empty
    assert "offline" in failed.error and failed.fetched_at == cached.fetched_at
    missing = load_quote_series(tmp_path, "TL2612", "1D", fetcher=Mock(side_effect=RuntimeError("offline")))
    assert missing.bars.empty and missing.error and not missing.from_cache and missing.fetched_at is None


@pytest.mark.parametrize("mutation", ["symbol", "frequency", "ticker", "source_symbol", "ohlc", "fetched_at"])
def test_cache_metadata_and_bars_are_validated_before_fallback(tmp_path, mutation):
    load_quote_series(tmp_path, "TL0", "1min", fetcher=lambda *args: bars())
    path = next((tmp_path / "data" / "market_quotes").glob("*.json"))
    payload = json.loads(path.read_text())
    if mutation == "symbol":
        payload["symbol"] = "TL2612"
    elif mutation == "frequency":
        payload["frequency"] = "1D"
    elif mutation == "fetched_at":
        payload["fetched_at"] = "not a date"
    elif mutation == "ohlc":
        payload["bars"][0]["high"] = 0
    else:
        payload["bars"][0][mutation] = "TL2612"
    path.write_text(json.dumps(payload))
    result = load_quote_series(tmp_path, "TL0", "1min", refresh=False)
    assert result.bars.empty and result.error and not result.from_cache


def test_invalid_new_data_cannot_overwrite_valid_cache(tmp_path):
    original = load_quote_series(tmp_path, "TL0", "1min", fetcher=lambda *args: bars())
    wrong = load_quote_series(tmp_path, "TL0", "1min", fetcher=lambda *args: bars("TL2612"))
    assert wrong.from_cache and wrong.error
    pd.testing.assert_frame_equal(wrong.bars, original.bars)
    empty = load_quote_series(tmp_path, "TL0", "1min", fetcher=lambda *args: pd.DataFrame())
    assert empty.error and empty.from_cache


def test_cache_write_failure_does_not_discard_valid_live_data(tmp_path, monkeypatch):
    monkeypatch.setattr("market_data.os.replace", Mock(side_effect=OSError("disk full")))
    result = load_quote_series(tmp_path, "TL0", "1min", fetcher=lambda *args: bars())
    assert not result.bars.empty and not result.from_cache
    assert "disk full" in result.error
    assert not list((tmp_path / "data" / "market_quotes").glob("*.tmp"))


@pytest.mark.parametrize("timeout", [None, 0, -1, float("inf"), True])
def test_timeout_must_be_bounded_and_positive(timeout):
    with pytest.raises(ValueError, match="timeout"):
        fetch_bars("TL0", timeout=timeout, session=Mock())


def test_provider_payload_rejects_missing_date_boolean_and_invalid_json():
    for changes in ({"d": None}, {"v": True}, {"c": "not-a-number"}):
        with pytest.raises(ValueError):
            bars(**changes)
    session = response_session([])
    session.get.return_value.text = "=({invalid});"
    with pytest.raises(ValueError, match="JSON"):
        fetch_bars("TL0", session=session)


def test_timezone_aware_provider_dates_are_converted_to_shanghai():
    frame = bars(d="2026-09-30T01:30:00+00:00")
    assert frame.date.iloc[0] == pd.Timestamp("2026-09-30 09:30")


@pytest.mark.parametrize("new_frame", [
    None,
    pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"]),
])
def test_fetcher_wrong_type_or_empty_typed_frame_reports_failure(tmp_path, new_frame):
    result = load_quote_series(tmp_path, "TL0", "1D", fetcher=lambda *args: new_frame)
    assert result.error and result.bars.empty


def test_missing_oi_and_intraday_payload_for_daily_frequency_are_rejected(tmp_path):
    result = load_quote_series(tmp_path, "TL0", "1min", fetcher=lambda *args: bars().drop(columns="oi"))
    assert result.error and "oi" in result.error
    result = load_quote_series(tmp_path, "TL0", "1D", fetcher=lambda *args: bars())
    assert result.error and "intraday" in result.error


def test_weekly_bars_and_invalid_aggregation_frequencies():
    assert len(resample_ohlc(bars(), "W")) == 1
    with pytest.raises(ValueError, match="positive"):
        resample_ohlc(bars(), "0min")
    plain = bars().drop(columns=["ticker", "source_symbol", "series_kind", "oi"])
    assert len(resample_ohlc(plain, "1D")) == 1


@pytest.mark.parametrize("timestamp", [None, "2026-09-30T10:00:00", "NaT"])
def test_invalid_cache_timestamp_and_network_failure_are_both_visible(tmp_path, timestamp):
    load_quote_series(tmp_path, "TL0", "1min", fetcher=lambda *args: bars())
    path = next((tmp_path / "data" / "market_quotes").glob("*.json"))
    payload = json.loads(path.read_text())
    payload["fetched_at"] = timestamp
    path.write_text(json.dumps(payload))
    result = load_quote_series(tmp_path, "TL0", "1min", fetcher=Mock(side_effect=RuntimeError("offline")))
    assert "offline" in result.error and "cache" in result.error and result.bars.empty


def test_failure_before_temp_file_creation_is_reported_without_losing_bars(tmp_path, monkeypatch):
    monkeypatch.setattr("market_data.tempfile.NamedTemporaryFile", Mock(side_effect=OSError("disk full")))
    result = load_quote_series(tmp_path, "TL0", "1min", fetcher=lambda *args: bars())
    assert not result.bars.empty and "disk full" in result.error


def test_default_fetcher_uses_requests_with_timeout(tmp_path, monkeypatch):
    session = response_session([quote_row()])
    monkeypatch.setattr("market_data.requests.get", session.get)
    result = load_quote_series(tmp_path, "TL0", "1min")
    assert not result.bars.empty and result.error is None
    assert session.get.call_args.kwargs["timeout"] == 15


def strict_session(rows=None, *, body=None, status=200, headers=None, chunks=None):
    """Streaming response fixture: strict tests never contact a provider."""
    raw = body if body is not None else ("=(" + json.dumps(rows) + ");").encode()
    response = Mock()
    response.status_code = status
    response.headers = headers or {}
    response.iter_content.return_value = chunks if chunks is not None else [raw]
    session = Mock()
    session.get.return_value = response
    return session, raw


def test_strict_capture_is_bounded_single_request_and_preserves_provenance():
    session, raw = strict_session([quote_row()])
    before = pd.Timestamp.now(tz="UTC")
    frame = fetch_bars("TL2612", strict=True, timeout=30, session=session)
    after = pd.Timestamp.now(tz="UTC")
    assert len(frame) == 1 and frame.ticker.tolist() == ["TL2612"]
    args, kwargs = session.get.call_args
    assert args[0].endswith("getFewMinLine")
    assert kwargs == {"params": {"symbol": "TL2612", "type": "1"},
                      "timeout": 30, "stream": True, "allow_redirects": False}
    session.get.assert_called_once()
    session.get.return_value.close.assert_called_once()
    capture = frame.attrs["source_capture"]
    assert capture == {"provider": "sina", "endpoint_id": "sina_futures_1min",
                       "symbol": "TL2612", "frequency": "1min",
                       "response_sha256": hashlib.sha256(raw).hexdigest(),
                       "fetched_at": capture["fetched_at"], "request_count": 1,
                       "bar_label": "unverified"}
    assert before <= pd.Timestamp(capture["fetched_at"]) <= after


@pytest.mark.parametrize("kwargs", [{"frequency": "1D"}, {"timeout": 30.01}, {"strict": "yes"}])
def test_strict_invalid_request_never_fetches(kwargs):
    session = Mock()
    with pytest.raises(ValueError):
        fetch_bars("TL2612", session=session, **({"strict": True} | kwargs))
    session.get.assert_not_called()


@pytest.mark.parametrize("status", [301, 302, 307, 403, 500])
def test_strict_never_follows_redirects_retries_or_reads_error_bodies(status):
    session, _ = strict_session([quote_row()], status=status)
    with pytest.raises((ValueError, requests.HTTPError)):
        fetch_bars("TL2612", strict=True, session=session)
    session.get.assert_called_once()
    session.get.return_value.iter_content.assert_not_called()
    session.get.return_value.close.assert_called_once()


@pytest.mark.parametrize("headers,chunks", [
    ({"Content-Length": str(2 * 1024 * 1024 + 1)}, []),
    ({"Content-Length": "invalid"}, []),
    ({"Content-Length": "-1"}, []),
    ({}, [b"x" * (2 * 1024 * 1024), b"x"]),
    ({"Content-Length": "10"}, [b"x" * (2 * 1024 * 1024 + 1)]),
])
def test_strict_caps_declared_and_actual_response_bytes(headers, chunks):
    session, _ = strict_session([], headers=headers, chunks=chunks)
    with pytest.raises(ValueError, match="size|length|limit"):
        fetch_bars("TL2612", strict=True, session=session)
    session.get.return_value.close.assert_called_once()


def test_strict_response_closes_after_stream_error():
    session, _ = strict_session([])
    session.get.return_value.iter_content.side_effect = requests.Timeout("stream timeout")
    with pytest.raises(requests.Timeout):
        fetch_bars("TL2612", strict=True, session=session)
    session.get.assert_called_once()
    session.get.return_value.close.assert_called_once()


def test_strict_rejects_nonbyte_stream_and_accepts_empty_chunks():
    session, raw = strict_session([quote_row()])
    session.get.return_value.iter_content.return_value = [b"", raw[:7], b"", raw[7:]]
    assert len(fetch_bars("TL2612", strict=True, session=session)) == 1
    session, _ = strict_session([], chunks=["not bytes"])
    with pytest.raises(ValueError, match="byte stream"):
        fetch_bars("TL2612", strict=True, session=session)
    session.get.return_value.close.assert_called_once()


def test_strict_closes_response_when_raise_for_status_raises():
    session, _ = strict_session([], status=403)
    session.get.return_value.raise_for_status.side_effect = requests.HTTPError("403 Forbidden")
    with pytest.raises(requests.HTTPError):
        fetch_bars("TL2612", strict=True, session=session)
    session.get.assert_called_once()
    session.get.return_value.iter_content.assert_not_called()
    session.get.return_value.close.assert_called_once()


@pytest.mark.parametrize("changes", [{}, {"c": "116.2"}])
def test_strict_rejects_identical_and_conflicting_duplicate_bars(changes):
    session, _ = strict_session([quote_row(), quote_row(**changes)])
    with pytest.raises(ValueError, match="duplicate|Duplicate"):
        fetch_bars("TL2612", strict=True, session=session)


@pytest.mark.parametrize("changes", [
    {"ticker": "TL0"}, {"source_symbol": "TL0"}, {"source": "other"},
    {"series_kind": "continuous"}, {"symbol": "TL0"}, {"date": "2026-09-29"},
    {"unexpected": "ignored?"}, {"v": True}, {"p": False},
    {"d": 1790725800000000000}, {"d": 1790725800.0}, {"d": True},
])
def test_strict_rejects_raw_identity_schema_and_type_anomalies(changes):
    session, _ = strict_session([quote_row() | changes])
    with pytest.raises(ValueError):
        fetch_bars("TL2612", strict=True, session=session)


def test_strict_accepts_matching_identity_but_requires_open_interest():
    session, _ = strict_session([quote_row(ticker="TL2612", source_symbol="TL2612",
                                         series_kind="contract", source="sina", symbol="TL2612")])
    assert fetch_bars("TL2612", strict=True, session=session).ticker.tolist() == ["TL2612"]
    row = quote_row()
    del row["p"]
    session, _ = strict_session([row])
    with pytest.raises(ValueError, match="oi|interest|schema|fields"):
        fetch_bars("TL2612", strict=True, session=session)


@pytest.mark.parametrize("body", [
    b'=( [{"d":"2026-09-30 09:31", "d":"2026-09-30 09:32"}] );',
    b"=(\xff);", b"<html>not JSONP</html>", b"=({invalid});",
])
def test_strict_rejects_duplicate_json_keys_encoding_and_malformed_payload(body):
    session, _ = strict_session(body=body)
    with pytest.raises(ValueError):
        fetch_bars("TL2612", strict=True, session=session)
    session.get.return_value.close.assert_called_once()


def test_strict_rejects_oversized_row_count_before_normalizing():
    session, _ = strict_session([quote_row()] * 5001)
    with pytest.raises(ValueError, match="5000|row|limit"):
        fetch_bars("TL2612", strict=True, session=session)


@pytest.mark.parametrize("value", [1 + 0j, 1 + 2j])
def test_numeric_validation_never_discards_imaginary_components(value):
    frame = bars()
    frame["volume"] = value
    with pytest.raises(ValueError, match="complex|real"):
        resample_ohlc(frame, "1D")
