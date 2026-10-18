import hashlib
from dataclasses import replace
from unittest.mock import Mock

import pandas as pd
import pytest

from recent_minutes import fetch_recent_minutes, commit_recent_capture


def provider_frame(label="end"):
    frame = pd.DataFrame({"date": pd.date_range("2026-09-30 09:31", periods=3, freq="min"),
        "ticker": "TL2612", "source_symbol": "TL2612", "series_kind": "contract", "source": "sina",
        "open": 100., "high": 101., "low": 99., "close": 100.5, "volume": 10., "oi": 1000.})
    if label == "start":
        frame["date"] -= pd.Timedelta(minutes=1)
    frame.attrs["source_capture"] = {"provider": "sina", "endpoint_id": "sina_futures_1min",
        "symbol": "TL2612", "frequency": "1min", "response_sha256": "a" * 64,
        "fetched_at": "2026-10-03T16:00:00+00:00", "request_count": 1, "bar_label": "unverified"}
    return frame


def capture(tmp_path, *, label="end", frame=None, now="2026-10-04", **kwargs):
    return fetch_recent_minutes(tmp_path, "TL2612", allow_network=True, bar_label=label,
        fetcher=Mock(return_value=provider_frame() if frame is None else frame), now=now, **kwargs)


def test_network_is_never_implicit(tmp_path):
    worker = Mock()
    with pytest.raises(ValueError, match="network"):
        fetch_recent_minutes(tmp_path, "TL2612", fetcher=worker)
    worker.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("symbol,label", [("TL0", "end"), ("T2612", "end"), ("TL2612", "guess"), ("TL2612,TL2703", "end")])
def test_invalid_request_rejected_before_network(tmp_path, symbol, label):
    worker = Mock()
    with pytest.raises(ValueError):
        fetch_recent_minutes(tmp_path, symbol, allow_network=True, bar_label=label, fetcher=worker)
    worker.assert_not_called()


def test_unknown_label_remains_preview_only(tmp_path):
    worker = Mock(return_value=provider_frame())
    item = fetch_recent_minutes(tmp_path, "TL2612", allow_network=True, fetcher=worker, now="2026-10-04")
    assert item.report["status"] == "review_required"
    assert item.csv_bytes is None
    assert item.report["trade_eligible"] is False
    assert item.report["history_backfilled"] is False
    assert worker.call_args.kwargs == {"frequency": "1min", "timeout": 15, "strict": True}
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError):
        commit_recent_capture(tmp_path, item, license_note="research", confirmed=True)


def test_declared_end_preview_is_validated_but_does_not_write(tmp_path):
    item = capture(tmp_path)
    assert item.report["status"] == "inspected"
    assert item.report["quality"]["status"] == "incomplete"
    assert item.report["is_stale"] is True
    assert item.report["semantics_verified"] is False
    assert item.csv_sha256 == hashlib.sha256(item.csv_bytes).hexdigest()
    assert not list(tmp_path.iterdir())


def test_start_labels_shift_once_and_preserve_capture_metadata(tmp_path):
    item = capture(tmp_path, label="start", frame=provider_frame("start"))
    assert item.report["normalized_start"] == "2026-09-30T09:31:00+08:00"
    assert item.report["timestamp_shift_seconds"] == 60
    assert item.source["response_sha256"] == "a" * 64


@pytest.mark.parametrize("mutate", [
    lambda f: f.assign(ticker="TL2703"),
    lambda f: f.assign(source="another"),
    lambda f: f.assign(date=pd.date_range("2027-01-01", periods=3, freq="min")),
    lambda f: pd.concat([f, f.iloc[[0]]]),
    lambda f: f.drop(columns="oi"),
    lambda f: f.assign(high=1.),
])
def test_bad_response_cannot_be_importable(tmp_path, mutate):
    frame = provider_frame()
    result = mutate(frame)
    result.attrs = frame.attrs
    with pytest.raises(ValueError):
        capture(tmp_path, frame=result)
    assert not list(tmp_path.iterdir())


def test_missing_transport_evidence_is_rejected(tmp_path):
    frame = provider_frame()
    frame.attrs = {}
    with pytest.raises(ValueError, match="capture"):
        capture(tmp_path, frame=frame)


def test_forming_bar_is_withheld_and_counted_not_fabricated(tmp_path):
    item = capture(tmp_path, now="2026-09-30 09:32:30+08:00")
    assert item.report["withheld_forming_rows"] == 1
    assert item.report["accepted_rows"] == 2
    assert item.report["received_rows"] == 3


def test_replayed_capture_cannot_admit_bars_after_its_receipt_time(tmp_path):
    frame = provider_frame().iloc[:2].copy()
    frame.attrs["source_capture"]["fetched_at"] = "2026-09-30T01:31:30Z"
    item = capture(tmp_path, frame=frame, now="2026-10-04")
    assert item.report["withheld_forming_rows"] == 1
    assert item.report["accepted_rows"] == 1
    assert item.report["normalized_end"] == "2026-09-30T09:31:00+08:00"


def test_commit_requires_consent_and_uses_captured_bytes_without_refetch(tmp_path):
    item = capture(tmp_path)
    with pytest.raises(ValueError, match="confirm"):
        commit_recent_capture(tmp_path, item, license_note="local research")
    assert not list(tmp_path.iterdir())
    result = commit_recent_capture(tmp_path, item, license_note="authorized personal research", confirmed=True)
    assert result["status"] == "imported"
    assert result["acquisition"]["response_sha256"] == "a" * 64
    assert result["acquisition"]["semantics_confirmed"] is True
    assert result["provider_verified"] is False
    assert not (tmp_path / "data/TL分钟级量价数据.pkl").exists()


def test_modified_capture_fails_before_persistence(tmp_path):
    item = replace(capture(tmp_path), csv_bytes=b"changed")
    with pytest.raises(ValueError, match="hash"):
        commit_recent_capture(tmp_path, item, license_note="research", confirmed=True)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("section,key,value", [
    ("report", "symbol", "TL2703"), ("report", "bar_label", "start"),
    ("source", "response_sha256", "b" * 64), ("source", "fetched_at", "2026-01-01T00:00:00Z"),
    ("report", "accepted_rows", 999),
])
def test_changed_capture_metadata_is_not_admitted(tmp_path, section, key, value):
    item = capture(tmp_path)
    getattr(item, section)[key] = value
    with pytest.raises(ValueError, match="hash"):
        commit_recent_capture(tmp_path, item, license_note="research", confirmed=True)
    assert not list(tmp_path.iterdir())


def test_network_failure_is_not_replaced_by_cached_success(tmp_path):
    worker = Mock(side_effect=TimeoutError("timeout"))
    with pytest.raises(TimeoutError):
        fetch_recent_minutes(tmp_path, "TL2612", allow_network=True, fetcher=worker)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("field,value", [("response_sha256", "not-a-hash"), ("fetched_at", "2099-01-01T00:00:00Z"),
    ("fetched_at", "2026-09-30"), ("request_count", True)])
def test_invalid_transport_metadata_is_rejected(tmp_path, field, value):
    frame = provider_frame()
    frame.attrs["source_capture"][field] = value
    with pytest.raises(ValueError, match="capture"):
        capture(tmp_path, frame=frame)


@pytest.mark.parametrize("field,value", [("date", 1), ("date", None), ("volume", True), ("oi", 10 + 1j)])
def test_raw_scalar_anomalies_cannot_be_laundered_by_conversion(tmp_path, field, value):
    frame = provider_frame()
    frame[field] = frame[field].astype(object)
    frame.loc[0, field] = value
    with pytest.raises(ValueError):
        capture(tmp_path, frame=frame)


def test_all_forming_rows_and_unexplained_future_rows_fail_closed(tmp_path):
    frame = provider_frame().iloc[:1].copy()
    with pytest.raises(ValueError, match="completed"):
        capture(tmp_path, frame=frame, now="2026-09-30 09:30:30")
    frame["date"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="future normalized"):
        capture(tmp_path, frame=frame, label="start", now="2026-09-30 09:31:30")


def test_capture_cannot_be_saved_into_another_project(tmp_path):
    item = capture(tmp_path)
    another = tmp_path / "other"
    another.mkdir()
    with pytest.raises(ValueError, match="another project"):
        commit_recent_capture(another, item, license_note="research", confirmed=True)
    assert not list(another.iterdir())


def test_invalid_asof_rejected_before_request(tmp_path):
    fetch = Mock()
    with pytest.raises(ValueError, match="as-of"):
        fetch_recent_minutes(tmp_path, "TL2612", allow_network=True, fetcher=fetch, now=pd.NaT)
    fetch.assert_not_called()


def test_default_fetcher_is_the_shared_strict_transport(tmp_path, monkeypatch):
    import market_data
    worker = Mock(return_value=provider_frame())
    monkeypatch.setattr(market_data, "fetch_bars", worker)
    result = fetch_recent_minutes(tmp_path, "TL2612", allow_network=True)
    assert result.report["status"] == "review_required"
    assert worker.call_args.kwargs["strict"] is True
