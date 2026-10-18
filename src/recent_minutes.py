"""One-shot public-window acquisition, then separately confirmed local admission.

Fetching never writes or falls back to a cache. The vendor's timestamp/quantity
semantics are not certified: unknown labels remain preview-only. Confirmation
is a caller assertion, not an independent license or market-data attestation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from minute_store import import_minute_csv


@dataclass(frozen=True)
class RecentMinuteCapture:
    report: dict
    source: dict
    csv_bytes: bytes | None
    csv_sha256: str | None
    capture_sha256: str


def _digest(report, source, csv_sha):
    return hashlib.sha256(json.dumps([report, source, csv_sha], sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _pack(report, source, payload=None):
    csv_sha = hashlib.sha256(payload).hexdigest() if payload is not None else None
    return RecentMinuteCapture(report, source, payload, csv_sha, _digest(report, source, csv_sha))


def _asof(now):
    value = pd.Timestamp.now(tz="Asia/Shanghai") if now is None else pd.Timestamp(now)
    if pd.isna(value):
        raise ValueError("invalid as-of timestamp")
    return value.tz_localize("Asia/Shanghai") if value.tzinfo is None else value.tz_convert("Asia/Shanghai")


def _capture_source(frame, symbol):
    source = frame.attrs.get("source_capture")
    expected = {"provider": "sina", "endpoint_id": "sina_futures_1min", "symbol": symbol,
                "frequency": "1min", "request_count": 1, "bar_label": "unverified"}
    if not isinstance(source, dict) or any(source.get(key) != value for key, value in expected.items()) or type(source.get("request_count")) is not int:
        raise ValueError("missing or inconsistent strict capture metadata")
    if not isinstance(source.get("response_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", source["response_sha256"]):
        raise ValueError("invalid capture response hash")
    stamp = pd.Timestamp(source.get("fetched_at"))
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("capture requires an aware acquisition timestamp")
    if stamp > pd.Timestamp.now(tz="UTC"):
        raise ValueError("capture acquisition timestamp is in the future")
    return {**expected, "response_sha256": source["response_sha256"], "fetched_at": stamp.tz_convert("UTC").isoformat()}


def fetch_recent_minutes(base_dir, symbol, *, allow_network=False, bar_label="unverified", fetcher=None, now=None):
    """Fetch one explicit TL contract once; no writes and no historical paging."""
    if allow_network is not True:
        raise ValueError("explicit network confirmation is required")
    if not isinstance(symbol, str) or not re.fullmatch(r"TL\d{2}(03|06|09|12)", symbol):
        raise ValueError("request one explicit TL quarterly contract, never TL0")
    if bar_label not in ("unverified", "start", "end"):
        raise ValueError("bar label must be unverified, start or end")
    asof = _asof(now)
    if fetcher is None:
        from market_data import fetch_bars
        fetcher = fetch_bars
    frame = fetcher(symbol, frequency="1min", timeout=15, strict=True)
    required = {"date", "ticker", "source_symbol", "series_kind", "source", "open", "high", "low", "close", "volume", "oi"}
    if not isinstance(frame, pd.DataFrame) or frame.empty or len(frame) > 5000 or not frame.columns.is_unique or not required.issubset(frame):
        raise ValueError("invalid bounded recent minute frame")
    for field, value in {"ticker": symbol, "source_symbol": symbol, "series_kind": "contract", "source": "sina"}.items():
        if not frame[field].eq(value).all():
            raise ValueError("recent response contract identity mismatch")
    source = _capture_source(frame, symbol)
    observed_cutoff = min(asof, pd.Timestamp(source["fetched_at"]).tz_convert("Asia/Shanghai")).floor("min")
    # Retain scalar evidence before any coercion, including injected providers.
    if frame.date.map(lambda item: isinstance(item, (bool, int, float, np.number))).any():
        raise ValueError("numeric timestamps require an explicit unit, not inference")
    dates = pd.to_datetime(frame.date, errors="raise", format="mixed")
    if dates.isna().any() or not pd.api.types.is_datetime64_any_dtype(dates):
        raise ValueError("invalid or mixed recent timestamps")
    dates = dates.dt.tz_localize("Asia/Shanghai") if dates.dt.tz is None else dates.dt.tz_convert("Asia/Shanghai")
    if not dates.eq(dates.dt.floor("min")).all() or dates.duplicated().any():
        raise ValueError("duplicate or sub-minute recent timestamps")
    for field in ("open", "high", "low", "close", "volume", "oi"):
        if frame[field].map(lambda item: isinstance(item, (bool, np.bool_, complex, np.complexfloating))).any():
            raise ValueError("boolean or complex recent numeric field")
    from market_data import _validate_prices
    _validate_prices(frame)
    if (dates > observed_cutoff + pd.Timedelta(minutes=1)).any():
        raise ValueError("future recent timestamp exceeds possible forming bar")
    report = {
        "schema_version": "f_agent.recent_minutes.v1", "status": "review_required",
        "project_root": str(Path(base_dir).resolve()),
        "source": source, "symbol": symbol, "received_rows": len(frame), "accepted_rows": 0,
        "provider_start": dates.min().isoformat(), "provider_end": dates.max().isoformat(),
        "as_of": asof.isoformat(), "bar_label": bar_label, "timestamp_shift_seconds": 60 if bar_label == "start" else 0,
        "completion_cutoff": observed_cutoff.isoformat(),
        "semantics_verified": False, "history_backfilled": False, "trade_eligible": False,
        "network_used": True, "production_modified": False, "persisted": False,
        "withheld_forming_rows": None,
        "limitations": ["Public short rolling window; no arbitrary historical backfill or freshness guarantee.",
                        "Contract identity comes from the requested endpoint, not independently attested vendor payload.",
                        "Timestamp labels, quantity units, license and calendar require independent review.",
                        "No retries, cache fallback, automatic scheduling, training, publication or orders."],
    }
    if bar_label == "unverified":
        report["reason"] = "Choose a verified-by-you bar label before canonical conversion; no time semantics inferred."
        return _pack(report, source)
    shifted = dates + pd.Timedelta(seconds=report["timestamp_shift_seconds"])
    if (shifted > observed_cutoff + pd.Timedelta(minutes=1)).any():
        raise ValueError("future normalized timestamp")
    complete = shifted <= observed_cutoff
    report["withheld_forming_rows"] = int((~complete).sum())
    if not complete.any():
        raise ValueError("no completed minute bars in response")
    normalized = frame.loc[complete, ["ticker", "open", "high", "low", "close", "volume", "oi"]].copy()
    normalized.insert(0, "date", shifted.loc[complete].dt.tz_localize(None))
    payload = normalized.to_csv(index=False, date_format="%Y-%m-%d %H:%M:%S", float_format="%.17g").encode("utf-8")
    validation = import_minute_csv(base_dir, payload, provider="sina-public-recent",
                                  license_note="Inspection only; no local persistence authorization asserted", now=asof)
    latest = shifted.loc[complete].max()
    age = (asof - latest).total_seconds()
    report.update(status="inspected", quality=validation["quality"], accepted_rows=len(normalized),
                  normalized_start=shifted.loc[complete].min().isoformat(), normalized_end=latest.isoformat(),
                  age_seconds=age, is_stale=age > 86400, canonical_sha256=hashlib.sha256(payload).hexdigest())
    return _pack(report, source, payload)


def commit_recent_capture(base_dir, capture, *, license_note, confirmed=False, now=None):
    """Persist precisely the inspected bytes, without another provider request."""
    if confirmed is not True:
        raise ValueError("confirm local-use rights and timestamp/quantity semantics before saving")
    if not isinstance(capture, RecentMinuteCapture) or capture.report.get("status") != "inspected" or capture.csv_bytes is None:
        raise ValueError("a validated capture with explicit bar semantics is required")
    if hashlib.sha256(capture.csv_bytes).hexdigest() != capture.csv_sha256:
        raise ValueError("capture CSV hash changed")
    if _digest(capture.report, capture.source, capture.csv_sha256) != capture.capture_sha256:
        raise ValueError("capture metadata hash changed")
    if capture.report["project_root"] != str(Path(base_dir).resolve()):
        raise ValueError("capture belongs to another project")
    acquisition = {"kind": "sina_recent", "symbol": capture.report["symbol"], "frequency": "1min",
        "response_sha256": capture.source["response_sha256"], "fetched_at": capture.source["fetched_at"],
        "bar_label": capture.report["bar_label"], "semantics_confirmed": True}
    return import_minute_csv(base_dir, capture.csv_bytes, provider="sina-public-recent", license_note=license_note,
                             persist=True, now=now, acquisition=acquisition)
