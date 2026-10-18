"""Strict, offline adapters for documented minute export formats.

The supported TqSdk DataDownloader K-line layout follows its official
``_get_headers``/``_get_data_cols``/``_nano_to_str`` implementation:
https://github.com/shinnytech/tqsdk-python/blob/master/tqsdk/tools/downloader.py
It contains no duration field. Declaring 60 seconds does not authenticate the
export's actual cadence; sparse observations cannot prove their bar duration.
"""
from __future__ import annotations

import csv
import io
import re

import numpy as np
import pandas as pd


_FIELDS = ["open", "high", "low", "close", "volume", "open_oi", "close_oi"]
_SYMBOL = re.compile(r"CFFEX\.TL\d{2}(?:03|06|09|12)\Z")
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
_DATETIME = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{9}\Z")
_NANO = re.compile(r"(?:0|[1-9]\d{0,18})\Z")


def _metadata(symbol, pairs):
    return {
        "schema_version": "f_agent.tqsdk_conversion.v1",
        "symbol": symbol,
        "source_columns": ["datetime", "datetime_nano"] + [f"{symbol}.{field}" for field in _FIELDS],
        "auxiliary_columns": [f"{symbol}.open_oi"],
        "source_timezone": "Asia/Shanghai",
        "source_bar_label": "start", "target_bar_label": "end",
        "timestamp_shift_seconds": 60, "declared_interval_seconds": 60,
        "interval_verified": False, "observed_one_minute_pairs": pairs,
    }


def validate_conversion(value):
    """Validate a fixed, non-secret provenance record; never trust free text."""
    if not isinstance(value, dict) or not isinstance(value.get("symbol"), str) or not _SYMBOL.fullmatch(value["symbol"]):
        raise ValueError("invalid TqSdk conversion symbol")
    pairs = value.get("observed_one_minute_pairs")
    if type(pairs) is not int or not 0 <= pairs < 500_000:
        raise ValueError("invalid TqSdk conversion observation count")
    expected = _metadata(value["symbol"], pairs)
    if value != expected or any(type(value[key]) is not type(item) for key, item in expected.items()):
        raise ValueError("invalid TqSdk conversion metadata")
    return expected


def parse_tqsdk_csv(payload, *, bar_seconds=None, max_bytes=25 * 1024 * 1024, max_rows=500_000):
    """Parse a single-contract DataDownloader CSV without filling any gaps.

    Nanoseconds are read directly as decimal integer strings, never through a
    float. The two original time columns must agree exactly. An independently
    declared 60-second period controls the explicit start-to-end conversion.
    Returns canonical bars and a bounded conversion provenance record. The
    caller retains the original bytes to preserve auxiliary ``open_oi`` values.
    """
    if type(bar_seconds) is not int or bar_seconds != 60:
        raise ValueError("TqSdk requires an explicit bar_seconds=60 declaration")
    if not isinstance(payload, bytes) or not payload or len(payload) > max_bytes:
        raise ValueError("empty TqSdk CSV or size limit exceeded")
    try:
        text = payload.decode("utf-8-sig")
        if "\x00" in text:
            raise ValueError("NUL byte in TqSdk CSV")
        reader = csv.reader(io.StringIO(text), strict=True)
        header = next(reader)
        if len(header) != 9 or len(set(header)) != len(header) or not header[2].endswith(".open"):
            raise ValueError("unsupported TqSdk single-contract K-line columns")
        symbol = header[2][:-5]
        if not _SYMBOL.fullmatch(symbol) or header != _metadata(symbol, 0)["source_columns"]:
            raise ValueError("only documented single CFFEX TL contract K-line columns are accepted")
        records = []
        for record in reader:
            if not record or len(record) != len(header) or any(not value for value in record):
                raise ValueError("empty or ragged TqSdk CSV record")
            records.append(record)
            if len(records) > max_rows:
                raise ValueError("TqSdk CSV row limit exceeded")
        if not records:
            raise ValueError("empty TqSdk CSV records")
    except (UnicodeError, csv.Error, StopIteration) as exc:
        raise ValueError("invalid UTF-8 TqSdk CSV") from exc

    raw = pd.DataFrame(records, columns=header)
    if not raw["datetime_nano"].map(lambda value: bool(_NANO.fullmatch(value))).all():
        raise ValueError("datetime_nano must contain exact unsigned int64 strings")
    nanos = raw["datetime_nano"].map(int)
    if any(value > np.iinfo(np.int64).max for value in nanos):
        raise ValueError("datetime_nano exceeds int64 range")
    if not raw["datetime"].map(lambda value: bool(_DATETIME.fullmatch(value))).all():
        raise ValueError("datetime must use the documented nine-digit Shanghai timestamp")
    try:
        local = pd.to_datetime(raw["datetime"], format="%Y-%m-%d %H:%M:%S.%f", errors="raise")
        precise = pd.to_datetime(nanos, unit="ns", utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    except (ValueError, OverflowError) as exc:
        raise ValueError("invalid TqSdk timestamp") from exc
    if not local.equals(precise):
        raise ValueError("datetime and datetime_nano disagree; precision loss is not accepted")
    if not (nanos % 60_000_000_000 == 0).all():
        raise ValueError("TqSdk minute timestamps must be minute-aligned")
    diffs = nanos.diff()
    if (diffs.iloc[1:] <= 0).any():
        raise ValueError("duplicate or out-of-order TqSdk timestamps")
    numeric = raw[header[2:]]
    if not numeric.map(lambda value: bool(_NUMBER.fullmatch(value))).all(axis=None):
        raise ValueError("TqSdk financial values must be explicit real numbers")
    try:
        numeric = numeric.astype(float)
    except (ValueError, OverflowError) as exc:
        raise ValueError("invalid TqSdk financial values") from exc
    numeric.columns = _FIELDS
    if not np.isfinite(numeric.to_numpy()).all() or (numeric[["volume", "open_oi", "close_oi"]] < 0).any(axis=None):
        raise ValueError("TqSdk volume/OI must be finite and nonnegative")
    prices = numeric[["open", "high", "low", "close"]]
    if (prices <= 0).any(axis=None) or (prices["low"] > prices.min(axis=1)).any() or (prices["high"] < prices.max(axis=1)).any():
        raise ValueError("invalid TqSdk OHLC envelope")
    # int64 timestamps remain exact through the comparison and offset. No
    # resampling, interpolation, forward fill, sorting or timezone guessing.
    end = precise + pd.Timedelta(seconds=60)
    frame = pd.DataFrame({"date": end, "trade_dt": end.dt.normalize(), "ticker": symbol.split(".")[1]})
    for field in ["open", "high", "low", "close", "volume"]:
        frame[field] = numeric[field]
    frame["oi"] = numeric["close_oi"]
    return frame, _metadata(symbol, int((diffs == 60_000_000_000).sum()))
