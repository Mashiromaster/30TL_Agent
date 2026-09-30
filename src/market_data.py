"""Identity-preserving Sina futures bars, aggregation, and isolated quote caches.

``TL0`` is the vendor's continuous series, not a delivery contract. It must never
be relabelled using open interest or merged into a concrete-contract history.
Minute responses contain a short rolling window; daily charts should request
``1D`` directly rather than present that window as complete daily history.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Callable

import numpy as np
import pandas as pd
import requests


VARIETY_CONFIG = {
    "TL": {"label": "TL (30年期)", "prefix": "TL", "file_name": "TL分钟级量价数据.pkl", "main_symbol": "TL0"},
    "T": {"label": "T (10年期)", "prefix": "T", "file_name": "T分钟级量价数据.pkl", "main_symbol": "T0"},
    "TF": {"label": "TF (5年期)", "prefix": "TF", "file_name": "TF分钟级量价数据.pkl", "main_symbol": "TF0"},
    "TS": {"label": "TS (2年期)", "prefix": "TS", "file_name": "TS分钟级量价数据.pkl", "main_symbol": "TS0"},
}

_BASE_URL = "https://stock2.finance.sina.com.cn/futures/api/jsonp.php/=/InnerFuturesNewService."
_ENDPOINTS = {"1min": "getFewMinLine", "1D": "getDailyKLine"}
_PRICE_COLUMNS = ["open", "high", "low", "close"]
_BAR_COLUMNS = ["date", "trade_dt", "ticker", "source_symbol", "series_kind",
                "open", "high", "low", "close", "volume", "oi", "time", "source"]
_SCHEMA_VERSION = 1


def _series_kind(symbol: str) -> str:
    return "continuous" if symbol in {cfg["main_symbol"] for cfg in VARIETY_CONFIG.values()} else "contract"


@dataclass(frozen=True)
class QuoteSnapshot:
    bars: pd.DataFrame
    error: str | None = None
    from_cache: bool = False
    fetched_at: str | None = None


def _validate_request(symbol: str, frequency: str) -> None:
    if not isinstance(symbol, str) or not re.fullmatch(r"(?:TL|TF|TS|T)(?:0|\d{2}(?:03|06|09|12))", symbol):
        raise ValueError(f"Unsupported treasury-futures symbol: {symbol!r}")
    if frequency not in _ENDPOINTS:
        raise ValueError(f"Unsupported frequency: {frequency!r}; use '1min' or '1D'")


def contract_candidates(variety: str, as_of=None) -> list[str]:
    """Return current and next two quarterly candidates, not verified listings.

    The current quarter may already have expired during its delivery month.
    A candidate is only usable after the provider actually returns valid bars.
    """
    if variety not in VARIETY_CONFIG:
        raise ValueError(f"Unsupported variety: {variety!r}")
    timestamp = pd.Timestamp.now(tz="Asia/Shanghai") if as_of is None else pd.Timestamp(as_of)
    if pd.isna(timestamp):
        raise ValueError("Invalid as_of timestamp")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("Asia/Shanghai")
    quarter_month = ((timestamp.month - 1) // 3 + 1) * 3
    start = timestamp.year * 12 + quarter_month - 1
    return [f"{variety}{(start + offset) // 12 % 100:02d}{(start + offset) % 12 + 1:02d}"
            for offset in (0, 3, 6)]


def _dates(values) -> pd.Series:
    try:
        dates = pd.to_datetime(values, errors="raise")
        if dates.isna().any():
            raise ValueError("Missing date")
        if dates.dt.tz is not None:
            dates = dates.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
        return dates
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f"Invalid bar date: {exc}") from exc


def _validate_prices(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["date", *_PRICE_COLUMNS, "volume"]
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"Bar schema missing fields: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Provider returned no bars")
    result = frame.copy()
    result["date"] = _dates(result["date"])
    numeric = [*_PRICE_COLUMNS, "volume"] + (["oi"] if "oi" in result else [])
    for column in numeric:
        try:
            if result[column].map(lambda value: isinstance(value, (bool, np.bool_))).any():
                raise ValueError("boolean is not a price or quantity")
            result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {column}: {exc}") from exc
        if not np.isfinite(result[column].to_numpy()).all():
            raise ValueError(f"Non-finite {column} in provider bars")
    if (result[_PRICE_COLUMNS] <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    if ((result["high"] < result[_PRICE_COLUMNS].max(axis=1)) |
            (result["low"] > result[_PRICE_COLUMNS].min(axis=1))).any():
        raise ValueError("Invalid OHLC extrema: low <= open/close <= high required")
    if (result["volume"] < 0).any() or ("oi" in result and (result["oi"] < 0).any()):
        raise ValueError("Volume and open interest must be non-negative")
    return result


def _normalize_bars(frame: pd.DataFrame, symbol: str, frequency: str) -> pd.DataFrame:
    """Validate a self-describing provider/cache frame; never repair identity."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Provider must return a DataFrame")
    frame = _validate_prices(frame)
    kind = _series_kind(symbol)
    for column, expected in (("ticker", symbol), ("source_symbol", symbol),
                             ("series_kind", kind), ("source", "sina")):
        if column not in frame or not frame[column].eq(expected).all():
            raise ValueError(f"Bar identity mismatch: {column} must be {expected!r}")
    if "oi" not in frame:
        raise ValueError("Bar schema missing fields: oi")
    if frequency == "1D" and not frame["date"].eq(frame["date"].dt.normalize()).all():
        raise ValueError("Daily series contains intraday timestamps")
    frame["trade_dt"] = frame["date"].dt.normalize()
    frame["time"] = frame["date"].dt.strftime("%H:%M")
    frame = frame.drop_duplicates(subset=["date", "ticker"], keep="last")
    return frame.sort_values("date", kind="stable").reset_index(drop=True)[_BAR_COLUMNS]


def fetch_bars(symbol: str, frequency: str = "1min", timeout: float = 15, session=None) -> pd.DataFrame:
    """Fetch one explicitly identified series with bounded HTTP I/O.

    Source: AKShare's documented Sina InnerFuturesNewService endpoints. JSONP
    is parsed as JSON data only; no vendor script is ever evaluated. Invalid or
    empty payloads raise, so callers cannot mistake a failure for fresh data.
    """
    _validate_request(symbol, frequency)
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a finite positive number")
    client = requests if session is None else session
    period = "1" if frequency == "1min" else pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y_%m_%d")
    response = client.get(_BASE_URL + _ENDPOINTS[frequency],
                          params={"symbol": symbol, "type": period}, timeout=timeout)
    response.raise_for_status()
    match = re.search(r"=\s*\((.*)\)\s*;?\s*$", response.text, re.DOTALL)
    if match is None:
        raise ValueError(f"{symbol} {frequency}: invalid Sina JSONP response")
    try:
        payload = json.loads(match.group(1))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{symbol} {frequency}: invalid Sina JSON payload") from exc
    if not isinstance(payload, list) or not payload or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"{symbol} {frequency}: no bars or unexpected provider schema")
    frame = pd.DataFrame(payload).rename(columns={
        "d": "date", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "p": "oi",
    })
    frame["ticker"] = frame["source_symbol"] = symbol
    frame["series_kind"] = _series_kind(symbol)
    frame["source"] = "sina"
    return _normalize_bars(frame, symbol, frequency)


def resample_ohlc(frame: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Aggregate one series using true OHLC; omit empty periods, never fill.

    Continuous and concrete contracts cannot share an output candle. The caller
    must select the series before resampling. Auxiliary fields retain the last
    observed value within each non-empty bin.
    """
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["date", *_PRICE_COLUMNS, "volume"])
    for identity in ("ticker", "source_symbol", "series_kind"):
        if identity in frame and (frame[identity].isna().any() or frame[identity].nunique() != 1):
            raise ValueError(f"Mixed or missing {identity}; select a single series before aggregation")
    result = _validate_prices(frame).drop_duplicates(subset="date", keep="last").sort_values("date")
    aggregation = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    for column in ("oi", "Market_Regime", "ticker", "source_symbol", "series_kind", "source"):
        if column in result:
            aggregation[column] = "last"
    offset = pd.tseries.frequencies.to_offset(freq)
    if offset.n <= 0:
        raise ValueError("Resampling frequency must be positive")
    try:
        intraday = offset.nanos < pd.Timedelta(days=1).value
    except ValueError:  # Non-fixed offsets (e.g. weekly/monthly) are not intraday.
        intraday = False
    if intraday:
        # Sina minute timestamps mark bar ends. Separate sessions ensure a wide
        # bin cannot straddle lunch, and actual labels remain inside plot hours.
        result["_actual_time"] = result["date"]
        aggregation["_actual_time"] = "first"
        pieces = []
        sessions = result["date"].dt.hour.ge(12)
        for (day, afternoon), group in result.groupby([result["date"].dt.normalize(), sessions]):
            anchor = day + (pd.Timedelta(hours=13) if afternoon else pd.Timedelta(hours=9, minutes=30))
            piece = group.set_index("date").resample(freq, origin=anchor, closed="right").agg(aggregation)
            piece = piece.dropna(subset=_PRICE_COLUMNS).reset_index(drop=True)
            pieces.append(piece.rename(columns={"_actual_time": "date"}))
        return pd.concat(pieces, ignore_index=True).sort_values("date").reset_index(drop=True)
    result = result.set_index("date").resample(freq).agg(aggregation)
    return result.dropna(subset=_PRICE_COLUMNS).reset_index()


def _read_cache(path: Path, symbol: str, frequency: str) -> tuple[pd.DataFrame, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (payload.get("schema_version") != _SCHEMA_VERSION or payload.get("symbol") != symbol or
            payload.get("frequency") != frequency or payload.get("provider") != "sina"):
        raise ValueError("Quote cache metadata mismatch")
    fetched_at = payload.get("fetched_at")
    if not isinstance(fetched_at, str):
        raise ValueError("Quote cache has no fetched_at")
    timestamp = pd.Timestamp(fetched_at)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("Quote cache fetched_at must be a timezone-aware timestamp")
    return _normalize_bars(pd.DataFrame(payload.get("bars")), symbol, frequency), fetched_at


def _write_cache(path: Path, frame: pd.DataFrame, symbol: str, frequency: str, fetched_at: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": _SCHEMA_VERSION, "provider": "sina", "symbol": symbol,
               "frequency": frequency, "fetched_at": fetched_at,
               "bars": json.loads(frame.to_json(orient="records", date_format="iso", double_precision=15))}
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=path.name + ".",
                                         suffix=".tmp", dir=path.parent, delete=False) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_quote_series(base_dir, symbol: str, frequency: str, refresh: bool = True,
                      fetcher: Callable | None = None) -> QuoteSnapshot:
    """Load a quote snapshot without reading or modifying legacy raw files.

    Refresh failure can fall back only to a validated same-symbol/frequency
    cache. ``error`` and the original ``fetched_at`` remain visible on fallback.
    A successful live response remains usable if its atomic cache write fails.
    """
    _validate_request(symbol, frequency)
    path = Path(base_dir) / "data" / "market_quotes" / f"{symbol}_{frequency}.json"
    cached = None
    cache_error = None
    if path.exists():
        try:
            cached = _read_cache(path, symbol, frequency)
        except (OSError, ValueError, TypeError, AttributeError, KeyError) as exc:
            cache_error = f"Invalid quote cache: {exc}"
    if not refresh:
        if cached is not None:
            return QuoteSnapshot(cached[0], from_cache=True, fetched_at=cached[1])
        return QuoteSnapshot(pd.DataFrame(columns=_BAR_COLUMNS),
                             error=cache_error or f"No cached bars for {symbol} {frequency}")
    try:
        frame = _normalize_bars((fetcher or fetch_bars)(symbol, frequency), symbol, frequency)
    except Exception as exc:
        error = f"{symbol} {frequency} refresh failed: {exc}"
        if cache_error:
            error += f"; {cache_error}"
        if cached is not None:
            return QuoteSnapshot(cached[0], error=error, from_cache=True, fetched_at=cached[1])
        return QuoteSnapshot(pd.DataFrame(columns=_BAR_COLUMNS), error=error)
    fetched_at = pd.Timestamp.now(tz="Asia/Shanghai").isoformat()
    error = None
    try:
        _write_cache(path, frame, symbol, frequency, fetched_at)
    except (OSError, ValueError, TypeError) as exc:
        error = f"Live bars loaded but quote cache write failed: {exc}"
    return QuoteSnapshot(frame, error=error, fetched_at=fetched_at)
