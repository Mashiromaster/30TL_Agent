# -*- coding: utf-8 -*-
"""Provider-neutral real-time A-share quote gateway.

The default Tencent web adapter and optional AKShare/Eastmoney adapter are
useful for research and live display.  Neither public-web source provides the
contractual timing and availability guarantees needed for automated orders,
so both fail closed with ``trade_eligible=False``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, time as wall_time
from typing import Any, Callable, Iterable, Protocol
from zoneinfo import ZoneInfo


SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")
SCHEMA_VERSION = 1


class QuoteError(RuntimeError):
    """Base exception for quote gateway failures."""


class InvalidSymbolError(QuoteError):
    """Raised when an A-share symbol cannot be normalized safely."""


class QuoteUnavailableError(QuoteError):
    """Raised when the provider cannot return a usable fresh response."""


@dataclass(frozen=True)
class AShareSymbol:
    code: str
    exchange: str

    @property
    def canonical(self) -> str:
        return f"{self.code}.{self.exchange}"


@dataclass(frozen=True)
class QuoteSnapshot:
    """Normalized quote snapshot shared by all future provider adapters."""

    symbol: str
    code: str
    exchange: str
    name: str | None
    last: float
    open: float | None
    high: float | None
    low: float | None
    prev_close: float | None
    average: float | None
    change: float | None
    change_pct: float | None
    volume_lots: float | None
    amount: float | None
    bid_prices: tuple[float | None, ...]
    bid_volume_shares: tuple[float | None, ...]
    ask_prices: tuple[float | None, ...]
    ask_volume_shares: tuple[float | None, ...]
    provider_timestamp: datetime | None
    received_at: datetime
    source: str
    market_phase: str
    trading_day_verified: bool
    is_stale: bool | None
    trade_eligible: bool
    quality_reasons: tuple[str, ...]
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("provider_timestamp", "received_at"):
            value = payload[key]
            payload[key] = value.isoformat() if value is not None else None
        for key in (
            "bid_prices",
            "bid_volume_shares",
            "ask_prices",
            "ask_volume_shares",
        ):
            payload[key] = list(payload[key])
        payload["quality_reasons"] = list(payload["quality_reasons"])
        return payload


class QuoteProvider(Protocol):
    """Minimal contract for snapshot, WebSocket, or broker quote adapters."""

    def get_quote(self, symbol: str) -> QuoteSnapshot: ...

    def get_quotes(self, symbols: Iterable[str]) -> list[QuoteSnapshot]: ...


_SUFFIX_TO_EXCHANGE = {
    "SH": "SH",
    "SSE": "SH",
    "XSHG": "SH",
    "SZ": "SZ",
    "SZSE": "SZ",
    "XSHE": "SZ",
    "BJ": "BJ",
    "BSE": "BJ",
    "XBEI": "BJ",
}


def _infer_exchange(code: str) -> str:
    if code.startswith(("600", "601", "603", "605", "688", "689")):
        return "SH"
    if code.startswith(("000", "001", "002", "003", "300", "301")):
        return "SZ"
    if code.startswith(("4", "8", "92")):
        return "BJ"
    raise InvalidSymbolError(
        f"{code!r} is not a recognized Shanghai, Shenzhen, or Beijing A-share stock code"
    )


def normalize_ashare_symbol(symbol: str) -> AShareSymbol:
    """Normalize common A-share formats to ``000001.SZ`` style."""

    raw = str(symbol).strip().upper().replace(" ", "")
    if not raw:
        raise InvalidSymbolError("Symbol cannot be empty")

    match = re.fullmatch(r"(\d{6})\.([A-Z]+)", raw)
    if match:
        code, suffix = match.groups()
        exchange = _SUFFIX_TO_EXCHANGE.get(suffix)
        if exchange is None:
            raise InvalidSymbolError(f"Unsupported exchange suffix: {suffix}")
        expected_exchange = _infer_exchange(code)
        if exchange != expected_exchange:
            raise InvalidSymbolError(
                f"Symbol {raw!r} conflicts with code exchange {expected_exchange}"
            )
        return AShareSymbol(code=code, exchange=exchange)

    match = re.fullmatch(r"([A-Z]+)(\d{6})", raw)
    if match:
        prefix, code = match.groups()
        exchange = _SUFFIX_TO_EXCHANGE.get(prefix)
        if exchange is None:
            raise InvalidSymbolError(f"Unsupported exchange prefix: {prefix}")
        expected_exchange = _infer_exchange(code)
        if exchange != expected_exchange:
            raise InvalidSymbolError(
                f"Symbol {raw!r} conflicts with code exchange {expected_exchange}"
            )
        return AShareSymbol(code=code, exchange=exchange)

    if re.fullmatch(r"\d{6}", raw):
        return AShareSymbol(code=raw, exchange=_infer_exchange(raw))

    raise InvalidSymbolError(
        f"Invalid A-share symbol {symbol!r}; examples: 600000, 600000.SH, SZ000001"
    )


def _localize(now: datetime | None) -> datetime:
    value = now or datetime.now(tz=SHANGHAI_TZ)
    if value.tzinfo is None:
        return value.replace(tzinfo=SHANGHAI_TZ)
    return value.astimezone(SHANGHAI_TZ)


def market_phase(now: datetime | None = None, trading_day: bool | None = None) -> str:
    """Return an advisory A-share market phase.

    When ``trading_day`` is omitted, only the weekday is checked.  Exchange
    holidays require a provider calendar and are deliberately not guessed.
    """

    current = _localize(now)
    if trading_day is False or (trading_day is None and current.weekday() >= 5):
        return "closed"

    clock = current.time().replace(tzinfo=None)
    if clock < wall_time(9, 15):
        return "pre_open"
    if clock < wall_time(9, 25):
        return "opening_auction"
    if clock < wall_time(9, 30):
        return "auction_break"
    if clock < wall_time(11, 30):
        return "continuous"
    if clock < wall_time(13, 0):
        return "lunch_break"
    if clock < wall_time(14, 57):
        return "continuous"
    if clock < wall_time(15, 0):
        return "closing_auction"
    return "closed"


def _number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip().replace(",", "")
        if value in {"", "-", "--", "None", "null"}:
            return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _shares_from_lots(value: Any) -> float | None:
    lots = _number(value)
    return lots * 100 if lots is not None else None


def _item_value_mapping(frame: Any) -> dict[str, Any]:
    if frame is None or not hasattr(frame, "columns"):
        raise QuoteUnavailableError("Provider returned no tabular quote response")
    if "item" not in frame.columns or "value" not in frame.columns:
        raise QuoteUnavailableError("Provider response is missing item/value columns")
    if len(frame) == 0:
        raise QuoteUnavailableError("Provider returned an empty quote response")
    return {
        str(row["item"]): row["value"]
        for _, row in frame[["item", "value"]].iterrows()
    }


class AkShareQuoteProvider:
    """Near-real-time Eastmoney snapshots through AKShare.

    This provider is suitable for prototyping and monitoring only.  AKShare's
    ``stock_bid_ask_em`` result has no source event timestamp, so freshness
    cannot be proved and ``trade_eligible`` is always false.
    """

    source = "akshare:eastmoney:stock_bid_ask_em"

    def __init__(
        self,
        fetch_quote: Callable[[str], Any] | None = None,
        clock: Callable[[], datetime] | None = None,
        min_interval_seconds: float = 0.5,
        max_batch_size: int = 50,
    ) -> None:
        if min_interval_seconds < 0:
            raise ValueError("min_interval_seconds must be non-negative")
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        self._fetch_quote = fetch_quote or self._akshare_fetch_quote
        self._clock = clock or (lambda: datetime.now(tz=SHANGHAI_TZ))
        self._min_interval_seconds = float(min_interval_seconds)
        self._max_batch_size = int(max_batch_size)
        self._last_request_monotonic: float | None = None

    @staticmethod
    def _akshare_fetch_quote(code: str) -> Any:
        import akshare as ak

        return ak.stock_bid_ask_em(symbol=code)

    def _rate_limit(self) -> None:
        if self._last_request_monotonic is not None:
            elapsed = time.monotonic() - self._last_request_monotonic
            remaining = self._min_interval_seconds - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_monotonic = time.monotonic()

    def get_quote(self, symbol: str) -> QuoteSnapshot:
        normalized = normalize_ashare_symbol(symbol)
        self._rate_limit()
        try:
            frame = self._fetch_quote(normalized.code)
        except QuoteError:
            raise
        except Exception as exc:
            raise QuoteUnavailableError(
                f"Quote fetch failed for {normalized.canonical}: {exc}"
            ) from exc

        values = _item_value_mapping(frame)
        last = _number(values.get("最新"))
        if last is None or last <= 0:
            raise QuoteUnavailableError(
                f"Provider returned no usable last price for {normalized.canonical}"
            )

        received_at = _localize(self._clock())
        phase = market_phase(received_at)
        reasons = ["provider_timestamp_missing", "trading_calendar_not_verified"]
        if phase != "continuous":
            reasons.append(f"market_phase_{phase}")

        return QuoteSnapshot(
            symbol=normalized.canonical,
            code=normalized.code,
            exchange=normalized.exchange,
            name=None,
            last=last,
            open=_number(values.get("今开")),
            high=_number(values.get("最高")),
            low=_number(values.get("最低")),
            prev_close=_number(values.get("昨收")),
            average=_number(values.get("均价")),
            change=_number(values.get("涨跌")),
            change_pct=_number(values.get("涨幅")),
            volume_lots=_number(values.get("总手")),
            amount=_number(values.get("金额")),
            bid_prices=tuple(_number(values.get(f"buy_{level}")) for level in range(1, 6)),
            bid_volume_shares=tuple(
                _number(values.get(f"buy_{level}_vol")) for level in range(1, 6)
            ),
            ask_prices=tuple(_number(values.get(f"sell_{level}")) for level in range(1, 6)),
            ask_volume_shares=tuple(
                _number(values.get(f"sell_{level}_vol")) for level in range(1, 6)
            ),
            provider_timestamp=None,
            received_at=received_at,
            source=self.source,
            market_phase=phase,
            trading_day_verified=False,
            is_stale=None,
            trade_eligible=False,
            quality_reasons=tuple(reasons),
        )

    def get_quotes(self, symbols: Iterable[str]) -> list[QuoteSnapshot]:
        requested = list(dict.fromkeys(str(symbol) for symbol in symbols))
        if not requested:
            raise InvalidSymbolError("At least one symbol is required")
        if len(requested) > self._max_batch_size:
            raise QuoteError(
                f"Requested {len(requested)} symbols; prototype provider limit is "
                f"{self._max_batch_size}"
            )
        return [self.get_quote(symbol) for symbol in requested]


class TencentQuoteProvider:
    """Tencent web quote snapshots with an explicit timeout and source time.

    This is one of the two web sources used by the open-source ``Ashare.py``
    project.  Direct parsing gives us a bounded request and the provider's
    timestamp, but it remains an undocumented public-web interface without an
    availability or latency SLA, so quotes stay ineligible for live orders.
    """

    source = "tencent_web:qt.gtimg.cn"
    endpoint = "https://qt.gtimg.cn/q={market_code}"

    def __init__(
        self,
        http_get: Callable[..., Any] | None = None,
        clock: Callable[[], datetime] | None = None,
        timeout_seconds: float = 5.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.2,
        stale_after_seconds: float = 15.0,
        max_batch_size: int = 50,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_retries < 0 or retry_backoff_seconds < 0:
            raise ValueError("retry settings must be non-negative")
        if stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be positive")
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        self._http_get = http_get or self._requests_get
        self._clock = clock or (lambda: datetime.now(tz=SHANGHAI_TZ))
        self._timeout_seconds = float(timeout_seconds)
        self._max_retries = int(max_retries)
        self._retry_backoff_seconds = float(retry_backoff_seconds)
        self._stale_after_seconds = float(stale_after_seconds)
        self._max_batch_size = int(max_batch_size)

    @staticmethod
    def _requests_get(url: str, **kwargs: Any) -> Any:
        import requests

        return requests.get(url, **kwargs)

    @staticmethod
    def _market_code(symbol: AShareSymbol) -> str:
        return f"{symbol.exchange.lower()}{symbol.code}"

    def _request_text(self, symbols: list[AShareSymbol]) -> str:
        market_codes = ",".join(self._market_code(symbol) for symbol in symbols)
        url = self.endpoint.format(market_code=market_codes)
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._http_get(
                    url,
                    timeout=self._timeout_seconds,
                    headers={"User-Agent": "F_Agent/1.0 real-time quote prototype"},
                )
                if hasattr(response, "raise_for_status"):
                    response.raise_for_status()
                if hasattr(response, "encoding"):
                    response.encoding = "gbk"
                text = getattr(response, "text", None)
                if not text:
                    raise QuoteUnavailableError("Tencent returned an empty response")
                return str(text)
            except Exception as exc:
                last_error = exc
                if attempt < self._max_retries:
                    time.sleep(self._retry_backoff_seconds * (2**attempt))
        raise QuoteUnavailableError(
            "Quote fetch failed for "
            f"{','.join(symbol.canonical for symbol in symbols)}: {last_error}"
        ) from last_error

    @staticmethod
    def _parse_values(text: str) -> list[str]:
        if '="' not in text:
            raise QuoteUnavailableError("Tencent response has an unknown wire format")
        payload = text.split('="', 1)[1].split('"', 1)[0]
        values = payload.split("~")
        if len(values) < 36:
            raise QuoteUnavailableError(
                f"Tencent response has {len(values)} fields; expected at least 36"
            )
        return values

    @staticmethod
    def _provider_time(value: str) -> datetime | None:
        try:
            parsed = datetime.strptime(value, "%Y%m%d%H%M%S")
        except (TypeError, ValueError):
            return None
        return parsed.replace(tzinfo=SHANGHAI_TZ)

    def _snapshot_from_values(
        self,
        normalized: AShareSymbol,
        values: list[str],
        received_at: datetime,
    ) -> QuoteSnapshot:
        last = _number(values[3])
        if last is None or last <= 0:
            raise QuoteUnavailableError(
                f"Provider returned no usable last price for {normalized.canonical}"
            )

        provider_timestamp = self._provider_time(values[30])
        phase = market_phase(received_at)
        is_stale: bool | None = None
        reasons = [
            "unofficial_web_source_no_sla",
            "provider_timestamp_semantics_unverified",
            "trading_calendar_not_verified",
        ]
        if provider_timestamp is None:
            reasons.append("provider_timestamp_missing")
        else:
            age_seconds = (received_at - provider_timestamp).total_seconds()
            if age_seconds < -5:
                is_stale = True
                reasons.append("provider_timestamp_in_future")
            else:
                is_stale = age_seconds > self._stale_after_seconds
            if is_stale and "provider_timestamp_in_future" not in reasons:
                reasons.append("quote_stale")
        if phase != "continuous":
            reasons.append(f"market_phase_{phase}")

        volume_lots = _number(values[6])
        amount_parts = values[35].split("/")
        amount = _number(amount_parts[2]) if len(amount_parts) >= 3 else None
        average = (
            amount / (volume_lots * 100)
            if amount is not None and volume_lots is not None and volume_lots > 0
            else None
        )

        return QuoteSnapshot(
            symbol=normalized.canonical,
            code=normalized.code,
            exchange=normalized.exchange,
            name=values[1] or None,
            last=last,
            open=_number(values[5]),
            high=_number(values[33]),
            low=_number(values[34]),
            prev_close=_number(values[4]),
            average=average,
            change=_number(values[31]),
            change_pct=_number(values[32]),
            volume_lots=volume_lots,
            amount=amount,
            bid_prices=tuple(_number(values[index]) for index in (9, 11, 13, 15, 17)),
            bid_volume_shares=tuple(
                _shares_from_lots(values[index]) for index in (10, 12, 14, 16, 18)
            ),
            ask_prices=tuple(_number(values[index]) for index in (19, 21, 23, 25, 27)),
            ask_volume_shares=tuple(
                _shares_from_lots(values[index]) for index in (20, 22, 24, 26, 28)
            ),
            provider_timestamp=provider_timestamp,
            received_at=received_at,
            source=self.source,
            market_phase=phase,
            trading_day_verified=False,
            is_stale=is_stale,
            trade_eligible=False,
            quality_reasons=tuple(reasons),
        )

    def get_quote(self, symbol: str) -> QuoteSnapshot:
        return self.get_quotes([symbol])[0]

    def get_quotes(self, symbols: Iterable[str]) -> list[QuoteSnapshot]:
        requested = [normalize_ashare_symbol(symbol) for symbol in symbols]
        requested = list({symbol.canonical: symbol for symbol in requested}.values())
        if not requested:
            raise InvalidSymbolError("At least one symbol is required")
        if len(requested) > self._max_batch_size:
            raise QuoteError(
                f"Requested {len(requested)} symbols; prototype provider limit is "
                f"{self._max_batch_size}"
            )
        lines = [line for line in self._request_text(requested).splitlines() if line.strip()]
        responses: dict[str, list[str]] = {}
        for line in lines:
            values = self._parse_values(line)
            responses[values[2]] = values
        missing = [symbol.canonical for symbol in requested if symbol.code not in responses]
        if missing:
            raise QuoteUnavailableError(
                f"Provider response omitted requested symbols: {','.join(missing)}"
            )
        received_at = _localize(self._clock())
        return [
            self._snapshot_from_values(symbol, responses[symbol.code], received_at)
            for symbol in requested
        ]


def watch_quotes(
    provider: QuoteProvider,
    symbols: Iterable[str],
    interval_seconds: float = 3.0,
    count: int = 0,
) -> Iterable[list[QuoteSnapshot]]:
    """Yield repeated snapshots; ``count=0`` means run until interrupted."""

    if interval_seconds < 0.5:
        raise ValueError("interval_seconds must be at least 0.5")
    if count < 0:
        raise ValueError("count must be non-negative")
    requested = list(symbols)
    emitted = 0
    while count == 0 or emitted < count:
        started = time.monotonic()
        yield provider.get_quotes(requested)
        emitted += 1
        if count and emitted >= count:
            break
        remaining = interval_seconds - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)


def _emit(snapshots: Iterable[QuoteSnapshot]) -> None:
    payload = [snapshot.to_dict() for snapshot in snapshots]
    print(json.dumps(payload[0] if len(payload) == 1 else payload, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only real-time A-share quote gateway")
    subparsers = parser.add_subparsers(dest="command", required=True)

    quote = subparsers.add_parser("quote", help="fetch one snapshot")
    quote.add_argument("symbols", nargs="+", help="e.g. 600000 000001.SZ")
    quote.add_argument(
        "--provider", choices=("tencent", "akshare"), default="tencent"
    )

    watch = subparsers.add_parser("watch", help="poll and print JSON lines")
    watch.add_argument("symbols", nargs="+", help="e.g. 600000 000001.SZ")
    watch.add_argument("--interval", type=float, default=3.0, help="poll interval seconds")
    watch.add_argument("--count", type=int, default=0, help="0 means until interrupted")
    watch.add_argument(
        "--provider", choices=("tencent", "akshare"), default="tencent"
    )
    return parser


def build_provider(name: str) -> QuoteProvider:
    if name == "tencent":
        return TencentQuoteProvider()
    if name == "akshare":
        return AkShareQuoteProvider()
    raise ValueError(f"Unsupported quote provider: {name}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    provider = build_provider(args.provider)
    try:
        if args.command == "quote":
            _emit(provider.get_quotes(args.symbols))
        else:
            for snapshots in watch_quotes(
                provider,
                args.symbols,
                interval_seconds=args.interval,
                count=args.count,
            ):
                _emit(snapshots)
    except KeyboardInterrupt:
        return 130
    except (QuoteError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
