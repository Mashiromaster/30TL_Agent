from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from realtime_quotes import (
    AkShareQuoteProvider,
    InvalidSymbolError,
    QuoteError,
    QuoteUnavailableError,
    TencentQuoteProvider,
    market_phase,
    normalize_ashare_symbol,
)


TZ = ZoneInfo("Asia/Shanghai")


def sample_frame(last=10.25):
    values = {
        "最新": last,
        "均价": 10.20,
        "涨幅": 1.49,
        "涨跌": 0.15,
        "总手": 12345,
        "金额": 12600000,
        "最高": 10.40,
        "最低": 9.98,
        "今开": 10.05,
        "昨收": 10.10,
    }
    for level in range(1, 6):
        values[f"buy_{level}"] = 10.25 - (level - 1) * 0.01
        values[f"buy_{level}_vol"] = level * 100
        values[f"sell_{level}"] = 10.26 + (level - 1) * 0.01
        values[f"sell_{level}_vol"] = level * 200
    return pd.DataFrame({"item": values.keys(), "value": values.values()})


def tencent_response(
    timestamp="20260922100005",
    last="9.04",
    code="600000",
    market="sh",
    name="浦发银行",
):
    values = [""] * 88
    updates = {
        0: "1",
        1: name,
        2: code,
        3: last,
        4: "9.01",
        5: "9.03",
        6: "532667",
        30: timestamp,
        31: "0.03",
        32: "0.33",
        33: "9.07",
        34: "8.97",
        35: "9.04/532667/480795737",
    }
    for level, (bid_index, ask_index) in enumerate(
        zip((9, 11, 13, 15, 17), (19, 21, 23, 25, 27)), start=1
    ):
        updates[bid_index] = str(9.04 - (level - 1) * 0.01)
        updates[bid_index + 1] = str(level * 10)
        updates[ask_index] = str(9.05 + (level - 1) * 0.01)
        updates[ask_index + 1] = str(level * 20)
    for index, value in updates.items():
        values[index] = value
    return f'v_{market}{code}="' + "~".join(values) + '";'


class FakeResponse:
    def __init__(self, text):
        self.text = text
        self.encoding = None

    def raise_for_status(self):
        return None


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("600000", "600000.SH"),
        ("600000.xshg", "600000.SH"),
        ("SH600000", "600000.SH"),
        ("000001", "000001.SZ"),
        ("000001.XSHE", "000001.SZ"),
        ("BJ430047", "430047.BJ"),
    ],
)
def test_normalize_ashare_symbol(raw, canonical):
    assert normalize_ashare_symbol(raw).canonical == canonical


@pytest.mark.parametrize(
    "raw", ["", "AAPL", "123", "600000.HK", "000001.SH", "510300"]
)
def test_rejects_invalid_symbol(raw):
    with pytest.raises(InvalidSymbolError):
        normalize_ashare_symbol(raw)


@pytest.mark.parametrize(
    ("hour", "minute", "expected"),
    [
        (9, 10, "pre_open"),
        (9, 20, "opening_auction"),
        (9, 27, "auction_break"),
        (10, 0, "continuous"),
        (12, 0, "lunch_break"),
        (14, 58, "closing_auction"),
        (15, 1, "closed"),
    ],
)
def test_market_phase(hour, minute, expected):
    now = datetime(2026, 9, 22, hour, minute, tzinfo=TZ)
    assert market_phase(now) == expected


def test_weekend_is_closed():
    assert market_phase(datetime(2026, 9, 20, 10, 0, tzinfo=TZ)) == "closed"


def test_akshare_provider_normalizes_snapshot_and_fails_closed():
    provider = AkShareQuoteProvider(
        fetch_quote=lambda code: sample_frame(),
        clock=lambda: datetime(2026, 9, 22, 10, 0, tzinfo=TZ),
        min_interval_seconds=0,
    )

    quote = provider.get_quote("SH600000")

    assert quote.symbol == "600000.SH"
    assert quote.last == 10.25
    assert quote.bid_prices == pytest.approx((10.25, 10.24, 10.23, 10.22, 10.21))
    assert quote.ask_volume_shares == pytest.approx((200, 400, 600, 800, 1000))
    assert quote.market_phase == "continuous"
    assert quote.provider_timestamp is None
    assert quote.is_stale is None
    assert quote.trade_eligible is False
    assert "provider_timestamp_missing" in quote.quality_reasons
    assert quote.to_dict()["received_at"] == "2026-09-22T10:00:00+08:00"


def test_provider_does_not_reuse_an_old_quote_after_failure():
    calls = 0

    def fetch(_code):
        nonlocal calls
        calls += 1
        if calls == 1:
            return sample_frame()
        raise OSError("network down")

    provider = AkShareQuoteProvider(fetch_quote=fetch, min_interval_seconds=0)
    assert provider.get_quote("600000").last == 10.25
    with pytest.raises(QuoteUnavailableError, match="network down"):
        provider.get_quote("600000")


@pytest.mark.parametrize("last", [None, "-", 0, float("nan")])
def test_provider_rejects_missing_or_nonpositive_last(last):
    provider = AkShareQuoteProvider(
        fetch_quote=lambda _code: sample_frame(last), min_interval_seconds=0
    )
    with pytest.raises(QuoteUnavailableError, match="no usable last price"):
        provider.get_quote("000001")


def test_batch_is_deduplicated_and_bounded():
    provider = AkShareQuoteProvider(
        fetch_quote=lambda _code: sample_frame(),
        min_interval_seconds=0,
        max_batch_size=2,
    )
    assert len(provider.get_quotes(["600000", "600000"])) == 1
    with pytest.raises(QuoteError, match="prototype provider limit"):
        provider.get_quotes(["600000", "000001", "300750"])


def test_tencent_provider_parses_source_time_and_depth():
    requested = []

    def get(url, **kwargs):
        requested.append((url, kwargs))
        return FakeResponse(tencent_response())

    provider = TencentQuoteProvider(
        http_get=get,
        clock=lambda: datetime(2026, 9, 22, 10, 0, 10, tzinfo=TZ),
        max_retries=0,
    )

    quote = provider.get_quote("600000.SH")

    assert requested[0][0].endswith("q=sh600000")
    assert requested[0][1]["timeout"] == 5.0
    assert quote.name == "浦发银行"
    assert quote.last == 9.04
    assert quote.provider_timestamp.isoformat() == "2026-09-22T10:00:05+08:00"
    assert quote.is_stale is False
    assert quote.bid_prices == pytest.approx((9.04, 9.03, 9.02, 9.01, 9.00))
    assert quote.ask_volume_shares == pytest.approx((2000, 4000, 6000, 8000, 10000))
    assert quote.trade_eligible is False
    assert "unofficial_web_source_no_sla" in quote.quality_reasons


def test_tencent_provider_marks_old_source_timestamp_stale():
    provider = TencentQuoteProvider(
        http_get=lambda *_args, **_kwargs: FakeResponse(
            tencent_response(timestamp="20260922095900")
        ),
        clock=lambda: datetime(2026, 9, 22, 10, 0, 0, tzinfo=TZ),
        max_retries=0,
        stale_after_seconds=15,
    )
    quote = provider.get_quote("600000")
    assert quote.is_stale is True
    assert "quote_stale" in quote.quality_reasons


def test_tencent_batch_uses_one_request_and_preserves_order():
    requested = []

    def get(url, **_kwargs):
        requested.append(url)
        return FakeResponse(
            tencent_response()
            + "\n"
            + tencent_response(code="000001", market="sz", name="平安银行")
        )

    provider = TencentQuoteProvider(
        http_get=get,
        clock=lambda: datetime(2026, 9, 22, 10, 0, 10, tzinfo=TZ),
        max_retries=0,
    )
    quotes = provider.get_quotes(["600000", "000001.SZ", "600000.SH"])

    assert len(requested) == 1
    assert requested[0].endswith("q=sh600000,sz000001")
    assert [quote.symbol for quote in quotes] == ["600000.SH", "000001.SZ"]
    assert [quote.name for quote in quotes] == ["浦发银行", "平安银行"]


def test_tencent_provider_marks_future_source_timestamp_unusable():
    provider = TencentQuoteProvider(
        http_get=lambda *_args, **_kwargs: FakeResponse(
            tencent_response(timestamp="20260922100100")
        ),
        clock=lambda: datetime(2026, 9, 22, 10, 0, 0, tzinfo=TZ),
        max_retries=0,
    )
    quote = provider.get_quote("600000")
    assert quote.is_stale is True
    assert "provider_timestamp_in_future" in quote.quality_reasons


def test_tencent_provider_retries_then_reports_failure(monkeypatch):
    attempts = 0

    def get(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise OSError("network down")

    monkeypatch.setattr("realtime_quotes.time.sleep", lambda _seconds: None)
    provider = TencentQuoteProvider(http_get=get, max_retries=2)
    with pytest.raises(QuoteUnavailableError, match="network down"):
        provider.get_quote("600000")
    assert attempts == 3
