import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from data_fetcher import CachePolicy, MacroDataFetcher


def _frame(start="2025-01-01", end="2025-01-10", value=1.0):
    dates = pd.date_range(start, end, freq="D")
    return pd.DataFrame({"date": dates, "value": value})


def test_force_refresh_replaces_raw_cache_and_manifest(tmp_path):
    initial = MacroDataFetcher(
        tmp_path, start_date="20250101", end_date="20250110"
    )
    initial._fetch_with_cache("sample", lambda: _frame(value=1.0))

    refreshed = MacroDataFetcher(
        tmp_path,
        start_date="20250101",
        end_date="20250110",
        cache_policy=CachePolicy(force_refresh=True),
    )
    result = refreshed._fetch_with_cache("sample", lambda: _frame(value=2.0))

    assert result["value"].eq(2.0).all()
    manifest = json.loads((tmp_path / "sample.meta.json").read_text())
    assert manifest["schema_version"] == CachePolicy().schema_version
    assert manifest["coverage_end"].startswith("2025-01-10")


def test_cache_with_insufficient_coverage_is_refetched(tmp_path):
    first = MacroDataFetcher(
        tmp_path, start_date="20250101", end_date="20250110"
    )
    first._fetch_with_cache("sample", lambda: _frame(end="2025-01-10"))

    calls = []
    wider = MacroDataFetcher(
        tmp_path, start_date="20250101", end_date="20250120"
    )
    result = wider._fetch_with_cache(
        "sample", lambda: calls.append(True) or _frame(end="2025-01-20", value=3.0)
    )

    assert calls == [True]
    assert result["date"].max() == pd.Timestamp("2025-01-20")


def test_failed_refresh_can_fall_back_to_valid_stale_cache(tmp_path):
    first = MacroDataFetcher(
        tmp_path, start_date="20250101", end_date="20250110"
    )
    expected = first._fetch_with_cache("sample", lambda: _frame(value=4.0))

    refresh = MacroDataFetcher(
        tmp_path,
        start_date="20250101",
        end_date="20250110",
        cache_policy=CachePolicy(force_refresh=True, allow_stale_on_error=True),
    )

    result = refresh._fetch_with_cache(
        "sample", lambda: (_ for _ in ()).throw(RuntimeError("offline"))
    )
    pd.testing.assert_frame_equal(result, expected)


def test_corrupt_cache_is_not_returned(tmp_path):
    (tmp_path / "sample.pkl").write_bytes(b"not-a-pickle")
    result = MacroDataFetcher(
        tmp_path, start_date="20250101", end_date="20250110"
    )._fetch_with_cache("sample", lambda: _frame(value=5.0))

    assert result["value"].eq(5.0).all()


def test_failed_refresh_does_not_return_cache_with_insufficient_coverage(tmp_path):
    first = MacroDataFetcher(
        tmp_path, start_date="20250101", end_date="20250110"
    )
    first._fetch_with_cache("sample", lambda: _frame(end="2025-01-10"))
    wider = MacroDataFetcher(
        tmp_path,
        start_date="20250101",
        end_date="20250201",
        cache_policy=CachePolicy(force_refresh=True, allow_stale_on_error=True),
    )

    result = wider._fetch_with_cache(
        "sample", lambda: (_ for _ in ()).throw(RuntimeError("offline"))
    )

    assert result.empty


def test_repo_fetch_parses_official_columns_and_chunks_one_year_requests(
    tmp_path, monkeypatch
):
    repo_calls = []

    def repo_rate_hist(start_date, end_date):
        repo_calls.append((start_date, end_date))
        return pd.DataFrame(
            {
                "date": pd.to_datetime([start_date, end_date]),
                "FR001": [1.7, 1.8],
                "FR007": [1.9, 2.0],
                "FDR007": [1.8, 1.9],
            }
        )

    fake_akshare = SimpleNamespace(
        repo_rate_hist=repo_rate_hist,
    )
    monkeypatch.setitem(sys.modules, "akshare", fake_akshare)
    fetcher = MacroDataFetcher(
        tmp_path,
        start_date="20250101",
        end_date="20260105",
        cache_policy=CachePolicy(force_refresh=True),
    )
    monkeypatch.setattr(fetcher, "_rate_limit", lambda: None)

    result = fetcher.fetch_repo_rates()

    assert len(repo_calls) == 2
    assert {"Repo_FR001", "Repo_FR007", "Repo_FDR007"}.issubset(result.columns)


def test_cross_market_fetch_merges_lpr_with_daily_spread(tmp_path, monkeypatch):
    fake_akshare = SimpleNamespace(
        bond_zh_us_rate=lambda **kwargs: pd.DataFrame(
            {
                "日期": pd.to_datetime(["2025-01-20", "2025-01-21"]),
                "中国国债收益率10年": [1.65, 1.66],
                "美国国债收益率10年": [4.60, 4.58],
            }
        ),
        macro_china_lpr=lambda: pd.DataFrame(
            {
                "TRADE_DATE": pd.to_datetime(["2025-01-20"]),
                "LPR1Y": [3.10],
                "LPR5Y": [3.60],
            }
        ),
        macro_china_swap_rate=lambda **kwargs: pd.DataFrame(),
    )
    monkeypatch.setitem(sys.modules, "akshare", fake_akshare)
    fetcher = MacroDataFetcher(
        tmp_path,
        start_date="20250120",
        end_date="20250121",
        cache_policy=CachePolicy(force_refresh=True),
    )
    monkeypatch.setattr(fetcher, "_rate_limit", lambda: None)

    result = fetcher.fetch_cross_market()

    assert {"CN_US_10Y_Spread", "LPR_1Y", "LPR_5Y"}.issubset(result.columns)
    first = result.loc[result["date"] == pd.Timestamp("2025-01-20")].iloc[0]
    assert first["CN_US_10Y_Spread"] == pytest.approx(-2.95)
    assert first["LPR_1Y"] == 3.10
