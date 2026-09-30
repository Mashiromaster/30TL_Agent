"""Training-cache updates must never disguise continuous quotes as contracts."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import update_market_data as updater


def bars(symbol="TL2612", date="2026-09-30 09:31:00", close=117.0):
    timestamp = pd.Timestamp(date)
    return pd.DataFrame({
        "date": [timestamp], "trade_dt": [timestamp.normalize()],
        "ticker": [symbol], "source_symbol": [symbol],
        "series_kind": ["contract"], "source": ["sina"],
        "open": [close], "high": [close + 0.1], "low": [close - 0.1],
        "close": [close], "volume": [10.0], "oi": [123456.0],
        "time": [timestamp.strftime("%H:%M")],
    })


def raw_path(base_dir):
    return base_dir / "data" / updater.VARIETY_CONFIG["TL"]["file_name"]


def test_default_base_dir_is_repository_not_windows_desktop():
    assert Path(updater.DEFAULT_BASE_DIR) == Path(__file__).resolve().parents[1]


def test_default_contracts_are_dynamic_and_never_request_continuous(monkeypatch):
    seen = []
    monkeypatch.setattr(updater, "contract_candidates", lambda variety, as_of=None: ["TL2612", "TL2703"], raising=False)

    def fetch(symbol, frequency):
        seen.append((symbol, frequency))
        return bars(symbol)

    results = updater.fetch_contracts("TL", fetcher=fetch)
    assert seen == [("TL2612", "1min"), ("TL2703", "1min")]
    assert [frame.iloc[0]["ticker"] for frame in results] == ["TL2612", "TL2703"]


@pytest.mark.parametrize("symbol", ["TL0", "T2612", "TL2613", "../TL2612"])
def test_reject_non_specific_or_wrong_variety_symbol_before_fetch(symbol):
    with pytest.raises(ValueError, match="合约"):
        updater.fetch_contracts("TL", symbols=[symbol], fetcher=lambda *args, **kwargs: pytest.fail("unexpected network request"))


def test_source_failure_does_not_discard_successful_contract(capsys):
    def fetch(symbol, frequency):
        if symbol == "TL2612":
            raise TimeoutError("request timed out")
        return bars(symbol)

    results = updater.fetch_contracts("TL", symbols=["TL2612", "TL2703"], fetcher=fetch)
    assert len(results) == 1
    assert results[0].iloc[0]["ticker"] == "TL2703"
    output = capsys.readouterr().out
    assert "TL2612" in output and "request timed out" in output


@pytest.mark.parametrize("column,value", [("ticker", "TL0"), ("source_symbol", "TL0"), ("series_kind", "continuous")])
def test_reject_unexpected_source_identity(column, value, capsys):
    frame = bars()
    frame[column] = value
    results = updater.fetch_contracts("TL", symbols=["TL2612"], fetcher=lambda *args, **kwargs: frame)
    assert results == []
    assert "TL2612" in capsys.readouterr().out


def test_update_creates_directory_deduplicates_and_keeps_money_field(tmp_path):
    assert updater.update_raw_data(tmp_path, "TL", symbols=["TL2612"], fetcher=lambda *args, **kwargs: bars()) is True
    frame = pd.read_pickle(raw_path(tmp_path))
    assert len(frame) == 1
    assert frame.iloc[0]["money"] == 117000.0
    assert bool(frame.iloc[0]["money_estimated"]) is True
    assert frame.iloc[0]["money_method"] == "legacy_close_x_volume_x_100"
    assert frame.iloc[0]["source_symbol"] == "TL2612"


def test_update_backs_up_and_preserves_legacy_rows_with_warning(tmp_path, capsys):
    target = raw_path(tmp_path)
    target.parent.mkdir()
    existing = bars("TL2609", "2026-06-02 10:00:00").drop(columns=["source_symbol", "series_kind", "source"])
    existing.to_pickle(target)
    original_bytes = target.read_bytes()

    assert updater.update_raw_data(tmp_path, "TL", symbols=["TL2612"], fetcher=lambda *args, **kwargs: bars()) is True
    backups = list(target.parent.glob(target.name + ".bak.*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original_bytes
    saved = pd.read_pickle(target)
    assert set(saved.ticker) == {"TL2609", "TL2612"}
    assert pd.isna(saved.loc[saved.ticker == "TL2609", "source_symbol"].iloc[0])
    assert "历史" in capsys.readouterr().out


def test_duplicate_minute_uses_new_verified_contract_quote(tmp_path):
    target = raw_path(tmp_path)
    target.parent.mkdir()
    bars(close=116.0).to_pickle(target)
    assert updater.update_raw_data(tmp_path, "TL", symbols=["TL2612"], fetcher=lambda *args, **kwargs: bars())
    saved = pd.read_pickle(target)
    assert len(saved) == 1
    assert saved.iloc[0]["close"] == 117.0


def test_unrelated_legacy_duplicate_rows_are_not_silently_cleaned(tmp_path):
    target = raw_path(tmp_path)
    target.parent.mkdir()
    legacy = bars("TL2609", "2026-06-02 10:00:00").drop(columns=["source_symbol", "series_kind", "source"])
    pd.concat([legacy, legacy], ignore_index=True).to_pickle(target)
    assert updater.update_raw_data(tmp_path, "TL", symbols=["TL2612"], fetcher=lambda *args, **kwargs: bars())
    saved = pd.read_pickle(target)
    assert len(saved.loc[saved.ticker == "TL2609"]) == 2
    assert len(saved.loc[saved.ticker == "TL2612"]) == 1


def test_failed_replace_leaves_existing_bytes_and_backup_untouched(tmp_path, monkeypatch, capsys):
    target = raw_path(tmp_path)
    target.parent.mkdir()
    bars(close=116.0).to_pickle(target)
    original_bytes = target.read_bytes()

    def fail_replace(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr(updater.os, "replace", fail_replace)
    assert updater.update_raw_data(tmp_path, "TL", symbols=["TL2612"], fetcher=lambda *args, **kwargs: bars()) is False
    assert target.read_bytes() == original_bytes
    assert list(target.parent.glob(target.name + ".bak.*"))[0].read_bytes() == original_bytes
    assert not list(target.parent.glob("*.tmp"))
    assert "disk unavailable" in capsys.readouterr().out


@pytest.mark.parametrize("failure_stage", ["backup", "serialize"])
def test_backup_or_serialization_failure_preserves_existing_bytes(tmp_path, monkeypatch, failure_stage):
    target = raw_path(tmp_path)
    target.parent.mkdir()
    bars(close=116.0).to_pickle(target)
    before = target.read_bytes()
    def fail(*args, **kwargs):
        raise OSError("disk full")
    if failure_stage == "backup":
        monkeypatch.setattr(updater.shutil, "copy2", fail)
    else:
        monkeypatch.setattr(pd.DataFrame, "to_pickle", fail)
    assert updater.update_raw_data(tmp_path, "TL", symbols=["TL2612"], fetcher=lambda *args, **kwargs: bars()) is False
    assert target.read_bytes() == before
    assert not list(target.parent.glob("*.tmp"))


def test_empty_fetch_does_not_rewrite_existing_file(tmp_path):
    target = raw_path(tmp_path)
    target.parent.mkdir()
    bars().to_pickle(target)
    before = target.read_bytes()
    assert updater.update_raw_data(tmp_path, "TL", symbols=["TL2612"], fetcher=lambda *args, **kwargs: pd.DataFrame()) is False
    assert target.read_bytes() == before
    assert not list(target.parent.glob(target.name + ".bak.*"))


def test_invalid_existing_cache_is_not_overwritten(tmp_path):
    target = raw_path(tmp_path)
    target.parent.mkdir()
    pd.DataFrame({"unrecognized": [1]}).to_pickle(target)
    before = target.read_bytes()
    assert updater.update_raw_data(tmp_path, "TL", symbols=["TL2612"], fetcher=lambda *args, **kwargs: pytest.fail("must inspect old cache first")) is False
    assert target.read_bytes() == before


@pytest.mark.parametrize("column,missing", [("date", pd.NaT), ("ticker", None)])
def test_null_legacy_identity_aborts_without_overwriting(tmp_path, column, missing):
    target = raw_path(tmp_path)
    target.parent.mkdir()
    existing = bars()
    existing[column] = missing
    existing.to_pickle(target)
    before = target.read_bytes()
    assert updater.update_raw_data(tmp_path, "TL", symbols=["TL2612"], fetcher=lambda *args, **kwargs: pytest.fail("invalid history must abort")) is False
    assert target.read_bytes() == before


def test_daily_only_calls_isolated_daily_cache_not_raw_updater(tmp_path, monkeypatch):
    calls = []
    def load(base_dir, symbol, frequency, refresh=True):
        calls.append((Path(base_dir), symbol, frequency, refresh))
        return SimpleNamespace(bars=bars("TL0"), error=None, from_cache=False, fetched_at="2026-09-30")

    monkeypatch.setattr(updater, "load_quote_series", load, raising=False)
    monkeypatch.setattr(updater, "update_raw_data", lambda *args, **kwargs: pytest.fail("daily quotes must not enter training raw"))
    assert updater.main(["--variety", "TL", "--base-dir", str(tmp_path), "--daily-only"]) == 0
    assert calls == [(tmp_path, "TL0", "1D", True)]
    assert not raw_path(tmp_path).exists()


def test_daily_only_reports_refresh_failure_instead_of_cache_success(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(updater, "load_quote_series", lambda *args, **kwargs: SimpleNamespace(bars=bars("TL0"), error="provider timeout", from_cache=True, fetched_at="2026-09-01"), raising=False)
    assert updater.main(["--base-dir", str(tmp_path), "--daily-only"]) == 1
    assert "provider timeout" in capsys.readouterr().out


@pytest.mark.parametrize("case", ["empty", "exception"])
def test_daily_only_fails_for_no_valid_daily_snapshot(tmp_path, monkeypatch, case):
    def load(*args, **kwargs):
        if case == "exception":
            raise OSError("cannot read cache")
        return SimpleNamespace(bars=pd.DataFrame(), error=None)
    monkeypatch.setattr(updater, "load_quote_series", load)
    assert updater.main(["--base-dir", str(tmp_path), "--daily-only"]) == 1


def test_cli_symbols_accept_space_or_comma_separated_values(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(updater, "update_raw_data", lambda base_dir, variety, symbols=None: calls.append((variety, symbols)) or True)
    assert updater.main(["--base-dir", str(tmp_path), "--symbols", "TL2612,TL2703", "TL2706"]) == 0
    assert calls == [("TL", ["TL2612", "TL2703", "TL2706"])]


def test_cli_rejects_symbols_together_with_daily_only(tmp_path):
    with pytest.raises(SystemExit) as exc:
        updater.main(["--base-dir", str(tmp_path), "--daily-only", "--symbols", "TL2612"])
    assert exc.value.code == 2


def test_cli_invalid_symbol_has_actionable_argument_error(tmp_path):
    with pytest.raises(SystemExit) as exc:
        updater.main(["--base-dir", str(tmp_path), "--symbols", "TL0"])
    assert exc.value.code == 2


@pytest.mark.parametrize("missing_field", ["date", "ticker", "source_symbol", "series_kind", "close", "oi"])
def test_incomplete_provider_record_is_rejected(missing_field):
    malformed = bars().drop(columns=[missing_field])
    assert updater.fetch_contracts("TL", symbols=["TL2612"], fetcher=lambda *args, **kwargs: malformed) == []


def test_missing_provider_date_is_rejected():
    malformed = bars()
    malformed["date"] = pd.NaT
    assert updater.fetch_contracts("TL", symbols=["TL2612"], fetcher=lambda *args, **kwargs: malformed) == []


def test_invalid_variety_is_rejected_before_fetch():
    with pytest.raises(ValueError, match="合约"):
        updater.fetch_contracts("unsupported", fetcher=lambda *args, **kwargs: pytest.fail("unexpected request"))


def test_string_symbols_are_normalized_and_requested_once():
    seen = []
    def fetch(symbol, frequency):
        seen.append(symbol)
        return bars(symbol)
    updater.fetch_contracts("TL", symbols="tl2612, TL2612", fetcher=fetch)
    assert seen == ["TL2612"]
