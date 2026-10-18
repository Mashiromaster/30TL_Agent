import hashlib
import json
from pathlib import Path
import warnings
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from factor_candidates import build_local_factor_candidate


def bars(days=5):
    frames = []
    for day_no, day in enumerate(pd.bdate_range("2026-05-11", periods=days)):
        dates = pd.date_range(day + pd.Timedelta(hours=9, minutes=31), periods=120, freq="min").append(
            pd.date_range(day + pd.Timedelta(hours=13, minutes=1), periods=135, freq="min")
        )
        for contract_no, ticker in enumerate(["TL2606", "TL2609"]):
            price = 100 + contract_no * 10 + day_no + np.arange(255) / 10000
            frames.append(pd.DataFrame({
                "date": dates, "trade_dt": day, "ticker": ticker,
                "open": price, "high": price + .01, "low": price - .01,
                "close": price, "volume": 10., "oi": 1000 - contract_no * 100,
                "money": 10000., "time": dates.strftime("%H:%M"),
            }))
    return pd.concat(frames, ignore_index=True)


def put_raw(root, frame):
    path = root / "data" / "TL分钟级量价数据.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(path)
    return path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def minute_snapshot(tmp_path, monkeypatch):
    directory = tmp_path / "artifacts/minute_snapshots/minset-test"
    directory.mkdir(parents=True)
    path = directory / "bars.pkl"
    data = bars()
    data.to_pickle(path)
    metadata = {"snapshot_id": "minset-test", "provider_verified": False,
                "acquired_at": "2026-05-18T00:00:00Z", "inputs": [{"batch_id": "minute-test"}]}
    (directory / "manifest.json").write_text(json.dumps(metadata))
    def load(base, snapshot_id):
        assert snapshot_id == "minset-test"
        return data.copy(), path, sha(path), metadata
    monkeypatch.setitem(sys.modules, "minute_store", SimpleNamespace(load_minute_snapshot=load))
    return path


def test_explicit_minute_snapshot_does_not_read_legacy_source(tmp_path, minute_snapshot):
    raw = tmp_path / "data/TL分钟级量价数据.pkl"
    raw.parent.mkdir()
    raw.write_bytes(b"must-not-load-this-legacy-file")
    result = build_local_factor_candidate(tmp_path, now="2026-05-18", minute_snapshot_id="minset-test")
    frame, manifest = load_result(result)
    assert manifest["source_file"] == str(minute_snapshot)
    assert manifest["source_sha256"] == sha(minute_snapshot)
    assert manifest["provenance"]["minute_snapshot_id"] == "minset-test"
    assert manifest["sources"]["market"]["provider_verified"] is False
    assert str(minute_snapshot.parent / "manifest.json") in manifest["input_sha256"]
    assert raw.read_bytes() == b"must-not-load-this-legacy-file"
    from research_readiness import verify_factor_provenance
    assert verify_factor_provenance(tmp_path, result["factor_path"], frame)["verified"]


def test_minute_snapshot_manifest_change_during_build_is_rejected(tmp_path, minute_snapshot, cheap_factors, monkeypatch):
    import factor_candidates
    calculate = factor_candidates.calculate_enhanced_factors
    def mutate(frame, **kwargs):
        (minute_snapshot.parent / "manifest.json").write_text('{"changed": true}')
        return calculate(frame, **kwargs)
    monkeypatch.setattr(factor_candidates, "calculate_enhanced_factors", mutate)
    result = build_local_factor_candidate(tmp_path, now="2026-05-18", minute_snapshot_id="minset-test")
    assert result["status"] == "failed"
    assert result["factor_path"] is None


def load_result(result):
    assert result["status"] == "built", result
    return pd.read_pickle(result["factor_path"]), json.loads(Path(result["manifest_path"]).read_text())


@pytest.fixture
def cheap_factors(monkeypatch):
    def calculate(frame, **kwargs):
        assert kwargs.get("df_tick") is None
        assert kwargs.get("df_basis") is None
        assert kwargs.get("df_macro") is None
        result = frame.copy()
        result["test_lag"] = result.adj_close.shift(1).fillna(0)
        return result
    monkeypatch.setattr("factor_candidates.calculate_enhanced_factors", calculate)


def test_real_factors_are_isolated_hashed_and_never_use_legacy_cache(tmp_path, monkeypatch):
    raw = put_raw(tmp_path, bars())
    protected = [raw]
    for name in ["data/main_contract_spliced.pkl", "outputs/df_factors.pkl", "models/model.pkl"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"do-not-read-or-overwrite")
        protected.append(path)
    before = {str(p): sha(p) for p in protected}
    import requests
    monkeypatch.setattr(requests.sessions.Session, "request", lambda *a, **k: pytest.fail("network used"))
    result = build_local_factor_candidate(tmp_path, now="2026-05-18 10:00")
    frame, manifest = load_result(result)
    assert "Short_Momentum_1D" in frame
    assert len(frame) == 4 * 255 - 500
    assert not any(c.startswith(("CNN_", "GP_")) for c in frame)
    assert before == {str(p): sha(p) for p in protected}
    assert Path(result["factor_path"]).parent.parent == tmp_path / "artifacts/factor_candidates"
    assert manifest["factor_sha256"] == sha(result["factor_path"])
    assert manifest["factor_file"] == "factors.pkl"
    assert manifest["source_sha256"] == sha(raw)
    provenance = frame.attrs["factor_provenance"]
    assert provenance == manifest["provenance"]
    assert provenance["schema_version"] == 1
    assert provenance["kind"] == "local_causal_rebuild_v1"
    assert provenance["pit_contract"] == "previous_completed_observed_day_oi_v1"
    assert provenance["candidate_id"] == result["candidate_id"]
    assert provenance["source_sha256"] == sha(raw)
    assert manifest["sources"]["tick"]["status"] == "excluded"
    assert manifest["sources"]["basis"]["status"] == "excluded"
    assert manifest["sources"]["macro"]["status"] == "excluded"
    assert result["coverage"]["end"].startswith("2026-05-15")
    assert result["warnings"]
    from research_readiness import verify_factor_provenance
    assert verify_factor_provenance(tmp_path, result["factor_path"], frame)["verified"]


def test_future_oi_never_changes_current_day_selection_and_prefix(tmp_path, cheap_factors):
    frame = bars()
    put_raw(tmp_path, frame)
    first, _ = load_result(build_local_factor_candidate(tmp_path, now="2026-05-18"))
    future = (frame.date.dt.normalize() >= pd.Timestamp("2026-05-13")) & frame.ticker.eq("TL2609")
    frame.loc[future, "oi"] = 99999
    put_raw(tmp_path, frame)
    second, _ = load_result(build_local_factor_candidate(tmp_path, now="2026-05-18"))
    cutoff = pd.Timestamp("2026-05-14")
    pd.testing.assert_frame_equal(
        first.loc[first.date < cutoff].reset_index(drop=True),
        second.loc[second.date < cutoff].reset_index(drop=True), check_flags=False,
    )
    assert first.loc[first.date.dt.day == 13, "ticker"].eq("TL2606").all()
    assert second.loc[second.date.dt.day == 14, "ticker"].eq("TL2609").all()
    assert second.loc[second.is_switch.eq(1), "raw_ret"].eq(0).all()


def test_append_future_prices_does_not_rewrite_adjustment(tmp_path, cheap_factors):
    frame = bars()
    put_raw(tmp_path, frame.loc[frame.date.dt.day <= 14])
    first, manifest = load_result(build_local_factor_candidate(tmp_path, now="2026-05-18"))
    put_raw(tmp_path, frame)
    second, new_manifest = load_result(build_local_factor_candidate(tmp_path, now="2026-05-18"))
    pd.testing.assert_frame_equal(first, second.iloc[:len(first)].reset_index(drop=True), check_flags=False)
    assert manifest["source_sha256"] != new_manifest["source_sha256"]
    assert manifest["candidate_id"] != new_manifest["candidate_id"]


def test_missing_selected_contract_and_partial_day_are_quarantined(tmp_path, cheap_factors):
    frame = bars(6)
    frame = frame.loc[~((frame.date.dt.day == 12) & frame.ticker.eq("TL2606"))]
    frame = frame.loc[~((frame.date == pd.Timestamp("2026-05-14 09:31")) & frame.ticker.eq("TL2606"))]
    put_raw(tmp_path, frame)
    result = build_local_factor_candidate(tmp_path, now="2026-05-20")
    output, manifest = load_result(result)
    assert not output.date.dt.day.isin([11, 12, 14]).any()
    audit = json.loads((Path(result["manifest_path"]).parent / "audit.json").read_text())
    reasons = {entry["reason"] for entry in audit["quarantined_days"]}
    assert "first_observed_day" in reasons
    assert "selected_contract_missing" in reasons
    assert "selected_contract_incomplete" in reasons
    assert manifest["coverage"]["quarantined_days"] >= 3


@pytest.mark.parametrize("problem", ["duplicate", "bad_high", "negative_volume", "negative_oi", "nan", "trade_date", "continuous", "subminute"])
def test_invalid_market_input_fails_with_diagnostic(tmp_path, problem):
    frame = bars()
    if problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif problem == "bad_high":
        frame.loc[0, "high"] = .5
    elif problem == "negative_volume":
        frame.loc[0, "volume"] = -1
    elif problem == "negative_oi":
        frame.loc[0, "oi"] = -1
    elif problem == "nan":
        frame.loc[0, "close"] = np.nan
    elif problem == "trade_date":
        frame.loc[0, "trade_dt"] = pd.Timestamp("2027-01-01")
    elif problem == "continuous":
        frame.loc[0, "ticker"] = "TL0"
    else:
        frame.loc[0, "date"] += pd.Timedelta(seconds=1)
    source = put_raw(tmp_path, frame)
    old_sha = sha(source)
    result = build_local_factor_candidate(tmp_path, now="2026-05-18")
    assert result["status"] == "failed"
    assert result["factor_path"] is None
    assert Path(result["manifest_path"]).is_file()
    assert json.loads(Path(result["manifest_path"]).read_text())["error"]
    assert old_sha == sha(source)


@pytest.mark.parametrize("column", ["open", "high", "low", "close", "volume", "oi"])
@pytest.mark.parametrize("value", [True, np.bool_(False), 10 + 2j, np.complex128(10 + 0j)])
def test_legacy_raw_rejects_boolean_and_complex_values_before_coercion(tmp_path, cheap_factors, column, value):
    frame = bars()
    frame[column] = frame[column].astype(object)
    frame.at[0, column] = value
    source = put_raw(tmp_path, frame)
    old_sha = sha(source)
    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        result = build_local_factor_candidate(tmp_path, now="2026-05-18")
    assert result["status"] == "failed"
    assert result["factor_path"] is None
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert "real non-boolean" in manifest["error"]
    assert old_sha == sha(source)
    assert not (Path(result["manifest_path"]).parent / "factors.pkl").exists()


@pytest.mark.parametrize("dtype", [bool, complex])
def test_legacy_raw_rejects_boolean_and_complex_column_dtypes(tmp_path, cheap_factors, dtype):
    frame = bars()
    frame["volume"] = frame.volume.astype(dtype)
    source = put_raw(tmp_path, frame)
    old_sha = sha(source)
    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        result = build_local_factor_candidate(tmp_path, now="2026-05-18")
    assert result["status"] == "failed"
    assert result["factor_path"] is None
    assert "real non-boolean" in json.loads(Path(result["manifest_path"]).read_text())["error"]
    assert old_sha == sha(source)


def test_legacy_raw_rejects_duplicate_columns_before_selection(tmp_path):
    frame = bars()
    frame = pd.concat([frame, frame[["volume"]]], axis=1)
    source = put_raw(tmp_path, frame)
    old_sha = sha(source)
    result = build_local_factor_candidate(tmp_path, now="2026-05-18")
    assert result["status"] == "failed"
    assert result["factor_path"] is None
    assert "duplicate market columns" in json.loads(Path(result["manifest_path"]).read_text())["error"]
    assert old_sha == sha(source)


@pytest.mark.parametrize("column", ["date", "trade_dt"])
@pytest.mark.parametrize("kind", ["int", "np_int", "float", "np_float", "bool", "np_bool"])
def test_legacy_raw_rejects_undeclared_numeric_timestamp_units(tmp_path, cheap_factors, column, kind):
    frame = bars()
    stamp = frame.at[0, column].value
    values = {"int": stamp, "np_int": np.int64(stamp), "float": float(stamp),
              "np_float": np.float64(stamp), "bool": True, "np_bool": np.bool_(False)}
    frame[column] = frame[column].astype(object)
    frame.at[0, column] = values[kind]
    source = put_raw(tmp_path, frame)
    old_sha = sha(source)
    result = build_local_factor_candidate(tmp_path, now="2026-05-18")
    assert result["status"] == "failed"
    assert result["factor_path"] is None
    assert "numeric timestamp units are not declared" in json.loads(Path(result["manifest_path"]).read_text())["error"]
    assert old_sha == sha(source)
    assert not (Path(result["manifest_path"]).parent / "factors.pkl").exists()


def test_legacy_raw_still_accepts_explicit_timestamp_strings(tmp_path, cheap_factors):
    frame = bars()
    frame["date"] = frame.date.dt.strftime("%Y-%m-%d %H:%M:%S")
    frame["trade_dt"] = frame.trade_dt.dt.strftime("%Y-%m-%d")
    put_raw(tmp_path, frame)
    result = build_local_factor_candidate(tmp_path, now="2026-05-18")
    assert result["status"] == "built"


def test_absent_or_insufficient_source_is_failed_not_fresh(tmp_path):
    result = build_local_factor_candidate(tmp_path, now="2026-05-18")
    assert result["status"] == "failed"
    assert result["factor_path"] is None
    put_raw(tmp_path, bars(1))
    result = build_local_factor_candidate(tmp_path, now="2026-05-18")
    assert result["status"] == "failed"


def test_now_excludes_future_and_in_progress_day(tmp_path, cheap_factors):
    put_raw(tmp_path, bars())
    frame, manifest = load_result(build_local_factor_candidate(tmp_path, now="2026-05-14 10:00+08:00"))
    assert frame.date.max() == pd.Timestamp("2026-05-13 15:15")
    assert manifest["coverage"]["excluded_not_completed_rows"] > 0


def test_legacy_macro_or_schema_only_metadata_is_excluded(tmp_path, cheap_factors):
    put_raw(tmp_path, bars())
    out = tmp_path / "outputs"
    out.mkdir()
    pd.DataFrame({"available_ts": [pd.Timestamp("2026-05-01")], "Macro_Test": [7.]}).to_pickle(out / "macro_factors.pkl")
    (out / "macro_factors.meta.json").write_text(json.dumps({"schema_version": 2, "factor_ttls": {"Macro_Test": 5}}))
    frame, manifest = load_result(build_local_factor_candidate(tmp_path, now="2026-05-18"))
    assert "Macro_Test" not in frame
    assert manifest["sources"]["macro"]["status"] == "excluded"


def test_verified_macro_respects_asof_ttl_and_preserves_missingness(tmp_path, cheap_factors):
    put_raw(tmp_path, bars())
    out = tmp_path / "outputs"
    out.mkdir()
    cache = out / "macro_factors.pkl"
    pd.DataFrame({
        "available_ts": pd.to_datetime(["2026-05-12 10:00", "2026-05-14 10:00"]),
        "_asof__Macro_Test": pd.to_datetime(["2026-05-12 10:00", "2026-05-14 10:00"]),
        "Macro_Test": [7., 9.],
    }).to_pickle(cache)
    (out / "macro_factors.meta.json").write_text(json.dumps({
        "schema_version": 2, "factor_ttls": {"Macro_Test": .25},
        "availability_verified": True, "vintage_verified": True,
        "cache_sha256": sha(cache),
    }))
    frame, manifest = load_result(build_local_factor_candidate(tmp_path, now="2026-05-18"))
    assert manifest["sources"]["macro"]["status"] == "included"
    assert frame.loc[frame.date == "2026-05-12 09:59", "Macro_Test"].isna().all()
    assert frame.loc[frame.date == "2026-05-12 10:00", "Macro_Test"].iloc[0] == 7
    assert frame.loc[frame.date == "2026-05-13 10:00", "Macro_Test"].isna().all()
    assert frame.loc[frame.date == "2026-05-14 10:00", "Macro_Test"].iloc[0] == 9
    assert "_asof__Macro_Test" in frame


def test_real_calculator_retains_prefix_when_later_days_are_appended(tmp_path):
    raw = bars(5)
    put_raw(tmp_path, raw.loc[raw.date.dt.day <= 14])
    first, _ = load_result(build_local_factor_candidate(tmp_path, now="2026-05-18"))
    put_raw(tmp_path, raw)
    second, _ = load_result(build_local_factor_candidate(tmp_path, now="2026-05-18"))
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.iloc[:len(first)].reset_index(drop=True))


def test_input_replacement_during_build_fails_before_factor_is_persisted(tmp_path, cheap_factors, monkeypatch):
    import factor_candidates
    source = put_raw(tmp_path, bars())
    original = factor_candidates.calculate_enhanced_factors
    def mutate_source(frame, **kwargs):
        changed = pd.read_pickle(source)
        changed.loc[0, "volume"] += 1
        changed.to_pickle(source)
        return original(frame, **kwargs)
    monkeypatch.setattr(factor_candidates, "calculate_enhanced_factors", mutate_source)
    result = build_local_factor_candidate(tmp_path, now="2026-05-18")
    assert result["status"] == "failed"
    assert result["factor_path"] is None
    assert "source changed" in " ".join(result["warnings"])
    assert not (Path(result["manifest_path"]).parent / "factors.pkl").exists()


def test_partial_previous_observed_day_cannot_be_skipped_for_oi_selection(tmp_path, cheap_factors):
    frame = bars()
    frame = frame.loc[frame.date.ne(pd.Timestamp("2026-05-12 14:00"))]
    put_raw(tmp_path, frame)
    result = build_local_factor_candidate(tmp_path, now="2026-05-18")
    output, _ = load_result(result)
    assert not output.date.dt.day.isin([12, 13]).any()
    audit = json.loads((Path(result["manifest_path"]).parent / "audit.json").read_text())
    assert any(entry["reason"] == "prior_observed_day_incomplete" for entry in audit["quarantined_days"])


def test_contract_without_closing_oi_is_explicitly_reported(tmp_path, cheap_factors):
    frame = bars()
    frame = frame.loc[~((frame.date == pd.Timestamp("2026-05-11 15:15")) & frame.ticker.eq("TL2606"))]
    put_raw(tmp_path, frame)
    result = build_local_factor_candidate(tmp_path, now="2026-05-18")
    output, _ = load_result(result)
    assert output.loc[output.date.dt.day.eq(12), "ticker"].eq("TL2609").all()
    audit = json.loads((Path(result["manifest_path"]).parent / "audit.json").read_text())
    assert audit["oi_observation_exclusions"][0]["contracts_without_closing_oi"] == ["TL2606"]


@pytest.mark.parametrize("input_kind", ["empty", "missing_column", "bad_date", "not_dataframe", "insufficient_warmup"])
def test_malformed_or_too_short_source_is_diagnostic_failure(tmp_path, input_kind):
    frame = bars(2)
    if input_kind == "empty":
        frame = frame.iloc[:0]
    elif input_kind == "missing_column":
        frame = frame.drop(columns="oi")
    elif input_kind == "bad_date":
        frame.loc[0, "date"] = pd.NaT
    elif input_kind == "not_dataframe":
        source = put_raw(tmp_path, frame)
        pd.to_pickle({"not": "bars"}, source)
    if input_kind != "not_dataframe":
        put_raw(tmp_path, frame)
    result = build_local_factor_candidate(tmp_path, now="2026-05-18")
    assert result["status"] == "failed"
    assert json.loads(Path(result["manifest_path"]).read_text())["error"]


@pytest.mark.parametrize("problem", ["version", "hash", "ttls", "negative_ttl", "missing_asof", "future_asof", "duplicate", "non_numeric", "future_events"])
def test_bad_macro_is_excluded_without_discarding_market_candidate(tmp_path, cheap_factors, problem):
    put_raw(tmp_path, bars())
    out = tmp_path / "outputs"
    out.mkdir()
    cache = out / "macro_factors.pkl"
    data = pd.DataFrame({"available_ts": pd.to_datetime(["2026-05-12"]),
                         "_asof__Macro_Test": pd.to_datetime(["2026-05-12"]), "Macro_Test": [1.]})
    metadata = {"schema_version": 2, "factor_ttls": {"Macro_Test": 5},
                "availability_verified": True, "vintage_verified": True}
    if problem == "version":
        metadata["schema_version"] = 1
    elif problem == "ttls":
        metadata["factor_ttls"] = {}
    elif problem == "negative_ttl":
        metadata["factor_ttls"]["Macro_Test"] = -1
    elif problem == "missing_asof":
        data = data.drop(columns="_asof__Macro_Test")
    elif problem == "future_asof":
        data["_asof__Macro_Test"] = pd.Timestamp("2026-05-13")
    elif problem == "duplicate":
        data = pd.concat([data, data])
    elif problem == "non_numeric":
        data["Macro_Test"] = "not a number"
    elif problem == "future_events":
        data["available_ts"] = data["_asof__Macro_Test"] = pd.Timestamp("2027-01-01")
    data.to_pickle(cache)
    metadata["cache_sha256"] = "incorrect" if problem == "hash" else sha(cache)
    (out / "macro_factors.meta.json").write_text(json.dumps(metadata))
    frame, manifest = load_result(build_local_factor_candidate(tmp_path, now="2026-05-18"))
    assert manifest["sources"]["macro"]["status"] == "excluded"
    assert "Macro_Test" not in frame


def test_audit_write_failure_never_returns_built(tmp_path, cheap_factors, monkeypatch):
    put_raw(tmp_path, bars())
    def denied(*args, **kwargs):
        raise OSError("test write denied")
    monkeypatch.setattr("factor_candidates._atomic_json", denied)
    result = build_local_factor_candidate(tmp_path, now="2026-05-18")
    assert result["status"] == "failed"
    assert result["factor_path"] is None
    assert "audit persistence failed" in " ".join(result["warnings"])


@pytest.mark.parametrize("timestamp", ["invalid", pd.NaT])
def test_invalid_now_remains_inspectable(tmp_path, timestamp):
    result = build_local_factor_candidate(tmp_path, now=timestamp)
    assert result["status"] == "failed"
    assert Path(result["manifest_path"]).is_file()


def test_observed_date_gaps_and_template_origin_are_not_calendar_certification(tmp_path, cheap_factors):
    frame = bars(6)
    put_raw(tmp_path, frame)
    result = build_local_factor_candidate(tmp_path, now="2026-05-20")
    _, manifest = load_result(result)
    audit = json.loads((Path(result["manifest_path"]).parent / "audit.json").read_text())
    assert audit["observed_day_gaps"] == [{"after": "2026-05-15", "before": "2026-05-18", "unobserved_calendar_days": 2}]
    assert manifest["coverage"]["calendar_verified"] is False
    assert manifest["coverage"]["template_source"] == "src/check_timestamp_format.py; observed local end-labelled bars"


@pytest.mark.parametrize("escape_level", ["artifacts", "factor_candidates"])
def test_candidate_directory_symlink_escape_is_rejected_before_external_write(tmp_path, cheap_factors, escape_level):
    base = tmp_path / "project"
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("unchanged")
    put_raw(base, bars())
    if escape_level == "artifacts":
        (base / "artifacts").symlink_to(outside, target_is_directory=True)
    else:
        (base / "artifacts").mkdir()
        (base / "artifacts/factor_candidates").symlink_to(outside, target_is_directory=True)
    result = build_local_factor_candidate(base, now="2026-05-18")
    assert result["status"] == "failed"
    assert result["factor_path"] is None
    assert result["manifest_path"] is None
    assert list(outside.iterdir()) == [sentinel]
    assert sentinel.read_text() == "unchanged"
    assert "candidate output" in " ".join(result["warnings"]).lower()


def test_candidate_output_redirect_during_calculation_is_rechecked(tmp_path, cheap_factors, monkeypatch):
    import factor_candidates
    base = tmp_path / "project"
    outside = tmp_path / "outside"
    outside.mkdir()
    put_raw(base, bars())
    original = factor_candidates.calculate_enhanced_factors
    def redirect_output(frame, **kwargs):
        candidate = next((base / "artifacts/factor_candidates").iterdir())
        candidate.rename(candidate.with_name("held-" + candidate.name))
        candidate.symlink_to(outside, target_is_directory=True)
        return original(frame, **kwargs)
    monkeypatch.setattr(factor_candidates, "calculate_enhanced_factors", redirect_output)
    result = build_local_factor_candidate(base, now="2026-05-18")
    assert result["status"] == "failed"
    assert result["factor_path"] is None
    assert result["manifest_path"] is None
    assert list(outside.iterdir()) == []


def test_internal_output_symlink_cannot_redirect_artifacts_to_production(tmp_path, cheap_factors):
    put_raw(tmp_path, bars())
    (tmp_path / "outputs").mkdir()
    (tmp_path / "artifacts").symlink_to(tmp_path / "outputs", target_is_directory=True)
    result = build_local_factor_candidate(tmp_path, now="2026-05-18")
    assert result["status"] == "failed"
    assert result["manifest_path"] is None
    assert list((tmp_path / "outputs").iterdir()) == []
