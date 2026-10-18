import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import minute_store as store


NOW = "2026-05-18 16:00"


def bars(day="2026-05-11", ticker="TL2606", rows=255):
    day = pd.Timestamp(day)
    dates = pd.date_range(day + pd.Timedelta(hours=9, minutes=31), periods=120, freq="min").append(
        pd.date_range(day + pd.Timedelta(hours=13, minutes=1), periods=135, freq="min")
    )[:rows]
    return pd.DataFrame({
        "date": dates, "trade_dt": day, "ticker": ticker, "open": 100.,
        "high": 101., "low": 99., "close": 100., "volume": 10., "oi": 100.,
    })


def csv_bytes(frame=None):
    return (bars() if frame is None else frame).to_csv(index=False).encode("utf-8")


def save_batch(root, frame=None):
    return store.import_minute_csv(root, csv_bytes(frame), provider="Local export",
                                   license_note="Research use declaration", persist=True, now=NOW)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def acquisition(**changes):
    value = {"kind": "sina_recent", "symbol": "TL2606", "frequency": "1min",
             "response_sha256": "a" * 64, "fetched_at": "2026-05-11T15:16:00+08:00",
             "bar_label": "start", "semantics_confirmed": True}
    return {**value, **changes}


def test_tqsdk_import_preserves_raw_auxiliary_columns_and_conversion_lineage(tmp_path):
    from test_minute_formats import tq_csv
    raw = tq_csv()
    inspected = store.import_minute_csv(tmp_path, raw, provider="tqsdk export", license_note="authorized export",
                                        input_format="tqsdk", bar_seconds=60, now=NOW)
    assert inspected["status"] == "inspected"
    assert inspected["source_format"] == "tqsdk"
    assert not list(tmp_path.iterdir())
    result = store.import_minute_csv(tmp_path, raw, provider="tqsdk export", license_note="authorized export",
                                     input_format="tqsdk", bar_seconds=60, persist=True, now=NOW)
    raw_path = Path(result["manifest_path"]).parent / "source.csv"
    assert raw_path.read_bytes() == raw
    assert result["source_sha256"] == sha(raw_path)
    assert result["data_sha256"] == sha(result["data_path"])
    assert result["source_sha256"] != result["data_sha256"]
    assert pd.read_csv(result["data_path"]).iloc[0]["date"] == "2026-05-11 09:31:00"
    snap = store.build_minute_snapshot(tmp_path, [result["batch_id"]], now=NOW)
    _, _, _, manifest = store.load_minute_snapshot(tmp_path, snap["snapshot_id"])
    source = manifest["inputs"][0]
    assert source["conversion"] == result["conversion"]
    assert source["source_format"] == "tqsdk"
    assert source["source_file"] == "source.csv"
    raw_path.write_bytes(raw + b"\n")
    with pytest.raises(ValueError, match="source.*hash"):
        store.build_minute_snapshot(tmp_path, [result["batch_id"]], now=NOW)


def test_canonical_acquisition_preserves_response_provenance_and_old_manifests(tmp_path):
    result = store.import_minute_csv(tmp_path, csv_bytes(), provider="Sina", license_note="research",
                                     acquisition=acquisition(), persist=True, now=NOW)
    snap = store.build_minute_snapshot(tmp_path, [result["batch_id"]], now=NOW)
    _, _, _, manifest = store.load_minute_snapshot(tmp_path, snap["snapshot_id"])
    assert manifest["inputs"][0]["acquisition"] == acquisition()
    assert result["source_format"] == "canonical"
    assert result["network_used"] is False
    assert result["bar_contract"]["semantics_verified"] is False
    assert not (Path(result["manifest_path"]).parent / "source.csv").exists()
    old = json.loads(Path(result["manifest_path"]).read_text())
    old.pop("source_format")
    old.pop("acquisition")
    Path(result["manifest_path"]).write_text(json.dumps(old))
    assert len(store.list_minute_batches(tmp_path)) == 1
    store.build_minute_snapshot(tmp_path, [result["batch_id"]], now=NOW)


@pytest.mark.parametrize("changes", [
    {"kind": "unknown"}, {"symbol": "TL0"}, {"symbol": "TL2609"},
    {"frequency": "15min"}, {"response_sha256": "abc"}, {"fetched_at": "2026-05-11"},
    {"fetched_at": "2036-05-11T09:00:00+08:00"}, {"fetched_at": "not timestamp"},
    {"bar_label": "unknown"}, {"semantics_confirmed": False}, {"semantics_confirmed": 1},
    {"token": "secret"},
])
def test_acquisition_metadata_is_bounded_exact_and_matches_bars(tmp_path, changes):
    with pytest.raises(ValueError):
        store.import_minute_csv(tmp_path, csv_bytes(), provider="Sina", license_note="research",
                                acquisition=acquisition(**changes), persist=True, now=NOW)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("options", [
    {"input_format": "unknown"}, {"input_format": None}, {"bar_seconds": 60},
    {"input_format": "tqsdk"}, {"input_format": "tqsdk", "bar_seconds": 900},
    {"input_format": "tqsdk", "bar_seconds": 60, "acquisition": acquisition()},
    {"acquisition": {}}, {"acquisition": "not a dict"},
])
def test_import_format_options_fail_closed(tmp_path, options):
    with pytest.raises(ValueError):
        store.import_minute_csv(tmp_path, csv_bytes(), provider="x", license_note="research", **options)


@pytest.mark.parametrize("changes", [
    {"source_format": "unknown"}, {"conversion": {}}, {"source_file": "../source.csv"},
    {"acquisition": {"token": "bad"}},
])
def test_optional_manifest_metadata_is_validated_on_build_and_load(tmp_path, changes):
    result = save_batch(tmp_path)
    path = Path(result["manifest_path"])
    manifest = json.loads(path.read_text())
    manifest.update(changes)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        store.build_minute_snapshot(tmp_path, [result["batch_id"]], now=NOW)
    path.write_text(json.dumps({k: v for k, v in result.items() if k not in {"data_path", "manifest_path"}}))
    snap = store.build_minute_snapshot(tmp_path, [result["batch_id"]], now=NOW)
    path = Path(snap["manifest_path"])
    manifest = json.loads(path.read_text())
    manifest["inputs"][0].update(changes)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        store.load_minute_snapshot(tmp_path, snap["snapshot_id"])


def tq_batch(tmp_path):
    from test_minute_formats import tq_csv
    return store.import_minute_csv(tmp_path, tq_csv(), provider="tqsdk", license_note="research",
                                   input_format="tqsdk", bar_seconds=60, persist=True, now=NOW)


@pytest.mark.parametrize("change", ["bad_file", "acquisition", "other_symbol", "excess_pairs"])
def test_tqsdk_metadata_contract_is_revalidated(tmp_path, change):
    from minute_formats import _metadata
    result = tq_batch(tmp_path)
    path = Path(result["manifest_path"])
    value = json.loads(path.read_text())
    if change == "bad_file":
        value["source_file"] = "../outside.csv"
    elif change == "acquisition":
        value["acquisition"] = acquisition()
    elif change == "other_symbol":
        value["conversion"] = _metadata("CFFEX.TL2609", 1)
    else:
        value["conversion"]["observed_one_minute_pairs"] = value["rows"]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        store.build_minute_snapshot(tmp_path, [result["batch_id"]], now=NOW)
    assert not (tmp_path / "artifacts").exists()


def test_tqsdk_original_source_is_required_and_obeys_combined_read_limit(tmp_path, monkeypatch):
    result = tq_batch(tmp_path)
    source = Path(result["manifest_path"]).parent / "source.csv"
    payload = source.read_bytes()
    normalized_size = Path(result["data_path"]).stat().st_size
    monkeypatch.setattr(store, "MAX_SNAPSHOT_BYTES", normalized_size + len(payload) - 1)
    with pytest.raises(ValueError, match="combined input size"):
        store.build_minute_snapshot(tmp_path, [result["batch_id"]], now=NOW)
    source.unlink()
    with pytest.raises(ValueError, match="unavailable"):
        store.build_minute_snapshot(tmp_path, [result["batch_id"]], now=NOW)


def test_tqsdk_trusted_period_still_uses_canonical_session_and_future_validation(tmp_path):
    from test_minute_formats import tq_csv
    for dates in [["2026-05-11 11:30:00"], ["2026-05-19 09:30:00"]]:
        with pytest.raises(ValueError):
            store.import_minute_csv(tmp_path, tq_csv(times=dates), provider="tqsdk", license_note="research",
                                    input_format="tqsdk", bar_seconds=60, persist=True, now=NOW)
    assert not list(tmp_path.iterdir())


def test_full_tqsdk_session_maps_to_255_end_labels_without_filling(tmp_path):
    from test_minute_formats import tq_csv
    times = [str(value - pd.Timedelta(minutes=1)) for value in bars().date]
    result = store.import_minute_csv(tmp_path, tq_csv(times=times), provider="tqsdk", license_note="research",
                                     input_format="tqsdk", bar_seconds=60, now=NOW)
    assert result["quality"]["status"] == "complete_observed"
    assert result["rows"] == 255
    assert result["conversion"]["observed_one_minute_pairs"] == 253


def test_acquisition_timestamps_in_manifests_fail_closed_if_naive(tmp_path):
    result = store.import_minute_csv(tmp_path, csv_bytes(), provider="Sina", license_note="research",
                                     acquisition=acquisition(), persist=True, now=NOW)
    path = Path(result["manifest_path"])
    value = json.loads(path.read_text())
    value["acquired_at"] = "2026-10-04T00:00:00"
    path.write_text(json.dumps(value))
    assert store.list_minute_batches(tmp_path) == []
    with pytest.raises(ValueError):
        store.build_minute_snapshot(tmp_path, [result["batch_id"]], now=NOW)


@pytest.mark.parametrize("fetched_at", ["2026-05-11T09:00:00+08:00", "2026-05-11T07:14:59Z"])
def test_acquisition_bars_cannot_postdate_real_response_capture(tmp_path, fetched_at):
    with pytest.raises(ValueError, match="capture"):
        store.import_minute_csv(tmp_path, csv_bytes(), provider="Sina", license_note="research",
                                acquisition=acquisition(fetched_at=fetched_at), persist=True, now="2027-01-01")
    assert not list(tmp_path.iterdir())


def test_acquisition_capture_accepts_closed_bars_in_utc(tmp_path):
    result = store.import_minute_csv(tmp_path, csv_bytes(), provider="Sina", license_note="research",
                                     acquisition=acquisition(fetched_at="2026-05-11T07:15:00Z"), now=NOW)
    assert result["rows"] == 255


def test_hash_updated_batch_cannot_bypass_acquisition_capture_bound(tmp_path):
    result = store.import_minute_csv(tmp_path, csv_bytes(), provider="Sina", license_note="research",
                                     acquisition=acquisition(), persist=True, now=NOW)
    data_path, manifest_path = Path(result["data_path"]), Path(result["manifest_path"])
    # Simulate a locally edited normalized batch whose integrity hash was also
    # updated. Its own as-of allows May 12, but the capture event was May 11.
    data_path.write_bytes(csv_bytes(bars("2026-05-12")))
    manifest = json.loads(manifest_path.read_text())
    manifest["data_sha256"] = sha(data_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="capture"):
        store.build_minute_snapshot(tmp_path, [result["batch_id"]], now="2027-01-01")
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("change", ["pairs", "bars"])
def test_tqsdk_build_replays_original_conversion_not_just_independent_hashes(tmp_path, change):
    result = tq_batch(tmp_path)
    manifest_path, data_path = Path(result["manifest_path"]), Path(result["data_path"])
    manifest = json.loads(manifest_path.read_text())
    if change == "pairs":
        # This is structurally legal (< 2 rows) but disagrees with source.csv.
        manifest["conversion"]["observed_one_minute_pairs"] = 0
    else:
        edited = pd.read_csv(data_path)
        edited.loc[0, "close"] = 100.25
        data_path.write_bytes(store._canonical_csv(edited))
        manifest["data_sha256"] = sha(data_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="conversion"):
        store.build_minute_snapshot(tmp_path, [result["batch_id"]], now=NOW)
    assert not (tmp_path / "artifacts").exists()


def test_materialized_tqsdk_snapshot_stays_independent_of_source_retention(tmp_path):
    result = tq_batch(tmp_path)
    snapshot = store.build_minute_snapshot(tmp_path, [result["batch_id"]], now=NOW)
    (Path(result["manifest_path"]).parent / "source.csv").unlink()
    frame, _, _, manifest = store.load_minute_snapshot(tmp_path, snapshot["snapshot_id"])
    assert len(frame) == 2
    assert manifest["inputs"][0]["source_sha256"] == result["source_sha256"]


def test_snapshot_loader_rejects_conversion_observation_count_over_input_rows(tmp_path, monkeypatch):
    result = tq_batch(tmp_path)
    snap = store.build_minute_snapshot(tmp_path, [result["batch_id"]], now=NOW)
    path = Path(snap["manifest_path"])
    value = json.loads(path.read_text())
    value["inputs"][0]["conversion"]["observed_one_minute_pairs"] = result["rows"]
    path.write_text(json.dumps(value))
    monkeypatch.setattr(pd, "read_pickle", lambda *args: pytest.fail("bad conversion reached unpickle"))
    with pytest.raises(ValueError):
        store.load_minute_snapshot(tmp_path, snap["snapshot_id"])


def test_default_import_is_read_only_inspection_and_never_network(tmp_path, monkeypatch):
    import requests
    monkeypatch.setattr(requests.sessions.Session, "request", lambda *a, **k: pytest.fail("network used"))
    result = store.import_minute_csv(tmp_path, csv_bytes(), provider="Vendor claim",
                                     license_note="Research only", now=NOW)
    assert result["status"] == "inspected"
    assert result["provider_verified"] is False
    assert result["coverage"]["rows"] == 255
    assert result["quality"]["calendar"]["verified"] is False
    assert result["batch_id"] is None
    assert result["data_path"] is None
    assert not list(tmp_path.iterdir())
    json.dumps(result, allow_nan=False)


def test_import_canonical_csv_hashes_and_acquisition_time_are_honest(tmp_path):
    source = tmp_path / "input.csv"
    source.write_bytes(csv_bytes().replace(b"2026-05-11 09:31:00", b"2026-05-11T01:31:00+00:00"))
    # Mixed aware/naive dates have ambiguous semantics and must not be accepted.
    with pytest.raises(ValueError):
        store.import_minute_csv(tmp_path, source, provider="x", license_note="research", now=NOW)
    frame = bars().drop(columns="trade_dt")
    frame["date"] = frame.date.dt.tz_localize("Asia/Shanghai").dt.tz_convert("UTC")
    source.write_bytes(csv_bytes(frame))
    result = store.import_minute_csv(tmp_path, source, provider="x", license_note="research", persist=True, now=NOW)
    assert result["status"] == "imported"
    path = Path(result["data_path"])
    assert path.parent.parent == tmp_path / "data/minute_batches"
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert manifest["source_sha256"] == sha(source)
    assert manifest["data_sha256"] == sha(path)
    assert manifest["data_file"] == "bars.csv"
    assert manifest["provider_verified"] is False
    assert manifest["license_verified"] is False
    assert pd.Timestamp(manifest["acquired_at"]) > pd.Timestamp("2026-05-18", tz="UTC")
    assert manifest["acquired_at"] != manifest["as_of"]
    canonical = pd.read_csv(path)
    assert canonical.iloc[0]["date"] == "2026-05-11 09:31:00"
    assert "trade_dt" in canonical
    assert source.read_bytes() == csv_bytes(frame)


@pytest.mark.parametrize("column,value", [
    ("ticker", "TL0"), ("ticker", "TL2605"), ("ticker", "../TL2606"),
    ("date", "bad"), ("date", "2027-05-11 09:31:00"),
    ("date", "2026-05-11 09:31:01"), ("date", "2026-05-11 12:01:00"),
    ("trade_dt", "2026-05-12"), ("open", 0), ("high", 99), ("low", 101),
    ("close", np.inf), ("volume", -1), ("oi", "bad"),
])
def test_invalid_csv_is_rejected_before_creating_artifacts(tmp_path, column, value):
    frame = bars()
    frame[column] = frame[column].astype(object)
    frame.at[0, column] = value
    with pytest.raises(ValueError):
        store.import_minute_csv(tmp_path, csv_bytes(frame), provider="x", license_note="research", persist=True, now=NOW)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("payload", [
    b"", b"not,csv\n1,2\n", b"\xff\xfe", b"date,date\n1,2\n",
    csv_bytes(bars().drop(columns="oi")), csv_bytes(pd.concat([bars(), bars().iloc[:1]])),
    csv_bytes(bars().assign(available_ts="2020-01-01")),
])
def test_invalid_structure_and_self_reported_availability_are_rejected(tmp_path, payload):
    with pytest.raises(ValueError):
        store.import_minute_csv(tmp_path, payload, provider="x", license_note="research", persist=True, now=NOW)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("provider,license_note", [("", "yes"), ("x", ""), ("x", "token=secret"), ("api_key: bad", "research")])
def test_metadata_requires_nonsecret_declarations(tmp_path, provider, license_note):
    with pytest.raises(ValueError):
        store.import_minute_csv(tmp_path, csv_bytes(), provider=provider, license_note=license_note, now=NOW)
    assert not list(tmp_path.iterdir())


def test_partial_current_day_is_preserved_but_never_complete(tmp_path):
    result = save_batch(tmp_path, bars("2026-05-18", rows=100))
    assert result["status"] == "imported"
    assert result["quality"]["status"] == "incomplete"
    assert result["coverage"]["current_day_rows"] == 100
    assert result["coverage"]["completed_rows"] == 0


def test_input_byte_row_and_source_format_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "MAX_CSV_BYTES", 10)
    with pytest.raises(ValueError, match="size"):
        save_batch(tmp_path)
    monkeypatch.setattr(store, "MAX_CSV_BYTES", 25 * 1024 * 1024)
    monkeypatch.setattr(store, "MAX_BATCH_ROWS", 5)
    with pytest.raises(ValueError, match="row"):
        save_batch(tmp_path)
    source = tmp_path / "unsafe.pkl"
    source.write_bytes(csv_bytes())
    with pytest.raises(ValueError, match="CSV"):
        store.import_minute_csv(tmp_path, source, provider="x", license_note="y", now=NOW)
    assert not (tmp_path / "data").exists()


def test_input_and_output_symlink_ancestors_are_rejected(tmp_path):
    source = tmp_path / "input.csv"
    source.write_bytes(csv_bytes())
    alias = tmp_path / "alias.csv"
    alias.symlink_to(source)
    with pytest.raises(ValueError, match="symlink"):
        store.import_minute_csv(tmp_path, alias, provider="x", license_note="y", now=NOW)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "data").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        save_batch(tmp_path)
    assert not list(outside.iterdir())


def test_lists_are_read_only_bounded_manifest_only_and_ignore_invalid_entries(tmp_path, monkeypatch):
    first = save_batch(tmp_path)
    second = save_batch(tmp_path, bars("2026-05-12"))
    monkeypatch.setattr(pd, "read_pickle", lambda *a, **k: pytest.fail("list deserialized pickle"))
    items = store.list_minute_batches(tmp_path, limit=1)
    assert len(items) == 1
    assert items[0]["batch_id"] in {first["batch_id"], second["batch_id"]}
    assert items[0]["coverage"]["rows"] == 255
    assert store.list_minute_snapshots(tmp_path) == []
    Path(first["manifest_path"]).write_text("{}")
    assert [item["batch_id"] for item in store.list_minute_batches(tmp_path)] == [second["batch_id"]]
    with pytest.raises(ValueError):
        store.list_minute_batches(tmp_path, limit=31)


def test_snapshot_merges_explicit_batches_and_deduplicates_identical_overlap(tmp_path):
    first = save_batch(tmp_path)
    second = save_batch(tmp_path, pd.concat([bars(), bars("2026-05-12")], ignore_index=True))
    protected = {item["data_path"]: sha(item["data_path"]) for item in [first, second]}
    result = store.build_minute_snapshot(tmp_path, [first["batch_id"], second["batch_id"]], now=NOW)
    assert result["status"] == "built"
    frame, path, digest, manifest = store.load_minute_snapshot(tmp_path, result["snapshot_id"])
    assert len(frame) == 510
    assert path == Path(result["data_path"])
    assert digest == sha(path)
    assert manifest["duplicate_identical_rows"] == 255
    assert manifest["include_legacy"] is False
    assert manifest["source_trust"] == "declared_unverified"
    assert len(manifest["inputs"]) == 2
    assert manifest["data_sha256"] == digest
    assert len(store.list_minute_snapshots(tmp_path)) == 1
    assert protected == {p: sha(p) for p in protected}


def test_snapshot_rejects_conflicts_before_any_output_and_never_keep_last(tmp_path):
    first = save_batch(tmp_path)
    frame = bars()
    frame.loc[0, "close"] = 100.5
    second = save_batch(tmp_path, frame)
    with pytest.raises(ValueError, match="conflict"):
        store.build_minute_snapshot(tmp_path, [first["batch_id"], second["batch_id"]], now=NOW)
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("ids", [[], ["../escape"], ["minute-wrong"], "minute-" + "a" * 32])
def test_snapshot_requires_explicit_valid_batch_ids(tmp_path, ids):
    with pytest.raises(ValueError):
        store.build_minute_snapshot(tmp_path, ids, now=NOW)
    assert not list(tmp_path.iterdir())


def test_snapshot_rejects_duplicate_id_and_batch_limit(tmp_path, monkeypatch):
    item = save_batch(tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        store.build_minute_snapshot(tmp_path, [item["batch_id"], item["batch_id"]], now=NOW)
    monkeypatch.setattr(store, "MAX_SELECTED_BATCHES", 1)
    with pytest.raises(ValueError, match="batch"):
        store.build_minute_snapshot(tmp_path, [item["batch_id"], "minute-" + "a" * 32], now=NOW)


def test_tampered_batch_hash_and_manifest_identity_are_rejected(tmp_path):
    item = save_batch(tmp_path)
    original = Path(item["data_path"]).read_bytes()
    Path(item["data_path"]).write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="hash"):
        store.build_minute_snapshot(tmp_path, [item["batch_id"]], now=NOW)
    Path(item["data_path"]).write_bytes(original)
    manifest = json.loads(Path(item["manifest_path"]).read_text())
    manifest["batch_id"] = "minute-" + "a" * 32
    Path(item["manifest_path"]).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"):
        store.build_minute_snapshot(tmp_path, [item["batch_id"]], now=NOW)
    assert not (tmp_path / "artifacts").exists()


def test_legacy_is_only_read_on_explicit_opt_in_and_preserves_trust_boundary(tmp_path, monkeypatch):
    path = tmp_path / "data/TL分钟级量价数据.pkl"
    path.parent.mkdir()
    bars().assign(extra=1).to_pickle(path)
    digest = sha(path)
    item = save_batch(tmp_path, bars("2026-05-12"))
    with monkeypatch.context() as scoped:
        scoped.setattr(pd, "read_pickle", lambda *a, **k: pytest.fail("unexpected legacy read"))
        store.build_minute_snapshot(tmp_path, [item["batch_id"]], now=NOW)
    result = store.build_minute_snapshot(tmp_path, [item["batch_id"]], include_legacy=True, now=NOW)
    frame, _, _, manifest = store.load_minute_snapshot(tmp_path, result["snapshot_id"])
    assert len(frame) == 510
    assert manifest["source_trust"] == "legacy_unverified"
    assert manifest["inputs"][0]["kind"] == "legacy_unverified"
    assert manifest["inputs"][0]["sha256"] == digest
    assert "extra" not in frame
    assert sha(path) == digest


def test_snapshot_loader_verifies_hash_before_pickle_and_forbids_path_escape(tmp_path, monkeypatch):
    item = save_batch(tmp_path)
    result = store.build_minute_snapshot(tmp_path, [item["batch_id"]], now=NOW)
    Path(result["data_path"]).write_bytes(b"pickle exploit not executed")
    monkeypatch.setattr(pd, "read_pickle", lambda *a, **k: pytest.fail("unverified pickle deserialization"))
    with pytest.raises(ValueError, match="hash"):
        store.load_minute_snapshot(tmp_path, result["snapshot_id"])
    with pytest.raises(ValueError):
        store.load_minute_snapshot(tmp_path, "../escape")


def test_output_snapshot_symlink_is_rejected_before_reading_inputs(tmp_path):
    item = save_batch(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "artifacts").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        store.build_minute_snapshot(tmp_path, [item["batch_id"]], now=NOW)
    assert not list(outside.iterdir())


@pytest.mark.parametrize("field,value", [
    ("source_sha256", "bad"), ("source_sha256", None), ("provider", ""),
    ("license_note", "token=secret"), ("bar_contract", {}), ("rows", True),
    ("as_of", None), ("acquired_at", None),
    ("coverage", {"rows": 2}), ("columns", []),
])
def test_batch_manifest_contract_tampering_fails_closed(tmp_path, field, value):
    item = save_batch(tmp_path)
    path = Path(item["manifest_path"])
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        store.build_minute_snapshot(tmp_path, [item["batch_id"]], now=NOW)
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("field,value", [
    ("inputs", []), ("inputs", [{"kind": "arbitrary_path", "file": "evil.pkl"}]),
    ("inputs", None), ("include_legacy", "yes"), ("source_trust", "verified"),
])
def test_snapshot_manifest_lineage_is_checked_before_pickle(tmp_path, monkeypatch, field, value):
    item = save_batch(tmp_path)
    result = store.build_minute_snapshot(tmp_path, [item["batch_id"]], now=NOW)
    path = Path(result["manifest_path"])
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(pd, "read_pickle", lambda *a, **k: pytest.fail("unverified lineage deserialized"))
    with pytest.raises(ValueError, match="manifest"):
        store.load_minute_snapshot(tmp_path, result["snapshot_id"])


@pytest.mark.parametrize("payload", [
    csv_bytes() + b"too,few,fields\n",
    csv_bytes().replace(b"TL2606", b"TL\x002606"),
])
def test_nul_and_ragged_csv_records_cannot_shift_financial_fields(tmp_path, payload):
    with pytest.raises(ValueError):
        store.import_minute_csv(tmp_path, payload, provider="x", license_note="research", now=NOW)


@pytest.mark.parametrize("argument,value", [("persist", "yes"), ("now", "bad"), ("now", pd.NaT)])
def test_import_control_arguments_fail_closed(tmp_path, argument, value):
    args = {"provider": "x", "license_note": "research", "now": NOW, argument: value}
    with pytest.raises(ValueError):
        store.import_minute_csv(tmp_path, csv_bytes(), **args)
    assert not list(tmp_path.iterdir())


def test_bad_paths_formats_missing_files_and_limit_are_rejected(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        store.import_minute_csv(tmp_path / "missing", csv_bytes(), provider="x", license_note="y")
    with pytest.raises(ValueError):
        store.import_minute_csv(tmp_path, object(), provider="x", license_note="y")
    with pytest.raises(ValueError):
        store.import_minute_csv(tmp_path, "missing.csv", provider="x", license_note="y")
    directory = tmp_path / "directory.csv"
    directory.mkdir()
    with pytest.raises(ValueError):
        store.import_minute_csv(tmp_path, directory, provider="x", license_note="y")
    source = tmp_path / "input.csv"
    source.write_bytes(csv_bytes())
    monkeypatch.setattr(store, "MAX_CSV_BYTES", 1)
    with pytest.raises(ValueError, match="size"):
        store.import_minute_csv(tmp_path, source, provider="x", license_note="y")


def test_manifest_encoding_is_bounded_before_any_output(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "MAX_MANIFEST_BYTES", 20)
    with pytest.raises(ValueError, match="manifest size"):
        save_batch(tmp_path)
    assert not list(tmp_path.iterdir())


def test_canonical_expansion_is_bounded_before_any_output(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "_canonical_csv", lambda frame: b"a" * (store.MAX_CSV_BYTES + 1))
    with pytest.raises(ValueError, match="normalized CSV size"):
        save_batch(tmp_path)
    assert not list(tmp_path.iterdir())


def test_snapshot_combined_byte_row_and_output_limits(tmp_path, monkeypatch):
    first = save_batch(tmp_path)
    second = save_batch(tmp_path, bars("2026-05-12"))
    with monkeypatch.context() as scoped:
        scoped.setattr(store, "MAX_SNAPSHOT_BYTES", 100)
        with pytest.raises(ValueError, match="combined input size"):
            store.build_minute_snapshot(tmp_path, [first["batch_id"]], now=NOW)
    with monkeypatch.context() as scoped:
        scoped.setattr(store, "MAX_SNAPSHOT_ROWS", 300)
        with pytest.raises(ValueError, match="combined input row"):
            store.build_minute_snapshot(tmp_path, [first["batch_id"], second["batch_id"]], now=NOW)
    with monkeypatch.context() as scoped:
        scoped.setattr(pd.DataFrame, "to_pickle", lambda frame, out: out.write(b"a" * 30_000))
        scoped.setattr(store, "MAX_SNAPSHOT_BYTES", 25_000)
        with pytest.raises(ValueError, match="snapshot size"):
            store.build_minute_snapshot(tmp_path, [first["batch_id"]], now=NOW)
    assert not (tmp_path / "artifacts").exists()


def test_loader_revalidates_frame_schema_and_counts_after_hash_check(tmp_path):
    item = save_batch(tmp_path)
    result = store.build_minute_snapshot(tmp_path, [item["batch_id"]], now=NOW)
    path, metadata = Path(result["data_path"]), Path(result["manifest_path"])
    manifest = json.loads(metadata.read_text())
    bars(rows=100).to_pickle(path)
    manifest["data_sha256"] = sha(path)
    metadata.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="row count"):
        store.load_minute_snapshot(tmp_path, result["snapshot_id"])
    pd.DataFrame({"bad": [1]}).to_pickle(path)
    manifest["data_sha256"] = sha(path)
    metadata.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="columns"):
        store.load_minute_snapshot(tmp_path, result["snapshot_id"])


def test_bad_internal_and_legacy_pickles_fail_without_publishing(tmp_path):
    path = tmp_path / "data/TL分钟级量价数据.pkl"
    path.parent.mkdir()
    path.write_bytes(b"not a pickle")
    with pytest.raises(ValueError, match="legacy pickle"):
        store.build_minute_snapshot(tmp_path, [], include_legacy=True, now=NOW)
    assert not (tmp_path / "artifacts").exists()
    item = save_batch(tmp_path)
    result = store.build_minute_snapshot(tmp_path, [item["batch_id"]], now=NOW)
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    Path(result["data_path"]).write_bytes(b"not a pickle")
    manifest["data_sha256"] = sha(result["data_path"])
    Path(result["manifest_path"]).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="internal snapshot pickle"):
        store.load_minute_snapshot(tmp_path, result["snapshot_id"])


@pytest.mark.parametrize("field,value", [
    ("sha256", "bad"), ("rows", -1), ("batch_id", "../escape"),
    ("manifest_sha256", "bad"), ("source_sha256", "bad"),
    ("provider", ""), ("acquired_at", None), ("kind", "unknown"),
])
def test_snapshot_input_manifest_details_fail_before_deserializing(tmp_path, monkeypatch, field, value):
    item = save_batch(tmp_path)
    result = store.build_minute_snapshot(tmp_path, [item["batch_id"]], now=NOW)
    path = Path(result["manifest_path"])
    manifest = json.loads(path.read_text())
    manifest["inputs"][0][field] = value
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(pd, "read_pickle", lambda *a, **k: pytest.fail("bad lineage deserialized"))
    with pytest.raises(ValueError, match="manifest"):
        store.load_minute_snapshot(tmp_path, result["snapshot_id"])


@pytest.mark.parametrize("field,value", [("duplicate_identical_rows", 1), ("duplicate_identical_rows", -1)])
def test_snapshot_merge_accounting_is_not_a_claim_without_validation(tmp_path, field, value):
    item = save_batch(tmp_path)
    result = store.build_minute_snapshot(tmp_path, [item["batch_id"]], now=NOW)
    path = Path(result["manifest_path"])
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="merge accounting"):
        store.load_minute_snapshot(tmp_path, result["snapshot_id"])


def test_legacy_source_identity_and_opt_in_cannot_be_laundered(tmp_path):
    source = tmp_path / "data/TL分钟级量价数据.pkl"
    source.parent.mkdir()
    bars().to_pickle(source)
    result = store.build_minute_snapshot(tmp_path, [], include_legacy=True, now=NOW)
    _, _, _, valid = store.load_minute_snapshot(tmp_path, result["snapshot_id"])
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    manifest["inputs"][0]["file"] = "evil.pkl"
    Path(result["manifest_path"]).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="legacy source"):
        store.load_minute_snapshot(tmp_path, result["snapshot_id"])
    valid["include_legacy"] = False
    valid["source_trust"] = "declared_unverified"
    Path(result["manifest_path"]).write_text(json.dumps(valid))
    with pytest.raises(ValueError, match="input limits"):
        store.load_minute_snapshot(tmp_path, result["snapshot_id"])


def test_lists_skip_bad_json_and_missing_data_and_do_not_scan_indefinitely(tmp_path, monkeypatch):
    first, second = save_batch(tmp_path), save_batch(tmp_path, bars("2026-05-12"))
    Path(first["manifest_path"]).write_text("{broken")
    Path(second["data_path"]).unlink()
    assert store.list_minute_batches(tmp_path) == []
    with pytest.raises(ValueError, match="manifest JSON"):
        store.build_minute_snapshot(tmp_path, [first["batch_id"]], now=NOW)
    monkeypatch.setattr(store, "MAX_SCAN_ENTRIES", 0)
    assert store.list_minute_batches(tmp_path) == []


def test_control_characters_and_nonboolean_legacy_opt_in_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="control"):
        store.import_minute_csv(tmp_path, csv_bytes(), provider="vendor\nsecret", license_note="research", now=NOW)
    with pytest.raises(ValueError, match="boolean"):
        store.build_minute_snapshot(tmp_path, [], include_legacy="yes", now=NOW)


def test_batch_row_count_mismatch_rejects_before_snapshot_output(tmp_path):
    item = save_batch(tmp_path)
    path = Path(item["manifest_path"])
    manifest = json.loads(path.read_text())
    manifest["rows"] = manifest["coverage"]["rows"] = 254
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="row count"):
        store.build_minute_snapshot(tmp_path, [item["batch_id"]], now=NOW)
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("column", ["volume", "oi"])
@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_legacy_boolean_numeric_values_are_not_laundered_by_coercion(tmp_path, column, value):
    source = tmp_path / "data/TL分钟级量价数据.pkl"
    source.parent.mkdir()
    frame = bars()
    frame[column] = frame[column].astype(object)
    frame.at[0, column] = value
    frame.to_pickle(source)
    digest = sha(source)
    with pytest.raises(ValueError, match="boolean_numeric"):
        store.build_minute_snapshot(tmp_path, [], include_legacy=True, now=NOW)
    assert sha(source) == digest
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("column", ["volume", "oi"])
@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_internal_snapshot_boolean_numeric_values_fail_revalidation(tmp_path, column, value):
    item = save_batch(tmp_path)
    result = store.build_minute_snapshot(tmp_path, [item["batch_id"]], now=NOW)
    frame = bars()
    frame[column] = frame[column].astype(object)
    frame.at[0, column] = value
    frame.to_pickle(result["data_path"])
    path = Path(result["manifest_path"])
    manifest = json.loads(path.read_text())
    manifest["data_sha256"] = sha(result["data_path"])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="boolean_numeric"):
        store.load_minute_snapshot(tmp_path, result["snapshot_id"])


def test_complex_legacy_numeric_values_are_rejected_without_discarding_imaginary_part(tmp_path):
    source = tmp_path / "data/TL分钟级量价数据.pkl"
    source.parent.mkdir()
    frame = bars()
    frame["volume"] = np.complex128(10 + 2j)
    frame.to_pickle(source)
    with pytest.raises(ValueError, match="nonreal_numeric"):
        store.build_minute_snapshot(tmp_path, [], include_legacy=True, now=NOW)
    assert not (tmp_path / "artifacts").exists()


def test_complex_internal_snapshot_numeric_values_fail_revalidation(tmp_path):
    item = save_batch(tmp_path)
    result = store.build_minute_snapshot(tmp_path, [item["batch_id"]], now=NOW)
    frame = bars()
    frame["oi"] = np.complex128(100 + 2j)
    frame.to_pickle(result["data_path"])
    path = Path(result["manifest_path"])
    manifest = json.loads(path.read_text())
    manifest["data_sha256"] = sha(result["data_path"])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="nonreal_numeric"):
        store.load_minute_snapshot(tmp_path, result["snapshot_id"])
