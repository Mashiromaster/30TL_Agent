"""Actual local components on synthetic bars; no online service or training."""
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from factor_candidates import build_local_factor_candidate
from minute_store import build_minute_snapshot, import_minute_csv, load_minute_snapshot
from prepare_market_data import audit_local_minutes
from research_readiness import inspect_research_readiness
from test_factor_candidates import bars


def test_csv_batches_snapshot_factors_and_readiness_are_one_isolated_lineage(tmp_path):
    frame = bars(8)[["date", "trade_dt", "ticker", "open", "high", "low", "close", "volume", "oi"]]
    clock = "2026-06-01"
    kwargs = {"provider": "synthetic-test-only", "license_note": "generated test fixture", "now": clock}
    preview = import_minute_csv(tmp_path, frame.to_csv(index=False).encode(), **kwargs)
    assert preview["status"] == "inspected"
    assert not list(tmp_path.iterdir())
    first = import_minute_csv(tmp_path, frame.iloc[:3000].to_csv(index=False).encode(), persist=True, **kwargs)
    second = import_minute_csv(tmp_path, frame.iloc[2000:].to_csv(index=False).encode(), persist=True, **kwargs)
    snapshot = build_minute_snapshot(tmp_path, [first["batch_id"], second["batch_id"]], now=clock)
    assert snapshot["duplicate_identical_rows"] == 1000
    loaded, path, digest, metadata = load_minute_snapshot(tmp_path, snapshot["snapshot_id"])
    assert len(loaded) == len(frame)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert metadata["provider_verified"] is False
    audit = audit_local_minutes(tmp_path, snapshot_id=snapshot["snapshot_id"], now=clock)
    assert audit["status"] == "complete_observed"
    candidate = build_local_factor_candidate(tmp_path, minute_snapshot_id=snapshot["snapshot_id"], now=clock)
    assert candidate["status"] == "built", candidate
    readiness = inspect_research_readiness(tmp_path, factor_path=candidate["factor_path"], now=clock)
    assert readiness["factors"]["pit_verified"] is True  # Local algorithm/hash contract only.
    assert readiness["factors"]["provenance"]["provenance"]["minute_snapshot_id"] == snapshot["snapshot_id"]
    assert readiness["status"] == "blocked"  # Too little data + no release, never fake ready.
    for filename in ["data/TL分钟级量价数据.pkl", "outputs/df_factors.pkl", "models/trained_model.pkl", "artifacts/models/current.json"]:
        assert not (tmp_path / filename).exists()


def test_real_cli_audit_handles_missing_legacy_without_creating_files(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        audit_local_minutes(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_legacy_audit_refuses_symlink_without_reading_target(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    base = tmp_path / "project"
    base.mkdir()
    (base / "data").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="non-redirected"):
        audit_local_minutes(base)


def test_legacy_audit_records_exact_bytes_and_is_read_only(tmp_path):
    frame = bars(2)
    path = tmp_path / "data/TL分钟级量价数据.pkl"
    path.parent.mkdir()
    frame.to_pickle(path)
    content = path.read_bytes()
    report = audit_local_minutes(tmp_path, now="2026-06-01")
    assert report["input"]["sha256"] == hashlib.sha256(content).hexdigest()
    assert report["input"]["kind"] == "legacy_unverified"
    assert path.read_bytes() == content


def test_tq_export_source_survives_snapshot_and_real_factor_pipeline(tmp_path):
    frame = bars(8).query("ticker == 'TL2606'").copy()
    starts = frame.date - pd.Timedelta(minutes=1)
    export = pd.DataFrame({
        "datetime": starts.dt.strftime("%Y-%m-%d %H:%M:%S") + ".000000000",
        "datetime_nano": [str(stamp.tz_localize("Asia/Shanghai").value) for stamp in starts],
    })
    for field in ("open", "high", "low", "close", "volume", "open_oi", "close_oi"):
        export["CFFEX.TL2606." + field] = frame["oi" if field.endswith("_oi") else field].to_numpy()
    payload = export.to_csv(index=False).encode()
    batch = import_minute_csv(tmp_path, payload, provider="synthetic-tq-export", license_note="test fixture",
                              input_format="tqsdk", bar_seconds=60, persist=True, now="2026-06-01")
    assert batch["source_sha256"] == hashlib.sha256(payload).hexdigest()
    assert (Path(batch["manifest_path"]).parent / "source.csv").read_bytes() == payload
    snapshot = build_minute_snapshot(tmp_path, [batch["batch_id"]], now="2026-06-01")
    assert snapshot["inputs"][0]["source_format"] == "tqsdk"
    candidate = build_local_factor_candidate(tmp_path, minute_snapshot_id=snapshot["snapshot_id"], now="2026-06-01")
    assert candidate["status"] == "built", candidate
    import json
    manifest = json.loads(Path(candidate["manifest_path"]).read_text())
    origin = manifest["sources"]["market"]["inputs"][0]
    assert origin["source_sha256"] == hashlib.sha256(payload).hexdigest()
    assert origin["source_format"] == "tqsdk"
    assert not (tmp_path / "outputs/df_factors.pkl").exists()


def test_recent_capture_admission_and_snapshot_keep_http_provenance(tmp_path):
    from recent_minutes import fetch_recent_minutes, commit_recent_capture
    from test_recent_minutes import provider_frame
    capture = fetch_recent_minutes(tmp_path, "TL2612", allow_network=True, bar_label="end", fetcher=lambda *a, **k: provider_frame())
    batch = commit_recent_capture(tmp_path, capture, license_note="synthetic test fixture", confirmed=True)
    snapshot = build_minute_snapshot(tmp_path, [batch["batch_id"]])
    origin = snapshot["inputs"][0]["acquisition"]
    assert origin["response_sha256"] == capture.source["response_sha256"]
    assert origin["fetched_at"] == capture.source["fetched_at"]
    assert snapshot["provider_verified"] is False
    assert not (tmp_path / "data/TL分钟级量价数据.pkl").exists()
