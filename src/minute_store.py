"""Isolated, explicitly selected minute data batches and research snapshots.

External interchange is bounded CSV only. Provider, license, bar semantics and
timestamps are declarations, not vendor/calendar/vintage certification. A CSV's
historical bar date is never used as its acquisition timestamp. Internal pickle
snapshots are trusted local artifacts; hashes detect corruption, not a malicious
local author who can rewrite both the artifact and its manifest.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import uuid

import pandas as pd

from minute_formats import parse_tqsdk_csv, validate_conversion
from minute_quality import analyze_minute_coverage


MAX_CSV_BYTES = 25 * 1024 * 1024
MAX_BATCH_ROWS = 500_000
MAX_SNAPSHOT_BYTES = 512 * 1024 * 1024
MAX_SNAPSHOT_ROWS = 2_000_000
MAX_SELECTED_BATCHES = 30
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_LIST_ITEMS = 30
MAX_SCAN_ENTRIES = 1000
_COLUMNS = ["date", "trade_dt", "ticker", "open", "high", "low", "close", "volume", "oi"]
_NUMERIC = ["open", "high", "low", "close", "volume", "oi"]
_SESSION_MINUTES = set(range(571, 691)) | set(range(781, 916))
_BATCH_SCHEMA = "f_agent.minute_batch.v1"
_SNAPSHOT_SCHEMA = "f_agent.minute_snapshot.v1"
_BAR_CONTRACT = {
    "timezone": "Asia/Shanghai", "bar_label": "end", "interval_seconds": 60,
    "volume_semantics": "bar", "oi_semantics": "end", "semantics_verified": False,
}
_LIMITATIONS = [
    "Provider and license are caller declarations, not independent verification.",
    "Canonical end-labelled 60-second bars, per-bar volume and end-bar OI are caller-declared semantics.",
    "Volume/OI units and single-versus-double-sided counting are not verified across providers; verify them before mixing sources.",
    "Acquisition time is recorded now; historical availability and revisions are not certified.",
    "Session completion uses an observed 255-minute template, not an exchange calendar.",
    "Partial sessions and current-day rows may be stored, but are not completed research sessions.",
    "No network fetch, production replacement, model training, publication or order is performed.",
]


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _limits():
    return {"csv_bytes": MAX_CSV_BYTES, "batch_rows": MAX_BATCH_ROWS,
            "snapshot_bytes": MAX_SNAPSHOT_BYTES, "snapshot_rows": MAX_SNAPSHOT_ROWS,
            "selected_batches": MAX_SELECTED_BATCHES, "manifest_bytes": MAX_MANIFEST_BYTES}


def _no_symlinks(path):
    """Check the lexical path before resolve() could conceal a symlink."""
    path = Path(path).expanduser().absolute()
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError("symlink paths are not permitted")
    return path


def _base(base_dir):
    path = _no_symlinks(base_dir)
    if not path.is_dir():
        raise ValueError("base directory must exist")
    return path


def _inside(base, *parts):
    path = base.joinpath(*parts)
    if not path.is_relative_to(base):
        raise ValueError("path is outside project")
    return _no_symlinks(path)


def _read_bounded(path, limit):
    path = _no_symlinks(path)
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    except OSError as exc:
        raise ValueError("required local file is unavailable") from exc
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        os.close(descriptor)
        reason = "input must be a regular file" if not stat.S_ISREG(info.st_mode) else "file size limit exceeded"
        raise ValueError(reason)
    with os.fdopen(descriptor, "rb") as handle:
        payload = handle.read(limit + 1)
    if len(payload) > limit:
        raise ValueError("file size limit exceeded")
    _no_symlinks(path)
    return payload


def _asof(now):
    try:
        value = pd.Timestamp.now(tz="Asia/Shanghai") if now is None else pd.Timestamp(now)
        if pd.isna(value):
            raise ValueError("invalid as-of timestamp")
        if value.tzinfo is None:
            value = value.tz_localize("Asia/Shanghai")
        return value.tz_convert("Asia/Shanghai")
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid as-of timestamp") from exc


def _declaration(value, name, limit):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} requires a bounded nonempty declaration")
    if any(ord(char) < 32 for char in value):
        raise ValueError(f"{name} may not contain control characters")
    if re.search(r"(?:api[_-]?key|token|password|secret|authorization)\s*[:=]|\bbearer\s+|\bsk-[A-Za-z0-9]{12,}", value, re.I):
        raise ValueError(f"{name} may not contain credentials")
    return value.strip()


def _validate_acquisition(value, *, acquired_at):
    keys = {"kind", "symbol", "frequency", "response_sha256", "fetched_at", "bar_label", "semantics_confirmed"}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("invalid acquisition metadata fields")
    if (value["kind"] != "sina_recent" or value["frequency"] != "1min"
            or not isinstance(value["symbol"], str) or not re.fullmatch(r"TL\d{2}(03|06|09|12)", value["symbol"])
            or not _valid_sha(value["response_sha256"]) or value["bar_label"] not in ("start", "end")
            or value["semantics_confirmed"] is not True):
        raise ValueError("invalid acquisition source declaration")
    fetched = value["fetched_at"]
    if (not isinstance(fetched, str) or len(fetched) > 48
            or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})", fetched)):
        raise ValueError("acquisition fetched_at requires an aware ISO timestamp")
    try:
        stamp = pd.Timestamp(fetched)
        acquired = pd.Timestamp(acquired_at)
        if (pd.isna(stamp) or stamp.tzinfo is None or pd.isna(acquired)
                or acquired.tzinfo is None or stamp > acquired):
            raise ValueError("acquisition fetched_at is invalid or after local acquisition")
    except (ValueError, OverflowError, TypeError) as exc:
        raise ValueError("invalid acquisition fetched_at") from exc
    return dict(value)


def _validate_source_metadata(value):
    # Older v1 canonical manifests have no adapter fields and remain readable.
    source_format = value.get("source_format", "canonical")
    if source_format not in ("canonical", "tqsdk"):
        raise ValueError("invalid source_format metadata")
    if source_format == "tqsdk":
        validate_conversion(value.get("conversion"))
        if type(value.get("rows")) is not int or value["conversion"]["observed_one_minute_pairs"] >= value["rows"]:
            raise ValueError("TqSdk conversion observation count exceeds source rows")
        if value.get("source_file") != "source.csv" or "acquisition" in value:
            raise ValueError("invalid TqSdk source retention metadata")
    elif "conversion" in value or "source_file" in value:
        raise ValueError("canonical sources may not carry TqSdk conversion fields")
    if "acquisition" in value:
        _validate_acquisition(value["acquisition"], acquired_at=value.get("acquired_at"))


def _verify_source_matches_frame(manifest, frame):
    if "acquisition" in manifest:
        if not frame["ticker"].eq(manifest["acquisition"]["symbol"]).all():
            raise ValueError("acquisition symbol disagrees with imported bars")
        captured = pd.Timestamp(manifest["acquisition"]["fetched_at"]).tz_convert("Asia/Shanghai").floor("min").tz_localize(None)
        if (frame["date"] > captured).any():
            raise ValueError("minute end timestamps postdate response capture time")
    if manifest.get("source_format") == "tqsdk":
        symbol = manifest["conversion"]["symbol"].split(".")[1]
        if not frame["ticker"].eq(symbol).all():
            raise ValueError("TqSdk conversion symbol disagrees with imported bars")
        if manifest["conversion"]["observed_one_minute_pairs"] >= len(frame):
            raise ValueError("TqSdk conversion observation count exceeds bars")


def _wall_time(series):
    try:
        values = pd.to_datetime(series, errors="raise", format="ISO8601")
        if isinstance(values.dtype, pd.DatetimeTZDtype):
            values = values.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
        if not pd.api.types.is_datetime64_any_dtype(values):
            raise ValueError("mixed date timezones")
        return values
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid or mixed timestamp formats") from exc


def _normalize_frame(frame, *, now, row_limit, allow_extra=False):
    if not isinstance(frame, pd.DataFrame) or frame.empty or len(frame) > row_limit:
        raise ValueError("empty input or row limit exceeded")
    required = set(_COLUMNS) - {"trade_dt"}
    if not frame.columns.is_unique or not required.issubset(frame.columns):
        raise ValueError("invalid minute data columns")
    if not allow_extra and not set(frame.columns).issubset(_COLUMNS):
        raise ValueError("only canonical CSV columns are accepted; availability is not self-reported")
    result = frame[[column for column in _COLUMNS if column in frame]].copy()
    result["date"] = _wall_time(result["date"])
    result["trade_dt"] = _wall_time(result["trade_dt"]) if "trade_dt" in result else result["date"].dt.normalize()
    # Preserve raw scalar types for validation: coercing an object column first
    # would turn bool/np.bool_ into 1.0 and erase evidence of an invalid input.
    report = analyze_minute_coverage(result, now=now)
    if not report["validation"]["valid"]:
        codes = ", ".join(sorted(report["validation"]["issues"]))
        raise ValueError(f"minute validation rejected: {codes}")
    for column in _NUMERIC:
        result[column] = pd.to_numeric(result[column], errors="raise")
    minute = result["date"].dt.hour * 60 + result["date"].dt.minute
    if not minute.isin(_SESSION_MINUTES).all():
        raise ValueError("off-session timestamp for declared end-labelled minute bars")
    # Normalize the optional trade date to an unambiguous calendar date only
    # after the analyzer has rejected a mismatched local trade date.
    result["trade_dt"] = result["date"].dt.normalize()
    result = result[_COLUMNS].sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)
    return result, report


def _parse_csv(payload, *, now):
    if not isinstance(payload, bytes) or not payload or len(payload) > MAX_CSV_BYTES:
        raise ValueError("empty CSV or CSV size limit exceeded")
    try:
        text = payload.decode("utf-8-sig")
        if "\x00" in text:
            raise ValueError("NUL byte in CSV")
        reader = csv.reader(io.StringIO(text), strict=True)
        header = next(reader)
        if len(header) != len(set(header)) or not header:
            raise ValueError("duplicate or empty CSV header")
        if not set(header).issubset(_COLUMNS) or not (set(_COLUMNS) - {"trade_dt"}).issubset(header):
            raise ValueError("invalid canonical CSV columns")
        # pandas may accept an extra field as an implicit index. Reject all
        # ragged records before parsing to avoid shifting the financial fields.
        row_count = 0
        for record in reader:
            if not record:
                continue
            row_count += 1
            if len(record) != len(header):
                raise ValueError("ragged CSV record")
            if row_count > MAX_BATCH_ROWS:
                raise ValueError("CSV row limit exceeded")
        frame = pd.read_csv(io.StringIO(text), dtype={"ticker": "str"}, float_precision="round_trip")
    except (UnicodeError, csv.Error, StopIteration, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError("invalid UTF-8 canonical CSV") from exc
    return _normalize_frame(frame, now=now, row_limit=MAX_BATCH_ROWS)


def _canonical_csv(frame):
    return frame.to_csv(index=False, date_format="%Y-%m-%d %H:%M:%S", float_format="%.17g").encode("utf-8")


def _encode_manifest(manifest):
    payload = json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")
    if len(payload) > MAX_MANIFEST_BYTES:
        raise ValueError("manifest size limit exceeded")
    return payload


def _write_bundle(base, parent_parts, identity, data_name, payload, manifest, *, source_payload=None):
    # All content and manifest validation happens before any directories exist.
    encoded_manifest = _encode_manifest(manifest)
    parent = _inside(base, *parent_parts)
    current = base
    for part in parent.relative_to(base).parts:
        current = _inside(current, part)
        current.mkdir(mode=0o700, exist_ok=True)
        _no_symlinks(current)
    directory = _inside(parent, identity)
    directory.mkdir(mode=0o700, exist_ok=False)
    files = [(data_name, payload)]
    if source_payload is not None:
        files.append(("source.csv", source_payload))
    files.append(("manifest.json", encoded_manifest))
    for name, content in files:
        destination = _inside(directory, name)
        with destination.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    return directory / data_name, directory / "manifest.json"


def _paths(base, identity, kind):
    prefix, parents = ("minute", ("data", "minute_batches")) if kind == "batch" else ("minset", ("artifacts", "minute_snapshots"))
    if not isinstance(identity, str) or not re.fullmatch(prefix + r"-[0-9a-f]{32}", identity):
        raise ValueError(f"invalid {kind} identity")
    return _inside(base, *parents, identity)


def _manifest(base, identity, kind):
    directory = _paths(base, identity, kind)
    payload = _read_bounded(directory / "manifest.json", MAX_MANIFEST_BYTES)
    try:
        manifest = json.loads(payload)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("invalid manifest JSON") from exc
    is_batch = kind == "batch"
    if not isinstance(manifest, dict) or (
        manifest.get("schema_version") != (_BATCH_SCHEMA if is_batch else _SNAPSHOT_SCHEMA)
        or manifest.get("batch_id" if is_batch else "snapshot_id") != identity
        or manifest.get("status") != ("imported" if is_batch else "built")
        or manifest.get("data_file") != ("bars.csv" if is_batch else "bars.pkl")
        or not _valid_sha(manifest.get("data_sha256"))
        or type(manifest.get("rows")) is not int
        or not 0 < manifest["rows"] <= (MAX_BATCH_ROWS if is_batch else MAX_SNAPSHOT_ROWS)
        or manifest.get("columns") != _COLUMNS
        or manifest.get("bar_contract") != _BAR_CONTRACT
        or manifest.get("provider_verified") is not False
        or manifest.get("license_verified") is not False
    ):
        raise ValueError("invalid or inconsistent manifest")
    if not isinstance(manifest.get("coverage"), dict) or manifest["coverage"].get("rows") != manifest["rows"]:
        raise ValueError("invalid manifest coverage")
    for name in ("as_of", "acquired_at"):
        if not isinstance(manifest.get(name), str):
            raise ValueError("missing manifest timestamp")
        _asof(manifest[name])
    if is_batch:
        if not _valid_sha(manifest.get("source_sha256")):
            raise ValueError("invalid manifest source hash")
        _declaration(manifest.get("provider"), "manifest provider", 200)
        _declaration(manifest.get("license_note"), "manifest license_note", 2000)
        _validate_source_metadata(manifest)
    else:
        _validate_lineage(manifest)
    return manifest, directory, _sha(payload)


def _validate_lineage(manifest):
    inputs = manifest.get("inputs")
    include_legacy = manifest.get("include_legacy")
    if not isinstance(include_legacy, bool) or not isinstance(inputs, list) or not 1 <= len(inputs) <= MAX_SELECTED_BATCHES + int(include_legacy):
        raise ValueError("invalid snapshot manifest inputs")
    expected_trust = "legacy_unverified" if include_legacy else "declared_unverified"
    if manifest.get("source_trust") != expected_trust:
        raise ValueError("invalid snapshot manifest source trust")
    seen, legacy_count, total_rows = set(), 0, 0
    for item in inputs:
        if not isinstance(item, dict) or not _valid_sha(item.get("sha256")) or type(item.get("rows")) is not int or item["rows"] <= 0:
            raise ValueError("invalid snapshot manifest input identity")
        total_rows += item["rows"]
        if item.get("kind") == "legacy_unverified":
            legacy_count += 1
            if item.get("file") != "data/TL分钟级量价数据.pkl" or item.get("historical_availability_verified") is not False:
                raise ValueError("invalid snapshot manifest legacy source")
        elif item.get("kind") == "declared_csv":
            batch_id = item.get("batch_id")
            if not isinstance(batch_id, str) or not re.fullmatch(r"minute-[0-9a-f]{32}", batch_id) or batch_id in seen:
                raise ValueError("invalid snapshot manifest batch identity")
            seen.add(batch_id)
            if not _valid_sha(item.get("manifest_sha256")) or not _valid_sha(item.get("source_sha256")):
                raise ValueError("invalid snapshot manifest batch hashes")
            _declaration(item.get("provider"), "manifest provider", 200)
            if not isinstance(item.get("acquired_at"), str):
                raise ValueError("invalid snapshot manifest acquisition time")
            _asof(item["acquired_at"])
            _validate_source_metadata(item)
        else:
            raise ValueError("invalid snapshot manifest source kind")
    duplicate_rows = manifest.get("duplicate_identical_rows")
    if legacy_count != int(include_legacy) or len(seen) > MAX_SELECTED_BATCHES or total_rows > MAX_SNAPSHOT_ROWS:
        raise ValueError("invalid snapshot manifest input limits")
    if type(duplicate_rows) is not int or duplicate_rows < 0 or total_rows - duplicate_rows != manifest["rows"]:
        raise ValueError("invalid snapshot manifest merge accounting")


def _result(manifest, *, data_path=None, manifest_path=None):
    return {**manifest, "data_path": str(data_path) if data_path else None,
            "manifest_path": str(manifest_path) if manifest_path else None}


def import_minute_csv(base_dir, csv_path, *, provider, license_note, persist=False, now=None,
                      input_format="canonical", bar_seconds=None, acquisition=None):
    """Inspect an explicitly declared CSV format, or persist an isolated batch.

    The default is read-only. CSV files must have a .csv suffix; byte uploads are
    parsed through the exact same validation path without temporary files.
    ``now`` controls validation only, never the genuine acquisition timestamp.
    TqSdk input requires ``bar_seconds=60`` and retains original source bytes.
    Optional canonical acquisition metadata records upstream provenance, not
    independently verified vendor identity, response authenticity or semantics.
    """
    base = _base(base_dir)
    provider = _declaration(provider, "provider", 200)
    license_note = _declaration(license_note, "license_note", 2000)
    as_of = _asof(now)
    if not isinstance(persist, bool):
        raise ValueError("persist must be an explicit boolean")
    if input_format not in ("canonical", "tqsdk"):
        raise ValueError("input_format must be canonical or tqsdk")
    if input_format == "canonical" and bar_seconds is not None:
        raise ValueError("canonical input does not accept a bar_seconds override")
    if input_format == "tqsdk" and acquisition is not None:
        raise ValueError("acquisition metadata only applies to canonical recent responses")
    if isinstance(csv_path, bytes):
        payload, source_name = csv_path, "uploaded.csv"
    else:
        if not isinstance(csv_path, (str, Path)):
            raise ValueError("input must be CSV path or bytes")
        path = Path(csv_path).expanduser()
        if not path.is_absolute():
            path = base / path
        if path.suffix.lower() != ".csv":
            raise ValueError("external inputs must be CSV, never pickle")
        payload, source_name = _read_bounded(path, MAX_CSV_BYTES), path.name
    acquired_at = pd.Timestamp.now(tz="UTC").isoformat()
    conversion = None
    if input_format == "tqsdk":
        converted, conversion = parse_tqsdk_csv(payload, bar_seconds=bar_seconds,
                                               max_bytes=MAX_CSV_BYTES, max_rows=MAX_BATCH_ROWS)
        frame, quality = _normalize_frame(converted, now=as_of, row_limit=MAX_BATCH_ROWS)
    else:
        frame, quality = _parse_csv(payload, now=as_of)
    normalized = _canonical_csv(frame)
    if len(normalized) > MAX_CSV_BYTES:
        raise ValueError("normalized CSV size limit exceeded")
    batch_id = "minute-" + uuid.uuid4().hex if persist else None
    manifest = {
        "schema_version": _BATCH_SCHEMA, "status": "imported" if persist else "inspected",
        "batch_id": batch_id, "data_file": "bars.csv", "source_name": source_name,
        "source_format": input_format,
        "source_sha256": _sha(payload), "data_sha256": _sha(normalized),
        "provider": provider, "provider_verified": False, "license_note": license_note,
        "license_verified": False, "acquired_at": acquired_at, "as_of": as_of.isoformat(),
        "acquired_at_semantics": "local_csv_read_time",
        "bar_contract": dict(_BAR_CONTRACT), "rows": len(frame), "columns": _COLUMNS,
        "quality": quality, "coverage": quality["coverage"], "limitations": list(_LIMITATIONS),
        "limits": _limits(),
        "production_modified": False, "network_used": False,
    }
    if conversion is not None:
        manifest.update(conversion=conversion, source_file="source.csv")
    if acquisition is not None:
        manifest["acquisition"] = _validate_acquisition(acquisition, acquired_at=acquired_at)
    _verify_source_matches_frame(manifest, frame)
    if not persist:
        return _result(manifest)
    paths = _write_bundle(base, ("data", "minute_batches"), batch_id, "bars.csv", normalized, manifest,
                          source_payload=payload if input_format == "tqsdk" else None)
    return _result(manifest, data_path=paths[0], manifest_path=paths[1])


def _list_artifacts(base_dir, kind, limit):
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_LIST_ITEMS:
        raise ValueError(f"list limit must be 1..{MAX_LIST_ITEMS}")
    base = _base(base_dir)
    parents = ("data", "minute_batches") if kind == "batch" else ("artifacts", "minute_snapshots")
    parent = _inside(base, *parents)
    if not parent.exists():
        return []
    entries = []
    with os.scandir(parent) as scanner:
        for index, entry in enumerate(scanner):
            if index >= MAX_SCAN_ENTRIES:
                break
            if not entry.is_symlink() and entry.is_dir(follow_symlinks=False):
                entries.append(entry)
    # Most recent within a bounded directory scan, not an exhaustive registry.
    entries.sort(key=lambda entry: (entry.stat(follow_symlinks=False).st_mtime_ns, entry.name), reverse=True)
    result = []
    for entry in entries[:MAX_LIST_ITEMS]:
        try:
            manifest, directory, _ = _manifest(base, entry.name, kind)
            data_path = _inside(directory, manifest["data_file"])
            if not data_path.is_file():
                continue
            result.append(_result(manifest, data_path=data_path, manifest_path=directory / "manifest.json"))
        except (ValueError, OSError):
            continue
        if len(result) == limit:
            break
    return result


def list_minute_batches(base_dir, limit=30):
    """Read at most 30 manifests from at most 1000 entries; never read bar data."""
    return _list_artifacts(base_dir, "batch", limit)


def list_minute_snapshots(base_dir, limit=30):
    """Bounded manifest-only list; membership does not certify artifact data."""
    return _list_artifacts(base_dir, "snapshot", limit)


def build_minute_snapshot(base_dir, batch_ids, *, include_legacy=False, now=None):
    """Freeze explicitly selected batches; conflicting overlapping bars reject.

    ``include_legacy=True`` is the only path that reads the known local legacy
    pickle. Unknown external pickle paths cannot enter this API.
    """
    base, as_of = _base(base_dir), _asof(now)
    _inside(base, "artifacts", "minute_snapshots")
    if not isinstance(include_legacy, bool):
        raise ValueError("include_legacy must be an explicit boolean")
    if not isinstance(batch_ids, (list, tuple)) or (not batch_ids and not include_legacy):
        raise ValueError("explicit batch selection is required")
    if len(batch_ids) > MAX_SELECTED_BATCHES:
        raise ValueError("batch selection limit exceeded")
    for batch_id in batch_ids:
        _paths(base, batch_id, "batch")
    if len(set(batch_ids)) != len(batch_ids):
        raise ValueError("duplicate batch identities")
    frames, inputs, total_bytes = [], [], 0
    if include_legacy:
        legacy_path = _inside(base, "data", "TL分钟级量价数据.pkl")
        payload = _read_bounded(legacy_path, MAX_SNAPSHOT_BYTES)
        total_bytes += len(payload)
        try:
            legacy = pd.read_pickle(io.BytesIO(payload))
        except Exception as exc:
            raise ValueError("invalid trusted local legacy pickle") from exc
        frame, _ = _normalize_frame(legacy, now=as_of, row_limit=MAX_SNAPSHOT_ROWS, allow_extra=True)
        frames.append(frame)
        inputs.append({"kind": "legacy_unverified", "file": "data/TL分钟级量价数据.pkl",
                       "sha256": _sha(payload), "rows": len(frame), "historical_availability_verified": False})
    for batch_id in batch_ids:
        manifest, directory, manifest_sha = _manifest(base, batch_id, "batch")
        payload = _read_bounded(directory / "bars.csv", MAX_CSV_BYTES)
        total_bytes += len(payload)
        if total_bytes > MAX_SNAPSHOT_BYTES:
            raise ValueError("combined input size limit exceeded")
        if _sha(payload) != manifest["data_sha256"]:
            raise ValueError("batch data hash mismatch")
        frame, _ = _parse_csv(payload, now=as_of)
        if len(frame) != manifest["rows"]:
            raise ValueError("batch manifest row count mismatch")
        _verify_source_matches_frame(manifest, frame)
        if manifest.get("source_format") == "tqsdk":
            original = _read_bounded(directory / "source.csv", MAX_CSV_BYTES)
            if _sha(original) != manifest["source_sha256"]:
                raise ValueError("TqSdk source data hash mismatch")
            total_bytes += len(original)
            if total_bytes > MAX_SNAPSHOT_BYTES:
                raise ValueError("combined input size limit exceeded")
            converted, conversion = parse_tqsdk_csv(original, bar_seconds=60,
                                                   max_bytes=MAX_CSV_BYTES, max_rows=MAX_BATCH_ROWS)
            converted, _ = _normalize_frame(converted, now=as_of, row_limit=MAX_BATCH_ROWS)
            if conversion != manifest["conversion"] or _canonical_csv(converted) != payload:
                raise ValueError("TqSdk conversion replay disagrees with canonical bars or metadata")
        frames.append(frame)
        inputs.append({"kind": "declared_csv", "batch_id": batch_id, "sha256": _sha(payload),
                       "manifest_sha256": manifest_sha, "source_sha256": manifest["source_sha256"],
                       "rows": len(frame), "provider": manifest["provider"], "acquired_at": manifest["acquired_at"]})
        for name in ("source_format", "conversion", "source_file", "acquisition"):
            if name in manifest:
                inputs[-1][name] = manifest[name]
        if sum(len(item) for item in frames) > MAX_SNAPSHOT_ROWS:
            raise ValueError("combined input row limit exceeded")
    combined = pd.concat(frames, ignore_index=True)
    unique = combined.drop_duplicates(subset=_COLUMNS)
    if unique.duplicated(["date", "ticker"], keep=False).any():
        raise ValueError("conflicting overlapping minute records; no keep-last policy")
    frame, quality = _normalize_frame(unique, now=as_of, row_limit=MAX_SNAPSHOT_ROWS)
    snapshot_id = "minset-" + uuid.uuid4().hex
    buffer = io.BytesIO()
    frame.to_pickle(buffer)
    payload = buffer.getvalue()
    if len(payload) > MAX_SNAPSHOT_BYTES:
        raise ValueError("snapshot size limit exceeded")
    manifest = {
        "schema_version": _SNAPSHOT_SCHEMA, "status": "built", "snapshot_id": snapshot_id,
        "data_file": "bars.pkl", "data_sha256": _sha(payload), "rows": len(frame), "columns": _COLUMNS,
        "inputs": inputs, "include_legacy": include_legacy, "duplicate_identical_rows": len(combined) - len(frame),
        "source_trust": "legacy_unverified" if include_legacy else "declared_unverified",
        "provider_verified": False, "license_verified": False, "bar_contract": dict(_BAR_CONTRACT),
        "acquired_at": pd.Timestamp.now(tz="UTC").isoformat(), "as_of": as_of.isoformat(),
        "acquired_at_semantics": "snapshot_materialized_time_not_historical_bar_availability",
        "quality": quality, "coverage": quality["coverage"], "limitations": list(_LIMITATIONS),
        "limits": _limits(),
        "production_modified": False, "network_used": False,
    }
    paths = _write_bundle(base, ("artifacts", "minute_snapshots"), snapshot_id, "bars.pkl", payload, manifest)
    return _result(manifest, data_path=paths[0], manifest_path=paths[1])


def load_minute_snapshot(base_dir, snapshot_id):
    """Validate identity/hash/schema before reading the internal trusted pickle."""
    base = _base(base_dir)
    manifest, directory, _ = _manifest(base, snapshot_id, "snapshot")
    path = _inside(directory, "bars.pkl")
    payload = _read_bounded(path, MAX_SNAPSHOT_BYTES)
    digest = _sha(payload)
    if digest != manifest["data_sha256"]:
        raise ValueError("snapshot data hash mismatch")
    try:
        raw = pd.read_pickle(io.BytesIO(payload))
    except Exception as exc:
        raise ValueError("invalid internal snapshot pickle") from exc
    frame, _ = _normalize_frame(raw, now=_asof(manifest["as_of"]), row_limit=MAX_SNAPSHOT_ROWS)
    if len(frame) != manifest["rows"]:
        raise ValueError("snapshot manifest row count mismatch")
    return frame, path, digest, manifest
