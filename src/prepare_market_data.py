"""Explicit minute-data operations; never train or publish.

CSV import is inspection-only unless --commit is present. Snapshot assembly is
an explicit immutable write, not a replacement of the old raw-data pickle.
Only fetch-recent with --allow-network accesses the single public endpoint.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys

import pandas as pd

from config import BASE_DIR
from intelligence_events import redact_sensitive_text


def import_minute_csv(*args, **kwargs):
    from minute_store import import_minute_csv as worker
    return worker(*args, **kwargs)


def list_minute_batches(*args, **kwargs):
    from minute_store import list_minute_batches as worker
    return worker(*args, **kwargs)


def list_minute_snapshots(*args, **kwargs):
    from minute_store import list_minute_snapshots as worker
    return worker(*args, **kwargs)


def build_minute_snapshot(*args, **kwargs):
    from minute_store import build_minute_snapshot as worker
    return worker(*args, **kwargs)


def fetch_recent_minutes(*args, **kwargs):
    from recent_minutes import fetch_recent_minutes as worker
    return worker(*args, **kwargs)


def commit_recent_capture(*args, **kwargs):
    from recent_minutes import commit_recent_capture as worker
    return worker(*args, **kwargs)


def audit_local_minutes(base_dir, *, snapshot_id=None, now=None):
    """Audit one trusted local source without creating any files."""
    from minute_quality import analyze_minute_coverage
    base = Path(base_dir).resolve()
    if snapshot_id:
        from minute_store import load_minute_snapshot
        frame, path, digest, manifest = load_minute_snapshot(base, snapshot_id)
        source = {"kind": "isolated_snapshot", "snapshot_id": snapshot_id,
                  "provider_verified": False, "manifest_path": str(path.parent / "manifest.json")}
    else:
        path = base / "data/TL分钟级量价数据.pkl"
        if (base / "data").is_symlink() or path.is_symlink() or not path.resolve().is_relative_to(base):
            raise ValueError("legacy minute input must be a local non-redirected file")
        if not path.is_file() or path.stat().st_size > 512 * 1024 * 1024:
            raise ValueError("legacy minute input missing or exceeds 512 MiB")
        content = path.read_bytes()
        if len(content) > 512 * 1024 * 1024:
            raise ValueError("legacy minute input exceeds 512 MiB")
        digest = hashlib.sha256(content).hexdigest()
        frame = pd.read_pickle(io.BytesIO(content))
        source = {"kind": "legacy_unverified", "provider_verified": False}
    result = analyze_minute_coverage(frame, now=now)
    result["input"] = {**source, "path": str(path), "sha256": digest}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="TL minute data: isolated import; optional one-shot public fetch; no production writes")
    parser.add_argument("--base-dir", default=BASE_DIR)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Read-only coverage check on legacy input or a snapshot")
    audit.add_argument("--snapshot-id")
    ingest = commands.add_parser("import", help="Inspect a canonical end-labelled CSV, optionally commit an isolated batch")
    ingest.add_argument("csv_path")
    ingest.add_argument("--provider", required=True, help="Declared data origin, not independent verification")
    ingest.add_argument("--license-note", required=True, help="Your local-use authorization description; no account secrets")
    ingest.add_argument("--commit", action="store_true", help="Explicitly write a new isolated batch after validation")
    ingest.add_argument("--input-format", choices=["canonical", "tqsdk"], default="canonical")
    ingest.add_argument("--bar-seconds", type=int, choices=[60], help="Required declaration for a TqSDK 60-second export")
    recent = commands.add_parser("fetch-recent", help="One explicit public contract window; no historical backfill")
    recent.add_argument("--symbol", required=True)
    recent.add_argument("--allow-network", action="store_true")
    recent.add_argument("--bar-label", choices=["unverified", "start", "end"], default="unverified")
    recent.add_argument("--commit", action="store_true")
    recent.add_argument("--confirm-semantics", action="store_true", help="Assert verified timestamp and per-bar volume/end OI semantics")
    recent.add_argument("--license-note", default="", help="Local-use authorization; never credentials")
    commands.add_parser("list", help="Read bounded batch/snapshot manifests only")
    assemble = commands.add_parser("snapshot", help="Write an isolated snapshot from explicitly selected batches")
    assemble.add_argument("--batch-id", action="append", default=[])
    assemble.add_argument("--include-legacy", action="store_true", help="Explicitly include unverified old local raw data; never overwrite it")
    args = parser.parse_args(argv)
    try:
        with redirect_stdout(sys.stderr):
            if args.command == "audit":
                result = audit_local_minutes(args.base_dir, snapshot_id=args.snapshot_id)
            elif args.command == "import":
                options = {"input_format": args.input_format, "bar_seconds": args.bar_seconds} if args.input_format != "canonical" or args.bar_seconds is not None else {}
                result = import_minute_csv(args.base_dir, args.csv_path, provider=args.provider,
                                           license_note=args.license_note, persist=args.commit, **options)
            elif args.command == "fetch-recent":
                if not args.allow_network:
                    raise ValueError("fetch-recent requires explicit --allow-network")
                if args.commit and (not args.confirm_semantics or not args.license_note.strip() or args.bar_label == "unverified"):
                    raise ValueError("commit requires --confirm-semantics, --license-note and a known --bar-label")
                capture = fetch_recent_minutes(args.base_dir, args.symbol, allow_network=True, bar_label=args.bar_label)
                batch = commit_recent_capture(args.base_dir, capture, license_note=args.license_note, confirmed=True) if args.commit else None
                result = {"status": batch["status"] if batch else capture.report["status"], "capture": capture.report, "batch": batch}
            elif args.command == "snapshot":
                if not args.batch_id and not args.include_legacy:
                    raise ValueError("explicit batch selection or --include-legacy is required")
                result = build_minute_snapshot(args.base_dir, args.batch_id, include_legacy=args.include_legacy)
            else:
                result = {"status": "listed", "batches": list_minute_batches(args.base_dir),
                          "snapshots": list_minute_snapshots(args.base_dir), "limit_per_kind": 30}
        code = 1 if isinstance(result, dict) and result.get("status") in {"failed", "rejected", "invalid"} else 0
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    except Exception as exc:
        payload = json.dumps({"status": "failed", "error": redact_sensitive_text(str(exc))[:1500]}, ensure_ascii=False)
        code = 1
    print(payload)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
