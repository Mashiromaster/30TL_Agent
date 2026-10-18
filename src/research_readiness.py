"""Read-only readiness and provenance checks; never repair, train or promote.

Candidate provenance verifies a recorded local construction contract, not an
exchange calendar, data vendor certification or the economic value of factors.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from intelligence_events import redact_sensitive_text
from model_release import ReleaseManager, load_manifest
from strategy_runtime import ModelRuntime, get_strategy_spec
from training_data import LabelSpec, SplitSpec, build_forward_labels, split_train_validation_holdout


def resolve_factor_path(base_dir, factor_path=None):
    base = Path(base_dir).resolve()
    path = Path(factor_path) if factor_path is not None else Path("outputs/df_factors.pkl")
    path = (path if path.is_absolute() else base / path).resolve()
    if not path.is_relative_to(base):
        raise ValueError("factor path must remain inside the project")
    if path != base / "outputs" / "df_factors.pkl" and not path.is_relative_to(base / "artifacts" / "factor_candidates"):
        raise ValueError("use the production factor file or an isolated factor candidate")
    if path.suffix != ".pkl":
        raise ValueError("factor snapshot must be a local .pkl file")
    return path


def load_factor_input(base_dir, factor_path=None):
    """Load trusted local factor bytes once, binding the frame to its digest."""
    path = resolve_factor_path(base_dir, factor_path)
    if path.stat().st_size > 512 * 1024 * 1024:
        raise ValueError("factor snapshot exceeds the 512 MiB local input limit")
    content = path.read_bytes()
    frame = pd.read_pickle(io.BytesIO(content))
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("factor snapshot is not a DataFrame")
    return frame, path, hashlib.sha256(content).hexdigest()


def verify_factor_provenance(base_dir, factor_path, frame, *, expected_sha256=None):
    result = {"verified": False, "reason": "unverified_factor_provenance", "manifest_path": "", "provenance": {}}
    try:
        path = resolve_factor_path(base_dir, factor_path)
        candidate_root = Path(base_dir).resolve() / "artifacts" / "factor_candidates"
        if path.parent.parent != candidate_root or path.name != "factors.pkl":
            raise ValueError("legacy factors have no verified candidate construction contract")
        manifest_path = path.parent / "manifest.json"
        if manifest_path.resolve().parent != path.parent or manifest_path.stat().st_size > 1_000_000:
            raise ValueError("invalid factor manifest path or size")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        provenance = manifest.get("provenance", {})
        if (manifest.get("schema_version") != 1 or manifest.get("status") != "built"
                or manifest.get("candidate_id") != path.parent.name
                or manifest.get("factor_file") != "factors.pkl"
                or not isinstance(provenance, dict)
                or provenance.get("schema_version") != 1
                or provenance.get("kind") != "local_causal_rebuild_v1"
                or provenance.get("pit_contract") != "previous_completed_observed_day_oi_v1"
                or provenance.get("candidate_id") != path.parent.name
                or provenance.get("source_sha256") != manifest.get("source_sha256")
                or not re.fullmatch(r"[a-f0-9]{64}", str(manifest.get("source_sha256", "")))):
            raise ValueError("factor manifest construction contract is invalid")
        if not isinstance(frame, pd.DataFrame) or frame.attrs.get("factor_provenance") != provenance:
            raise ValueError("factor frame provenance does not match manifest")
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != manifest.get("factor_sha256"):
            raise ValueError("factor artifact hash does not match manifest")
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("loaded factor snapshot changed before provenance verification")
        result.update(verified=True, reason="verified_local_construction_contract",
                      manifest_path=str(manifest_path), provenance=provenance,
                      coverage=manifest.get("coverage", {}), sources=manifest.get("sources", {}))
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        result["reason"] = redact_sensitive_text(str(exc))
    return result


def _utc(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("missing timestamp")
    return (stamp.tz_localize("Asia/Shanghai") if stamp.tzinfo is None else stamp).tz_convert("UTC")


def _partition(frame):
    return {"rows": len(frame), "days": int(frame.date.dt.normalize().nunique()),
            "start": frame.date.min().isoformat(), "end": frame.date.max().isoformat()}


def resolve_managed_runtime(base_dir, *, strategy_id=None):
    """Same managed authority as inference, without legacy loading or stdout."""
    base = Path(base_dir)
    manager = ReleaseManager(base / "artifacts" / "models", legacy_model_path=base / "models" / "trained_model.pkl")
    signature = manager.current_signature()
    if signature is None or signature.source != "release":
        raise ValueError("canonical_model_unavailable: a verified managed release is required")
    manifest = load_manifest(signature.manifest_path, verify_artifacts=True)
    spec = get_strategy_spec(strategy_id)
    if manifest.get("metadata", {}).get("strategy_fingerprint") != spec.fingerprint():
        raise ValueError("managed model strategy fingerprint does not match selected policy")
    runtime = ModelRuntime.from_path(signature.artifact_path)
    if runtime.metadata["model_release_id"] != signature.release_id:
        raise ValueError("model artifact release identity does not match managed pointer")
    return runtime, signature


def inspect_research_readiness(base_dir, *, factor_path=None, now=None, max_age_hours=24, strategy_id=None):
    if isinstance(max_age_hours, bool) or not isinstance(max_age_hours, (int, float)) or not np.isfinite(max_age_hours) or max_age_hours <= 0:
        raise ValueError("max_age_hours must be finite and positive")
    clock = _utc(now if now is not None else datetime.now(timezone.utc))
    base = Path(base_dir).resolve()
    report = {"schema_version": "f_agent.readiness.v1", "checked_at": clock.isoformat(), "status": "blocked",
              "factors": {}, "training": {"eligible": False, "valid_rows": 0, "valid_days": 0,
                  "days_without_valid_labels": 0, "partitions": {}},
              "model": {"status": "missing", "release_id": None, "reason": ""},
              "checks": [], "next_actions": [], "execution_authorized": False}

    def check(name, status, detail, action=""):
        report["checks"].append({"id": name, "status": status, "detail": redact_sensitive_text(str(detail)), "action": action})
        if status != "pass" and action and action not in report["next_actions"]:
            report["next_actions"].append(action)

    frame, valid_input, verified, samples_ready = None, False, False, False
    try:
        frame, path, digest = load_factor_input(base, factor_path)
        report["factors"].update(path=str(path), sha256=digest, rows=len(frame), pit_verified=False)
        if frame.empty or not frame.columns.is_unique or not {"date", "ticker", "close", "adj_close"}.issubset(frame):
            raise ValueError("factor data requires unique columns, rows and date/ticker/close/adj_close")
        frame = frame.copy().reset_index(drop=True)
        dates = frame.date.map(_utc)
        frame["date"] = dates.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
        frame = frame.sort_values("date", kind="stable").reset_index(drop=True)
        if frame.duplicated(["date", "ticker"]).any() or not frame.ticker.astype(str).str.fullmatch(r"TL\d{3,4}").all():
            raise ValueError("invalid or duplicate timestamp/contract identities")
        for column in ("close", "adj_close"):
            values = pd.to_numeric(frame[column], errors="coerce")
            if not (np.isfinite(values) & values.gt(0)).all():
                raise ValueError(f"invalid {column} values")
        from feature_registry import discover_feature_groups
        feature_count = len(discover_feature_groups(frame).all_features)
        if not feature_count:
            raise ValueError("no usable numeric model features")
        report["factors"].update(start=frame.date.min().isoformat(), end=frame.date.max().isoformat(),
            observed_days=int(frame.date.dt.normalize().nunique()), feature_count=feature_count,
            age_hours=float((clock - dates.max()).total_seconds() / 3600))
        check("factor_schema", "pass", f"{len(frame)} rows; {feature_count} usable candidate features")
        valid_input = bool((dates <= clock).all())
        check("factor_time", "pass" if valid_input else "blocked", "timestamp cutoff verified" if valid_input else "future factor rows exist",
              "重建截至当前时点的因果因子候选")
        fresh = 0 <= report["factors"]["age_hours"] <= max_age_hours
        check("freshness", "pass" if fresh else "blocked", f"market age {report['factors']['age_hours']:.2f}h; limit {max_age_hours}h",
              "补齐经授权且保留合约身份的分钟行情，再构建候选；日线不能补分钟")
        provenance = verify_factor_provenance(base, path, frame, expected_sha256=digest)
        verified = provenance["verified"]
        report["factors"].update(pit_verified=verified, provenance=provenance)
        check("factor_provenance", "pass" if verified else "blocked", provenance["reason"], "构建独立本地因果因子候选，禁止直接复用来源不明的旧CNN缓存")
        coverage = provenance.get("coverage", {})
        gap = coverage.get("largest_observed_day_gap_days", 0) if isinstance(coverage, dict) else 0
        if isinstance(gap, (int, float)) and gap > 7:
            check("coverage_gap", "warning", f"largest observed day gap: {gap} days; exchange calendar not certified",
                  "审查长时间缺口和隔离日；训练资格不等于连续行情或可发布性认证")
        labels = build_forward_labels(frame, LabelSpec())
        labelled = labels.loc[labels.label_valid & np.isfinite(labels.Target_Ret)].copy()
        valid_days = int(labelled.date.dt.normalize().nunique())
        report["training"].update(valid_rows=len(labelled), valid_days=valid_days,
            days_without_valid_labels=report["factors"]["observed_days"] - valid_days)
        try:
            train, validation, holdout = split_train_validation_holdout(labelled, SplitSpec())
            train = train.loc[train.date >= validation.date.min() - pd.Timedelta(days=9 * 30)]
            if train.empty:
                raise ValueError("lookback-limited training set is empty")
            report["training"]["partitions"] = {name: _partition(part) for name, part in
                (("train", train), ("validation", validation), ("sealed_holdout", holdout))}
            samples_ready = True
            check("training_samples", "pass", f"{valid_days} days with valid 30-bar labels; purged split available")
        except ValueError as exc:
            check("training_samples", "blocked", exc, "补齐足够完整分钟交易日；不能缩短封存集来绕过门禁")
        if report["training"]["days_without_valid_labels"]:
            check("sparse_days", "warning", f"{report['training']['days_without_valid_labels']} observed days have no valid labels",
                  "检查样本稀疏期与有效标签覆盖，不能只按日历跨度宣称样本充足")
    except Exception as exc:
        check("factor_input", "blocked", exc, "检查本地行情源并构建独立因子候选；不启用旧模型回退")

    report["training"]["eligible"] = bool(valid_input and verified and samples_ready)
    try:
        runtime, signature = resolve_managed_runtime(base, strategy_id=strategy_id)
        report["model"].update(status="verified", release_id=signature.release_id, reason="managed integrity and strategy verified",
                               kind=runtime.metadata["selected_model_kind"])
        check("managed_model", "pass", signature.release_id)
        from feature_registry import validate_feature_schema
        if frame is not None:
            # Includes persisted GP reconstruction; no model prediction or fit.
            prepared = runtime._with_gp_features(frame)
            validate_feature_schema(prepared, runtime.features)
            check("model_features", "pass", "selected factor schema matches managed model")
    except Exception as exc:
        pointer_exists = (base / "artifacts" / "models" / "current.json").exists()
        legacy_exists = (base / "models" / "trained_model.pkl").exists()
        report["model"].update(status="invalid" if pointer_exists else "legacy_only" if legacy_exists else "missing", reason=redact_sensitive_text(str(exc)))
        check("managed_model", "blocked", exc, "先训练独立候选并审查门禁结果；模型发布仍需显式操作")
    report["status"] = "ready" if all(item["status"] != "blocked" for item in report["checks"]) else "blocked"
    return report
