# -*- coding: utf-8 -*-
"""Immutable model release manifests and an atomic active-release pointer.

This module is intentionally independent from training.  Training code may
write a candidate directory and its manifest; promotion verifies every byte,
materialises an immutable release, and only then switches ``current.json``.
An existing ``models/trained_model.pkl`` can optionally be maintained as a
compatibility projection, but it is never treated as release metadata.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Mapping


MANIFEST_SCHEMA_VERSION = 1
POINTER_SCHEMA_VERSION = 1
GATE_ATTESTATION_SCHEMA_VERSION = 1
BASELINE_MIN_OBSERVATION_DAYS = 20
BASELINE_MIN_TOTAL_RETURN = 0.0
BASELINE_MIN_SHARPE = 0.0
BASELINE_MAX_DRAWDOWN = 0.35
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class ModelReleaseError(RuntimeError):
    """Base error for candidate validation and release promotion."""


class ManifestValidationError(ModelReleaseError):
    """A manifest is malformed or refers outside its artifact directory."""


class ArtifactIntegrityError(ModelReleaseError):
    """An artifact no longer matches its declared size or SHA-256 digest."""


class GateRejectedError(ModelReleaseError):
    """Promotion was attempted without an explicit passing release gate."""


class ReleaseConflictError(ModelReleaseError):
    """An immutable release ID already refers to a different candidate."""


def _serialized_release(method):
    """Serialize release mutations across threads and local processes."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._release_lock():
            return method(self, *args, **kwargs)

    return wrapped


@dataclass(frozen=True)
class ReleaseSignature:
    """Read-only identity of either a managed release or a legacy model."""

    source: str
    read_only: bool
    release_id: str | None
    artifact_name: str
    artifact_path: Path
    artifact_sha256: str
    manifest_path: Path | None = None
    manifest_sha256: str | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fsync_directory(path: Path) -> None:
    """Best-effort directory fsync so renames survive a power loss."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    try:
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    except OSError:
        pass


def sha256_file(path: str | os.PathLike, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _validate_release_id(release_id: str) -> str:
    value = str(release_id)
    if not _SAFE_ID.fullmatch(value):
        raise ManifestValidationError(
            "release_id must be a single safe path segment"
        )
    return value


def _artifact_path(base_dir: Path, relative_path: str) -> Path:
    relative = Path(str(relative_path))
    if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
        raise ManifestValidationError(
            f"artifact path must be relative and contained: {relative_path!r}"
        )
    base_resolved = base_dir.resolve()
    resolved = (base_dir / relative).resolve()
    try:
        resolved.relative_to(base_resolved)
    except ValueError as exc:
        raise ManifestValidationError(
            f"artifact path escapes manifest directory: {relative_path!r}"
        ) from exc
    return resolved


def _stage_json(destination: Path, payload: Mapping) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return temporary_path
    except BaseException:
        try:
            temporary_path.unlink()
        except OSError:
            pass
        raise


def _atomic_write_json(destination: Path, payload: Mapping) -> Path:
    temporary_path = _stage_json(destination, payload)
    try:
        os.replace(temporary_path, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        try:
            temporary_path.unlink()
        except OSError:
            pass
        raise
    return destination


def _load_gate_attestation(candidate_path: Path, candidate: Mapping) -> dict:
    """Validate a hashed evaluation report instead of trusting a caller bool."""
    artifacts = candidate["artifacts"]
    for required_name in ("model", "evaluation"):
        if required_name not in artifacts:
            raise GateRejectedError(
                f"candidate gate requires {required_name!r} artifact"
            )
    evaluation_path = _artifact_path(
        candidate_path.parent, artifacts["evaluation"]["path"]
    )
    try:
        evaluation_bytes = evaluation_path.read_bytes()
        evaluation_record = artifacts["evaluation"]
        if len(evaluation_bytes) != evaluation_record["size_bytes"]:
            raise GateRejectedError("evaluation evidence size changed")
        if hashlib.sha256(evaluation_bytes).hexdigest() != evaluation_record["sha256"]:
            raise GateRejectedError("evaluation evidence hash changed")
        report = json.loads(evaluation_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateRejectedError("cannot read hashed evaluation evidence") from exc
    if not isinstance(report, dict):
        raise GateRejectedError("evaluation evidence must be an object")
    attestation = report.get("gate_attestation")
    if not isinstance(attestation, dict):
        raise GateRejectedError("evaluation has no gate_attestation")
    if attestation.get("schema_version") != GATE_ATTESTATION_SCHEMA_VERSION:
        raise GateRejectedError("unsupported gate attestation schema")

    metadata = candidate.get("metadata", {})
    expected = {
        "release_id": candidate["release_id"],
        "run_id": candidate["run_id"],
        "strategy_id": metadata.get("strategy_id"),
        "strategy_fingerprint": metadata.get("strategy_fingerprint"),
        "model_sha256": artifacts["model"]["sha256"],
    }
    for name, value in expected.items():
        if not isinstance(value, str) or not value:
            raise GateRejectedError(f"candidate is missing gate binding {name}")
        if attestation.get(name) != value:
            raise GateRejectedError(f"gate attestation {name} mismatch")
        report_name = name if name != "model_sha256" else None
        if report_name and report.get(report_name) != value:
            raise GateRejectedError(f"evaluation {report_name} mismatch")

    if metadata.get("release_gate_passed") is not True:
        raise GateRejectedError("candidate metadata does not record a passing gate")
    if report.get("release_gate_passed") is not True:
        raise GateRejectedError("evaluation release gate did not pass")
    if report.get("promotion_eligible") is not True:
        raise GateRejectedError("validation promotion gate did not pass")
    if report.get("strategy_gate_passed") is not True:
        raise GateRejectedError("strategy risk gate did not pass")

    holdout_ic = report.get("sealed_holdout_ic")
    if (
        not isinstance(holdout_ic, (int, float))
        or isinstance(holdout_ic, bool)
        or not math.isfinite(float(holdout_ic))
        or float(holdout_ic) <= 0.0
    ):
        raise GateRejectedError("sealed holdout IC veto did not pass")

    policy = attestation.get("policy")
    metrics = report.get("strategy_backtest")
    if not isinstance(policy, dict) or not isinstance(metrics, dict):
        raise GateRejectedError("gate policy or strategy metrics are missing")
    required_policy = (
        "min_observation_days",
        "min_total_return",
        "min_sharpe",
        "max_drawdown_limit",
    )
    if any(name not in policy for name in required_policy):
        raise GateRejectedError("gate policy is incomplete")
    required_metrics = (
        "observation_days",
        "total_return",
        "sharpe",
        "annual_volatility",
        "max_drawdown",
        "active_ratio",
    )
    if any(name not in metrics for name in required_metrics):
        raise GateRejectedError("strategy metrics are incomplete")
    numeric_values = [policy[name] for name in required_policy] + [
        metrics[name] for name in required_metrics
    ]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in numeric_values
    ):
        raise GateRejectedError("gate policy and metrics must be finite numbers")
    if metrics.get("strategy_fingerprint") != expected["strategy_fingerprint"]:
        raise GateRejectedError("backtest strategy fingerprint mismatch")
    if metrics.get("strategy_id") != expected["strategy_id"]:
        raise GateRejectedError("backtest strategy_id mismatch")
    if metrics.get("model_release_id") != expected["release_id"]:
        raise GateRejectedError("backtest model release mismatch")
    if float(policy["min_observation_days"]) % 1 != 0:
        raise GateRejectedError("minimum observation days must be an integer")
    if int(policy["min_observation_days"]) < BASELINE_MIN_OBSERVATION_DAYS:
        raise GateRejectedError("gate policy weakens minimum observation days")
    if float(policy["min_total_return"]) < BASELINE_MIN_TOTAL_RETURN:
        raise GateRejectedError("gate policy weakens total return threshold")
    if float(policy["min_sharpe"]) < BASELINE_MIN_SHARPE:
        raise GateRejectedError("gate policy weakens Sharpe threshold")
    max_drawdown_limit = float(policy["max_drawdown_limit"])
    if not 0.0 < max_drawdown_limit <= BASELINE_MAX_DRAWDOWN:
        raise GateRejectedError("gate policy weakens max drawdown threshold")
    for name in ("min_total_return", "min_sharpe", "max_drawdown_limit"):
        if report.get(name) != policy[name]:
            raise GateRejectedError(f"evaluation {name} policy mismatch")
    if int(metrics["observation_days"]) < int(policy["min_observation_days"]):
        raise GateRejectedError("insufficient strategy observation days")
    if not 0.0 < float(metrics["active_ratio"]) <= 1.0:
        raise GateRejectedError("strategy active ratio is invalid")
    if float(metrics["annual_volatility"]) < 0.0:
        raise GateRejectedError("strategy annual volatility is invalid")
    if float(metrics["total_return"]) <= float(policy["min_total_return"]):
        raise GateRejectedError("cost-adjusted total return gate did not pass")
    if float(metrics["sharpe"]) <= float(policy["min_sharpe"]):
        raise GateRejectedError("cost-adjusted Sharpe gate did not pass")
    if not -1.0 <= float(metrics["max_drawdown"]) <= 0.0:
        raise GateRejectedError("strategy max drawdown is invalid")
    if float(metrics["max_drawdown"]) < -max_drawdown_limit:
        raise GateRejectedError("max drawdown gate did not pass")
    if attestation.get("gate_passed") is not True:
        raise GateRejectedError("gate attestation is not passing")
    return dict(attestation)


def _artifact_record(path: Path, relative_path: str) -> dict:
    if not path.is_file():
        raise ManifestValidationError(f"artifact does not exist: {relative_path}")
    return {
        "path": Path(relative_path).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def write_candidate_manifest(
    candidate_dir: str | os.PathLike,
    *,
    release_id: str,
    run_id: str,
    artifacts: Mapping[str, str | os.PathLike],
    metadata: Mapping | None = None,
) -> Path:
    """Hash candidate artifacts and atomically write ``manifest.json``."""

    directory = Path(candidate_dir)
    release_id = _validate_release_id(release_id)
    if not artifacts:
        raise ManifestValidationError("candidate manifest requires artifacts")

    records = {}
    for name, relative_path in sorted(artifacts.items()):
        artifact_name = str(name).strip()
        if not artifact_name:
            raise ManifestValidationError("artifact names must be non-empty")
        relative_text = Path(relative_path).as_posix()
        path = _artifact_path(directory, relative_text)
        records[artifact_name] = _artifact_record(path, relative_text)

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_type": "candidate",
        "release_id": release_id,
        "run_id": str(run_id),
        "created_at": _utc_now(),
        "metadata": dict(metadata or {}),
        "artifacts": records,
    }
    return _atomic_write_json(directory / "manifest.json", manifest)


def load_manifest(
    manifest_path: str | os.PathLike,
    *,
    verify_artifacts: bool = True,
) -> dict:
    """Load, validate, and optionally verify every artifact in a manifest."""

    path = Path(manifest_path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestValidationError(f"cannot read manifest: {path}") from exc

    if not isinstance(manifest, dict):
        raise ManifestValidationError("manifest must be a JSON object")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestValidationError("unsupported manifest schema_version")
    if manifest.get("manifest_type") not in {"candidate", "release"}:
        raise ManifestValidationError("manifest_type must be candidate or release")
    _validate_release_id(manifest.get("release_id", ""))
    if not isinstance(manifest.get("run_id"), str):
        raise ManifestValidationError("manifest requires run_id")
    if not isinstance(manifest.get("metadata", {}), dict):
        raise ManifestValidationError("metadata must be an object")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ManifestValidationError("manifest requires an artifacts object")
    for name, record in artifacts.items():
        if not isinstance(name, str) or not name or not isinstance(record, dict):
            raise ManifestValidationError("invalid artifact record")
        relative_path = record.get("path")
        expected_hash = record.get("sha256")
        expected_size = record.get("size_bytes")
        if not isinstance(relative_path, str):
            raise ManifestValidationError(f"artifact {name} requires path")
        artifact = _artifact_path(path.parent, relative_path)
        if not isinstance(expected_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_hash
        ):
            raise ManifestValidationError(f"artifact {name} has invalid sha256")
        if not isinstance(expected_size, int) or expected_size < 0:
            raise ManifestValidationError(f"artifact {name} has invalid size_bytes")
        if verify_artifacts:
            if not artifact.is_file():
                raise ArtifactIntegrityError(f"artifact missing: {relative_path}")
            actual_size = artifact.stat().st_size
            if actual_size != expected_size:
                raise ArtifactIntegrityError(
                    f"artifact size mismatch for {relative_path}: "
                    f"{actual_size} != {expected_size}"
                )
            actual_hash = sha256_file(artifact)
            if actual_hash != expected_hash:
                raise ArtifactIntegrityError(
                    f"artifact sha256 mismatch for {relative_path}: "
                    f"{actual_hash} != {expected_hash}"
                )

    if manifest["manifest_type"] == "release":
        if manifest.get("gate_passed") is not True:
            raise ManifestValidationError("release manifest requires gate_passed=true")
        source_hash = manifest.get("source_candidate_sha256")
        if not isinstance(source_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", source_hash
        ):
            raise ManifestValidationError(
                "release manifest requires source_candidate_sha256"
            )
        attestation = manifest.get("gate_attestation")
        if (
            not isinstance(attestation, dict)
            or attestation.get("schema_version")
            != GATE_ATTESTATION_SCHEMA_VERSION
            or attestation.get("gate_passed") is not True
        ):
            raise ManifestValidationError(
                "release manifest requires a passing gate_attestation"
            )
        verified_attestation = _load_gate_attestation(path, manifest)
        if verified_attestation != attestation:
            raise ManifestValidationError(
                "release gate_attestation differs from hashed evaluation evidence"
            )
    return manifest


class ReleaseManager:
    """Promote verified candidates and resolve the active model signature."""

    def __init__(
        self,
        artifacts_root: str | os.PathLike,
        *,
        legacy_model_path: str | os.PathLike | None = None,
        candidate_root: str | os.PathLike | None = None,
        projection_roots: tuple[str | os.PathLike, ...] | None = None,
    ):
        self.artifacts_root = Path(artifacts_root)
        self.candidate_root = (
            Path(candidate_root)
            if candidate_root is not None
            else self.artifacts_root.parent / "experiments"
        )
        self.legacy_model_path = (
            Path(legacy_model_path) if legacy_model_path is not None else None
        )
        if projection_roots is None:
            if (
                self.artifacts_root.name == "models"
                and self.artifacts_root.parent.name == "artifacts"
            ):
                project_root = self.artifacts_root.parent.parent
            else:
                project_root = self.artifacts_root.parent
            projection_roots = (
                project_root / "models",
                project_root / "outputs",
            )
        self.projection_roots = tuple(Path(path) for path in projection_roots)
        # Kept injectable for deterministic interruption tests.
        self._replace = os.replace

    @contextmanager
    def _release_lock(self):
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        lock_path = self.artifacts_root / ".release.lock"
        with lock_path.open("a+b") as handle:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                unlock = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except ImportError:  # pragma: no cover - Windows fallback
                import msvcrt

                handle.seek(0)
                if handle.read(1) == b"":
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                unlock = lambda: msvcrt.locking(
                    handle.fileno(), msvcrt.LK_UNLCK, 1
                )
            try:
                yield
            finally:
                unlock()

    @property
    def releases_dir(self) -> Path:
        return self.artifacts_root / "releases"

    @property
    def current_path(self) -> Path:
        return self.artifacts_root / "current.json"

    @_serialized_release
    def promote(
        self,
        candidate_manifest_path: str | os.PathLike,
        *,
        project_legacy: bool = False,
        legacy_artifact_name: str = "model",
    ) -> ReleaseSignature:
        """Promote a candidate after an explicit gate and switch it active.

        ``current.json`` is the sole authoritative switch and is replaced in
        one filesystem operation.  The optional legacy file is refreshed only
        afterwards as a non-authoritative, best-effort compatibility view.
        """

        candidate_path = Path(candidate_manifest_path)
        if candidate_path.name != "manifest.json":
            raise ManifestValidationError("candidate must be a manifest.json file")
        try:
            candidate_path.resolve().relative_to(self.candidate_root.resolve())
        except ValueError as exc:
            raise ManifestValidationError(
                "candidate manifest is outside the configured experiments root"
            ) from exc
        candidate_hash = sha256_file(candidate_path)
        candidate = load_manifest(candidate_path, verify_artifacts=True)
        if candidate["manifest_type"] != "candidate":
            raise ManifestValidationError("promotion requires a candidate manifest")
        gate_attestation = _load_gate_attestation(candidate_path, candidate)
        if sha256_file(candidate_path) != candidate_hash:
            raise ArtifactIntegrityError("candidate manifest changed during promotion")
        release_id = candidate["release_id"]

        release_dir = self.releases_dir / release_id
        release_manifest_path = release_dir / "manifest.json"
        if release_dir.exists():
            release_manifest = load_manifest(
                release_manifest_path, verify_artifacts=True
            )
            if release_manifest.get("source_candidate_sha256") != candidate_hash:
                raise ReleaseConflictError(
                    f"release_id {release_id!r} already belongs to another candidate"
                )
        else:
            release_manifest = self._materialize_release(
                candidate_path,
                candidate,
                candidate_hash,
                gate_attestation,
                release_dir,
            )

        if legacy_artifact_name not in release_manifest["artifacts"]:
            raise ManifestValidationError(
                f"release has no {legacy_artifact_name!r} compatibility artifact"
            )
        if project_legacy and self.legacy_model_path is None:
            raise ManifestValidationError(
                "project_legacy requires legacy_model_path"
            )

        release_manifest_hash = sha256_file(release_manifest_path)
        pointer = {
            "schema_version": POINTER_SCHEMA_VERSION,
            "release_id": release_id,
            "manifest_path": release_manifest_path.relative_to(
                self.artifacts_root
            ).as_posix(),
            "manifest_sha256": release_manifest_hash,
            "source_candidate_sha256": candidate_hash,
            "model_artifact": legacy_artifact_name,
            "switched_at": _utc_now(),
        }
        self._switch_active(
            pointer,
            project_legacy=project_legacy,
            legacy_artifact_name=legacy_artifact_name,
        )
        signature = self.current_signature()
        if signature is None:  # Defensive: a successful switch must resolve.
            raise ModelReleaseError("active release disappeared after promotion")
        return signature

    @_serialized_release
    def activate_release(
        self,
        release_id: str,
        *,
        project_legacy: bool = False,
        legacy_artifact_name: str = "model",
    ) -> ReleaseSignature:
        """Atomically reactivate a previously verified immutable release."""
        release_id = _validate_release_id(release_id)
        manifest_path = self.releases_dir / release_id / "manifest.json"
        manifest = load_manifest(manifest_path, verify_artifacts=True)
        if manifest["manifest_type"] != "release":
            raise ManifestValidationError("activation requires a release manifest")
        if manifest["release_id"] != release_id:
            raise ManifestValidationError("release directory/id mismatch")
        if legacy_artifact_name not in manifest["artifacts"]:
            raise ManifestValidationError(
                f"release has no {legacy_artifact_name!r} compatibility artifact"
            )
        if project_legacy and self.legacy_model_path is None:
            raise ManifestValidationError(
                "project_legacy requires legacy_model_path"
            )

        manifest_hash = sha256_file(manifest_path)
        pointer = {
            "schema_version": POINTER_SCHEMA_VERSION,
            "release_id": release_id,
            "manifest_path": manifest_path.relative_to(
                self.artifacts_root
            ).as_posix(),
            "manifest_sha256": manifest_hash,
            "source_candidate_sha256": manifest["source_candidate_sha256"],
            "model_artifact": legacy_artifact_name,
            "switched_at": _utc_now(),
        }
        self._switch_active(
            pointer,
            project_legacy=project_legacy,
            legacy_artifact_name=legacy_artifact_name,
        )
        signature = self.current_signature()
        if signature is None:
            raise ModelReleaseError("release disappeared after activation")
        return signature

    def _materialize_release(
        self,
        candidate_path: Path,
        candidate: dict,
        candidate_hash: str,
        gate_attestation: dict,
        release_dir: Path,
    ) -> dict:
        self.releases_dir.mkdir(parents=True, exist_ok=True)
        stage = Path(
            tempfile.mkdtemp(
                prefix=f".{candidate['release_id']}.", dir=self.releases_dir
            )
        )
        try:
            for record in candidate["artifacts"].values():
                source = _artifact_path(candidate_path.parent, record["path"])
                destination = stage / record["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                _fsync_file(destination)
                _fsync_directory(destination.parent)

            release_manifest = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "manifest_type": "release",
                "release_id": candidate["release_id"],
                "run_id": candidate["run_id"],
                "created_at": candidate.get("created_at"),
                "promoted_at": _utc_now(),
                "metadata": dict(candidate.get("metadata", {})),
                "artifacts": candidate["artifacts"],
                "gate_passed": True,
                "gate_attestation": gate_attestation,
                "source_candidate_sha256": candidate_hash,
            }
            _atomic_write_json(stage / "manifest.json", release_manifest)
            load_manifest(stage / "manifest.json", verify_artifacts=True)
            os.replace(stage, release_dir)
            _fsync_directory(self.releases_dir)
            return load_manifest(
                release_dir / "manifest.json", verify_artifacts=True
            )
        finally:
            if stage.exists():
                shutil.rmtree(stage, ignore_errors=True)

    @staticmethod
    def _discard(path: Path | None) -> None:
        if path is None:
            return
        try:
            path.unlink()
        except OSError:
            pass

    def _switch_active(
        self,
        pointer: dict,
        *,
        project_legacy: bool,
        legacy_artifact_name: str,
    ) -> None:
        """Atomically switch authority, then refresh a compatibility view.

        A process kill can occur between any two filesystem operations.  By
        making the pointer the only authoritative operation and projecting the
        legacy file afterwards, every crash state still resolves to exactly
        one verified managed release.  The compatibility file may lag, which
        is safe because managed consumers never read it as release metadata.
        """
        pointer_temp = _stage_json(self.current_path, pointer)
        try:
            self._replace(pointer_temp, self.current_path)
            pointer_temp = None
            _fsync_directory(self.current_path.parent)
        finally:
            self._discard(pointer_temp)

        if project_legacy:
            assert self.legacy_model_path is not None
            try:
                self.project_active_artifact(
                    legacy_artifact_name, self.legacy_model_path
                )
            except Exception as exc:
                warnings.warn(
                    "active release switched, but the non-authoritative legacy "
                    "projection was not refreshed "
                    f"({type(exc).__name__})",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def current_signature(self) -> ReleaseSignature | None:
        """Resolve managed metadata, or fingerprint an unmanaged legacy model."""

        if self.current_path.exists():
            try:
                pointer = json.loads(self.current_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ManifestValidationError("cannot read current release pointer") from exc
            if pointer.get("schema_version") != POINTER_SCHEMA_VERSION:
                raise ManifestValidationError("unsupported current pointer schema")
            release_id = _validate_release_id(pointer.get("release_id", ""))
            manifest_relative = pointer.get("manifest_path")
            if not isinstance(manifest_relative, str):
                raise ManifestValidationError("current pointer requires manifest_path")
            manifest_path = _artifact_path(
                self.artifacts_root, manifest_relative
            )
            expected_manifest_hash = pointer.get("manifest_sha256")
            if sha256_file(manifest_path) != expected_manifest_hash:
                raise ArtifactIntegrityError("active manifest sha256 mismatch")
            manifest = load_manifest(manifest_path, verify_artifacts=True)
            if manifest["manifest_type"] != "release":
                raise ManifestValidationError("current pointer must target a release")
            if manifest["release_id"] != release_id:
                raise ManifestValidationError("current pointer release_id mismatch")
            if manifest.get("source_candidate_sha256") != pointer.get(
                "source_candidate_sha256"
            ):
                raise ManifestValidationError("current pointer candidate mismatch")

            artifact_name = pointer.get("model_artifact", "model")
            if artifact_name not in manifest["artifacts"]:
                raise ManifestValidationError("current model artifact is missing")
            record = manifest["artifacts"][artifact_name]
            artifact_path = _artifact_path(manifest_path.parent, record["path"])

            return ReleaseSignature(
                source="release",
                read_only=False,
                release_id=release_id,
                artifact_name=artifact_name,
                artifact_path=artifact_path,
                artifact_sha256=record["sha256"],
                manifest_path=manifest_path,
                manifest_sha256=expected_manifest_hash,
            )

        if self.legacy_model_path is not None and self.legacy_model_path.is_file():
            return ReleaseSignature(
                source="legacy",
                read_only=True,
                release_id=None,
                artifact_name="legacy_model",
                artifact_path=self.legacy_model_path.resolve(),
                artifact_sha256=sha256_file(self.legacy_model_path),
            )
        return None

    def active_artifact_path(self, artifact_name: str) -> Path | None:
        """Resolve a verified artifact from the active managed release.

        Legacy models have no multi-artifact manifest, so this deliberately
        returns ``None`` for them instead of guessing companion paths.
        """
        signature = self.current_signature()
        if signature is None or signature.source != "release":
            return None
        assert signature.manifest_path is not None
        manifest = load_manifest(signature.manifest_path, verify_artifacts=True)
        record = manifest["artifacts"].get(str(artifact_name))
        if record is None:
            return None
        return _artifact_path(signature.manifest_path.parent, record["path"])

    def project_active_artifact(
        self,
        artifact_name: str,
        destination: str | os.PathLike,
    ) -> Path:
        """Atomically create a non-authoritative compatibility projection."""
        source = self.active_artifact_path(artifact_name)
        if source is None:
            raise ManifestValidationError(
                f"active release has no artifact {artifact_name!r}"
            )
        destination_path = Path(destination)
        destination_resolved = destination_path.resolve()
        if not any(
            self._is_within(destination_resolved, root.resolve())
            for root in self.projection_roots
        ):
            raise ManifestValidationError(
                "compatibility projection destination is outside allowed roots"
            )
        if destination_path.is_symlink():
            raise ManifestValidationError(
                "compatibility projection destination must not be a symlink"
            )
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination_path.name}.",
            suffix=".tmp",
            dir=destination_path.parent,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            shutil.copy2(source, temporary_path)
            _fsync_file(temporary_path)
            if sha256_file(temporary_path) != sha256_file(source):
                raise ArtifactIntegrityError(
                    f"compatibility projection sha256 mismatch: {artifact_name}"
                )
            os.replace(temporary_path, destination_path)
            _fsync_directory(destination_path.parent)
        except BaseException:
            self._discard(temporary_path)
            raise
        return destination_path

    @staticmethod
    def _is_within(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False


__all__ = [
    "ArtifactIntegrityError",
    "BASELINE_MAX_DRAWDOWN",
    "BASELINE_MIN_OBSERVATION_DAYS",
    "BASELINE_MIN_SHARPE",
    "BASELINE_MIN_TOTAL_RETURN",
    "GateRejectedError",
    "ManifestValidationError",
    "ModelReleaseError",
    "ReleaseConflictError",
    "ReleaseManager",
    "ReleaseSignature",
    "load_manifest",
    "sha256_file",
    "write_candidate_manifest",
]
