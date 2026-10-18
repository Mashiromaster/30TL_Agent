# -*- coding: utf-8 -*-
"""SQLite-backed event ledger for TL sentiment intelligence.

This module is intentionally independent from trading, inference, model release,
and broker code.  It stores untrusted external documents, source cursors,
structured event assessments, alerts, and monitor run summaries under the
``data/intelligence`` namespace.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


ASSESSMENT_SCHEMA_VERSION = "event-assessment-v1"
DEFAULT_RULE_MODEL_VERSION = "deterministic-tl-rules-v2"
ALERT_STATUSES = {"open", "suppressed", "reported"}
SOURCE_CONTENT_STALE_AFTER_SECONDS = 72 * 60 * 60
_BLOCKED_RESPONSE_KEYS = {
    "trade_action",
    "tool_calls",
    "function_call",
    "orders",
    "order",
    "broker",
    "position_size",
    "target_position",
}
_SECRET_RESPONSE_KEYS = {
    "api_key",
    "apikey",
    "token",
    "access_token",
    "authorization",
    "password",
    "secret",
}
_SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|token|authorization|password|secret)"
    r"(\s*[:=]\s*|\s+)([^\s,;]+)"
)
_BEARER_PATTERN = re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+")
_AUTHORIZATION_PATTERN = re.compile(
    r"(?i)authorization\s*[:=]\s*[^\r\n,;]+"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _safe_json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def redact_sensitive_text(value: object) -> str:
    """Remove common credential forms before persisting or displaying text."""
    text = re.sub(r"https://open\.feishu\.cn/open-apis/bot/v2/hook/[^\s\"'<>]+",
                  "[REDACTED_WEBHOOK]", str(value))
    text = _AUTHORIZATION_PATTERN.sub("Authorization=[REDACTED]", text)
    text = _BEARER_PATTERN.sub("Bearer [REDACTED]", text)
    text = _SECRET_PATTERN.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)
    return text


def _redact_error(value: object) -> str:
    return redact_sensitive_text(value)[:1000]


def _hash_parts(*parts: object, length: int = 32) -> str:
    payload = "\x1f".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _normalize_event_text(title: str, content: str) -> str:
    text = f"{title or ''} {content or ''}".lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    return text[:800]


def _sanitize_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not value:
        return {}
    clean: dict[str, Any] = {}
    for key, item in value.items():
        key_text = str(key)
        normalized_key = key_text.lower()
        if normalized_key in _BLOCKED_RESPONSE_KEYS:
            continue
        if (
            normalized_key in _SECRET_RESPONSE_KEYS
            or normalized_key.endswith("_api_key")
            or normalized_key.endswith("_token")
        ):
            clean[key_text] = "[REDACTED]"
            continue
        if isinstance(item, Mapping):
            nested = _sanitize_mapping(item)
            if nested:
                clean[key_text] = nested
        elif isinstance(item, list):
            clean[key_text] = [
                _sanitize_mapping(x)
                if isinstance(x, Mapping)
                else redact_sensitive_text(x) if isinstance(x, str) else x
                for x in item
            ]
        elif isinstance(item, str):
            clean[key_text] = redact_sensitive_text(item)
        else:
            clean[key_text] = item
    return clean


@dataclass(frozen=True)
class DocumentRecord:
    source_name: str
    source_document_id: str
    title: str
    content: str
    url: str = ""
    published_at: str | None = None
    received_at: str | None = None
    available_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    document_id: str = ""
    content_hash: str = ""
    event_cluster_id: str = ""
    inserted_at: str = ""

    def normalized(
        self,
        *,
        source_name: str | None = None,
        now: str | None = None,
    ) -> "DocumentRecord":
        timestamp = now or utc_now_iso()
        normalized_source = source_name or self.source_name
        received_at = self.received_at or timestamp
        # Point-in-time availability is first receipt by this system; a
        # provider's publication claim remains separate provenance.
        available_at = self.available_at or received_at
        received_dt = _parse_iso(received_at)
        available_dt = _parse_iso(available_at)
        if (
            received_dt is not None
            and available_dt is not None
            and available_dt < received_dt
        ):
            available_at = received_at
        content_hash = self.content_hash or _hash_parts(self.title, self.content)
        source_document_id = (
            self.source_document_id
            or self.url
            or content_hash
        )
        document_id = self.document_id or _hash_parts(
            normalized_source,
            source_document_id,
            content_hash,
            length=40,
        )
        event_cluster_id = self.event_cluster_id or _hash_parts(
            _normalize_event_text(self.title, self.content),
            length=24,
        )
        return replace(
            self,
            source_name=normalized_source,
            source_document_id=source_document_id,
            document_id=document_id,
            content_hash=content_hash,
            event_cluster_id=event_cluster_id,
            received_at=received_at,
            available_at=available_at,
            inserted_at=self.inserted_at or timestamp,
        )


@dataclass(frozen=True)
class SourceCursor:
    source_name: str
    cursor_value: str
    updated_at: str


@dataclass(frozen=True)
class SourceHealth:
    source_name: str
    last_success_at: str | None = None
    last_checked_at: str | None = None
    last_error: str = ""
    source_stale: bool = False
    content_stale: bool = False
    last_document_at: str | None = None
    content_baseline_at: str | None = None
    document_count: int = 0


@dataclass
class EventAssessment:
    document_id: str
    event_cluster_id: str
    event_type: str
    severity: str
    policy_stance: str = "unclear"
    bond_impact: str = "unclear"
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    affected_assets: list[str] = field(default_factory=list)
    tenors: list[str] = field(default_factory=list)
    novelty: float = 1.0
    channels: Mapping[str, float] = field(default_factory=dict)
    prompt_injection_flag: bool = False
    model_version: str = DEFAULT_RULE_MODEL_VERSION
    schema_version: str = ASSESSMENT_SCHEMA_VERSION
    assessed_at: str = ""
    expires_at: str | None = None
    missing_reason: str = ""
    raw_response: Mapping[str, Any] = field(default_factory=dict)
    assessment_id: str = ""

    def normalized(self, *, now: str | None = None) -> "EventAssessment":
        timestamp = now or utc_now_iso()
        self.assessed_at = self.assessed_at or timestamp
        self.raw_response = _sanitize_mapping(self.raw_response)
        self.assessment_id = self.assessment_id or _hash_parts(
            self.document_id,
            self.event_cluster_id,
            self.schema_version,
            self.model_version,
            self.event_type,
            self.severity,
            self.policy_stance,
            self.bond_impact,
            self.confidence,
            _stable_json(self.evidence),
            _stable_json(self.affected_assets),
            _stable_json(self.tenors),
            self.novelty,
            _stable_json(self.channels),
            self.prompt_injection_flag,
            self.expires_at,
            self.missing_reason,
            _stable_json(self.raw_response),
            length=40,
        )
        return self


@dataclass(frozen=True)
class Alert:
    alert_id: str
    event_cluster_id: str
    first_assessment_id: str
    last_assessment_id: str
    severity: str
    status: str
    reason: str = ""
    created_at: str = ""
    updated_at: str = ""


@dataclass
class MonitorRun:
    run_id: str
    started_at: str
    finished_at: str = ""
    status: str = "running"
    sources_checked: int = 0
    sources_failed: int = 0
    sources_content_stale: int = 0
    documents_seen: int = 0
    documents_inserted: int = 0
    documents_dropped: int = 0
    assessments_created: int = 0
    alerts_opened: int = 0
    report_path: str = ""


class _ClosingConnection(sqlite3.Connection):
    """sqlite's default context manager commits, but does not close the handle."""

    def __exit__(self, *args):
        try:
            return super().__exit__(*args)
        finally:
            self.close()


class IntelligenceEventStore:
    """Small sqlite repository for intelligence documents and assessments."""

    def __init__(self, base_dir: str | Path, *, db_path: str | Path | None = None,
                 read_only: bool = False):
        self.base_dir = Path(base_dir).resolve()
        allowed_root = (self.base_dir / "data" / "intelligence").resolve()
        if db_path is None:
            resolved_db = allowed_root / "events.sqlite"
        else:
            resolved_db = Path(db_path).resolve()
        try:
            resolved_db.relative_to(allowed_root)
        except ValueError as exc:
            raise ValueError(
                "intelligence db_path must stay under data/intelligence"
            ) from exc
        self.db_path = resolved_db
        self.read_only = read_only
        if read_only:
            if not self.db_path.is_file():
                raise FileNotFoundError("intelligence event store does not exist")
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        if self.read_only:
            conn = sqlite3.connect(self.db_path.as_uri() + "?mode=ro", uri=True,
                                   timeout=5.0, factory=_ClosingConnection)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only = ON")
            return conn
        conn = sqlite3.connect(str(self.db_path), timeout=5.0, factory=_ClosingConnection)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    source_name TEXT NOT NULL,
                    source_document_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    url TEXT NOT NULL DEFAULT '',
                    published_at TEXT,
                    received_at TEXT,
                    available_at TEXT,
                    content_hash TEXT NOT NULL,
                    event_cluster_id TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    inserted_at TEXT NOT NULL,
                    UNIQUE(source_name, source_document_id, content_hash)
                );

                CREATE INDEX IF NOT EXISTS idx_documents_source
                    ON documents(source_name, inserted_at);
                CREATE INDEX IF NOT EXISTS idx_documents_cluster
                    ON documents(event_cluster_id);

                CREATE TABLE IF NOT EXISTS source_cursors (
                    source_name TEXT PRIMARY KEY,
                    cursor_value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS source_health (
                    source_name TEXT PRIMARY KEY,
                    last_success_at TEXT,
                    last_checked_at TEXT,
                    last_error TEXT NOT NULL DEFAULT '',
                    source_stale INTEGER NOT NULL DEFAULT 0,
                    content_stale INTEGER NOT NULL DEFAULT 0,
                    last_document_at TEXT,
                    content_baseline_at TEXT,
                    document_count INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS assessments (
                    assessment_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    event_cluster_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    policy_stance TEXT NOT NULL,
                    bond_impact TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence_json TEXT NOT NULL,
                    affected_assets_json TEXT NOT NULL,
                    tenors_json TEXT NOT NULL,
                    novelty REAL NOT NULL,
                    channels_json TEXT NOT NULL,
                    prompt_injection_flag INTEGER NOT NULL,
                    model_version TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    assessed_at TEXT NOT NULL,
                    expires_at TEXT,
                    missing_reason TEXT NOT NULL DEFAULT '',
                    raw_response_json TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id)
                );

                CREATE INDEX IF NOT EXISTS idx_assessments_cluster
                    ON assessments(event_cluster_id, assessed_at);

                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    event_cluster_id TEXT NOT NULL UNIQUE,
                    first_assessment_id TEXT NOT NULL,
                    last_assessment_id TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(first_assessment_id) REFERENCES assessments(assessment_id),
                    FOREIGN KEY(last_assessment_id) REFERENCES assessments(assessment_id)
                );

                CREATE TABLE IF NOT EXISTS monitor_runs (
                    run_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    sources_checked INTEGER NOT NULL,
                    sources_failed INTEGER NOT NULL,
                    sources_content_stale INTEGER NOT NULL DEFAULT 0,
                    documents_seen INTEGER NOT NULL,
                    documents_inserted INTEGER NOT NULL,
                    documents_dropped INTEGER NOT NULL DEFAULT 0,
                    assessments_created INTEGER NOT NULL,
                    alerts_opened INTEGER NOT NULL,
                    report_path TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS notification_outbox (
                    delivery_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    assessment_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at TEXT NOT NULL,
                    last_error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(assessment_id) REFERENCES assessments(assessment_id)
                );
                CREATE INDEX IF NOT EXISTS idx_outbox_due
                    ON notification_outbox(status, next_attempt_at);
                """
            )
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(assessments)").fetchall()
            }
            if "expires_at" not in columns:
                conn.execute("ALTER TABLE assessments ADD COLUMN expires_at TEXT")
            run_columns = {
                row["name"]
                for row in conn.execute(
                    "PRAGMA table_info(monitor_runs)"
                ).fetchall()
            }
            if "documents_dropped" not in run_columns:
                conn.execute(
                    "ALTER TABLE monitor_runs ADD COLUMN "
                    "documents_dropped INTEGER NOT NULL DEFAULT 0"
                )
            if "sources_content_stale" not in run_columns:
                conn.execute(
                    "ALTER TABLE monitor_runs ADD COLUMN "
                    "sources_content_stale INTEGER NOT NULL DEFAULT 0"
                )
            health_columns = {
                row["name"]
                for row in conn.execute(
                    "PRAGMA table_info(source_health)"
                ).fetchall()
            }
            if "content_stale" not in health_columns:
                conn.execute(
                    "ALTER TABLE source_health ADD COLUMN "
                    "content_stale INTEGER NOT NULL DEFAULT 0"
                )
            if "last_document_at" not in health_columns:
                conn.execute(
                    "ALTER TABLE source_health ADD COLUMN last_document_at TEXT"
                )
            if "content_baseline_at" not in health_columns:
                conn.execute(
                    "ALTER TABLE source_health ADD COLUMN content_baseline_at TEXT"
                )
        self._migrate_document_versions()

    def _migrate_document_versions(self) -> None:
        """Replace the old uniqueness constraint without changing historical IDs/FKs."""
        with self._connect() as conn:
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute("BEGIN IMMEDIATE")
            indexes = conn.execute("PRAGMA index_list(documents)").fetchall()
            old_constraint = any(
                [row["name"] for row in conn.execute(
                    'PRAGMA index_info("' + index["name"].replace('"', '""') + '")'
                )] == ["source_name", "source_document_id"]
                for index in indexes if index["unique"]
            )
            if old_constraint:
                conn.execute("""CREATE TABLE documents_v2 (
                    document_id TEXT PRIMARY KEY, source_name TEXT NOT NULL,
                    source_document_id TEXT NOT NULL, title TEXT NOT NULL,
                    content TEXT NOT NULL, url TEXT NOT NULL DEFAULT '',
                    published_at TEXT, received_at TEXT, available_at TEXT,
                    content_hash TEXT NOT NULL, event_cluster_id TEXT NOT NULL,
                    metadata_json TEXT NOT NULL, inserted_at TEXT NOT NULL,
                    UNIQUE(source_name, source_document_id, content_hash))""")
                conn.execute("INSERT INTO documents_v2 SELECT * FROM documents")
                conn.execute("DROP TABLE documents")
                conn.execute("ALTER TABLE documents_v2 RENAME TO documents")
                conn.execute("CREATE INDEX idx_documents_source ON documents(source_name, inserted_at)")
                conn.execute("CREATE INDEX idx_documents_cluster ON documents(event_cluster_id)")
            if conn.execute("PRAGMA foreign_key_check").fetchone():
                raise ValueError("intelligence ledger migration violates foreign keys")
            conn.commit()
            conn.execute("PRAGMA foreign_keys = ON")

    def commit_provider_batch(
        self,
        source_name: str,
        documents: Iterable[DocumentRecord],
        *,
        next_cursor: str | None,
        fetched_at: str | None = None,
        before_cursor_update: Callable[[], None] | None = None,
    ) -> list[DocumentRecord]:
        """Atomically persist raw documents and advance the provider cursor.

        If any step between document insertion and cursor update fails, both the
        newly inserted documents and the cursor update are rolled back.
        """

        checked_at = fetched_at or utc_now_iso()
        normalized_docs = [
            doc.normalized(source_name=source_name, now=checked_at)
            for doc in documents
        ]
        inserted: list[DocumentRecord] = []
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            for doc in normalized_docs:
                previous = conn.execute(
                    "SELECT * FROM documents WHERE source_name = ? AND source_document_id = ? "
                    "ORDER BY rowid DESC LIMIT 1", (source_name, doc.source_document_id)
                ).fetchone()
                if previous is not None and previous["content_hash"] != doc.content_hash:
                    previous_assessment = conn.execute(
                        "SELECT event_cluster_id FROM assessments WHERE document_id=? ORDER BY rowid DESC LIMIT 1",
                        (previous["document_id"],),
                    ).fetchone()
                    doc = replace(doc, event_cluster_id=(previous_assessment["event_cluster_id"]
                                  if previous_assessment else previous["event_cluster_id"]),
                                  metadata={**doc.metadata, "revision_of": previous["document_id"]})
                result = conn.execute(
                    """
                    INSERT OR IGNORE INTO documents (
                        document_id, source_name, source_document_id, title, content,
                        url, published_at, received_at, available_at, content_hash,
                        event_cluster_id, metadata_json, inserted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc.document_id,
                        doc.source_name,
                        doc.source_document_id,
                        doc.title,
                        doc.content,
                        doc.url,
                        doc.published_at,
                        doc.received_at,
                        doc.available_at,
                        doc.content_hash,
                        doc.event_cluster_id,
                        _stable_json(dict(doc.metadata)),
                        doc.inserted_at,
                    ),
                )
                if result.rowcount == 1:
                    inserted.append(doc)

            if before_cursor_update is not None:
                before_cursor_update()

            if next_cursor is not None:
                conn.execute(
                    """
                    INSERT INTO source_cursors(source_name, cursor_value, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(source_name) DO UPDATE SET
                        cursor_value=excluded.cursor_value,
                        updated_at=excluded.updated_at
                    """,
                    (source_name, next_cursor, checked_at),
                )

            source_document_count = conn.execute(
                "SELECT COUNT(*) AS n FROM documents WHERE source_name = ?",
                (source_name,),
            ).fetchone()["n"]
            previous_health = conn.execute(
                "SELECT * FROM source_health WHERE source_name = ?",
                (source_name,),
            ).fetchone()
            last_document_at = (
                checked_at
                if inserted
                else (
                    previous_health["last_document_at"]
                    if previous_health is not None else None
                )
            )
            previous_content_baseline = (
                previous_health["content_baseline_at"]
                if previous_health is not None else None
            )
            content_baseline_at = (
                checked_at
                if inserted
                else (
                    previous_content_baseline
                    or last_document_at
                    or (
                        previous_health["last_success_at"]
                        if previous_health is not None else None
                    )
                    or checked_at
                )
            )
            checked_dt = _parse_iso(checked_at)
            baseline_dt = _parse_iso(content_baseline_at)
            content_stale = bool(
                checked_dt is not None
                and baseline_dt is not None
                and (checked_dt - baseline_dt).total_seconds()
                > SOURCE_CONTENT_STALE_AFTER_SECONDS
            )

            conn.execute(
                """
                INSERT INTO source_health(
                    source_name, last_success_at, last_checked_at,
                    last_error, source_stale, content_stale,
                    last_document_at, content_baseline_at, document_count
                ) VALUES (?, ?, ?, '', 0, ?, ?, ?, ?)
                ON CONFLICT(source_name) DO UPDATE SET
                    last_success_at=excluded.last_success_at,
                    last_checked_at=excluded.last_checked_at,
                    last_error='',
                    source_stale=0,
                    content_stale=excluded.content_stale,
                    last_document_at=excluded.last_document_at,
                    content_baseline_at=excluded.content_baseline_at,
                    document_count=excluded.document_count
                """,
                (
                    source_name,
                    checked_at,
                    checked_at,
                    1 if content_stale else 0,
                    last_document_at,
                    content_baseline_at,
                    source_document_count,
                ),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return inserted

    def record_source_failure(
        self,
        source_name: str,
        *,
        error: str,
        checked_at: str | None = None,
    ) -> None:
        timestamp = checked_at or utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO source_health(
                    source_name, last_success_at, last_checked_at,
                    last_error, source_stale, document_count
                ) VALUES (?, NULL, ?, ?, 1, 0)
                ON CONFLICT(source_name) DO UPDATE SET
                    last_checked_at=excluded.last_checked_at,
                    last_error=excluded.last_error,
                    source_stale=1
                """,
                (source_name, timestamp, _redact_error(error)),
            )

    def get_cursor(self, source_name: str) -> SourceCursor | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM source_cursors WHERE source_name = ?",
                (source_name,),
            ).fetchone()
        if row is None:
            return None
        return SourceCursor(
            source_name=row["source_name"],
            cursor_value=row["cursor_value"],
            updated_at=row["updated_at"],
        )

    def get_source_health(self, source_name: str) -> SourceHealth | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM source_health WHERE source_name = ?",
                (source_name,),
            ).fetchone()
        return self._row_to_health(row) if row is not None else None

    def list_source_health(self) -> list[SourceHealth]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM source_health ORDER BY source_name"
            ).fetchall()
        return [self._row_to_health(row) for row in rows]

    def list_documents(
        self,
        *,
        source_name: str | None = None,
        limit: int | None = None,
    ) -> list[DocumentRecord]:
        sql = "SELECT * FROM documents"
        params: list[Any] = []
        if source_name:
            sql += " WHERE source_name = ?"
            params.append(source_name)
        sql += " ORDER BY inserted_at, document_id"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_document(row) for row in rows]

    def list_unassessed_documents(
        self,
        *,
        source_name: str | None = None,
        limit: int | None = None,
        model_version_prefix: str | None = None,
        resume_semantic_budget: bool = False,
    ) -> list[DocumentRecord]:
        version_clause = " AND a.model_version LIKE ?" if model_version_prefix else ""
        budget_clause = (" AND COALESCE(json_extract(a.raw_response_json, '$.semantic.status'), '') "
                         "!= 'budget_exhausted'") if resume_semantic_budget else ""
        sql = f"""
            SELECT d.* FROM documents d
            WHERE NOT EXISTS (SELECT 1 FROM assessments a WHERE a.document_id = d.document_id
                              {version_clause} {budget_clause})
        """
        params: list[Any] = []
        if model_version_prefix:
            params.append(model_version_prefix + "%")
        if source_name:
            sql += " AND d.source_name = ?"
            params.append(source_name)
        sql += " ORDER BY d.inserted_at, d.document_id"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_document(row) for row in rows]

    @staticmethod
    def _insert_assessment(conn, assessment: EventAssessment) -> None:
        conn.execute(
            """
            INSERT INTO assessments (
                assessment_id, document_id, event_cluster_id, event_type,
                severity, policy_stance, bond_impact, confidence,
                evidence_json, affected_assets_json, tenors_json, novelty,
                channels_json, prompt_injection_flag, model_version,
                schema_version, assessed_at, expires_at, missing_reason,
                raw_response_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                assessment.assessment_id,
                assessment.document_id,
                assessment.event_cluster_id,
                assessment.event_type,
                assessment.severity,
                assessment.policy_stance,
                assessment.bond_impact,
                float(assessment.confidence),
                _stable_json(assessment.evidence),
                _stable_json(assessment.affected_assets),
                _stable_json(assessment.tenors),
                float(assessment.novelty),
                _stable_json(dict(assessment.channels)),
                1 if assessment.prompt_injection_flag else 0,
                assessment.model_version,
                assessment.schema_version,
                assessment.assessed_at,
                assessment.expires_at,
                assessment.missing_reason,
                _stable_json(_sanitize_mapping(assessment.raw_response)),
            ),
        )

    @staticmethod
    def _upsert_alert_conn(conn, assessment: EventAssessment) -> str:
        timestamp = utc_now_iso()
        alert_id = _hash_parts("alert", assessment.event_cluster_id, length=32)
        reason = (
            f"{assessment.event_type}:{assessment.policy_stance}/"
            f"{assessment.bond_impact}"
        )
        existing = conn.execute(
            "SELECT * FROM alerts WHERE event_cluster_id = ?",
            (assessment.event_cluster_id,),
        ).fetchone()
        if existing is None:
            conn.execute(
                """
                INSERT INTO alerts (
                    alert_id, event_cluster_id, first_assessment_id,
                    last_assessment_id, severity, status, reason,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'open', ?, ?, ?)
                """,
                (
                    alert_id,
                    assessment.event_cluster_id,
                    assessment.assessment_id,
                    assessment.assessment_id,
                    assessment.severity,
                    reason,
                    timestamp,
                    timestamp,
                ),
            )
        else:
            alert_id = existing["alert_id"]
            severity = _max_severity(existing["severity"], assessment.severity)
            conn.execute(
                """
                UPDATE alerts
                SET last_assessment_id = ?, severity = ?, status = 'open',
                    reason = ?, updated_at = ?
                WHERE alert_id = ?
                """,
                (
                    assessment.assessment_id,
                    severity,
                    reason,
                    timestamp,
                    alert_id,
                ),
            )
        return str(alert_id)

    def record_assessment(
        self,
        assessment: EventAssessment,
        *,
        open_alert: bool = False,
    ) -> tuple[str, bool, Alert | None]:
        """Insert an assessment and its optional alert in one write transaction."""
        assessment = assessment.normalized()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM assessments WHERE assessment_id = ?",
                (assessment.assessment_id,),
            ).fetchone()
            if existing is not None:
                assessment = self._row_to_assessment(existing)
                created = False
            else:
                cluster_exists = conn.execute(
                    "SELECT * FROM assessments "
                    "WHERE event_cluster_id = ? ORDER BY rowid DESC LIMIT 1",
                    (assessment.event_cluster_id,),
                ).fetchone()
                changed = cluster_exists is not None and (
                    _safe_json_loads(cluster_exists["raw_response_json"], {}).get("fact_status")
                    != assessment.raw_response.get("fact_status")
                    or cluster_exists["bond_impact"] != assessment.bond_impact
                )
                if cluster_exists is not None and not changed and assessment.novelty != 0.0:
                    assessment.novelty = 0.0
                    assessment.assessment_id = ""
                    assessment = assessment.normalized()
                    existing = conn.execute(
                        "SELECT * FROM assessments WHERE assessment_id = ?",
                        (assessment.assessment_id,),
                    ).fetchone()
                if existing is not None:
                    assessment = self._row_to_assessment(existing)
                    created = False
                else:
                    self._insert_assessment(conn, assessment)
                    created = True
            alert_id = None
            if open_alert:
                if assessment.prompt_injection_flag:
                    raise ValueError("quarantined assessment cannot open an alert")
                alert_id = self._upsert_alert_conn(conn, assessment)
            alert_row = (
                conn.execute(
                    "SELECT * FROM alerts WHERE alert_id = ?", (alert_id,)
                ).fetchone()
                if alert_id is not None else None
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        alert = self._row_to_alert(alert_row) if alert_row is not None else None
        return assessment.assessment_id, created, alert

    def add_assessment(self, assessment: EventAssessment) -> str:
        assessment_id, _, _ = self.record_assessment(assessment)
        return assessment_id

    def has_assessment_for_cluster(self, event_cluster_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM assessments WHERE event_cluster_id = ? LIMIT 1",
                (event_cluster_id,),
            ).fetchone()
        return row is not None

    def get_assessment(self, assessment_id: str) -> EventAssessment | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM assessments WHERE assessment_id = ?",
                (assessment_id,),
            ).fetchone()
        return self._row_to_assessment(row) if row is not None else None

    def list_assessments(self) -> list[EventAssessment]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM assessments ORDER BY assessed_at, assessment_id"
            ).fetchall()
        return [self._row_to_assessment(row) for row in rows]

    def recent_event_records(self, limit: int = 200) -> list[dict]:
        """Bounded evidence view; full revision history remains in the ledger."""
        from dataclasses import asdict

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT a.*, d.title, d.content, d.url, d.source_name, d.published_at, "
                "d.available_at, d.metadata_json FROM assessments a "
                "JOIN documents d ON d.document_id = a.document_id "
                "WHERE NOT EXISTS (SELECT 1 FROM assessments newer WHERE newer.document_id=a.document_id "
                "AND newer.rowid > a.rowid) "
                "ORDER BY d.rowid DESC, a.rowid DESC LIMIT ?", (max(1, min(int(limit), 1000)),)
            ).fetchall()
        return [{**asdict(self._row_to_assessment(row)),
                 **{key: row[key] for key in ("title", "content", "url", "source_name", "published_at", "available_at")},
                 "document_metadata": _safe_json_loads(row["metadata_json"], {})} for row in rows]

    def enqueue_notification(self, *, delivery_id, task_id, channel, assessment_id, payload, now):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO notification_outbox "
                "(delivery_id, task_id, channel, assessment_id, payload_json, next_attempt_at, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(delivery_id) DO UPDATE SET "
                "assessment_id=excluded.assessment_id, payload_json=excluded.payload_json "
                "WHERE notification_outbox.status IN ('pending', 'failed')",
                (delivery_id, task_id, channel, assessment_id, _stable_json(_sanitize_mapping(payload)), now, now, now),
            )

    def list_deliveries(self, limit=100) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM notification_outbox ORDER BY rowid DESC LIMIT ?",
                                (max(1, min(int(limit), 1000)),)).fetchall()
        return [{**dict(row), "payload": _safe_json_loads(row["payload_json"], {})} for row in rows]

    def claim_delivery(self, *, channel, now, lease_until, active_tasks) -> dict | None:
        """Lease a due item so concurrent monitors do not both send it."""
        if not active_tasks:
            return None
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            placeholders = ",".join("?" for _ in active_tasks)
            row = conn.execute(
                f"SELECT * FROM notification_outbox WHERE channel = ? AND task_id IN ({placeholders}) "
                "AND status IN ('pending', 'failed', 'sending') AND attempts < 5 "
                "AND julianday(next_attempt_at) <= julianday(?) ORDER BY rowid LIMIT 1",
                (channel, *active_tasks, now),
            ).fetchone()
            if row is None:
                return None
            conn.execute("UPDATE notification_outbox SET status='sending', attempts=attempts+1, "
                         "next_attempt_at=?, updated_at=? WHERE delivery_id=?",
                         (lease_until, now, row["delivery_id"]))
        return {**dict(row), "attempts": row["attempts"] + 1,
                "payload": _safe_json_loads(row["payload_json"], {})}

    def finish_delivery(self, delivery_id, *, status, now, next_attempt_at=None, error=""):
        if status not in {"sent", "failed", "local", "cancelled"}:
            raise ValueError("invalid delivery status")
        with self._connect() as conn:
            conn.execute("UPDATE notification_outbox SET status=?, updated_at=?, next_attempt_at=?, "
                         "last_error=? WHERE delivery_id=? AND status='sending'",
                         (status, now, next_attempt_at or now, _redact_error(error), delivery_id))

    def get_document(self, document_id: str) -> DocumentRecord | None:
        """Retrieve the exact immutable source revision for an assessment."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE document_id = ?", (document_id,)).fetchone()
        return self._row_to_document(row) if row is not None else None

    def query_assessments(
        self,
        *,
        as_of: str,
        assets: Iterable[str] | None = None,
        limit: int = 100,
    ) -> list[EventAssessment]:
        """Return only events available and unexpired at ``as_of``."""
        as_of_dt = _parse_iso(as_of)
        if as_of_dt is None:
            raise ValueError("as_of must be an ISO timestamp")
        requested = sorted({str(asset) for asset in (assets or []) if str(asset)})
        bounded_limit = max(1, min(int(limit), 1000))
        asset_clause = ""
        params: list[Any] = [as_of] * 6
        if requested:
            placeholders = ",".join("?" for _ in requested)
            asset_clause = f"""
                  AND EXISTS (
                    SELECT 1
                    FROM json_each(a.affected_assets_json)
                    WHERE json_each.value IN ({placeholders})
                  )
            """
            params.extend(requested)
        params.append(bounded_limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT a.*
                FROM assessments a
                JOIN documents d ON d.document_id = a.document_id
                WHERE d.available_at IS NOT NULL
                  AND julianday(d.available_at) <= julianday(?)
                  AND julianday(a.assessed_at) <= julianday(?)
                  AND (
                    a.expires_at IS NULL
                    OR julianday(a.expires_at) > julianday(?)
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM documents newer
                    JOIN assessments newer_a ON newer_a.document_id = newer.document_id
                    WHERE newer.source_name = d.source_name
                      AND newer.source_document_id = d.source_document_id
                      AND newer.rowid > d.rowid
                      AND julianday(newer.available_at) <= julianday(?)
                      AND julianday(newer_a.assessed_at) <= julianday(?)
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM assessments newer_assessment
                    WHERE newer_assessment.document_id = a.document_id
                      AND newer_assessment.rowid > a.rowid
                      AND julianday(newer_assessment.assessed_at) <= julianday(?)
                  )
                {asset_clause}
                ORDER BY julianday(d.available_at) DESC,
                         julianday(a.assessed_at) DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [self._row_to_assessment(row) for row in rows]

    def upsert_alert_for_assessment(self, assessment_id: str) -> Alert:
        assessment = self.get_assessment(assessment_id)
        if assessment is None:
            raise KeyError(f"assessment not found: {assessment_id}")
        if assessment.prompt_injection_flag:
            raise ValueError("quarantined assessment cannot open an alert")
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            alert_id = self._upsert_alert_conn(conn, assessment)
            row = conn.execute(
                "SELECT * FROM alerts WHERE alert_id = ?", (alert_id,)
            ).fetchone()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        if row is None:
            raise RuntimeError("alert upsert failed")
        return self._row_to_alert(row)

    def set_alert_status(self, alert_id: str, status: str) -> None:
        if status not in ALERT_STATUSES:
            raise ValueError(f"invalid alert status: {status}")
        with self._connect() as conn:
            result = conn.execute(
                """
                UPDATE alerts
                SET status = ?, updated_at = ?
                WHERE alert_id = ?
                """,
                (status, utc_now_iso(), alert_id),
            )
        if result.rowcount != 1:
            raise KeyError(f"alert not found: {alert_id}")

    def get_alert(self, alert_id: str) -> Alert | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM alerts WHERE alert_id = ?",
                (alert_id,),
            ).fetchone()
        return self._row_to_alert(row) if row is not None else None

    def list_alerts(self) -> list[Alert]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM alerts ORDER BY updated_at, alert_id"
            ).fetchall()
        return [self._row_to_alert(row) for row in rows]

    def record_monitor_run(self, run: MonitorRun) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO monitor_runs (
                    run_id, started_at, finished_at, status, sources_checked,
                    sources_failed, sources_content_stale, documents_seen,
                    documents_inserted,
                    documents_dropped, assessments_created, alerts_opened,
                    report_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.started_at,
                    run.finished_at,
                    run.status,
                    run.sources_checked,
                    run.sources_failed,
                    run.sources_content_stale,
                    run.documents_seen,
                    run.documents_inserted,
                    run.documents_dropped,
                    run.assessments_created,
                    run.alerts_opened,
                    run.report_path,
                ),
            )

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            document_id=row["document_id"],
            source_name=row["source_name"],
            source_document_id=row["source_document_id"],
            title=row["title"],
            content=row["content"],
            url=row["url"],
            published_at=row["published_at"],
            received_at=row["received_at"],
            available_at=row["available_at"],
            content_hash=row["content_hash"],
            event_cluster_id=row["event_cluster_id"],
            metadata=_safe_json_loads(row["metadata_json"], {}),
            inserted_at=row["inserted_at"],
        )

    @staticmethod
    def _row_to_health(row: sqlite3.Row) -> SourceHealth:
        return SourceHealth(
            source_name=row["source_name"],
            last_success_at=row["last_success_at"],
            last_checked_at=row["last_checked_at"],
            last_error=row["last_error"],
            source_stale=bool(row["source_stale"]),
            content_stale=bool(row["content_stale"]),
            last_document_at=row["last_document_at"],
            content_baseline_at=row["content_baseline_at"],
            document_count=int(row["document_count"]),
        )

    @staticmethod
    def _row_to_assessment(row: sqlite3.Row) -> EventAssessment:
        return EventAssessment(
            assessment_id=row["assessment_id"],
            document_id=row["document_id"],
            event_cluster_id=row["event_cluster_id"],
            event_type=row["event_type"],
            severity=row["severity"],
            policy_stance=row["policy_stance"],
            bond_impact=row["bond_impact"],
            confidence=float(row["confidence"]),
            evidence=_safe_json_loads(row["evidence_json"], []),
            affected_assets=_safe_json_loads(row["affected_assets_json"], []),
            tenors=_safe_json_loads(row["tenors_json"], []),
            novelty=float(row["novelty"]),
            channels=_safe_json_loads(row["channels_json"], {}),
            prompt_injection_flag=bool(row["prompt_injection_flag"]),
            model_version=row["model_version"],
            schema_version=row["schema_version"],
            assessed_at=row["assessed_at"],
            expires_at=row["expires_at"],
            missing_reason=row["missing_reason"],
            raw_response=_safe_json_loads(row["raw_response_json"], {}),
        )

    @staticmethod
    def _row_to_alert(row: sqlite3.Row) -> Alert:
        return Alert(
            alert_id=row["alert_id"],
            event_cluster_id=row["event_cluster_id"],
            first_assessment_id=row["first_assessment_id"],
            last_assessment_id=row["last_assessment_id"],
            severity=row["severity"],
            status=row["status"],
            reason=row["reason"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


def _max_severity(left: str, right: str) -> str:
    order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    return left if order.get(left, 0) >= order.get(right, 0) else right
