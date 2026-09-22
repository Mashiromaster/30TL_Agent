# -*- coding: utf-8 -*-
"""Minimal TL sentiment monitoring loop.

The monitor is a research/intelligence component only:

provider -> DocumentRecord -> EventAssessment -> de-dup/event cluster
-> Alert/report -> later ResearchAgent integration.

It deliberately does not import trading, broker, inference, or model release
modules.  Default persistence is limited to ``data/intelligence`` and
``outputs/intelligence`` under the supplied base directory.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol
from uuid import uuid4

from intelligence_events import (
    DEFAULT_RULE_MODEL_VERSION,
    DocumentRecord,
    EventAssessment,
    IntelligenceEventStore,
    MonitorRun,
    SourceCursor,
    redact_sensitive_text,
    utc_now_iso,
)


PROMPT_INJECTION_PATTERNS = {
    "ignore_instructions": re.compile(
        r"(ignore\s+(all\s+)?(previous|above)\s+instructions|忽略.{0,12}(指令|规则|系统))",
        re.IGNORECASE,
    ),
    "system_prompt": re.compile(r"(system\s*prompt|系统提示词|开发者消息)", re.IGNORECASE),
    "tool_call": re.compile(r"(tool_calls?|function_call|调用工具|执行工具)", re.IGNORECASE),
    "trade_command": re.compile(r"(place_order|下单|立刻买入|立即买入|卖出开仓|买入开仓)", re.IGNORECASE),
}


@dataclass(frozen=True)
class ProviderDocument:
    source_document_id: str
    title: str
    content: str
    url: str = ""
    published_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderFetchResult:
    documents: list[ProviderDocument]
    next_cursor: str | None
    fetched_at: str | None = None


class NewsProvider(Protocol):
    name: str

    def fetch_since(self, cursor: SourceCursor | None) -> ProviderFetchResult:
        ...


class ScriptedNewsProvider:
    """Deterministic provider for tests and dry-run demos."""

    def __init__(self, name: str, batches: Iterable[ProviderFetchResult | Exception]):
        self.name = name
        self._batches = list(batches)
        self.cursor_values_seen: list[str | None] = []

    def fetch_since(self, cursor: SourceCursor | None) -> ProviderFetchResult:
        self.cursor_values_seen.append(cursor.cursor_value if cursor else None)
        if not self._batches:
            return ProviderFetchResult([], next_cursor=cursor.cursor_value if cursor else None)
        batch = self._batches.pop(0)
        if isinstance(batch, Exception):
            raise batch
        return batch


class EastmoneyTLNewsProvider:
    """Default 东方财富 TL/bond news provider.

    ``akshare`` is imported only inside ``fetch_since`` so importing this module
    never performs network work.  Tests should use ``ScriptedNewsProvider``.
    """

    name = "eastmoney_tl"
    keywords = (
        "国债",
        "TL",
        "30年",
        "债券",
        "利率",
        "央行",
        "收益率",
        "货币",
        "LPR",
        "MLF",
        "逆回购",
        "降准",
        "降息",
        "债市",
        "国债期货",
        "公开市场",
        "OMO",
        "流动性",
        "资金面",
        "地方债",
        "专项债",
    )

    def fetch_since(self, cursor: SourceCursor | None) -> ProviderFetchResult:
        import pandas as pd

        import akshare as ak

        # stock_news_em() is a single-equity endpoint whose default symbol is
        # 603777.  The global 7x24 feed is the appropriate broad macro source
        # for the TL/bond keyword filter below.
        df = ak.stock_info_global_em()
        if df is None or len(df) == 0:
            return ProviderFetchResult([], next_cursor=cursor.cursor_value if cursor else None)

        df = df.copy()
        col_map: dict[Any, str] = {}
        for col in df.columns:
            name = str(col).lower()
            if "标题" in str(col) or "title" in name:
                col_map[col] = "title"
            elif (
                "内容" in str(col)
                or "摘要" in str(col)
                or "content" in name
                or "summary" in name
            ):
                col_map[col] = "content"
            elif "时间" in str(col) or "date" in name or "time" in name:
                col_map[col] = "published_at"
            elif "来源" in str(col) or "source" in name:
                col_map[col] = "source"
            elif "链接" in str(col) or "url" in name:
                col_map[col] = "url"
        df = df.rename(columns=col_map)
        for required in ("title", "content"):
            if required not in df.columns:
                df[required] = ""
        if "url" not in df.columns:
            df["url"] = ""
        def normalize_published(value):
            try:
                timestamp = pd.Timestamp(value)
            except (TypeError, ValueError):
                return pd.NaT
            if pd.isna(timestamp):
                return pd.NaT
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("Asia/Shanghai")
            return timestamp.tz_convert("UTC")

        if "published_at" in df.columns:
            df["published_at"] = df["published_at"].map(normalize_published)
        else:
            df["published_at"] = pd.NaT

        pattern = "|".join(map(re.escape, self.keywords))
        mask = df["title"].astype(str).str.contains(pattern, case=False, na=False)
        mask |= df["content"].astype(str).str.contains(pattern, case=False, na=False)
        df = df[mask].copy()

        cursor_value = cursor.cursor_value if cursor else None
        cursor_timestamp = cursor_value
        if cursor_value:
            try:
                decoded_cursor = json.loads(cursor_value)
            except (TypeError, json.JSONDecodeError):
                decoded_cursor = None
            if isinstance(decoded_cursor, dict):
                cursor_timestamp = decoded_cursor.get("published_at")
        if cursor_timestamp:
            cursor_ts = normalize_published(cursor_timestamp)
            if pd.notna(cursor_ts):
                # Re-read the cursor timestamp inclusively.  Stable document
                # IDs plus the store's unique constraint de-duplicate repeats,
                # while a second item published at the same timestamp cannot
                # be lost between polls.
                df = df[df["published_at"].isna() | (df["published_at"] >= cursor_ts)]

        docs: list[ProviderDocument] = []
        max_seen: tuple[Any, str] | None = None
        for idx, row in df.iterrows():
            published = row.get("published_at")
            published_at = (
                published.tz_convert("Asia/Shanghai").isoformat()
                if pd.notna(published)
                else None
            )
            title = str(row.get("title", "") or "")
            content = str(row.get("content", "") or "")
            url = str(row.get("url", "") or "")
            stable_material = "\n".join((published_at or "", title, content, url))
            source_document_id = url or (
                "eastmoney-fastnews:"
                f"{sha256(stable_material.encode('utf-8')).hexdigest()}"
            )
            if published_at:
                candidate_cursor = (pd.Timestamp(published), source_document_id)
                if max_seen is None or candidate_cursor > max_seen:
                    max_seen = candidate_cursor
            docs.append(
                ProviderDocument(
                    source_document_id=source_document_id,
                    title=title,
                    content=content,
                    url=url,
                    published_at=published_at,
                    metadata={"provider": self.name, "source": str(row.get("source", "") or "")},
                )
            )
        next_cursor = cursor_value
        if max_seen is not None:
            next_cursor = json.dumps(
                {
                    "published_at": max_seen[0].isoformat(),
                    "source_document_id": max_seen[1],
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        return ProviderFetchResult(docs, next_cursor=next_cursor, fetched_at=utc_now_iso())


class TLRuleAnalyzer:
    easing_keywords = ("降准", "降息", "净投放", "释放流动性", "宽松", "呵护流动性")
    tightening_keywords = ("加息", "升息", "提准", "净回笼", "收紧", "资金面收紧")
    bullish_keywords = ("利好", "收益率下行", "债市走强", "资金面宽松", "降准", "降息")
    bearish_keywords = ("利空", "收益率上行", "债市调整", "供给压力", "增发", "通胀")
    liquidity_keywords = ("流动性", "资金面", "MLF", "OMO", "逆回购", "降准", "净投放")
    growth_keywords = ("PMI", "经济", "增长", "地产", "消费", "投资")
    inflation_keywords = ("CPI", "PPI", "通胀", "价格")
    supply_keywords = ("地方债", "专项债", "供给", "发行", "增发")

    def assess(self, document: DocumentRecord) -> EventAssessment:
        text = f"{document.title}\n{document.content}"
        policy_stance = self._choice(
            text,
            positive=self.easing_keywords,
            negative=self.tightening_keywords,
            positive_label="easing",
            negative_label="tightening",
            neutral_label="neutral",
        )
        bond_impact = self._choice(
            text,
            positive=self.bullish_keywords,
            negative=self.bearish_keywords,
            positive_label="bullish",
            negative_label="bearish",
            neutral_label="neutral",
        )
        channels = {
            "liquidity": self._channel_score(text, self.liquidity_keywords),
            "growth": self._channel_score(text, self.growth_keywords),
            "inflation": self._channel_score(text, self.inflation_keywords),
            "supply": self._channel_score(text, self.supply_keywords),
        }
        prompt_patterns = [
            name for name, pattern in PROMPT_INJECTION_PATTERNS.items() if pattern.search(text)
        ]
        matched = self._matched_keywords(text)
        event_type = "policy_liquidity" if channels["liquidity"] else "market_news"
        severity = self._severity(policy_stance, bond_impact, channels, matched)
        confidence = self._confidence(severity, matched)
        assets = self._assets(text)
        tenors = self._tenors(text)
        missing_reason = ""
        if prompt_patterns:
            # Quarantined text remains available for audit, but it must not
            # create directional intelligence, alerts, or downstream model
            # input.  Regexes are telemetry; the authority boundary is the
            # quarantine behavior itself.
            event_type = "quarantined_untrusted_content"
            severity = "low"
            confidence = 0.0
            policy_stance = "unclear"
            bond_impact = "unclear"
            channels = {name: 0.0 for name in channels}
            matched = []
            missing_reason = "prompt_injection_quarantined"

        return EventAssessment(
            document_id=document.document_id,
            event_cluster_id=document.event_cluster_id,
            event_type=event_type,
            severity=severity,
            policy_stance=policy_stance,
            bond_impact=bond_impact,
            confidence=confidence,
            evidence=self._evidence(text, matched),
            affected_assets=assets,
            tenors=tenors,
            novelty=1.0,
            channels=channels,
            prompt_injection_flag=bool(prompt_patterns),
            model_version=DEFAULT_RULE_MODEL_VERSION,
            missing_reason=missing_reason,
            raw_response={
                "source": "deterministic_tl_rules",
                "matched_keywords": matched,
                "prompt_injection_patterns": prompt_patterns,
                "scores": {
                    "policy_stance": policy_stance,
                    "bond_impact": bond_impact,
                    "severity": severity,
                },
            },
        )

    @staticmethod
    def _contains_any(text: str, keywords: Iterable[str]) -> bool:
        lower = text.lower()
        return any(keyword.lower() in lower for keyword in keywords)

    def _choice(
        self,
        text: str,
        *,
        positive: Iterable[str],
        negative: Iterable[str],
        positive_label: str,
        negative_label: str,
        neutral_label: str,
    ) -> str:
        has_positive = self._contains_any(text, positive)
        has_negative = self._contains_any(text, negative)
        if has_positive and not has_negative:
            return positive_label
        if has_negative and not has_positive:
            return negative_label
        if has_positive and has_negative:
            return "mixed"
        return neutral_label

    def _channel_score(self, text: str, keywords: Iterable[str]) -> float:
        return 1.0 if self._contains_any(text, keywords) else 0.0

    def _matched_keywords(self, text: str) -> list[str]:
        pool = (
            self.easing_keywords
            + self.tightening_keywords
            + self.bullish_keywords
            + self.bearish_keywords
            + self.liquidity_keywords
            + self.growth_keywords
            + self.inflation_keywords
            + self.supply_keywords
        )
        lower = text.lower()
        return sorted({keyword for keyword in pool if keyword.lower() in lower})

    @staticmethod
    def _severity(
        policy_stance: str,
        bond_impact: str,
        channels: Mapping[str, float],
        matched: list[str],
    ) -> str:
        if (
            policy_stance in {"easing", "tightening"}
            and bond_impact in {"bullish", "bearish"}
            and channels.get("liquidity", 0) > 0
        ):
            return "high"
        if matched and bond_impact in {"bullish", "bearish", "mixed"}:
            return "medium"
        return "low"

    @staticmethod
    def _confidence(severity: str, matched: list[str]) -> float:
        base = {"high": 0.82, "medium": 0.68, "low": 0.45}.get(severity, 0.4)
        return min(0.95, base + min(len(matched), 4) * 0.02)

    @staticmethod
    def _assets(text: str) -> list[str]:
        if re.search(r"(TL|30年|三十年|超长期|国债期货)", text, re.IGNORECASE):
            return ["TL"]
        if "国债" in text or "债券" in text or "债市" in text:
            return ["CGB"]
        return []

    @staticmethod
    def _tenors(text: str) -> list[str]:
        if re.search(r"(TL|30年|三十年|超长期)", text, re.IGNORECASE):
            return ["30Y"]
        if re.search(r"(10年|十年|T合约)", text, re.IGNORECASE):
            return ["10Y"]
        if re.search(r"(5年|五年|TF合约)", text, re.IGNORECASE):
            return ["5Y"]
        return []

    @staticmethod
    def _evidence(text: str, matched: list[str]) -> list[str]:
        if not matched:
            return []
        sentences = re.split(r"[。！？!?；;\n]+", text)
        evidence: list[str] = []
        for sentence in sentences:
            stripped = sentence.strip()
            if not stripped:
                continue
            if any(keyword in stripped for keyword in matched):
                evidence.append(stripped[:240])
            if len(evidence) >= 3:
                break
        return evidence


class SentimentMonitor:
    MAX_DOCUMENTS_PER_PROVIDER = 200
    MAX_BATCH_BYTES = 2 * 1024 * 1024
    MAX_TITLE_CHARS = 500
    MAX_CONTENT_CHARS = 20_000
    MAX_URL_CHARS = 2_048
    MAX_METADATA_BYTES = 8_192
    MAX_UNASSESSED_PER_PROVIDER = 500

    def __init__(
        self,
        base_dir: str | Path,
        *,
        providers: Iterable[NewsProvider] | None = None,
        store: IntelligenceEventStore | None = None,
        analyzer: TLRuleAnalyzer | None = None,
        jev_adapter: Any | None = None,
        now_fn=utc_now_iso,
    ):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data" / "intelligence"
        self.output_dir = self.base_dir / "outputs" / "intelligence"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.store = store or IntelligenceEventStore(self.base_dir)
        self.providers = list(providers) if providers is not None else [EastmoneyTLNewsProvider()]
        self.analyzer = analyzer or TLRuleAnalyzer()
        self.jev_adapter = jev_adapter
        self.now_fn = now_fn

    def run_once(self) -> MonitorRun:
        started_at = self.now_fn()
        run = MonitorRun(run_id=self._run_id(started_at), started_at=started_at)
        alert_clusters_touched: set[str] = set()
        for provider in self.providers:
            run.sources_checked += 1
            cursor = self.store.get_cursor(provider.name)
            try:
                result = provider.fetch_since(cursor)
            except Exception as exc:
                run.sources_failed += 1
                self.store.record_source_failure(
                    provider.name,
                    error=redact_sensitive_text(exc),
                    checked_at=self.now_fn(),
                )
                continue

            fetched_at = result.fetched_at or self.now_fn()
            run.documents_seen += len(result.documents)
            bounded_documents, dropped = self._bound_provider_documents(
                result.documents
            )
            run.documents_dropped += dropped
            records = [
                self._to_document_record(provider.name, document, fetched_at=fetched_at)
                for document in bounded_documents
            ]
            inserted = self.store.commit_provider_batch(
                provider.name,
                records,
                next_cursor=result.next_cursor,
                fetched_at=fetched_at,
            )
            run.documents_inserted += len(inserted)
            source_health = self.store.get_source_health(provider.name)
            if source_health is not None and source_health.content_stale:
                run.sources_content_stale += 1

            for document in self.store.list_unassessed_documents(
                source_name=provider.name,
                limit=self.MAX_UNASSESSED_PER_PROVIDER,
            ):
                assessment = self._assess(document)
                should_alert = (
                    not assessment.prompt_injection_flag
                    and assessment.severity in {"medium", "high", "critical"}
                )
                _assessment_id, created, alert = self.store.record_assessment(
                    assessment, open_alert=should_alert
                )
                run.assessments_created += int(created)
                if alert is not None and created:
                    if assessment.event_cluster_id not in alert_clusters_touched:
                        alert_clusters_touched.add(assessment.event_cluster_id)
                        run.alerts_opened += 1

        run.finished_at = self.now_fn()
        if run.sources_checked == 0:
            run.status = "failed"
        elif run.sources_failed == run.sources_checked:
            run.status = "failed"
        elif (
            run.sources_failed
            or run.sources_content_stale
            or run.documents_dropped
        ):
            run.status = "partial_failure"
        else:
            run.status = "ok"
        report_path = self._write_report(run)
        run.report_path = str(report_path)
        self.store.record_monitor_run(run)
        return run

    def _bound_provider_documents(self, documents):
        bounded = []
        dropped = 0
        total_bytes = 0
        for index, document in enumerate(documents):
            if index >= self.MAX_DOCUMENTS_PER_PROVIDER:
                dropped += 1
                continue
            title = str(document.title or "")[: self.MAX_TITLE_CHARS]
            content = str(document.content or "")[: self.MAX_CONTENT_CHARS]
            url = str(document.url or "")[: self.MAX_URL_CHARS]
            source_document_id = str(document.source_document_id or "")
            if len(source_document_id) > 512:
                source_document_id = (
                    "bounded:"
                    f"{sha256(source_document_id.encode('utf-8')).hexdigest()}"
                )
            try:
                metadata_text = json.dumps(
                    dict(document.metadata),
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
            except (TypeError, ValueError):
                metadata_text = "{}"
            if len(metadata_text.encode("utf-8")) > self.MAX_METADATA_BYTES:
                metadata = {"metadata_truncated": True}
            else:
                metadata = json.loads(metadata_text)
            document_bytes = len(
                "\n".join((title, content, url)).encode("utf-8")
            )
            if total_bytes + document_bytes > self.MAX_BATCH_BYTES:
                dropped += 1
                continue
            total_bytes += document_bytes
            bounded.append(
                ProviderDocument(
                    source_document_id=source_document_id,
                    title=title,
                    content=content,
                    url=url,
                    published_at=(
                        str(document.published_at)[:128]
                        if document.published_at is not None else None
                    ),
                    metadata=metadata,
                )
            )
        return bounded, dropped

    def _to_document_record(
        self,
        source_name: str,
        document: ProviderDocument,
        *,
        fetched_at: str,
    ) -> DocumentRecord:
        return DocumentRecord(
            source_name=source_name,
            source_document_id=document.source_document_id,
            title=document.title,
            content=document.content,
            url=document.url,
            published_at=document.published_at,
            received_at=fetched_at,
            # Live availability starts when this system actually received the
            # document; publication time is retained separately for research.
            available_at=fetched_at,
            metadata=dict(document.metadata),
        )

    def _assess(self, document: DocumentRecord) -> EventAssessment:
        assessment = self.analyzer.assess(document)
        if assessment.prompt_injection_flag:
            return assessment
        if self.jev_adapter is None:
            return assessment
        try:
            jev_features = self.jev_adapter.analyze_document(
                document.content,
                metadata={
                    "source": document.source_name,
                    "title": document.title,
                    "published_at": document.published_at,
                },
            )
        except Exception as exc:
            assessment.model_version = f"{assessment.model_version}+jev-failed"
            assessment.missing_reason = (
                f"jev failed: {redact_sensitive_text(exc)}"
            )
            return assessment

        if not jev_features:
            assessment.model_version = f"{assessment.model_version}+jev-empty"
            assessment.missing_reason = "jev returned no assessment"
            return assessment

        allowed_jev = {
            key: value
            for key, value in dict(jev_features).items()
            if (
                str(key).startswith("JEV_")
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            )
        }
        if allowed_jev:
            raw = dict(assessment.raw_response)
            raw["jev_features"] = allowed_jev
            assessment.raw_response = raw
            model = getattr(self.jev_adapter, "model", "jev")
            assessment.model_version = f"{assessment.model_version}+{model}"
            if float(allowed_jev.get("JEV_Impact_Level", 0.0)) >= 3.0:
                assessment.severity = "high"
        return assessment

    def _write_report(self, run: MonitorRun) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / f"{run.run_id}.json"
        payload = {
            "run_id": run.run_id,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "status": run.status,
            "counts": {
                "sources_checked": run.sources_checked,
                "sources_failed": run.sources_failed,
                "sources_content_stale": run.sources_content_stale,
                "documents_seen": run.documents_seen,
                "documents_inserted": run.documents_inserted,
                "documents_dropped": run.documents_dropped,
                "assessments_created": run.assessments_created,
                "alerts_opened": run.alerts_opened,
            },
            "source_health": [
                {
                    "source_name": health.source_name,
                    "last_success_at": health.last_success_at,
                    "last_checked_at": health.last_checked_at,
                    "last_error": health.last_error,
                    "source_stale": health.source_stale,
                    "content_stale": health.content_stale,
                    "last_document_at": health.last_document_at,
                    "content_baseline_at": health.content_baseline_at,
                    "document_count": health.document_count,
                }
                for health in self.store.list_source_health()
            ],
            "open_alerts": [
                {
                    "alert_id": alert.alert_id,
                    "event_cluster_id": alert.event_cluster_id,
                    "severity": alert.severity,
                    "status": alert.status,
                    "reason": alert.reason,
                    "updated_at": alert.updated_at,
                }
                for alert in self.store.list_alerts()
                if alert.status == "open"
            ],
        }
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=".sentiment_monitor.", suffix=".tmp", dir=self.output_dir
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(temporary_path, report_path)
        except Exception:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
            raise
        return report_path

    @staticmethod
    def _run_id(started_at: str) -> str:
        compact = re.sub(r"[^0-9A-Za-z]+", "", started_at)
        if not compact:
            compact = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"sentiment_monitor_{compact}_{uuid4().hex[:10]}"
