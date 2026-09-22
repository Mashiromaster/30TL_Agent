import pytest

from intelligence_events import (
    DocumentRecord,
    EventAssessment,
    IntelligenceEventStore,
)


def _doc(
    *,
    source_name="eastmoney_tl",
    source_document_id="doc-1",
    title="央行降准",
    content="央行宣布降准，释放长期流动性，利好TL。",
    url="https://example.test/news/1",
):
    return DocumentRecord(
        source_name=source_name,
        source_document_id=source_document_id,
        title=title,
        content=content,
        url=url,
        published_at="2026-09-23T09:30:00+08:00",
        received_at="2026-09-23T09:31:00+08:00",
        available_at="2026-09-23T09:30:00+08:00",
        metadata={"provider": source_name},
    )


def test_provider_batch_rolls_back_document_and_cursor_atomically(tmp_path):
    store = IntelligenceEventStore(tmp_path)

    with pytest.raises(RuntimeError, match="boom"):
        store.commit_provider_batch(
            source_name="eastmoney_tl",
            documents=[_doc()],
            next_cursor="cursor-1",
            fetched_at="2026-09-23T09:32:00+08:00",
            before_cursor_update=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )

    assert store.list_documents() == []
    assert store.get_cursor("eastmoney_tl") is None
    assert store.get_source_health("eastmoney_tl") is None

    inserted = store.commit_provider_batch(
        source_name="eastmoney_tl",
        documents=[_doc()],
        next_cursor="cursor-1",
        fetched_at="2026-09-23T09:32:00+08:00",
    )

    assert len(inserted) == 1
    assert inserted[0].content == "央行宣布降准，释放长期流动性，利好TL。"
    assert store.get_cursor("eastmoney_tl").cursor_value == "cursor-1"
    assert store.get_source_health("eastmoney_tl").source_stale is False


def test_document_dedupe_keeps_cross_source_records_and_independent_cursors(tmp_path):
    store = IntelligenceEventStore(tmp_path)
    em_doc = _doc(source_name="eastmoney_tl", source_document_id="same-story")
    sina_doc = _doc(
        source_name="sina_tl",
        source_document_id="same-story",
        url="https://example.test/sina/1",
    )

    assert len(
        store.commit_provider_batch(
            "eastmoney_tl", [em_doc], next_cursor="em-cursor-1"
        )
    ) == 1
    assert (
        store.commit_provider_batch(
            "eastmoney_tl", [em_doc], next_cursor="em-cursor-2"
        )
        == []
    )
    assert len(
        store.commit_provider_batch("sina_tl", [sina_doc], next_cursor="sina-cursor-1")
    ) == 1

    docs = store.list_documents()
    assert len(docs) == 2
    assert {doc.source_name for doc in docs} == {"eastmoney_tl", "sina_tl"}
    assert store.get_cursor("eastmoney_tl").cursor_value == "em-cursor-2"
    assert store.get_cursor("sina_tl").cursor_value == "sina-cursor-1"
    assert store.get_source_health("eastmoney_tl").document_count == 1
    assert store.get_source_health("sina_tl").document_count == 1


def test_source_failure_marks_stale_without_clearing_last_known_good(tmp_path):
    store = IntelligenceEventStore(tmp_path)
    store.commit_provider_batch("eastmoney_tl", [_doc()], next_cursor="cursor-ok")

    store.record_source_failure(
        "eastmoney_tl",
        error="timeout",
        checked_at="2026-09-23T10:00:00+08:00",
    )

    health = store.get_source_health("eastmoney_tl")
    assert health.source_stale is True
    assert health.last_error == "timeout"
    assert store.get_cursor("eastmoney_tl").cursor_value == "cursor-ok"
    assert len(store.list_documents()) == 1


def test_successful_empty_fetch_tracks_content_freshness_separately(tmp_path):
    store = IntelligenceEventStore(tmp_path)
    store.commit_provider_batch(
        "eastmoney_tl",
        [],
        next_cursor="empty-1",
        fetched_at="2026-09-20T09:00:00+08:00",
    )
    first = store.get_source_health("eastmoney_tl")
    assert first.source_stale is False
    assert first.content_stale is False
    assert first.last_document_at is None
    assert first.content_baseline_at == "2026-09-20T09:00:00+08:00"

    store.commit_provider_batch(
        "eastmoney_tl",
        [],
        next_cursor="empty-2",
        fetched_at="2026-09-24T09:00:01+08:00",
    )
    later = store.get_source_health("eastmoney_tl")
    assert later.source_stale is False
    assert later.content_stale is True
    assert later.last_success_at == "2026-09-24T09:00:01+08:00"
    assert later.content_baseline_at == "2026-09-20T09:00:00+08:00"

    store.commit_provider_batch(
        "eastmoney_tl",
        [],
        next_cursor="empty-3",
        fetched_at="2026-09-24T09:10:00+08:00",
    )
    still_stale = store.get_source_health("eastmoney_tl")
    assert still_stale.content_stale is True
    assert still_stale.content_baseline_at == "2026-09-20T09:00:00+08:00"


def test_alert_lifecycle_can_suppress_report_and_reopen_same_event_cluster(tmp_path):
    store = IntelligenceEventStore(tmp_path)
    doc = store.commit_provider_batch("eastmoney_tl", [_doc()], next_cursor="c1")[0]

    first = EventAssessment(
        document_id=doc.document_id,
        event_cluster_id="cluster-liquidity-easing",
        event_type="policy_liquidity",
        severity="high",
        policy_stance="easing",
        bond_impact="bullish",
        confidence=0.82,
        evidence=["央行宣布降准"],
        affected_assets=["TL"],
    )
    first_id = store.add_assessment(first)
    alert = store.upsert_alert_for_assessment(first_id)
    assert alert.status == "open"

    store.set_alert_status(alert.alert_id, "suppressed")
    assert store.get_alert(alert.alert_id).status == "suppressed"
    store.set_alert_status(alert.alert_id, "reported")
    assert store.get_alert(alert.alert_id).status == "reported"

    second = EventAssessment(
        document_id=doc.document_id,
        event_cluster_id="cluster-liquidity-easing",
        event_type="policy_liquidity",
        severity="high",
        policy_stance="easing",
        bond_impact="bullish",
        confidence=0.9,
        evidence=["释放长期流动性"],
        affected_assets=["TL"],
    )
    reopened = store.upsert_alert_for_assessment(store.add_assessment(second))

    assert reopened.alert_id == alert.alert_id
    assert reopened.status == "open"
    assert reopened.last_assessment_id != first_id


def test_assessment_identity_is_content_addressed_not_wall_clock():
    common = {
        "document_id": "doc-1",
        "event_cluster_id": "cluster-1",
        "event_type": "policy_liquidity",
        "severity": "high",
        "policy_stance": "easing",
        "bond_impact": "bullish",
        "confidence": 0.8,
        "evidence": ["央行降准"],
        "affected_assets": ["TL"],
        "tenors": ["30Y"],
        "channels": {"liquidity": 1.0},
        "raw_response": {"JEV_Impact_Level": 3.0},
    }
    first = EventAssessment(
        **common, assessed_at="2026-09-23T09:00:00+08:00"
    ).normalized()
    repeated = EventAssessment(
        **common, assessed_at="2026-09-23T09:05:00+08:00"
    ).normalized()
    changed_tenor = EventAssessment(
        **{**common, "tenors": ["10Y"]},
        assessed_at="2026-09-23T09:05:00+08:00",
    ).normalized()

    assert repeated.assessment_id == first.assessment_id
    assert changed_tenor.assessment_id != first.assessment_id


def test_assessment_raw_response_redacts_secrets(tmp_path):
    store = IntelligenceEventStore(tmp_path)
    document = store.commit_provider_batch(
        "eastmoney_tl", [_doc()], next_cursor="c1"
    )[0]
    assessment = EventAssessment(
        document_id=document.document_id,
        event_cluster_id=document.event_cluster_id,
        event_type="market_news",
        severity="low",
        raw_response={
            "api_key": "sk-sensitive",
            "debug": "Authorization: Bearer abc.def and token=secret-value",
            "nested": {"access_token": "another-secret"},
        },
    )

    stored = store.get_assessment(store.add_assessment(assessment))

    assert stored.raw_response["api_key"] == "[REDACTED]"
    assert "abc.def" not in stored.raw_response["debug"]
    assert "secret-value" not in stored.raw_response["debug"]
    assert stored.raw_response["nested"]["access_token"] == "[REDACTED]"


def test_event_query_respects_received_time_and_expiry(tmp_path):
    store = IntelligenceEventStore(tmp_path)
    doc = store.commit_provider_batch(
        "eastmoney_tl",
        [_doc()],
        next_cursor="c1",
        fetched_at="2026-09-23T09:31:00+08:00",
    )[0]
    assert doc.available_at == "2026-09-23T09:31:00+08:00"

    assessment = EventAssessment(
        document_id=doc.document_id,
        event_cluster_id=doc.event_cluster_id,
        event_type="policy_liquidity",
        severity="high",
        affected_assets=["TL"],
        assessed_at="2026-09-23T09:32:00+08:00",
        expires_at="2026-09-23T11:00:00+08:00",
    )
    store.add_assessment(assessment)

    assert store.query_assessments(as_of="2026-09-23T09:30:59+08:00") == []
    assert store.query_assessments(as_of="2026-09-23T01:30:59+00:00") == []
    assert store.query_assessments(as_of="2026-09-23T09:31:30+08:00") == []
    assert len(store.query_assessments(as_of="2026-09-23T10:00:00+08:00")) == 1
    assert store.query_assessments(as_of="2026-09-23T01:31:00Z") == []
    assert len(store.query_assessments(as_of="2026-09-23T01:32:00Z")) == 1
    assert store.query_assessments(as_of="2026-09-23T11:00:00+08:00") == []
