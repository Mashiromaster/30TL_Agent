import pandas as pd

from rag_tool import Document, RAGAnalyzer


def test_rag_index_export_keeps_only_real_publication_timestamps(tmp_path):
    analyzer = RAGAnalyzer(tmp_path)
    documents = [
        Document(
            "央行公告降准",
            metadata={
                "source": "PBC",
                "title": "公告",
                "release_ts": "2025-01-03 17:05:00",
            },
        ),
        Document(
            "只有抓取时间，不得进入回测",
            metadata={
                "source": "unknown",
                "fetched_at": "2025-01-04",
            },
        ),
        Document(
            "月报参考期不是发布日期",
            metadata={"source": "CFFEX", "period": "202501"},
        ),
    ]

    path = analyzer._write_semantic_event_cache(documents)
    result = pd.read_pickle(path)

    assert len(result) == 1
    assert result.loc[0, "date"] == pd.Timestamp("2025-01-03 17:05:00")
    assert result.loc[0, "content"] == "央行公告降准"
    assert "fetched_at" not in result.columns


def test_text_report_loader_deduplicates_exported_and_source_cache(tmp_path):
    from text_factor import _load_reports_with_dates

    rag_dir = tmp_path / "data" / "rag"
    reports_dir = rag_dir / "reports"
    reports_dir.mkdir(parents=True)
    duplicate = pd.DataFrame(
        {
            "date": ["2025-01-03 17:05:00"],
            "source": ["PBC"],
            "title": ["公告"],
            "content": ["央行公告降准"],
        }
    )
    duplicate.to_pickle(reports_dir / "source.pkl")
    duplicate.to_pickle(rag_dir / "semantic_events.pkl")

    result = _load_reports_with_dates(tmp_path)

    assert len(result) == 1


def test_empty_rag_refresh_clears_previous_semantic_event_snapshot(
    tmp_path, monkeypatch
):
    analyzer = RAGAnalyzer(tmp_path)
    analyzer._write_semantic_event_cache(
        [
            Document(
                "旧事件",
                metadata={"source": "old", "date": "2025-01-01"},
            )
        ]
    )
    monkeypatch.setattr(analyzer.crawler, "fetch_all", lambda **kwargs: [])
    monkeypatch.setattr(
        analyzer.vector_store,
        "get_stats",
        lambda: {"total_chunks": 0},
    )

    analyzer.build_index(force_refresh=False)

    events = pd.read_pickle(tmp_path / "data" / "rag" / "semantic_events.pkl")
    assert events.empty
