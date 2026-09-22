import pytest

from rag_tool import RAGAnalyzer, ResearchCrawler


class _Store:
    def get_stats(self):
        return {"total_chunks": 3}

    def search(self, _question, *, top_k, filter_dict):
        if filter_dict["level"] == "document":
            return {
                "documents": [["过去政策原文"]],
                "metadatas": [[{
                    "level": "document",
                    "parent_doc_id": "past",
                    "source": "PBC",
                    "title": "过去",
                    "published_at": "2026-09-20T10:00:00+08:00",
                }]],
            }
        return {
            "documents": [["过去政策摘录", "未来政策摘录", "无日期摘录"]],
            "metadatas": [[
                {
                    "level": "passage", "parent_doc_id": "past",
                    "chunk_id": "p1", "source": "PBC", "title": "过去",
                    "published_at": "2026-09-20T10:00:00+08:00",
                },
                {
                    "level": "passage", "parent_doc_id": "future",
                    "chunk_id": "p2", "source": "PBC", "title": "未来",
                    "published_at": "2026-09-25T10:00:00+08:00",
                },
                {
                    "level": "passage", "parent_doc_id": "undated",
                    "chunk_id": "p3", "source": "unknown", "title": "无日期",
                },
            ]],
            "distances": [[0.1, 0.1, 0.1]],
        }


def test_rag_as_of_excludes_future_and_undated_documents():
    analyzer = RAGAnalyzer.__new__(RAGAnalyzer)
    analyzer.vector_store = _Store()
    analyzer.api_key = ""

    result = analyzer.query(
        "货币政策如何？",
        as_of="2026-09-23T12:00:00+08:00",
    )

    assert "过去政策" in result["context"]
    assert "未来政策" not in result["context"]
    assert "无日期" not in result["context"]
    assert result["as_of"] == "2026-09-23T12:00:00+08:00"


def test_naive_timestamps_use_shanghai_and_date_only_is_conservative():
    as_of = RAGAnalyzer._timestamp_utc("2026-09-23 12:00:00")

    assert RAGAnalyzer._metadata_visible_as_of(
        {"published_at": "2026-09-23T11:59:00+08:00"}, as_of
    )
    assert not RAGAnalyzer._metadata_visible_as_of(
        {"published_at": "2026-09-23T12:01:00+08:00"}, as_of
    )
    assert not RAGAnalyzer._metadata_visible_as_of(
        {"date": "2026-09-23"}, as_of
    )
    next_day = RAGAnalyzer._timestamp_utc("2026-09-24 00:00:00")
    assert RAGAnalyzer._metadata_visible_as_of(
        {"date": "2026-09-23"}, next_day
    )


def test_safe_fetch_rejects_loopback_even_if_host_is_allowlisted(tmp_path):
    crawler = ResearchCrawler(tmp_path)

    with pytest.raises(ValueError, match="non-public address"):
        crawler._validate_fetch_url(
            "https://127.0.0.1/private", {"127.0.0.1"}
        )


def test_safe_fetch_revalidates_and_rejects_cross_domain_redirect(
    tmp_path, monkeypatch
):
    crawler = ResearchCrawler(tmp_path)
    monkeypatch.setattr(
        ResearchCrawler,
        "_resolved_addresses",
        staticmethod(lambda _host: {"93.184.216.34"}),
    )

    class RedirectResponse:
        status_code = 302
        headers = {"location": "https://evil.example/internal"}
        url = "https://www.pbc.gov.cn/report"

        def close(self):
            pass

    class Session:
        trust_env = True

        def get(self, *_args, **_kwargs):
            return RedirectResponse()

        def close(self):
            pass

    with pytest.raises(ValueError, match="not allowlisted"):
        crawler._safe_fetch(
            "https://www.pbc.gov.cn/report",
            allowed_hosts={"pbc.gov.cn"},
            session=Session(),
        )


def test_local_rag_skips_symlinked_files(tmp_path):
    root = tmp_path / "agentic_rag"
    root.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("敏感内容" * 100, encoding="utf-8")
    link = root / "linked.md"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")

    frame = ResearchCrawler(tmp_path / "cache").fetch_local_agentic_rag(tmp_path)

    assert frame.empty
