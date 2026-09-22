import pandas as pd

from text_factor import build_sentiment_factor
from text_factor import _load_reports_with_dates


def test_text_factor_discovers_nested_report_cache_and_uses_release_time(tmp_path):
    reports = tmp_path / "data" / "rag" / "reports"
    reports.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2025-01-03", "2025-01-06 10:15:00"],
            "content": ["降准 流动性充裕", "紧缩 流动性紧张"],
        }
    ).to_pickle(reports / "reports.pkl")

    result = build_sentiment_factor(tmp_path)

    assert result["available_ts"].tolist() == [
        pd.Timestamp("2025-01-06 09:00:00"),
        pd.Timestamp("2025-01-06 10:15:00"),
    ]
    assert result.attrs["factor_ttls"]["Policy_Sentiment"] == 10


def test_report_timestamp_falls_back_per_row_and_normalizes_timezone(tmp_path):
    reports = tmp_path / "data" / "rag"
    reports.mkdir(parents=True)
    pd.DataFrame(
        {
            "release_ts": [None, "2025-01-03 17:00:00+00:00"],
            "date": ["2025-01-02", "2025-01-03"],
            "content": ["第一条", "第二条"],
        }
    ).to_pickle(reports / "mixed_dates.pkl")

    result = _load_reports_with_dates(tmp_path)

    assert result["pub_ts"].tolist() == [
        pd.Timestamp("2025-01-02 00:00:00"),
        pd.Timestamp("2025-01-04 01:00:00"),
    ]


class _FakeJevClient:
    model = "jev-1.13.0"
    schema_version = "macro-document-v1"
    mode = "cache"

    def analyze_document(self, text, metadata=None):
        if "失败" in text:
            return None
        return {
            "JEV_Policy_Easing_P": 0.8,
            "JEV_Policy_Tightening_P": 0.1,
            "JEV_Policy_Direction": 0.7,
            "JEV_Bond_Bullish_P": 0.75,
            "JEV_Bond_Bearish_P": 0.05,
            "JEV_Bond_Direction": 0.7,
            "JEV_Impact_Level": 3.0,
            "JEV_Policy_Confidence": 0.9,
            "JEV_Bond_Confidence": 0.85,
            "JEV_Impact_Confidence": 0.8,
            "JEV_New_Information_P": 0.9,
            "JEV_Liquidity_Channel_P": 0.8,
            "JEV_Growth_Channel_P": 0.2,
            "JEV_Inflation_Channel_P": 0.1,
            "JEV_Supply_Channel_P": 0.1,
        }


def test_text_factor_adds_jev_events_without_erasing_last_good_value(tmp_path):
    reports = tmp_path / "data" / "rag" / "reports"
    reports.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2025-01-03", "2025-01-07", "2025-01-08"],
            "content": ["央行降准", "API失败", "继续宽松"],
            "source": ["PBC", "news", "PBC"],
        }
    ).to_pickle(reports / "reports.pkl")

    result = build_sentiment_factor(tmp_path, jev_client=_FakeJevClient())

    assert "JEV_Bond_Direction" in result.columns
    failed_event = result.loc[
        result["available_ts"] == pd.Timestamp("2025-01-08 09:00:00")
    ].iloc[0]
    assert failed_event["JEV_Bond_Direction"] == 0.7
    assert failed_event["_asof__JEV_Bond_Direction"] == pd.Timestamp(
        "2025-01-06 09:00:00"
    )
    assert result.attrs["factor_ttls"]["JEV_Bond_Direction"] == 10
    assert result.attrs["jev_model"] == "jev-1.13.0"
    assert result.attrs["text_factor_metadata"] == {
        "jev_model": "jev-1.13.0",
        "jev_schema_version": "macro-document-v1",
        "jev_mode": "cache",
        "documents_attempted": 3,
        "documents_succeeded": 2,
        "document_coverage": 2 / 3,
        "jev_features": sorted(
            col for col in result.columns if col.startswith("JEV_")
        ),
    }
