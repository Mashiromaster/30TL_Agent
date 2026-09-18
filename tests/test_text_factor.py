import pandas as pd

from text_factor import build_sentiment_factor


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
