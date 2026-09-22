import pandas as pd

from artifact_audit import build_factor_artifact_audit


def test_factor_audit_records_jev_provenance_without_secrets():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02 09:30", "2025-01-02 09:31"]),
            "close": [100.0, 100.1],
            "JEV_Bond_Direction": [0.7, 0.7],
        }
    )
    frame.attrs["text_factor_metadata"] = {
        "jev_model": "jev-1.13.0",
        "jev_schema_version": "macro-document-v1",
        "jev_mode": "cache",
        "documents_attempted": 10,
        "documents_succeeded": 9,
        "document_coverage": 0.9,
        "jev_features": ["JEV_Bond_Direction"],
    }

    report = build_factor_artifact_audit(frame)

    assert report["schema_version"] == 2
    assert report["text_factor_metadata"] == frame.attrs["text_factor_metadata"]
    assert "api_key" not in str(report).lower()
