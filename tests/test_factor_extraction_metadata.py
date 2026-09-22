import pandas as pd

from factor_extraction import merge_macro_to_minute


def test_text_factor_provenance_survives_point_in_time_merge():
    bars = pd.DataFrame(
        {"date": pd.to_datetime(["2025-01-02 08:59", "2025-01-02 09:00"])}
    )
    events = pd.DataFrame(
        {
            "available_ts": pd.to_datetime(["2025-01-02 09:00"]),
            "JEV_Bond_Direction": [0.5],
        }
    )
    events.attrs["factor_ttls"] = {"JEV_Bond_Direction": 10}
    events.attrs["text_factor_metadata"] = {
        "jev_model": "jev-1.13.0",
        "jev_features": ["JEV_Bond_Direction"],
    }

    result = merge_macro_to_minute(bars, events)

    assert result["JEV_Bond_Direction"].tolist() == [0.0, 0.5]
    assert result.attrs["text_factor_metadata"] == events.attrs[
        "text_factor_metadata"
    ]
