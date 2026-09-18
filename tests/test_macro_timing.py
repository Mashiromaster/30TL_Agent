import pandas as pd

from macro_timing import (
    assign_available_ts,
    combine_factor_events,
    factor_asof_column,
    merge_point_in_time,
)


def test_explicit_release_timestamp_always_wins():
    raw = pd.DataFrame(
        {
            "date": ["2025-08-01"],
            "release_ts": ["2025-09-12 17:00:00"],
            "value": [1.0],
        }
    )

    result = assign_available_ts(raw, release_kind="monetary")

    assert result.loc[0, "available_ts"] == pd.Timestamp("2025-09-12 17:00:00")


def test_monthly_fallbacks_are_conservative_and_not_period_start():
    period = pd.DataFrame({"date": ["2025-08-01"], "value": [1.0]})

    pmi = assign_available_ts(period, release_kind="pmi")
    cpi = assign_available_ts(period, release_kind="cpi")
    monetary = assign_available_ts(period, release_kind="monetary")

    assert pmi.loc[0, "available_ts"] == pd.Timestamp("2025-08-31 09:30:00")
    assert cpi.loc[0, "available_ts"] == pd.Timestamp("2025-09-15 09:30:00")
    assert monetary.loc[0, "available_ts"] == pd.Timestamp("2025-09-20 17:00:00")


def test_release_calendar_overrides_fallback_for_matching_series_and_period():
    period = pd.DataFrame({"date": ["2025-08-01"], "value": [1.0]})
    calendar = pd.DataFrame(
        {
            "series": ["cpi"],
            "period_date": ["2025-08-01"],
            "release_ts": ["2025-09-10 09:30:00"],
        }
    )

    result = assign_available_ts(
        period, release_kind="cpi", release_calendar=calendar, series_name="cpi"
    )

    assert result.loc[0, "available_ts"] == pd.Timestamp("2025-09-10 09:30:00")


def test_point_in_time_merge_never_exposes_future_values_and_enforces_ttl():
    minute = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-02 09:29:00",
                    "2025-01-02 09:30:00",
                    "2025-01-04 09:30:00",
                    "2025-01-10 09:30:00",
                ]
            )
        }
    )
    events = pd.DataFrame(
        {
            "available_ts": [pd.Timestamp("2025-01-02 09:30:00")],
            "Daily_Factor": [2.0],
            factor_asof_column("Daily_Factor"): [pd.Timestamp("2025-01-02 09:30:00")],
        }
    )

    result = merge_point_in_time(
        minute,
        events,
        factor_ttls={"Daily_Factor": pd.Timedelta(days=5)},
    )

    assert result["Daily_Factor"].tolist() == [0.0, 2.0, 2.0, 0.0]
    assert result.loc[0, "_macro_coverage_count"] == 0
    assert result.loc[3, "_macro_stale_count"] == 1


def test_combined_event_stream_tracks_each_factors_own_update_time():
    curve = pd.DataFrame(
        {
            "available_ts": pd.to_datetime(["2025-01-02 09:00", "2025-01-03 09:00"]),
            "Curve": [1.0, 1.5],
        }
    )
    monthly = pd.DataFrame(
        {
            "available_ts": pd.to_datetime(["2025-01-02 10:00"]),
            "Macro": [5.0],
        }
    )

    combined = combine_factor_events([curve, monthly])

    assert combined["available_ts"].tolist() == list(
        pd.to_datetime(["2025-01-02 09:00", "2025-01-02 10:00", "2025-01-03 09:00"])
    )
    last = combined.iloc[-1]
    assert last["Curve"] == 1.5
    assert last["Macro"] == 5.0
    assert last[factor_asof_column("Curve")] == pd.Timestamp("2025-01-03 09:00")
    assert last[factor_asof_column("Macro")] == pd.Timestamp("2025-01-02 10:00")


def test_timezone_aware_events_merge_with_local_naive_market_bars():
    bars = pd.DataFrame({"date": pd.to_datetime(["2025-01-02 09:29", "2025-01-02 09:30"])})
    events = pd.DataFrame(
        {
            "available_ts": pd.to_datetime(["2025-01-02 09:30+08:00"]),
            "Macro": [3.0],
        }
    )

    result = merge_point_in_time(bars, events)

    assert result["Macro"].tolist() == [0.0, 3.0]


def test_multiple_event_stream_merges_accumulate_diagnostics():
    bars = pd.DataFrame({"date": pd.to_datetime(["2025-01-02 10:00"])})
    macro = pd.DataFrame(
        {"available_ts": pd.to_datetime(["2025-01-02 09:00"]), "Macro": [1.0]}
    )
    text = pd.DataFrame(
        {"available_ts": pd.to_datetime(["2025-01-02 09:30"]), "Text": [2.0]}
    )

    once = merge_point_in_time(bars, macro)
    twice = merge_point_in_time(once, text)

    assert twice.loc[0, "_macro_coverage_count"] == 2
    assert twice.loc[0, "_macro_stale_count"] == 0
