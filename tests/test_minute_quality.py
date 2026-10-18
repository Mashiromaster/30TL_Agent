import json
import warnings

import numpy as np
import pandas as pd
import pytest

from minute_quality import analyze_minute_coverage


def bars(day="2026-05-11", ticker="TL2606"):
    day = pd.Timestamp(day)
    dates = pd.date_range(day + pd.Timedelta(hours=9, minutes=31), periods=120, freq="min").append(
        pd.date_range(day + pd.Timedelta(hours=13, minutes=1), periods=135, freq="min")
    )
    return pd.DataFrame({
        "date": dates, "trade_dt": day, "ticker": ticker, "open": 100.,
        "high": 101., "low": 99., "close": 100., "volume": 10., "oi": 100.,
    })


def audit(frame, **kwargs):
    return analyze_minute_coverage(frame, now="2026-05-18 10:00", **kwargs)


def test_complete_session_read_only_json_and_not_calendar_certified():
    frame = bars()
    before = frame.copy(deep=True)
    report = audit(frame)
    assert report["schema_version"] == "f_agent.minute_quality.v1"
    assert report["status"] == "complete_observed"
    assert report["validation"]["valid"] is True
    assert report["coverage"]["rows"] == 255
    assert report["coverage"]["complete_contract_days"] == 1
    assert report["coverage"]["completed_rows"] == 255
    assert report["calendar"]["source"] == "observed_only"
    assert report["calendar"]["verified"] is False
    assert report["calendar"]["missing_dates_count"] is None
    assert report["contract_day_issues"] == []
    assert report["template"]["expected_minutes"] == 255
    assert report["limitations"]
    json.dumps(report, allow_nan=False)
    pd.testing.assert_frame_equal(frame, before)


def test_missing_extra_and_duplicate_bars_are_not_masked_by_row_count():
    frame = bars().drop(index=[0, 1]).reset_index(drop=True)
    duplicate = frame.iloc[[0]].copy()
    extra = frame.iloc[[0]].copy()
    extra["date"] = pd.Timestamp("2026-05-11 12:00")
    frame = pd.concat([frame, duplicate, extra], ignore_index=True)
    report = audit(frame)
    assert report["status"] == "invalid"
    assert report["validation"]["issues"]["duplicate_date_ticker"] == 2
    item = report["contract_day_issues"][0]
    assert item["row_count"] == 255
    assert item["missing_minutes_count"] == 2
    assert item["extra_minutes_count"] == 1
    assert item["duplicate_rows"] == 1
    assert item["session_complete"] is False
    assert item["missing_minutes"] == ["09:31", "09:32"]


@pytest.mark.parametrize("column,value,code", [
    ("ticker", "TL0", "invalid_contract"),
    ("ticker", "TL2605", "invalid_contract"),
    ("ticker", None, "invalid_contract"),
    ("date", "bad", "invalid_date"),
    ("trade_dt", "bad", "invalid_trade_dt"),
    ("trade_dt", "2026-05-12", "trade_date_mismatch"),
    ("date", "2026-05-11 09:31:01", "subminute_timestamp"),
    ("close", np.inf, "nonfinite_numeric"),
    ("close", "bad", "nonfinite_numeric"),
    ("open", 0, "invalid_ohlc"),
    ("high", 99.5, "invalid_ohlc"),
    ("low", 100.5, "invalid_ohlc"),
    ("volume", -1, "negative_volume_or_oi"),
    ("oi", -1, "negative_volume_or_oi"),
])
def test_invalid_rows_fail_closed(column, value, code):
    frame = bars()
    frame[column] = frame[column].astype(object)
    frame.at[0, column] = value
    report = audit(frame)
    assert report["status"] == "invalid"
    assert report["validation"]["issues"][code] == 1
    assert report["validation"]["invalid_rows"] == 1
    json.dumps(report, allow_nan=False)


def test_groups_are_per_contract_not_combined_session():
    first, second = bars().iloc[:120], bars(ticker="TL2609").iloc[120:]
    report = audit(pd.concat([first, second], ignore_index=True))
    assert report["coverage"]["contract_days"] == 2
    assert report["coverage"]["complete_contract_days"] == 0
    assert report["coverage"]["incomplete_contract_days"] == 2
    assert len(report["contract_day_issues"]) == 2
    assert report["status"] == "incomplete"


def test_current_day_is_withheld_even_after_close_and_future_rows_flagged():
    frame = pd.concat([bars("2026-05-11"), bars("2026-05-12"), bars("2026-05-13")])
    report = analyze_minute_coverage(frame, now="2026-05-12 18:00")
    assert report["coverage"]["current_day_rows"] == 255
    assert report["coverage"]["future_rows"] == 255
    assert report["coverage"]["completed_rows"] == 255
    assert report["validation"]["issues"]["future_timestamp"] == 255
    assert report["status"] == "invalid"


def test_today_complete_template_is_not_completed_session():
    report = analyze_minute_coverage(bars(), now="2026-05-11 18:00")
    assert report["status"] == "incomplete"
    assert report["coverage"]["complete_contract_days"] == 1
    assert report["coverage"]["complete_completed_contract_days"] == 0
    assert report["contract_day_issues"][0]["completion"] == "current_day_withheld"


def test_utc_timestamps_are_converted_to_shanghai_wall_clock():
    frame = bars()
    frame["date"] = frame.date.dt.tz_localize("Asia/Shanghai").dt.tz_convert("UTC")
    frame["trade_dt"] = frame.trade_dt.dt.tz_localize("Asia/Shanghai").dt.tz_convert("UTC")
    report = analyze_minute_coverage(frame, now="2026-05-11T16:00:00+00:00")
    assert report["status"] == "complete_observed"
    assert report["coverage"]["start"] == "2026-05-11T09:31:00+08:00"


def test_observed_gaps_are_calendar_gaps_not_assumed_missing_trade_days():
    frame = pd.concat([bars("2026-05-08"), bars("2026-05-11")])
    report = audit(frame)
    assert report["calendar"]["missing_dates"] == []
    assert report["calendar"]["missing_dates_count"] is None
    assert report["observed_gaps"][0]["unobserved_calendar_days"] == 2
    assert report["status"] == "complete_observed"


def test_caller_calendar_reports_missing_dates_and_excludes_today_future():
    report = audit(bars(), expected_dates=["2026-05-11", "2026-05-12", "2026-05-18", "2026-05-19"])
    assert report["calendar"]["source"] == "caller_provided"
    assert report["calendar"]["verified"] is False
    assert report["calendar"]["missing_dates"] == ["2026-05-12"]
    assert report["calendar"]["missing_dates_count"] == 1
    assert report["calendar"]["not_completed_expected_dates_count"] == 2
    assert report["status"] == "incomplete"


def test_caller_calendar_does_not_silently_drop_invalid_expected_dates():
    report = audit(bars(), expected_dates=["2026-05-11", "oops"])
    assert report["status"] == "invalid"
    assert report["validation"]["issues"]["invalid_expected_dates"] == 1


@pytest.mark.parametrize("frame,code", [
    (None, "not_a_dataframe"),
    (pd.DataFrame(), "empty_input"),
    (pd.DataFrame({"date": ["2026-05-11"]}), "missing_columns"),
])
def test_bad_schema_produces_structured_failure(frame, code):
    report = audit(frame)
    assert report["status"] == "invalid"
    assert code in report["validation"]["issues"]
    json.dumps(report, allow_nan=False)


def test_duplicate_columns_and_invalid_now_fail_closed():
    frame = bars()
    frame.columns = list(frame.columns[:-1]) + ["date"]
    assert audit(frame)["validation"]["issues"]["duplicate_columns"] == 1
    report = analyze_minute_coverage(bars(), now="oops")
    assert report["validation"]["issues"]["invalid_as_of"] == 1


def test_reports_are_bounded_without_truncating_total_problem_counts():
    frame = pd.concat([bars(day).iloc[:1] for day in pd.date_range("2024-01-01", periods=210)])
    report = audit(frame, expected_dates=pd.date_range("2023-01-01", periods=600))
    assert report["coverage"]["incomplete_contract_days"] == 210
    assert report["coverage"]["problem_contract_days"] == 210
    assert len(report["contract_day_issues"]) == 200
    assert report["truncated"]["contract_day_issues"] is True
    assert report["calendar"]["missing_dates_count"] == 390
    assert len(report["calendar"]["missing_dates"]) == 200
    assert report["truncated"]["missing_dates"] is True
    assert len(report["contract_day_issues"][0]["missing_minutes"]) <= 20


@pytest.mark.parametrize("value", [pd.NaT, None])
def test_null_asof_is_rejected_or_uses_explicit_current_time(value):
    report = analyze_minute_coverage(bars(), now=value)
    if value is pd.NaT:
        assert report["validation"]["issues"] == {"invalid_as_of": 1}
    else:
        assert report["as_of"]


@pytest.mark.parametrize("expected", ["2026-05-11", 123])
def test_non_iterable_calendar_and_string_calendar_are_invalid(expected):
    assert audit(bars(), expected_dates=expected)["validation"]["issues"]["invalid_expected_dates"] == 1


def test_mixed_timezone_column_fails_closed_instead_of_shifting_naive_rows():
    frame = bars()
    frame["date"] = frame.date.astype(object)
    frame.at[0, "date"] = pd.Timestamp("2026-05-11 01:31", tz="UTC")
    report = audit(frame)
    assert report["status"] == "invalid"
    assert report["validation"]["issues"]["invalid_date"] > 0


def test_integer_timestamp_is_not_assumed_nanoseconds_from_unix_epoch():
    frame = bars()
    frame["date"] = frame.date.astype(object)
    frame.at[0, "date"] = 1778482260000000000
    report = audit(frame)
    assert report["validation"]["issues"]["invalid_date"] == 1


def test_large_gap_list_is_bounded_and_largest_gap_is_not_dropped():
    days = list(pd.date_range("2020-01-01", periods=210, freq="3D")) + [pd.Timestamp("2025-01-01")]
    frame = pd.concat([bars(day).iloc[:1] for day in days])
    report = audit(frame)
    assert report["coverage"]["observed_gap_count"] == 210
    assert len(report["observed_gaps"]) == 200
    assert report["truncated"]["observed_gaps"] is True
    assert report["observed_gaps"][0]["unobserved_calendar_days"] == report["coverage"]["largest_unobserved_calendar_gap_days"]


def test_duplicate_and_empty_caller_calendar_are_explicit_not_autoinferred():
    report = audit(bars(), expected_dates=["2026-05-11", "2026-05-11"])
    assert report["calendar"]["expected_dates_count"] == 1
    assert report["calendar"]["missing_dates_count"] == 0
    empty = audit(bars(), expected_dates=[])
    assert empty["calendar"]["expected_dates_count"] == 0
    assert empty["status"] == "complete_observed"


def test_subminute_extra_sample_preserves_fractional_timestamp():
    frame = bars()
    frame["date"] = frame.date.astype("datetime64[ns]")
    frame.loc[0, "date"] += pd.Timedelta(nanoseconds=1)
    report = audit(frame)
    assert report["validation"]["issues"]["subminute_timestamp"] == 1
    assert report["contract_day_issues"][0]["extra_minutes"] == ["09:31:00.000000001"]


def test_mixed_timezone_offsets_are_not_silently_treated_as_same_session():
    frame = bars()
    frame["date"] = frame.date.astype(object)
    frame.at[0, "date"] = pd.Timestamp("2026-05-11 01:31", tz="UTC")
    frame.at[1, "date"] = pd.Timestamp("2026-05-11 09:32", tz="Asia/Shanghai")
    report = audit(frame)
    assert report["status"] == "invalid"


def test_repair_ranges_cover_all_completed_problems_not_only_samples():
    days = pd.date_range("2024-01-01", periods=210)
    frame = pd.concat([bars(day).iloc[:1] for day in days])
    report = audit(frame)
    assert len(report["contract_day_issues"]) == 200
    assert report["repair_summary"]["contract_days"] == 210
    assert report["repair_summary"]["observed_rows"] == 210
    assert report["repair_summary"]["missing_minutes_total"] == 210 * 254
    assert report["repair_summary"]["range_count"] == 1
    item = report["repair_ranges"][0]
    assert item["ticker"] == "TL2606"
    assert item["start_date"] == "2024-01-01"
    assert item["end_date"] == days[-1].date().isoformat()
    assert item["contract_days"] == 210
    assert item["missing_minutes_total"] == 210 * 254
    assert item["calendar_verified"] is False
    assert item["action"] == "inspect_source_coverage"


def test_repair_ranges_split_at_observed_complete_contract_days_and_ignore_today():
    frame = pd.concat([
        bars("2026-05-11").iloc[:1], bars("2026-05-12"), bars("2026-05-13").iloc[:2],
        bars("2026-05-13", ticker="TL2609").iloc[:3], bars("2026-05-18").iloc[:4],
    ])
    report = audit(frame)
    assert report["repair_summary"]["range_count"] == 3
    assert report["repair_summary"]["contract_days"] == 3
    assert report["repair_summary"]["observed_rows"] == 6
    assert [item["start_date"] for item in report["repair_ranges"]] == ["2026-05-11", "2026-05-13", "2026-05-13"]
    assert report["truncated"]["repair_ranges"] is False


def test_repair_range_output_bound_preserves_all_range_totals():
    frame = pd.concat([bars("2025-01-01", ticker=f"TL{year:02d}{month}").iloc[:1]
                       for year in range(60) for month in ("03", "06", "09", "12")])
    report = audit(frame)
    assert report["repair_summary"]["range_count"] == 240
    assert report["repair_summary"]["contract_days"] == 240
    assert len(report["repair_ranges"]) == 200
    assert report["truncated"]["repair_ranges"] is True


@pytest.mark.parametrize("column", ["open", "high", "low", "close", "volume", "oi"])
@pytest.mark.parametrize("value", [True, np.bool_(False)])
def test_boolean_prices_volume_and_open_interest_are_not_numbers(column, value):
    frame = bars()
    frame[column] = frame[column].astype(object)
    frame.at[0, column] = value
    report = audit(frame)
    assert report["status"] == "invalid"
    assert report["validation"]["issues"]["boolean_numeric"] == 1
    assert report["validation"]["invalid_rows"] == 1


def test_entire_boolean_numeric_column_is_rejected():
    frame = bars()
    frame["volume"] = True
    report = audit(frame)
    assert report["validation"]["issues"]["boolean_numeric"] == 255


@pytest.mark.parametrize("column", ["open", "high", "low", "close", "volume", "oi"])
@pytest.mark.parametrize("value", [10 + 2j, np.complex64(10 + 2j), 10 + 0j])
def test_object_mixed_complex_numeric_values_are_rejected_without_cast_warning(column, value):
    frame = bars()
    frame[column] = frame[column].astype(object)
    frame.at[0, column] = value
    before = frame.copy(deep=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        report = audit(frame)
    assert report["status"] == "invalid"
    assert report["validation"]["issues"]["nonreal_numeric"] == 1
    assert report["validation"]["invalid_rows"] == 1
    json.dumps(report, allow_nan=False)
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("column", ["open", "high", "low", "close", "volume", "oi"])
def test_complex_dtype_is_rejected_without_losing_imaginary_parts(column):
    frame = bars()
    frame[column] = frame[column].astype(complex)
    frame.at[0, column] = 10 + 2j
    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        report = audit(frame)
    assert report["validation"]["valid"] is False
    assert report["validation"]["issues"]["nonreal_numeric"] == 255
