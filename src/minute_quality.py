"""Read-only coverage diagnostics for canonical TL minute bars.

The 255-minute template is an explicitly declared local convention, not an
exchange calendar or a certification of last-trading-day hours. No input is
repaired, inferred, persisted, or fetched by this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


_COLUMNS = ("date", "trade_dt", "ticker", "open", "high", "low", "close", "volume", "oi")
_NUMERIC = ("open", "high", "low", "close", "volume", "oi")
_SESSION_MINUTES = frozenset((*range(571, 691), *range(781, 916)))
_LIMIT = 200
_MINUTE_LIMIT = 20
_TZ = "Asia/Shanghai"
_LIMITATIONS = [
    "Read-only research diagnostics: no input repair, download, persistence, trading or model promotion.",
    "255 end-labelled minutes (09:31-11:30 and 13:01-15:15) are a local template, not verified exchange rules.",
    "Last-trading-day and historical session exceptions are not certified; a template mismatch is not proof of vendor data loss.",
    "Without an explicit calendar, unobserved dates cannot be distinguished from holidays or market closures.",
    "Caller-provided expected dates are an assertion, not an independently verified exchange calendar.",
    "A contract absent for a whole day cannot be detected without a contract listing/expiry calendar.",
    "All current-day rows are withheld from completed-session counts even after the nominal close.",
    "Coverage does not certify vendor identity, historical revision vintages, price accuracy or tradability.",
    "Problem samples are bounded; aggregate counts include all rows and contract-days.",
    "Repair ranges join consecutive observed incomplete contract-days, can span unobserved dates, and request source inspection only; they are not a certified download calendar.",
]


def _asof(now):
    stamp = pd.Timestamp.now(tz=_TZ) if now is None else pd.Timestamp(now)
    if pd.isna(stamp):
        raise ValueError("invalid as-of")
    return stamp.tz_localize(_TZ) if stamp.tzinfo is None else stamp.tz_convert(_TZ)


def _times(values):
    """Uniform timezone only; naive values mean declared Shanghai wall time."""
    values = pd.Series(values).reset_index(drop=True)
    numeric = values.map(lambda value: isinstance(value, (int, float, bool, np.number)))
    try:
        parsed = pd.to_datetime(values.mask(numeric), errors="coerce", format="mixed")
        if isinstance(parsed.dtype, pd.DatetimeTZDtype):
            return parsed.dt.tz_convert(_TZ).dt.tz_localize(None)
        if pd.api.types.is_datetime64_any_dtype(parsed.dtype):
            return parsed
    except (ValueError, TypeError, OverflowError):
        pass
    return pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")


def _iso(stamp):
    return None if pd.isna(stamp) else pd.Timestamp(stamp).tz_localize(_TZ).isoformat()


def _clock(minute):
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _base_report():
    return {
        "schema_version": "f_agent.minute_quality.v1", "as_of": None,
        "timezone": _TZ, "status": "invalid", "read_only": True,
        "validation": {"valid": False, "invalid_rows": 0, "issues": {}, "example_row_positions": []},
        "coverage": {
            "rows": 0, "valid_timestamp_rows": 0, "start": None, "end": None,
            "observed_days": 0, "contract_count": 0, "contract_days": 0,
            "complete_contract_days": 0, "incomplete_contract_days": 0,
            "complete_completed_contract_days": 0, "incomplete_completed_contract_days": 0,
            "problem_contract_days": 0, "problem_days": 0,
            "current_day_rows": 0, "future_rows": 0, "completed_rows": 0,
            "duplicate_rows": 0, "observed_gap_count": 0,
            "largest_unobserved_calendar_gap_days": 0,
        },
        "calendar": {
            "source": "observed_only", "verified": False, "expected_dates_count": None,
            "completed_expected_dates_count": None, "not_completed_expected_dates_count": None,
            "missing_dates_count": None, "missing_dates": [],
        },
        "template": {"id": "tl_local_end_labelled_255_v1", "label": "end",
                     "expected_minutes": 255, "sessions": ["09:31-11:30", "13:01-15:15"],
                     "exchange_calendar_verified": False},
        "contract_day_issues": [], "observed_gaps": [], "repair_ranges": [],
        "repair_summary": {"range_count": 0, "contract_days": 0, "observed_rows": 0,
                           "missing_minutes_total": 0, "extra_minutes_total": 0, "duplicate_rows": 0},
        "truncated": {"contract_day_issues": False, "observed_gaps": False,
                      "missing_dates": False, "repair_ranges": False},
        "sample_limits": {"contract_day_issues": _LIMIT, "observed_gaps": _LIMIT,
                          "missing_dates": _LIMIT, "minutes_per_issue": _MINUTE_LIMIT,
                          "repair_ranges": _LIMIT,
                          "validation_row_positions": 20},
        "limitations": list(_LIMITATIONS),
    }


def analyze_minute_coverage(frame, *, now=None, expected_dates=None) -> dict:
    """Return JSON-safe diagnostics without mutating ``frame`` or doing I/O.

    ``complete_observed`` means only that supplied historical contract-days
    match the local template. It never attests global market completeness.
    Schema/row failures return ``invalid`` with issue counts; counts can overlap.
    Duplicate validation counts include all members of a duplicate key, while
    coverage/contract-day ``duplicate_rows`` count only excess rows.
    ``expected_dates`` is optional, iterable, and treated as Shanghai dates;
    current/future expected dates are reported but excluded from missing dates.
    """
    report = _base_report()
    issues = report["validation"]["issues"]
    try:
        asof = _asof(now)
    except (ValueError, TypeError, OverflowError):
        issues["invalid_as_of"] = 1
        return report
    report["as_of"] = asof.isoformat()
    today = asof.tz_localize(None).normalize()
    if not isinstance(frame, pd.DataFrame):
        issues["not_a_dataframe"] = 1
        return report
    report["coverage"]["rows"] = len(frame)
    if frame.empty:
        issues["empty_input"] = 1
        return report
    if frame.columns.duplicated().any():
        issues["duplicate_columns"] = int(frame.columns.duplicated().sum())
        return report
    missing = sorted(set(_COLUMNS) - set(frame.columns))
    if missing:
        issues["missing_columns"] = len(missing)
        report["validation"]["missing_columns"] = missing
        return report

    data = frame.loc[:, _COLUMNS].copy().reset_index(drop=True)
    invalid = pd.Series(False, index=data.index)

    def flag(code, mask):
        nonlocal invalid
        mask = pd.Series(mask, index=data.index).fillna(True).astype(bool)
        count = int(mask.sum())
        if count:
            issues[code] = count
            invalid |= mask

    data["date"] = _times(data.date)
    data["trade_dt"] = _times(data.trade_dt)
    valid_date = data.date.notna()
    flag("invalid_date", ~valid_date)
    flag("invalid_trade_dt", data.trade_dt.isna())
    flag("trade_date_mismatch", valid_date & data.trade_dt.notna() &
         data.date.dt.normalize().ne(data.trade_dt.dt.normalize()))
    flag("subminute_timestamp", valid_date & data.date.ne(data.date.dt.floor("min")))
    valid_contract = data.ticker.astype("string").str.fullmatch(r"TL\d{2}(03|06|09|12)").fillna(False)
    flag("invalid_contract", ~valid_contract)
    duplicated = data.duplicated(["date", "ticker"], keep=False) & valid_date & valid_contract
    flag("duplicate_date_ticker", duplicated)
    booleans = pd.Series(False, index=data.index)
    nonreal = pd.Series(False, index=data.index)
    for column in _NUMERIC:
        if pd.api.types.is_bool_dtype(data[column].dtype):
            booleans |= data[column].notna()
        elif not pd.api.types.is_numeric_dtype(data[column].dtype):
            booleans |= data[column].map(lambda value: isinstance(value, (bool, np.bool_)))
        if pd.api.types.is_complex_dtype(data[column].dtype):
            # Reject the declared complex type, including zero-imaginary values;
            # replacing the entire column avoids any lossy float conversion.
            nonreal |= True
            data[column] = np.nan
        elif not pd.api.types.is_numeric_dtype(data[column].dtype):
            complex_values = data[column].map(lambda value: isinstance(value, (complex, np.complexfloating)))
            nonreal |= complex_values
            data[column] = pd.to_numeric(data[column].mask(complex_values), errors="coerce")
        else:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    flag("boolean_numeric", booleans)
    flag("nonreal_numeric", nonreal)
    numeric = data.loc[:, _NUMERIC].to_numpy(dtype=float, na_value=np.nan)
    finite = np.isfinite(numeric).all(axis=1)
    flag("nonfinite_numeric", ~finite)
    prices = numeric[:, :4]
    flag("invalid_ohlc", finite & (
        (prices <= 0).any(axis=1) | (numeric[:, 1] < prices.max(axis=1)) |
        (numeric[:, 2] > prices.min(axis=1))))
    flag("negative_volume_or_oi", finite & (numeric[:, 4:] < 0).any(axis=1))
    future = valid_date & data.date.gt(asof.tz_localize(None))
    flag("future_timestamp", future)
    current = valid_date & data.date.dt.normalize().eq(today)
    completed = valid_date & data.date.dt.normalize().lt(today)
    valid_rows = valid_date & valid_contract
    observed = sorted(data.loc[valid_date, "date"].dt.normalize().unique())
    coverage = report["coverage"]
    coverage.update({
        "valid_timestamp_rows": int(valid_date.sum()),
        "start": _iso(data.date.min()), "end": _iso(data.date.max()),
        "observed_days": len(observed), "contract_count": int(data.loc[valid_contract, "ticker"].nunique()),
        "current_day_rows": int(current.sum()), "future_rows": int(future.sum()),
        "completed_rows": int(completed.sum()),
        "duplicate_rows": int((data.duplicated(["date", "ticker"]) & valid_rows).sum()),
    })
    problems = []
    problem_days = set()
    repair_ranges = []
    active_repairs = {}
    grouped = data.loc[valid_rows].assign(_day=data.loc[valid_rows, "date"].dt.normalize())
    for (day, ticker), group in grouped.groupby(["_day", "ticker"], sort=True):
        aligned = group.date.eq(group.date.dt.floor("min"))
        minutes = group.date.dt.hour * 60 + group.date.dt.minute
        seen = set(minutes[aligned].astype(int))
        missing_minutes = sorted(_SESSION_MINUTES - seen)
        off_template = ~aligned | ~minutes.isin(_SESSION_MINUTES)
        extra = sorted({stamp.isoformat().split("T", 1)[1]
                        for stamp in group.loc[off_template, "date"]})
        extra = [stamp[:-3] if stamp.endswith(":00") else stamp for stamp in extra]
        duplicates = int(group.date.duplicated().sum())
        session_complete = not (missing_minutes or extra or duplicates) and len(group) == 255
        coverage["contract_days"] += 1
        coverage["complete_contract_days" if session_complete else "incomplete_contract_days"] += 1
        if day < today:
            coverage["complete_completed_contract_days" if session_complete else "incomplete_completed_contract_days"] += 1
            if session_complete:
                active_repairs.pop(ticker, None)
            else:
                if ticker not in active_repairs:
                    active_repairs[ticker] = {
                        "ticker": str(ticker), "start_date": day.date().isoformat(),
                        "end_date": day.date().isoformat(), "contract_days": 0,
                        "observed_rows": 0, "missing_minutes_total": 0,
                        "extra_minutes_total": 0, "duplicate_rows": 0,
                        "calendar_verified": False, "action": "inspect_source_coverage",
                    }
                    repair_ranges.append(active_repairs[ticker])
                repair = active_repairs[ticker]
                repair["end_date"] = day.date().isoformat()
                increments = {"contract_days": 1, "observed_rows": len(group),
                              "missing_minutes_total": len(missing_minutes),
                              "extra_minutes_total": len(extra), "duplicate_rows": duplicates}
                for key, count in increments.items():
                    repair[key] += count
                    report["repair_summary"][key] += count
        bad_rows = int(invalid.loc[group.index].sum())
        if not session_complete or bad_rows or day >= today:
            coverage["problem_contract_days"] += 1
            problem_days.add(day)
            if len(problems) < _LIMIT:
                problems.append({
                    "date": day.date().isoformat(), "ticker": str(ticker), "row_count": len(group),
                    "unique_timestamps": int(group.date.nunique()), "duplicate_rows": duplicates,
                    "missing_minutes_count": len(missing_minutes), "extra_minutes_count": len(extra),
                    "missing_minutes": [_clock(minute) for minute in missing_minutes[:_MINUTE_LIMIT]],
                    "extra_minutes": extra[:_MINUTE_LIMIT],
                    "minute_samples_truncated": len(missing_minutes) > _MINUTE_LIMIT or len(extra) > _MINUTE_LIMIT,
                    "invalid_rows": bad_rows, "session_complete": session_complete,
                    "completion": "completed_observed_day" if day < today else (
                        "current_day_withheld" if day == today else "future_day"),
                })
    coverage["problem_days"] = len(problem_days)
    report["contract_day_issues"] = problems
    report["truncated"]["contract_day_issues"] = coverage["problem_contract_days"] > _LIMIT
    report["repair_summary"]["range_count"] = len(repair_ranges)
    report["repair_ranges"] = sorted(repair_ranges, key=lambda item: (item["start_date"], item["ticker"]))[:_LIMIT]
    report["truncated"]["repair_ranges"] = len(repair_ranges) > _LIMIT

    gaps = []
    for left, right in zip(observed, observed[1:]):
        gap = (pd.Timestamp(right) - pd.Timestamp(left)).days - 1
        if gap > 0:
            coverage["observed_gap_count"] += 1
            coverage["largest_unobserved_calendar_gap_days"] = max(coverage["largest_unobserved_calendar_gap_days"], gap)
            gaps.append({"after": pd.Timestamp(left).date().isoformat(),
                         "before": pd.Timestamp(right).date().isoformat(), "unobserved_calendar_days": gap})
    report["observed_gaps"] = sorted(gaps, key=lambda gap: (-gap["unobserved_calendar_days"], gap["after"]))[:_LIMIT]
    report["truncated"]["observed_gaps"] = len(gaps) > _LIMIT

    if expected_dates is not None:
        report["calendar"]["source"] = "caller_provided"
        try:
            if isinstance(expected_dates, (str, bytes)):
                raise ValueError("calendar must be iterable dates, not a string")
            dates = _times(list(expected_dates))
            if dates.isna().any():
                issues["invalid_expected_dates"] = int(dates.isna().sum())
            else:
                expected = set(dates.dt.normalize())
                prior = {day for day in expected if day < today}
                missing_dates = sorted(prior - {pd.Timestamp(day) for day in observed})
                report["calendar"].update({
                    "expected_dates_count": len(expected), "completed_expected_dates_count": len(prior),
                    "not_completed_expected_dates_count": len(expected - prior),
                    "missing_dates_count": len(missing_dates),
                    "missing_dates": [day.date().isoformat() for day in missing_dates[:_LIMIT]],
                })
                report["truncated"]["missing_dates"] = len(missing_dates) > _LIMIT
        except (ValueError, TypeError, OverflowError):
            issues["invalid_expected_dates"] = 1
    report["validation"].update({"valid": not issues, "invalid_rows": int(invalid.sum()),
                                 "example_row_positions": invalid[invalid].index[:20].tolist()})
    if issues:
        report["status"] = "invalid"
    elif coverage["problem_contract_days"] or report["calendar"]["missing_dates_count"]:
        report["status"] = "incomplete"
    else:
        report["status"] = "complete_observed"
    return report
