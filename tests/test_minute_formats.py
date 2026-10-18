import csv
import io

import pandas as pd
import pytest

from minute_formats import parse_tqsdk_csv, validate_conversion


def tq_csv(times=None, symbol="CFFEX.TL2606", edits=None, extras=None):
    times = times or ["2026-05-11 09:30:00", "2026-05-11 09:31:00"]
    fields = ["open", "high", "low", "close", "volume", "open_oi", "close_oi"]
    header = ["datetime", "datetime_nano"] + [f"{symbol}.{key}" for key in fields]
    rows = []
    for value in times:
        stamp = pd.Timestamp(value).tz_localize("Asia/Shanghai")
        rows.append([stamp.strftime("%Y-%m-%d %H:%M:%S") + ".000000000", str(stamp.value),
                     "100.0", "101.0", "99.0", "100.5", "10", "101", "102"])
    for row, col, value in edits or []:
        rows[row][col] = value
    for name, value in extras or []:
        header.append(name)
        for row in rows:
            row.append(value)
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode()


def parse(payload=None, **kwargs):
    return parse_tqsdk_csv(payload or tq_csv(), bar_seconds=60, **kwargs)


def test_tqsdk_time_integer_precision_end_shift_and_fields():
    frame, conversion = parse()
    assert list(frame.columns) == ["date", "trade_dt", "ticker", "open", "high", "low", "close", "volume", "oi"]
    assert frame.iloc[0].date == pd.Timestamp("2026-05-11 09:31:00")
    assert frame.iloc[0].ticker == "TL2606"
    assert frame.iloc[0].oi == 102
    assert conversion["timestamp_shift_seconds"] == 60
    assert conversion["interval_verified"] is False
    assert conversion["auxiliary_columns"] == ["CFFEX.TL2606.open_oi"]
    assert validate_conversion(conversion) == conversion


@pytest.mark.parametrize("period", [None, 0, 1, 15, 900, True, "60", 60.0])
def test_period_requires_explicit_exact_60_seconds(period):
    with pytest.raises(ValueError, match="60"):
        parse_tqsdk_csv(tq_csv(), bar_seconds=period)


@pytest.mark.parametrize("symbol", ["KQ.m@CFFEX.TL", "CFFEX.TL0", "CFFEX.TL2605", "SHFE.TL2606", "cffex.TL2606"])
def test_only_individual_tl_quarterly_contract_supported(symbol):
    with pytest.raises(ValueError):
        parse(tq_csv(symbol=symbol))


@pytest.mark.parametrize("column,value", [
    (1, "1.7784566e18"), (1, "1778453400000000000.0"), (1, "9223372036854775808"),
    (1, "1778453400000000001"), (1, "-1"), (1, ""), (1, " 1778453400000000000"),
    (0, "2026-05-11 09:30:00"), (0, "2026-05-11T09:30:00.000000000+08:00"),
    (0, "2026-05-11 09:30:00.000000001"), (0, "bad"),
    (2, "True"), (2, "NaN"), (2, "inf"), (2, "#N/A"), (2, "100+1j"),
    (6, "-1"), (7, "-1"), (7, "NaN"), (7, ""), (8, "-1"), (4, "0"),
    (3, "90"), (2, ""),
    (0, "2026-02-30 09:30:00.000000000"), (2, "1e309"),
])
def test_invalid_source_values_never_coerce_or_fill(column, value):
    with pytest.raises(ValueError):
        parse(tq_csv(edits=[(0, column, value)]))


@pytest.mark.parametrize("payload", [
    b"", b"\xff", b"\x00", b"datetime,datetime\n1,1\n", b"one,two\n1\n",
    tq_csv(extras=[("CFFEX.TL2609.open", "100")]),
    tq_csv(extras=[("CFFEX.TL2606.id", "1")]),
    tq_csv().replace(b",CFFEX.TL2606.open_oi", b",unknown"),
    tq_csv().replace(b"100.0,101.0", b"100.0,101.0,103.0"),
    tq_csv().splitlines()[0] + b"\n",
])
def test_unknown_multisymbol_and_ragged_formats_reject(payload):
    with pytest.raises(ValueError):
        parse_tqsdk_csv(payload, bar_seconds=60)


def test_duplicate_out_of_order_and_subminute_reject():
    for times in [
        ["2026-05-11 09:30:00"] * 2,
        ["2026-05-11 09:31:00", "2026-05-11 09:30:00"],
        ["2026-05-11 09:30:01", "2026-05-11 09:31:01"],
    ]:
        with pytest.raises(ValueError):
            parse(tq_csv(times=times))


def test_gaps_are_not_filled_and_single_rows_remain_declared():
    frame, conversion = parse(tq_csv(times=["2026-05-11 09:30:00", "2026-05-11 09:31:00", "2026-05-11 10:00:00"]))
    assert len(frame) == 3
    assert conversion["observed_one_minute_pairs"] == 1
    frame, conversion = parse(tq_csv(times=["2026-05-11 09:30:00"]))
    assert len(frame) == 1
    assert conversion["observed_one_minute_pairs"] == 0
    frame, conversion = parse(tq_csv(times=["2026-05-11 09:30:00", "2026-05-11 09:45:00"]))
    assert len(frame) == 2
    assert conversion["interval_verified"] is False


def test_payload_and_row_bounds():
    with pytest.raises(ValueError, match="size"):
        parse(max_bytes=10)
    with pytest.raises(ValueError, match="row"):
        parse(max_rows=1)


@pytest.mark.parametrize("field,value", [
    ("schema_version", "unknown"), ("timestamp_shift_seconds", 0),
    ("declared_interval_seconds", 900), ("interval_verified", True),
    ("source_timezone", "UTC"), ("source_columns", ["bad"]),
    ("symbol", "CFFEX.TL0"), ("observed_one_minute_pairs", -1),
    ("auxiliary_columns", []), ("arbitrary_secret", "token=x"),
])
def test_conversion_metadata_is_fixed_and_validated(field, value):
    _, conversion = parse()
    conversion[field] = value
    with pytest.raises(ValueError):
        validate_conversion(conversion)


def test_conversion_rejects_non_dictionary_wrong_types_and_untrusted_symbol():
    for value in [None, [], {"symbol": "CFFEX.TL0"}]:
        with pytest.raises(ValueError):
            validate_conversion(value)
    for changes in [{"observed_one_minute_pairs": True}, {"timestamp_shift_seconds": 60.0}]:
        _, value = parse()
        value.update(changes)
        with pytest.raises(ValueError):
            validate_conversion(value)
