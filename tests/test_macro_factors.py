import numpy as np
import pandas as pd
import pytest

from macro_factors import MacroFactorComputer


def _monthly_frame(column, values, kind):
    dates = pd.date_range("2022-01-01", periods=len(values), freq="MS")
    release_ts = []
    for date in dates:
        if kind == "pmi":
            release_ts.append(date + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=9, minutes=30))
        elif kind == "cpi":
            release_ts.append(date + pd.offsets.MonthBegin(1) + pd.Timedelta(days=14, hours=9, minutes=30))
        else:
            release_ts.append(date + pd.offsets.MonthBegin(1) + pd.Timedelta(days=19, hours=17))
    return pd.DataFrame({"date": dates, "available_ts": release_ts, column: values})


def test_monthly_factors_stay_on_native_release_frequency():
    n = 36
    raw = {
        "pmi": _monthly_frame("PMI_Manufacturing", np.linspace(48, 52, n), "pmi"),
        "cpi": _monthly_frame("CPI_YoY", np.linspace(-0.5, 2.0, n), "cpi"),
        "m2": _monthly_frame("M2_YoY", np.linspace(8, 12, n), "monetary"),
    }

    result = MacroFactorComputer().compute_all(raw)

    assert "available_ts" in result.columns
    assert "Macro_State_Composite" in result.columns
    assert "M2_State_ZScore" in result.columns
    assert "CPI_Acceleration_3M" in result.columns
    assert len(result) <= n * 3
    assert result["available_ts"].is_monotonic_increasing


def test_daily_curve_factor_is_available_once_without_double_shift():
    curve = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "available_ts": pd.to_datetime(["2025-01-03 09:00", "2025-01-06 09:00"]),
            "Yield_1Y": [1.5, 1.6],
            "Yield_5Y": [1.8, 1.9],
            "Yield_10Y": [2.0, 2.1],
            "Yield_30Y": [2.3, 2.4],
        }
    )

    result = MacroFactorComputer().compute_all({"yield_curve": curve})

    first = result[result["available_ts"] == pd.Timestamp("2025-01-03 09:00")].iloc[0]
    assert first["YC_Slope_10Y_1Y"] == 0.5
    assert first["YC_Slope_30Y_10Y"] == pytest.approx(0.3)


def test_placeholders_and_constant_zero_factors_are_not_emitted():
    cross = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "available_ts": pd.to_datetime(["2025-01-03 09:00", "2025-01-06 09:00"]),
            "CN_US_10Y_Spread": [-1.8, -1.7],
        }
    )

    result = MacroFactorComputer().compute_all({"cross_market": cross})

    assert "Credit_Spread_Proxy" not in result.columns
    factor_cols = [
        c for c in result.columns if not c.startswith("_") and c != "available_ts"
    ]
    assert not any(result[c].fillna(0).eq(0).all() for c in factor_cols)


def test_lpr_is_emitted_as_a_monthly_policy_factor():
    cross = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-20", "2025-02-20", "2025-03-20"]),
            "available_ts": pd.to_datetime(
                ["2025-01-20 09:30", "2025-02-20 09:30", "2025-03-20 09:30"]
            ),
            "LPR_1Y": [3.10, 3.10, 3.00],
            "LPR_5Y": [3.60, 3.60, 3.50],
        }
    )
    computer = MacroFactorComputer()

    result = computer.compute_all({"cross_market": cross})

    assert "LPR_1Y_Level" in result.columns
    assert "LPR_1Y_Change" in result.columns
    assert "LPR_5Y_1Y_Spread" in result.columns
    assert computer.factor_metadata["LPR_1Y_Level"].frequency == "monthly"
    assert computer.factor_ttls["LPR_1Y_Level"] == pd.Timedelta(days=50)


def test_repo_levels_changes_and_market_segmentation_spread_are_emitted():
    dates = pd.bdate_range("2025-01-02", periods=25)
    repo = pd.DataFrame(
        {
            "date": dates,
            "Repo_FR007": np.linspace(1.8, 2.2, len(dates)),
            "Repo_FDR007": np.linspace(1.7, 2.0, len(dates)),
        }
    )
    computer = MacroFactorComputer()

    result = computer.compute_all({"repo": repo})

    assert "Repo_FR007_Level" in result.columns
    assert "Repo_FR007_Change_5D" in result.columns
    assert "Repo_FR007_Deviation" in result.columns
    assert "Repo_FR007_FDR007_Spread" in result.columns
    assert computer.factor_ttls["Repo_FR007_Level"] == pd.Timedelta(days=5)


def test_future_observations_cannot_change_past_factor_values():
    dates = pd.bdate_range("2025-01-01", periods=40)
    curve = pd.DataFrame(
        {
            "date": dates,
            "Yield_1Y": np.linspace(1.0, 1.4, 40),
            "Yield_5Y": np.linspace(1.4, 1.9, 40),
            "Yield_10Y": np.linspace(1.8, 2.5, 40),
            "Yield_30Y": np.linspace(2.2, 3.1, 40),
        }
    )
    prefix = MacroFactorComputer().compute_all({"yield_curve": curve.iloc[:25]})
    full = MacroFactorComputer().compute_all({"yield_curve": curve})
    cutoff = prefix["available_ts"].max()
    past = full.loc[full["available_ts"] <= cutoff]
    common = [
        col for col in prefix.columns
        if col in past.columns and not col.startswith("_asof__")
    ]

    pd.testing.assert_frame_equal(
        prefix[common].reset_index(drop=True),
        past[common].reset_index(drop=True),
    )
