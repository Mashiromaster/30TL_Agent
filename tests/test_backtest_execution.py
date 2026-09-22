from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from backtest import Backtester
from backtest_v2 import run_rank_backtest
from strategy_runtime import get_strategy_spec


def test_backtest_uses_effective_lagged_position_for_returns():
    rows = 80
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-05 09:30", periods=rows, freq="min"),
            "close": 100 + np.arange(rows, dtype=float) * 0.1,
            "Pred_Ret": np.r_[np.linspace(-1, 1, rows - 10), np.full(10, 10.0)],
            "Market_Regime": np.zeros(rows, dtype=int),
            "Model_Used": "fake",
        }
    )
    spec = replace(
        get_strategy_spec("tl_inference_breakout_legacy_v1"),
        lookback_bars=8,
        min_periods=4,
    )

    tester = Backtester(frame, strategy_spec=spec)
    tester.run_strategy()

    expected_position = tester.df["Target_Position"].shift(1).fillna(0.0)
    pd.testing.assert_series_equal(
        tester.df["Position"], expected_position.rename("Position")
    )
    expected_net = (
        tester.df["Held_Position"] * tester.df["Asset_Ret"]
        - tester.df["Cost"]
    )
    np.testing.assert_allclose(tester.df["Net_Ret"], expected_net)
    expected_trades = tester.df["Executed_Position"].diff().fillna(
        tester.df["Executed_Position"]
    ).abs()
    np.testing.assert_allclose(tester.df["Trades"], expected_trades)
    np.testing.assert_allclose(tester.df["Strategy_Ret"], expected_net)


def test_execution_bar_price_jump_is_not_credited_before_close_entry():
    rows = 80
    dates = pd.date_range("2026-01-05 09:30", periods=rows, freq="min")
    spec = replace(
        get_strategy_spec("tl_inference_breakout_legacy_v1"),
        lookback_bars=8,
        min_periods=4,
    )
    frame = pd.DataFrame(
        {
            "date": dates,
            "close": np.full(rows, 100.0),
            "Pred_Ret": np.linspace(-1.0, 10.0, rows),
            "Market_Regime": np.zeros(rows, dtype=int),
            "Model_Used": "fake",
        }
    )
    preliminary = Backtester(frame, strategy_spec=spec)
    preliminary.run_strategy()
    signal_index = int(
        np.flatnonzero(preliminary.df["Target_Position"].to_numpy() != 0)[0]
    )
    execution_index = signal_index + spec.execution_lag_bars

    frame.loc[execution_index:, "close"] = 200.0
    tester = Backtester(frame, strategy_spec=spec)
    tester.run_strategy()

    assert tester.df.iloc[execution_index]["Asset_Ret"] == 1.0
    assert tester.df.iloc[execution_index]["Executed_Position"] != 0.0
    assert tester.df.iloc[execution_index]["Held_Position"] == 0.0
    assert tester.df.iloc[execution_index]["Net_Ret"] == pytest.approx(
        -tester.df.iloc[execution_index]["Cost"]
    )


def test_legacy_rank_entrypoint_rejects_unversioned_parameter_overrides():
    with pytest.raises(ValueError, match="versioned StrategySpec"):
        run_rank_backtest(rank_pct=0.25)

    with pytest.raises(ValueError, match="versioned StrategySpec"):
        run_rank_backtest(smooth_span=80)
