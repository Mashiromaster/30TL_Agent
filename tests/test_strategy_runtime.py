from dataclasses import replace

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from strategy_runtime import (
    DEFAULT_STRATEGY_ID,
    STRATEGY_SPECS,
    SignalDecision,
    SignalPolicy,
    get_strategy_spec,
)


def _prediction_frame(rows=800):
    phase = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-02 09:30", periods=rows, freq="min"),
            "close": 100.0 + phase * 0.001,
            "Pred_Ret": np.sin(phase / 17.0) + 0.2 * np.cos(phase / 5.0),
            "Market_Regime": (phase.astype(int) // 150) % 3,
            "Model_Used": "flat_combined",
        }
    )


def test_registry_has_one_default_and_two_explicit_legacy_specs():
    assert DEFAULT_STRATEGY_ID == "tl_regime_breakout_v1"
    assert set(STRATEGY_SPECS) == {
        "tl_regime_breakout_v1",
        "tl_inference_breakout_legacy_v1",
        "tl_rank_sweetspot_legacy_v1",
    }
    assert get_strategy_spec() is STRATEGY_SPECS[DEFAULT_STRATEGY_ID]


def test_strategy_fingerprint_is_stable_and_changes_with_semantics():
    spec = get_strategy_spec()

    assert spec.fingerprint() == spec.fingerprint()
    assert spec.fingerprint() == get_strategy_spec("tl_regime_breakout_v1").fingerprint()
    assert replace(spec, execution_lag_bars=2).fingerprint() != spec.fingerprint()


def test_signal_policy_rejects_unsorted_and_duplicate_timestamps():
    policy = SignalPolicy(get_strategy_spec())
    frame = _prediction_frame(600)

    unsorted = frame.copy()
    unsorted.loc[[10, 11], "date"] = unsorted.loc[[11, 10], "date"].to_numpy()
    with pytest.raises(ValueError, match="strictly increasing"):
        policy.evaluate_frame(unsorted)

    duplicated = frame.copy()
    duplicated.loc[11, "date"] = duplicated.loc[10, "date"]
    with pytest.raises(ValueError, match="duplicate"):
        policy.evaluate_frame(duplicated)


def test_future_rows_do_not_change_historical_policy_outputs():
    policy = SignalPolicy(get_strategy_spec())
    frame = _prediction_frame(800)

    prefix = policy.evaluate_frame(frame.iloc[:700])
    full = policy.evaluate_frame(frame)
    stable_columns = [
        "Pred_Smooth",
        "Upper_Q",
        "Lower_Q",
        "Raw_Signal",
        "Signal_Confirmed",
        "Confidence",
        "Target_Position",
        "Effective_Position",
        "Execution_Lag_Bars",
        "Effective_From",
        "Decision_Ready",
        "Missing_Reason",
    ]

    pdt.assert_frame_equal(
        prefix[stable_columns].reset_index(drop=True),
        full.iloc[:700][stable_columns].reset_index(drop=True),
    )


def test_execution_position_is_lagged_one_bar():
    spec = replace(
        get_strategy_spec("tl_inference_breakout_legacy_v1"),
        lookback_bars=8,
        min_periods=4,
    )
    evaluated = SignalPolicy(spec).evaluate_frame(_prediction_frame(80))

    expected = evaluated["Target_Position"].shift(1).fillna(0.0)
    pdt.assert_series_equal(
        evaluated["Effective_Position"],
        expected.rename("Effective_Position"),
    )
    assert (evaluated["Target_Position"] != 0).any()


def test_latest_returns_typed_decision_with_provenance():
    spec = replace(
        get_strategy_spec("tl_rank_sweetspot_legacy_v1"),
        lookback_bars=20,
        min_periods=20,
        confirm_bars=3,
    )
    decision = SignalPolicy(spec).latest(
        _prediction_frame(100), model_release_id="release-test"
    )

    assert isinstance(decision, SignalDecision)
    assert decision.strategy_id == spec.strategy_id
    assert decision.strategy_fingerprint == spec.fingerprint()
    assert decision.model_release_id == "release-test"
    assert decision.effective_from is None
    assert decision.execution_lag_bars == 1
    assert decision.to_dict()["execution_lag_bars"] == 1
    assert decision.to_dict()["suggested_weight"] == decision.target_weight


def test_insufficient_history_is_explicit_not_an_implicit_flat_decision():
    spec = replace(
        get_strategy_spec("tl_inference_breakout_legacy_v1"),
        lookback_bars=20,
        min_periods=20,
    )

    decision = SignalPolicy(spec).latest(_prediction_frame(10))

    assert decision.direction == 0
    assert decision.decision_ready is False
    assert decision.missing_reason == "insufficient_policy_history"


def test_decision_id_changes_when_same_timestamp_decision_content_changes():
    spec = replace(
        get_strategy_spec("tl_inference_breakout_legacy_v1"),
        lookback_bars=8,
        min_periods=4,
    )
    frame = _prediction_frame(80)
    changed = frame.copy()
    changed.loc[changed.index[-1], "Pred_Ret"] += 50.0

    original = SignalPolicy(spec).latest(frame, model_release_id="legacy")
    revised = SignalPolicy(spec).latest(changed, model_release_id="legacy")

    assert original.as_of == revised.as_of
    assert original.decision_id != revised.decision_id
