import pytest

from LightGBM_model import _strategy_release_gate


def _metrics(**overrides):
    values = {
        "total_return": 0.01,
        "sharpe": 0.5,
        "annual_volatility": 0.08,
        "max_drawdown": -0.12,
        "active_ratio": 0.4,
        "observation_days": 60,
    }
    values.update(overrides)
    return values


def test_strategy_release_gate_requires_cost_adjusted_profitability():
    passed, reason = _strategy_release_gate(_metrics(total_return=-0.03))

    assert passed is False
    assert reason == "total_return_below_threshold"


def test_strategy_release_gate_accepts_positive_return_and_sharpe():
    passed, reason = _strategy_release_gate(
        _metrics(total_return=0.03, sharpe=0.4)
    )

    assert passed is True
    assert reason == "passed"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_total_return": -0.01},
        {"min_sharpe": -0.01},
        {"max_drawdown_limit": 0.36},
    ],
)
def test_strategy_release_policy_can_only_tighten_baseline(kwargs):
    with pytest.raises(ValueError):
        _strategy_release_gate(_metrics(), **kwargs)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"active_ratio": 0.0}, "strategy_never_active"),
        ({"observation_days": 19}, "insufficient_observation_days"),
        ({"max_drawdown": -0.36}, "max_drawdown_breach"),
        ({"sharpe": 0.0, "total_return": 0.03}, "sharpe_below_threshold"),
        ({"annual_volatility": float("nan")}, "non_finite_metrics"),
        ({"annual_volatility": -0.01}, "invalid_annual_volatility"),
        ({"active_ratio": 1.1}, "invalid_active_ratio"),
        ({"max_drawdown": 0.01}, "invalid_max_drawdown"),
    ],
)
def test_strategy_release_gate_fails_closed(overrides, reason):
    passed, actual_reason = _strategy_release_gate(_metrics(**overrides))

    assert passed is False
    assert actual_reason == reason
