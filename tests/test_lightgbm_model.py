import numpy as np
import pandas as pd

from LightGBM_model import (
    _date_block_stability,
    _release_gate_passed,
    _validation_promotion_eligible,
    prepare_training_data,
)
from training_data import LabelSpec


def test_prepare_training_data_uses_registry_and_adjusted_price_labels():
    rows = 80
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=rows, freq="min"),
            "ticker": "TL",
            "close": np.r_[np.arange(40.0), np.arange(1000.0, 1040.0)],
            "adj_close": np.arange(100.0, 180.0),
            "Spread_Mean": np.linspace(1, 2, rows),
            "YC_Slope_10Y_1Y": np.linspace(0.1, 0.2, rows),
            "GP_001_alpha": np.linspace(-1, 1, rows),
            "constant": 1.0,
        }
    )

    model, features = prepare_training_data(
        frame,
        label_spec=LabelSpec(
            horizon_bars=2, entry_lag_bars=0, allow_overnight=True
        ),
    )

    expected = (102.0 / 100.0 - 1) * 100
    assert model.iloc[0]["Target_Ret"] == expected
    assert "Spread_Mean" in features
    assert "YC_Slope_10Y_1Y" in features
    assert "GP_001_alpha" in features
    assert "constant" not in features


def test_validation_stability_uses_five_contiguous_date_blocks():
    days = pd.date_range("2025-01-01", periods=10, freq="D")
    frame = pd.DataFrame(
        {
            "date": np.repeat(days, 12),
            "Target_Ret": np.tile(np.arange(12.0), 10),
        }
    )
    prediction = frame["Target_Ret"].to_numpy()

    report = _date_block_stability(frame, prediction, blocks=5)

    assert report["fold_count"] == 5
    assert report["positive_fold_count"] == 5
    assert report["same_sign_ratio"] == 1.0


def test_sealed_holdout_can_only_veto_a_validation_qualified_candidate():
    stable = {
        "fold_count": 5,
        "positive_fold_count": 3,
        "median_ic": 0.01,
    }
    eligible = _validation_promotion_eligible(stable)

    assert eligible is True
    assert _release_gate_passed(eligible, 0.005) is True
    assert _release_gate_passed(eligible, -0.001) is False
    assert _release_gate_passed(False, 0.10) is False
