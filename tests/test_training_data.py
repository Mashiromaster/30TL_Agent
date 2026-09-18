import pandas as pd
import pytest

from training_data import (
    LabelSpec,
    SplitSpec,
    build_forward_labels,
    split_train_validation_holdout,
)


def test_forward_labels_do_not_cross_ticker_or_trading_day():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-02 14:58", "2025-01-02 14:59", "2025-01-03 09:30",
                    "2025-01-02 14:58", "2025-01-02 14:59", "2025-01-03 09:30",
                ]
            ),
            "ticker": ["TL2503"] * 3 + ["TL2506"] * 3,
            "adj_close": [100, 101, 200, 300, 303, 600],
        }
    )

    result = build_forward_labels(
        frame,
        LabelSpec(horizon_bars=1, entry_lag_bars=0, allow_overnight=False),
    )

    assert result.loc[0, "Target_Ret"] == pytest.approx(1.0)
    assert bool(result.loc[1, "label_valid"]) is False
    assert bool(result.loc[2, "label_valid"]) is False
    assert result.loc[3, "Target_Ret"] == pytest.approx(1.0)


def test_purged_date_split_has_no_label_overlap_and_seals_holdout():
    dates = pd.date_range("2025-01-01", periods=12, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates,
            "ticker": "TL",
            "adj_close": range(100, 112),
        }
    )
    labelled = build_forward_labels(
        frame,
        LabelSpec(horizon_bars=1, entry_lag_bars=0, allow_overnight=True),
    )

    train, validation, holdout = split_train_validation_holdout(
        labelled,
        SplitSpec(validation_days=3, sealed_holdout_days=3, min_train_days=3),
    )

    assert train["date"].dt.normalize().nunique() >= 3
    assert train["label_end_ts"].max() < validation["date"].min()
    assert validation["label_end_ts"].max() < holdout["date"].min()
    assert set(train.index).isdisjoint(validation.index)
    assert set(validation.index).isdisjoint(holdout.index)


def test_label_builder_requires_adjusted_price_by_default():
    frame = pd.DataFrame(
        {"date": pd.to_datetime(["2025-01-01", "2025-01-02"]), "close": [1, 2]}
    )

    try:
        build_forward_labels(frame, LabelSpec())
    except ValueError as exc:
        assert "adj_close" in str(exc)
    else:
        raise AssertionError("raw close must not be silently used for labels")
