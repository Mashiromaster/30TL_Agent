import numpy as np
import pandas as pd
import pytest

from genetic_factor_miner import GeneticFactorMiner


def _data(rows=160):
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {"a": rng.normal(size=rows), "b": rng.normal(size=rows)}
    )
    target = frame["a"] + frame["b"] + rng.normal(scale=0.05, size=rows)
    return frame, target


def test_gp_reuses_training_scaler_and_freezes_validation_selection():
    train_x, train_y = _data()
    validation_x, validation_y = _data()
    miner = GeneticFactorMiner(
        population_size=40,
        generations=2,
        hall_of_fame=8,
        n_components=5,
        n_jobs=1,
    ).fit(train_x, train_y)
    mean_before = miner.scaler_.mean_.copy()
    miner.select(validation_x, validation_y, min_abs_ic=0.0, max_factors=3)

    transformed = miner.transform(validation_x * 100)

    assert np.array_equal(miner.scaler_.mean_, mean_before)
    assert 1 <= transformed.shape[1] <= 3
    assert all(name.startswith("GP_") for name in transformed.columns)


def test_gp_rejects_feature_order_drift():
    train_x, train_y = _data()
    miner = GeneticFactorMiner(
        population_size=30,
        generations=1,
        hall_of_fame=5,
        n_components=3,
        n_jobs=1,
    ).fit(train_x, train_y)

    with pytest.raises(ValueError, match="schema/order"):
        miner.select(train_x[["b", "a"]], train_y, min_abs_ic=0.0)
