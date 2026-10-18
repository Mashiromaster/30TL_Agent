import json
import pickle
from dataclasses import asdict
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import genetic_discovery
from genetic_discovery import (
    DiscoveryResult,
    prepare_discovery,
    run_discovery,
    save_discovery,
)
from training_data import LabelSpec, SplitSpec


LABEL = LabelSpec(horizon_bars=1, entry_lag_bars=1)
SPLIT = SplitSpec(validation_days=10, sealed_holdout_days=10, min_train_days=20)


@pytest.fixture
def frame():
    dates = [day + pd.Timedelta(hours=10, minutes=minute)
             for day in pd.bdate_range("2025-01-01", periods=60)
             for minute in range(8)]
    values = np.arange(len(dates), dtype=float)
    return pd.DataFrame({
        "date": dates,
        "ticker": "TL2506",
        "adj_close": 100 + values * .02 + np.sin(values / 4) * .03,
        "signal": np.cos(values / 4),
        "noise": np.sin(values / 13),
        "Target_Ret": -999.,
        "label_valid": False,
        "label_start_ts": pd.Timestamp("1999-01-01"),
        "label_end_ts": pd.Timestamp("1999-01-01"),
        "next_close": values + 500,
        "future_close": values + 900,
        "Target_30": values,
        "label_answer": values,
        "Pred_Smooth": values,
        "Pred_Custom": values,
        "Should_Trade": values % 2,
    })


class FakeMiner:
    feature_names_in_ = ["signal", "noise"]

    def __init__(self):
        self.transformed = []

    def transform(self, frame):
        self.transformed.append(frame.copy())
        return pd.DataFrame({"GP_000": frame["signal"] * 2}, index=frame.index)


def fake_report(programs=True):
    return {
        "selection_basis": "validation_only",
        "feature_names": ["signal", "noise"],
        "programs": [{"name": "GP_000", "expression": "add(signal, signal)",
                      "validation_ic": .15, "complexity": 19,
                      "candidate_index": 2}] if programs else [],
    }


def test_prepare_rebuilds_labels_and_preserves_input(frame):
    original = frame.copy(deep=True)
    train, validation, holdout = prepare_discovery(frame, label_spec=LABEL, split_spec=SPLIT)
    pd.testing.assert_frame_equal(frame, original)
    assert train["Target_Ret"].iloc[0] == pytest.approx(
        (frame["adj_close"].iloc[2] / frame["adj_close"].iloc[1] - 1) * 100)
    assert train["label_valid"].all()
    assert train["label_end_ts"].max() < validation["date"].min()
    assert validation["label_end_ts"].max() < holdout["date"].min()
    assert set(train["date"]).isdisjoint(validation["date"])
    assert set(validation["date"]).isdisjoint(holdout["date"])
    assert holdout["date"].dt.normalize().nunique() == 10


def test_prepare_purges_cross_boundary_labels():
    sample = pd.DataFrame({"date": pd.bdate_range("2025-01-01", periods=15),
                           "ticker": "TL2506", "adj_close": np.arange(100., 115.),
                           "signal": np.arange(15)})
    train, validation, holdout = prepare_discovery(
        sample, label_spec=LabelSpec(horizon_bars=2, entry_lag_bars=0, allow_overnight=True),
        split_spec=SplitSpec(validation_days=4, sealed_holdout_days=4, min_train_days=3))
    assert train["label_end_ts"].max() < validation["date"].min()
    assert validation["label_end_ts"].max() < holdout["date"].min()
    assert len(train) == 3
    assert len(validation) == 2


def test_sparse_dates_are_removed_before_partition_boundaries(frame, monkeypatch):
    sparse = frame.groupby(frame["date"].dt.normalize()).head(1).iloc[-30:].copy()
    sparse["date"] += pd.Timedelta(days=300)
    sparse["ticker"] = [f"TL_SPARSE_{index}" for index in range(len(sparse))]
    combined = pd.concat([frame, sparse], ignore_index=True)
    monkeypatch.setattr(genetic_discovery, "mine_genetic_factors",
                        Mock(return_value=(FakeMiner(), fake_report())))
    result = run_discovery(combined, label_spec=LABEL, split_spec=SPLIT)
    assert result.report["data_quality"]["input_days"] == 90
    assert result.report["data_quality"]["labeled_days"] == 60
    assert result.report["data_quality"]["days_without_valid_labels"] == 30
    assert result.report["partitions"]["validation"]["days"] == 10
    assert result.report["partitions"]["sealed_holdout"]["days"] == 10
    assert result.factors["date"].max() < sparse["date"].min()


@pytest.mark.parametrize("column", ["adj_close", "date", "ticker"])
def test_required_columns_are_not_invented(frame, column):
    frame["close"] = frame["adj_close"]
    with pytest.raises(ValueError, match=column):
        prepare_discovery(frame.drop(columns=column), label_spec=LABEL, split_spec=SPLIT)


@pytest.mark.parametrize("column,value", [("date", pd.NaT), ("ticker", None),
                                         ("ticker", "  "), ("adj_close", np.inf),
                                         ("adj_close", 0), ("adj_close", np.nan)])
def test_bad_input_fails_explicitly(frame, column, value):
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match=column):
        prepare_discovery(frame, label_spec=LABEL, split_spec=SPLIT)


def test_duplicate_contract_timestamp_is_rejected(frame):
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        prepare_discovery(frame, label_spec=LABEL, split_spec=SPLIT)


@pytest.mark.parametrize("split", [SplitSpec(validation_days=0),
                                    SplitSpec(sealed_holdout_days=0),
                                    SplitSpec(min_train_days=0)])
def test_invalid_partition_spec_is_rejected(frame, split):
    with pytest.raises(ValueError, match="positive"):
        prepare_discovery(frame, label_spec=LABEL, split_spec=split)


def test_short_history_fails_before_fitting(frame, monkeypatch):
    mine = Mock()
    monkeypatch.setattr(genetic_discovery, "mine_genetic_factors", mine)
    with pytest.raises(ValueError, match="240 trading days"):
        run_discovery(frame)
    mine.assert_not_called()


def test_run_uses_validation_filters_leaks_and_never_transforms_holdout(frame, monkeypatch):
    original = frame.copy(deep=True)
    miner = FakeMiner()
    report = fake_report()
    mine = Mock(return_value=(miner, report))
    monkeypatch.setattr(genetic_discovery, "mine_genetic_factors", mine)
    result = run_discovery(frame, label_spec=LABEL, split_spec=SPLIT,
                           population_size=30, generations=1)
    train, validation = mine.call_args.args
    assert len(mine.call_args.args) == 2
    for partition in (train, validation):
        for forbidden in ("next_close", "future_close", "Target_30", "label_answer",
                          "Pred_Custom", "Pred_Smooth", "Should_Trade"):
            assert forbidden not in partition.columns
    assert mine.call_args.kwargs["n_jobs"] == 1
    assert result.miner is miner
    assert len(miner.transformed) == 1
    assert list(miner.transformed[0].columns) == ["signal", "noise"]
    assert len(miner.transformed[0]) == len(train) + len(validation)
    assert list(result.factors.columns) == ["date", "ticker", "partition", "GP_000"]
    assert set(result.factors["partition"]) == {"train", "validation"}
    assert result.report["holdout_evaluated"] is False
    assert result.report["n_final"] == 1
    assert result.report["programs"] == report["programs"]
    assert result.report["selection_basis"] == "validation_only"
    assert result.report["label_spec"] == asdict(LABEL)
    assert result.report["split_spec"] == asdict(SPLIT)
    assert set(result.report["partitions"]) == {"train", "validation", "sealed_holdout"}
    assert result.factors["date"].max() < pd.Timestamp(result.report["partitions"]["sealed_holdout"]["start"])
    assert "timestamp" in result.report
    assert result.report["mining_config"] == {
        "population_size": 30, "generations": 1, "max_new_factors": 20,
        "min_ic_threshold": .01, "n_jobs": 1}
    assert "timestamp" not in report
    pd.testing.assert_frame_equal(frame, original)


def test_no_winners_does_not_call_transform(frame, monkeypatch):
    miner = Mock()
    monkeypatch.setattr(genetic_discovery, "mine_genetic_factors",
                        Mock(return_value=(miner, fake_report(False))))
    result = run_discovery(frame, label_spec=LABEL, split_spec=SPLIT)
    miner.transform.assert_not_called()
    assert result.report["n_final"] == 0
    assert list(result.factors.columns) == ["date", "ticker", "partition"]


def test_custom_label_metadata_cannot_become_gp_features(frame, monkeypatch):
    sample = frame.rename(columns={"date": "timestamp", "ticker": "contract",
                                   "adj_close": "adjusted_price"})
    mine = Mock(return_value=(FakeMiner(), fake_report()))
    monkeypatch.setattr(genetic_discovery, "mine_genetic_factors", mine)
    custom_label = LabelSpec(horizon_bars=1, entry_lag_bars=1, timestamp_col="timestamp",
                             ticker_col="contract", price_col="adjusted_price")
    result = run_discovery(sample, label_spec=custom_label, split_spec=SPLIT)
    for partition in mine.call_args.args:
        assert {"date", "ticker", "adj_close"}.issubset(partition.columns)
        assert "adjusted_price" not in partition.columns
    assert list(result.factors.columns) == ["date", "ticker", "partition", "GP_000"]


def test_save_is_isolated_unique_and_roundtrips(frame, tmp_path, monkeypatch):
    trained = tmp_path / "data" / "df_factors.pkl"
    trained.parent.mkdir()
    trained.write_bytes(b"existing training data")
    monkeypatch.setattr(genetic_discovery, "mine_genetic_factors",
                        Mock(return_value=(FakeMiner(), fake_report())))
    result = run_discovery(frame, label_spec=LABEL, split_spec=SPLIT)
    assert not (tmp_path / "outputs").exists()
    first = save_discovery(result, tmp_path)
    second = save_discovery(result, tmp_path)
    assert first != second
    assert first.parent == tmp_path / "outputs" / "genetic_factors"
    assert {path.name for path in first.iterdir()} == {"factors.pkl", "miner.pkl", "report.json"}
    pd.testing.assert_frame_equal(pd.read_pickle(first / "factors.pkl"), result.factors)
    assert json.loads((first / "report.json").read_text())["holdout_evaluated"] is False
    with (first / "miner.pkl").open("rb") as handle:
        assert isinstance(pickle.load(handle), FakeMiner)
    assert trained.read_bytes() == b"existing training data"
    assert not (tmp_path / "models").exists()


def test_save_failure_does_not_publish_partial_results(tmp_path, monkeypatch):
    monkeypatch.setattr(genetic_discovery.pickle, "dump", Mock(side_effect=OSError("disk full")))
    result = DiscoveryResult(FakeMiner(), fake_report(), pd.DataFrame())
    with pytest.raises(OSError, match="disk full"):
        save_discovery(result, tmp_path)
    root = tmp_path / "outputs" / "genetic_factors"
    assert list(root.iterdir()) == []


def test_save_json_failure_does_not_publish_partial_results(tmp_path):
    result = DiscoveryResult(FakeMiner(), {"validation_ic": float("nan")}, pd.DataFrame())
    with pytest.raises(ValueError):
        save_discovery(result, tmp_path)
    assert list((tmp_path / "outputs" / "genetic_factors").iterdir()) == []


@pytest.mark.parametrize("sample,error", [(None, TypeError), (pd.DataFrame(), ValueError)])
def test_missing_input_is_explicit(sample, error):
    with pytest.raises(error):
        prepare_discovery(sample)


def test_duplicate_column_is_rejected(frame):
    sample = pd.concat([frame, frame[["signal"]]], axis=1)
    with pytest.raises(ValueError, match="duplicate"):
        prepare_discovery(sample, label_spec=LABEL, split_spec=SPLIT)


def test_nonparseable_timestamp_is_rejected(frame):
    frame["date"] = frame["date"].astype(str)
    frame.loc[0, "date"] = "not-a-date"
    with pytest.raises(ValueError, match="date"):
        prepare_discovery(frame, label_spec=LABEL, split_spec=SPLIT)


def test_save_rejects_output_symlink_outside_project(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (project / "outputs").symlink_to(external, target_is_directory=True)
    result = DiscoveryResult(FakeMiner(), fake_report(), pd.DataFrame())
    with pytest.raises(ValueError, match="outside"):
        save_discovery(result, project)
    assert list(external.iterdir()) == []


def test_small_real_gplearn_discovery(frame):
    pytest.importorskip("gplearn")
    result = run_discovery(frame, population_size=24, generations=1, max_new_factors=2,
                           min_ic_threshold=0., label_spec=LABEL, split_spec=SPLIT, n_jobs=1)
    assert result.report["n_final"] >= 1
    assert result.report["holdout_evaluated"] is False
    assert result.miner.n_jobs == 1
    assert set(result.report["feature_names"]) == {"signal", "noise"}
    assert len(result.factors) == sum(result.report["partitions"][name]["rows"]
                                      for name in ("train", "validation"))
