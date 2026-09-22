import pickle

import numpy as np
import pandas as pd
import pytest

from inference import SignalGenerator
from model_release import ModelReleaseError
from strategy_runtime import ModelRuntime


class _IdentityScaler:
    def transform(self, values):
        return np.asarray(values, dtype=float)


class _SumModel:
    def __init__(self, multiplier=1.0):
        self.multiplier = multiplier

    def predict(self, values):
        return np.asarray(values, dtype=float).sum(axis=1) * self.multiplier


class _HierarchicalModel:
    def components(self, frame):
        micro = frame["x"].to_numpy(dtype=float) * 2.0
        macro = frame["macro"].to_numpy(dtype=float)
        return pd.DataFrame(
            {
                "micro_alpha": micro,
                "macro_state": macro,
                "gate_weight": np.ones(len(frame)),
                "event_alpha": macro,
                "prediction": micro + macro,
            },
            index=frame.index,
        )


class _FakeGPMiner:
    selected_programs_ = ["x_plus_seed"]
    feature_names_in_ = ["seed"]

    def transform(self, frame):
        return {"GP_001_alpha": frame["seed"].to_numpy(dtype=float) + 1.0}


def _write_artifact(tmp_path, artifact):
    path = tmp_path / "trained_model.pkl"
    with path.open("wb") as handle:
        pickle.dump(artifact, handle)
    return path


def test_flat_runtime_predicts_and_exposes_current_metadata(tmp_path):
    path = _write_artifact(
        tmp_path,
        {
            "artifact_schema_version": 3,
            "model_release_id": "release-123",
            "model_base": _SumModel(),
            "model_active": None,
            "scaler": _IdentityScaler(),
            "features": ["x", "y"],
            "selected_model_kind": "flat_combined",
        },
    )
    runtime = ModelRuntime.from_path(path)
    frame = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})

    result = runtime.predict_frame(frame)

    assert result["Pred_Ret"].tolist() == [4.0, 6.0]
    assert result["Model_Used"].tolist() == ["flat_combined", "flat_combined"]
    assert runtime.metadata["artifact_schema_version"] == 3
    assert runtime.metadata["model_release_id"] == "release-123"
    assert runtime.metadata["legacy"] is False


def test_hierarchical_runtime_uses_components_from_selected_model(tmp_path):
    path = _write_artifact(
        tmp_path,
        {
            "artifact_schema_version": 3,
            "model_base": _SumModel(),
            "model_active": None,
            "scaler": _IdentityScaler(),
            "features": ["x", "macro"],
            "selected_model_kind": "hierarchical",
            "hierarchical_model": _HierarchicalModel(),
        },
    )
    runtime = ModelRuntime.from_path(path)

    result = runtime.predict_frame(
        pd.DataFrame({"x": [1.0, 2.0], "macro": [0.5, -0.5]})
    )

    assert result["Pred_Ret"].tolist() == [2.5, 3.5]
    assert result["Model_Used"].tolist() == ["hierarchical", "hierarchical"]
    assert result["micro_alpha"].tolist() == [2.0, 4.0]
    assert result["macro_state"].tolist() == [0.5, -0.5]


def test_runtime_reconstructs_gp_features_and_rejects_missing_gp_inputs(tmp_path):
    path = _write_artifact(
        tmp_path,
        {
            "artifact_schema_version": 3,
            "model_base": _SumModel(),
            "model_active": None,
            "scaler": _IdentityScaler(),
            "features": ["x", "GP_001_alpha"],
            "selected_model_kind": "flat_combined",
            "gp_miner": _FakeGPMiner(),
        },
    )
    runtime = ModelRuntime.from_path(path)

    result = runtime.predict_frame(pd.DataFrame({"x": [2.0], "seed": [4.0]}))
    assert result["Pred_Ret"].tolist() == [7.0]
    assert runtime.metadata["has_gp_miner"] is True

    with pytest.raises(ValueError, match="GP input schema"):
        runtime.predict_frame(pd.DataFrame({"x": [2.0]}))


def test_legacy_runtime_routes_active_regimes_and_marks_metadata(tmp_path):
    path = _write_artifact(
        tmp_path,
        {
            "model_base": _SumModel(1.0),
            "model_active": _SumModel(10.0),
            "scaler": _IdentityScaler(),
            "features": ["x"],
            "config": "legacy-config",
        },
    )
    runtime = ModelRuntime.from_path(path)

    result = runtime.predict_frame(
        pd.DataFrame({"x": [1.0, 2.0, 3.0], "Market_Regime": [0, 1, 2]})
    )

    assert result["Pred_Ret"].tolist() == [1.0, 20.0, 30.0]
    assert result["Model_Used"].tolist() == ["base", "active", "active"]
    assert runtime.metadata["artifact_schema_version"] == 1
    assert runtime.metadata["model_release_id"] == "legacy-unversioned"
    assert runtime.metadata["legacy"] is True


def test_production_facade_rejects_unmanaged_legacy_pickle_by_default(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    path = _write_artifact(
        model_dir,
        {
            "model_base": _SumModel(),
            "model_active": None,
            "scaler": _IdentityScaler(),
            "features": ["x"],
        },
    )

    with pytest.raises(ModelReleaseError, match="unmanaged legacy pickle"):
        SignalGenerator(path)

    offline = SignalGenerator(path, allow_legacy=True)
    assert offline.model_release_id == "legacy-unversioned"


def test_dashboard_fallback_does_not_report_unmanaged_pickle_as_loaded(tmp_path):
    from dashboard_v2 import _build_fallback_status

    legacy = tmp_path / "models" / "trained_model.pkl"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"not-a-managed-release")

    status = _build_fallback_status(tmp_path)

    assert status["base_model_loaded"] is False


def test_runtime_rejects_feature_schema_drift(tmp_path):
    path = _write_artifact(
        tmp_path,
        {
            "model_base": _SumModel(),
            "model_active": None,
            "scaler": _IdentityScaler(),
            "features": ["x", "missing"],
        },
    )
    runtime = ModelRuntime.from_path(path)

    with pytest.raises(ValueError, match="model feature schema"):
        runtime.predict_frame(pd.DataFrame({"x": [1.0]}))
