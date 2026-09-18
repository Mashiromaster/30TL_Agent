import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from feature_registry import FeatureGroups
from hierarchical_model import HierarchicalMacroMicroModel


def test_macro_layer_trains_once_per_day_and_exposes_components():
    rows = 60
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=rows, freq="h"),
            "micro": np.linspace(-1, 1, rows),
            "macro": np.repeat([0.0, 1.0, -1.0], [24, 24, 12]),
        }
    )
    frame["Target_Ret"] = 2 * frame["micro"] + 0.5 * frame["macro"]
    groups = FeatureGroups(
        micro=("micro",), market=(), macro=("macro",), regime=(), gp=()
    )
    model = HierarchicalMacroMicroModel(
        micro_model=LinearRegression(), macro_model=LinearRegression()
    )

    model.fit(frame, feature_groups=groups)
    components = model.components(frame)

    assert model.macro_training_rows_ == frame["date"].dt.normalize().nunique()
    assert list(components.columns) == [
        "micro_alpha", "macro_state", "gate_weight", "event_alpha", "prediction"
    ]
    assert np.isfinite(components.to_numpy()).all()
    assert np.allclose(model.predict(frame), components["prediction"])


def test_model_rejects_schema_drift():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=20, freq="h"),
            "micro": np.arange(20.0),
            "macro": np.arange(20.0),
            "Target_Ret": np.arange(20.0),
        }
    )
    groups = FeatureGroups(("micro",), (), ("macro",), (), ())
    model = HierarchicalMacroMicroModel(
        micro_model=LinearRegression(), macro_model=LinearRegression()
    ).fit(frame, feature_groups=groups)

    try:
        model.predict(frame.drop(columns="macro"))
    except ValueError as exc:
        assert "macro" in str(exc)
    else:
        raise AssertionError("schema drift must fail loudly")
