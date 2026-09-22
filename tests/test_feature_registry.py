import pandas as pd

from feature_registry import discover_feature_groups


def test_feature_registry_separates_macro_micro_gp_and_drops_constants():
    frame = pd.DataFrame(
        {
            "Spread_Mean": [1.0, 2.0, 3.0],
            "YC_Slope_10Y_1Y": [0.1, 0.2, 0.3],
            "JEV_Bond_Direction": [-0.2, 0.1, 0.5],
            "GP_001_alpha": [3.0, 2.0, 1.0],
            "Market_Regime": [0, 1, 2],
            "RSI": [30.0, 40.0, 50.0],
            "constant": [1.0, 1.0, 1.0],
            "_macro_stale_count": [0, 1, 2],
            "ticker": ["a", "a", "a"],
        }
    )

    groups = discover_feature_groups(frame)

    assert groups.micro == ("Spread_Mean",)
    assert "YC_Slope_10Y_1Y" in groups.macro
    assert "JEV_Bond_Direction" in groups.macro
    assert "_macro_stale_count" in groups.macro
    assert groups.gp == ("GP_001_alpha",)
    assert groups.regime == ("Market_Regime",)
    assert groups.market == ("RSI",)
    assert "constant" not in groups.all_features
