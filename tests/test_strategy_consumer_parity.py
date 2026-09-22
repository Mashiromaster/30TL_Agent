import json

import numpy as np
import pandas as pd
import pytest

from backtest import Backtester
from inference import SignalGenerator
from memory import TradingMemory
from research_agent import ResearchAgent
from signal_dashboard import SignalDashboard
from strategy_runtime import SignalPolicy, get_strategy_spec


class _Runtime:
    metadata = {"model_release_id": "release-parity"}
    model_base = object()
    model_active = None
    scaler = object()
    features = ()
    hierarchical_model = None
    selected_model_kind = "fake"
    gp_miner = None

    def predict_frame(self, factors):
        result = factors.copy()
        result["Pred_Ret"] = np.sin(np.arange(len(result)) / 17.0)
        result["Model_Used"] = "fake"
        return result


def test_all_strategy_consumers_share_the_same_base_decision(tmp_path):
    rows = 900
    timestamps = pd.date_range("2026-01-05 09:30", periods=rows, freq="min")
    factors = pd.DataFrame(
        {
            "date": timestamps,
            "trade_dt": timestamps.strftime("%Y-%m-%d"),
            "close": 100 + np.arange(rows) * 0.001,
            "Market_Regime": np.zeros(rows, dtype=int),
            "Target_Ret": np.cos(np.arange(rows) / 23.0) * 0.001,
        }
    )
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    factors.to_pickle(output_dir / "df_factors.pkl")

    spec = get_strategy_spec()
    policy = SignalPolicy(spec)
    runtime = _Runtime()
    generator = SignalGenerator(
        tmp_path / "models" / "trained_model.pkl",
        runtime=runtime,
        policy=policy,
    )
    cli_signal, predictions = generator.generate_signal(factors)
    predictions.to_pickle(output_dir / "df_predictions.pkl")

    dashboard = SignalDashboard.__new__(SignalDashboard)
    dashboard.base_dir = str(tmp_path)
    dashboard.model = generator
    dashboard_signal = dashboard.generate_live_signal(factors)

    tester = Backtester(predictions, strategy_spec=spec)
    tester.run_strategy()

    memory = TradingMemory(tmp_path, policy=policy)
    memory.backfill_from_predictions()
    memory_record = json.loads(
        (output_dir / "trade_memory.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    )

    agent = ResearchAgent(
        tmp_path,
        runtime=runtime,
        policy=policy,
        fusion_runner=lambda **_kwargs: None,
    )
    report = agent.run_cycle(
        rebuild_factors=False,
        use_llm=False,
        monitor_sentiment=False,
        persist=False,
    )

    assert dashboard_signal["decision_id"] == cli_signal["decision_id"]
    assert report["base_signal"]["decision_id"] == cli_signal["decision_id"]
    assert int(tester.df.iloc[-1]["Direction"]) == cli_signal["direction"]
    assert memory_record["direction"] == cli_signal["direction"]
    assert memory_record["decision_id"] == cli_signal["decision_id"]
    assert memory_record["model_release_id"] == "release-parity"
    assert tester.model_release_id == "release-parity"
    assert tester.df.iloc[-1]["Target_Weight"] == pytest.approx(
        cli_signal["suggested_weight"], abs=5e-5
    )
    assert tester.df.iloc[-1]["Effective_Position"] == tester.df.iloc[-2][
        "Target_Position"
    ]
    assert dashboard_signal["strategy_fingerprint"] == spec.fingerprint()
    assert report["base_signal"]["model_release_id"] == "release-parity"
