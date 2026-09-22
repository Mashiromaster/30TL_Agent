import ast
from types import SimpleNamespace

import numpy as np
import pandas as pd

from research_agent import ResearchAgent
import research_agent
from strategy_runtime import SignalPolicy, get_strategy_spec


class _Runtime:
    metadata = {"model_release_id": "release-test"}

    def predict_frame(self, frame):
        result = frame.copy()
        result["Pred_Ret"] = np.sin(np.arange(len(result)) / 15)
        result["Model_Used"] = "fake"
        return result


class _Monitor:
    def __init__(self, fail=False, status="ok", error="source unavailable"):
        self.fail = fail
        self.status = status
        self.error = error
        self.store = SimpleNamespace(
            list_alerts=lambda: [],
            list_source_health=lambda: [],
            list_assessments=lambda: [],
        )

    def run_once(self):
        if self.fail:
            raise RuntimeError(self.error)
        return SimpleNamespace(
            run_id="monitor-1",
            status=self.status,
            documents_inserted=2,
            assessments_created=2,
            alerts_opened=1,
            report_path="monitor.json",
        )


def _factor_file(tmp_path):
    dates = pd.date_range("2026-01-05 09:30", periods=700, freq="min")
    frame = pd.DataFrame(
        {
            "date": dates,
            "close": 100 + np.arange(len(dates)) * 0.01,
            "Market_Regime": np.zeros(len(dates), dtype=int),
        }
    )
    output = tmp_path / "outputs"
    output.mkdir()
    frame.to_pickle(output / "df_factors.pkl")


def test_research_agent_includes_monitor_and_canonical_decision_without_writes(tmp_path):
    _factor_file(tmp_path)
    policy = SignalPolicy(get_strategy_spec())
    agent = ResearchAgent(
        tmp_path,
        runtime=_Runtime(),
        policy=policy,
        monitor=_Monitor(),
        fusion_runner=lambda **_kwargs: None,
    )

    report = agent.run_cycle(
        rebuild_factors=False,
        use_llm=False,
        monitor_sentiment=True,
        persist=False,
    )

    assert report["step_status"]["sentiment_monitor"].startswith("ok")
    assert report["step_status"]["signal"] == "ok"
    assert report["base_signal"]["strategy_id"] == policy.spec.strategy_id
    assert report["base_signal"]["decision_id"].startswith("decision:")
    assert not list((tmp_path / "outputs").glob("research_cycle_*.json"))
    assert not (tmp_path / "outputs" / "trade_memory.jsonl").exists()


def test_monitor_failure_does_not_block_deterministic_tl_decision(tmp_path):
    _factor_file(tmp_path)
    agent = ResearchAgent(
        tmp_path,
        runtime=_Runtime(),
        policy=SignalPolicy(get_strategy_spec()),
        monitor=_Monitor(fail=True),
        fusion_runner=lambda **_kwargs: None,
    )

    report = agent.run_cycle(
        rebuild_factors=False,
        use_llm=False,
        monitor_sentiment=True,
        persist=False,
    )

    assert report["step_status"]["sentiment_monitor"].startswith("error:")
    assert report["step_status"]["signal"] == "ok"
    assert report["base_signal"] is not None


def test_research_report_redacts_credentials_from_step_errors(tmp_path):
    _factor_file(tmp_path)
    agent = ResearchAgent(
        tmp_path,
        runtime=_Runtime(),
        policy=SignalPolicy(get_strategy_spec()),
        monitor=_Monitor(
            fail=True,
            error="Authorization: Bearer secret-token; api_key=sk-sensitive",
        ),
        fusion_runner=lambda **_kwargs: None,
    )

    report = agent.run_cycle(
        rebuild_factors=False,
        use_llm=False,
        monitor_sentiment=True,
        persist=False,
    )

    status = report["step_status"]["sentiment_monitor"]
    assert "secret-token" not in status
    assert "sk-sensitive" not in status
    assert "[REDACTED]" in status


def test_partial_monitor_is_reported_as_warning_without_mutating_base_signal(
    tmp_path,
):
    _factor_file(tmp_path)
    agent = ResearchAgent(
        tmp_path,
        runtime=_Runtime(),
        policy=SignalPolicy(get_strategy_spec()),
        monitor=_Monitor(status="partial_failure"),
        fusion_runner=lambda **_kwargs: None,
    )

    report = agent.run_cycle(
        rebuild_factors=False,
        use_llm=False,
        monitor_sentiment=True,
        persist=False,
    )

    assert report["step_status"]["sentiment_monitor"] == (
        "warning:status=partial_failure"
    )
    assert report["step_status"]["signal"] == "ok"


def test_research_agent_import_graph_has_no_execution_capability():
    tree = ast.parse(open(research_agent.__file__, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])

    assert imported.isdisjoint(
        {"broker", "order_executor", "xttrader", "qmt", "vnpy"}
    )
