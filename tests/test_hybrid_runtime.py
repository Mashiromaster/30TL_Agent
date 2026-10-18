from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import hybrid_runtime as hybrid
from hybrid_runtime import HybridConfig, load_hybrid_run, list_hybrid_runs, run_hybrid_research


NOW = pd.Timestamp("2026-09-30T07:00:00Z")


@pytest.fixture
def frame():
    return pd.DataFrame({"date": pd.date_range("2026-09-30 13:00", periods=120, freq="min"),
                         "ticker": "TL2612", "close": 100 + np.arange(120) * .01,
                         "signal": np.sin(np.arange(120)), "Market_Regime": 0})


class Runtime:
    metadata = {"model_release_id": "release-test", "legacy": False,
                "selected_model_kind": "flat_combined", "features": ["signal"]}

    def predict_frame(self, frame):
        result = frame.copy()
        result["Pred_Ret"] = .2
        result["Model_Used"] = "flat_combined"
        return result


class Policy:
    def latest(self, frame, *, model_release_id):
        signal = {"timestamp": str(frame["date"].iloc[-1]), "direction": 1,
                  "suggested_weight": .8, "confidence": .75, "predicted_return": .2,
                  "decision_ready": True, "missing_reason": "", "model_release_id": model_release_id,
                  "strategy_id": "test", "strategy_fingerprint": "fingerprint", "decision_id": "test-decision"}
        return SimpleNamespace(to_dict=lambda: signal)


def evidence(**kwargs):
    return {"id": "event-1", "title": "政策公告", "source": "official", "url": "https://example.org/news",
            "published_at": "2026-09-30T04:00:00Z", "available_at": "2026-09-30T04:01:00Z",
            "assessed_at": "2026-09-30T04:02:00Z", "expires_at": "2026-10-01T00:00:00Z",
            "affected_assets": ["TL"], "bond_impact": "bullish", "severity": "medium",
            "confidence": .8, "summary": "政策新闻摘要", "prompt_injection_flag": False, **kwargs}


def reply(output, status="ok", **kwargs):
    return {"status": status, "model": "test-model", "source": "online", "output": output,
            "reason": "", "usage": {}, **kwargs}


class Adapters:
    def route(self, state):
        return reply({"route": "deep_review", "confidence": .9})

    def semantic(self, state):
        return reply({"action": "reduce", "confidence": .9, "evidence_sufficient": .9,
                      "probabilities": {"maintain": .1, "reduce": .8, "abstain": .1}})

    def review(self, state, *, role):
        return reply({"recommendation": "maintain", "confidence": .9, "summary": role,
                      "evidence_ids": ["event-1"], "counter_evidence_ids": [], "risks": [], "limitations": []})


def run(tmp_path, frame, **kwargs):
    params = dict(factors=frame, runtime=Runtime(), policy=Policy(), now=NOW, persist=False,
                  evidence_loader=lambda **_: [evidence()])
    params.update(kwargs)
    return run_hybrid_research(tmp_path, **params)


def test_offline_is_read_only_and_never_invokes_external_adapters(tmp_path, frame):
    adapters = Mock()
    original = frame.copy(deep=True)
    trace = run(tmp_path, frame, adapters=adapters)
    assert not adapters.mock_calls
    assert trace["baseline"]["direction"] == 1
    assert trace["decision"]["execution_authorized"] is False
    assert trace["decision"]["baseline_unchanged"] is True
    assert trace["opinions"]["llm"]["status"] == "skipped"
    assert trace["budget"]["attempted_calls"] == 0
    assert list(tmp_path.iterdir()) == []
    pd.testing.assert_frame_equal(frame, original)


def test_online_arbitrates_conservatively_without_touching_baseline(tmp_path, frame):
    trace = run(tmp_path, frame, adapters=Adapters(), config=HybridConfig(network_mode="online"))
    assert trace["decision"]["recommendation"] == "reduce"
    assert trace["decision"]["proposed_direction"] == 1
    assert trace["decision"]["proposed_weight"] == pytest.approx(.4)
    assert trace["baseline"]["suggested_weight"] == .8
    assert trace["decision"]["conflicts"]
    assert trace["budget"]["attempted_calls"] == 4
    assert all(stage["elapsed_ms"] >= 0 for stage in trace["stages"])
    assert all(stage["started_at"] and stage["finished_at"] for stage in trace["stages"])
    assert {s["stage_id"] for s in trace["stages"]} >= {"snapshot", "advisor_plan", "quant", "evidence", "router", "jev", "llm", "advisor", "risk", "arbitration"}


def test_future_factor_rows_never_reach_quant(tmp_path, frame):
    cutoff = frame.date.iloc[60]
    runtime = Mock(metadata=Runtime.metadata)
    runtime.predict_frame.side_effect = Runtime().predict_frame
    trace = run(tmp_path, frame, runtime=runtime, as_of=cutoff)
    supplied = runtime.predict_frame.call_args.args[0]
    assert supplied.date.max() <= cutoff
    assert trace["snapshot"]["excluded_future_rows"] == 59
    assert trace["snapshot"]["rows"] == 61


@pytest.mark.parametrize("change", [
    {"available_at": "2026-10-01T00:00:00Z"}, {"assessed_at": "2026-10-01T00:00:00Z"},
    {"published_at": "2026-10-01T00:00:00Z"}, {"available_at": None},
    {"expires_at": "2026-09-30T00:00:00Z"}, {"prompt_injection_flag": True},
    {"affected_assets": ["AAPL"]},
])
def test_evidence_must_be_contemporaneous_and_not_quarantined(tmp_path, frame, change):
    trace = run(tmp_path, frame, evidence_loader=lambda **_: [evidence(**change)])
    assert trace["evidence"] == []
    assert sum(trace["evidence_exclusions"].values()) == 1
    assert trace["decision"]["status"] == "review_required"


def test_stale_data_remains_visible_but_proposal_is_blocked(tmp_path, frame):
    trace = run(tmp_path, frame, now=NOW + pd.Timedelta(days=30))
    assert trace["snapshot"]["stale"]
    assert trace["baseline"]["direction"] == 1
    assert trace["decision"]["status"] == "blocked"
    assert trace["decision"]["proposed_weight"] == 0
    assert "stale_market_snapshot" in trace["decision"]["blocked_reasons"]


def test_missing_release_has_full_trace_not_fake_prediction(tmp_path, frame):
    trace = run(tmp_path, frame, runtime=None)
    assert trace["baseline"] is None
    assert trace["opinions"]["quant"]["status"] == "error"
    assert trace["decision"]["status"] == "blocked"
    assert trace["decision"]["proposed_weight"] == 0


def test_failed_expert_and_budget_exhaustion_are_explicit(tmp_path, frame):
    adapters = Adapters()
    adapters.semantic = Mock(side_effect=TimeoutError("Authorization: Bearer secret-token"))
    trace = run(tmp_path, frame, adapters=adapters, config=HybridConfig(network_mode="online", max_model_calls=2))
    assert trace["opinions"]["jev"]["status"] == "error"
    assert "secret-token" not in str(trace)
    assert trace["opinions"]["llm"]["status"] == "skipped"
    assert "budget" in trace["opinions"]["llm"]["reason"]
    assert trace["budget"]["attempted_calls"] == 2
    assert trace["baseline"]["direction"] == 1


def test_unknown_reference_or_action_cannot_override_strategy(tmp_path, frame):
    adapters = Adapters()
    adapters.review = Mock(return_value=reply({"recommendation": "buy", "confidence": 1,
                                              "evidence_ids": ["invented"], "order": "full_position"}))
    trace = run(tmp_path, frame, adapters=adapters, config=HybridConfig(network_mode="online"))
    assert trace["opinions"]["llm"]["status"] == "error"
    assert trace["baseline"]["suggested_weight"] == .8
    assert trace["decision"]["execution_authorized"] is False


def test_no_evidence_blocks_external_review_calls(tmp_path, frame):
    adapters = Mock()
    trace = run(tmp_path, frame, adapters=adapters, config=HybridConfig(network_mode="online"), evidence_loader=lambda **_: [])
    # The router and model reviews must not invent a news view when evidence is absent.
    assert not adapters.mock_calls
    assert trace["decision"]["recommendation"] == "abstain"
    assert trace["decision"]["status"] == "review_required"


def test_result_persistence_is_unique_read_only_and_traversal_safe(tmp_path, frame):
    first = run(tmp_path, frame, persist=True)
    second = run(tmp_path, frame, persist=True)
    assert first["run_id"] != second["run_id"]
    assert first["persistence"]["status"] == "saved"
    saved = load_hybrid_run(tmp_path, first["run_id"])
    assert saved["baseline"] == first["baseline"]
    assert len(list_hybrid_runs(tmp_path)) == 2
    assert not (tmp_path / "outputs" / "trade_memory.jsonl").exists()
    with pytest.raises(ValueError):
        load_hybrid_run(tmp_path, "../../outside")


def test_failed_persistence_is_reported_not_success(tmp_path, frame, monkeypatch):
    monkeypatch.setattr(hybrid, "save_hybrid_run", Mock(side_effect=OSError("disk full")))
    trace = run(tmp_path, frame, persist=True)
    assert trace["persistence"]["status"] == "error"
    assert "disk full" in trace["persistence"]["reason"]


def test_invalid_snapshot_skips_all_models_and_is_diagnosable(tmp_path):
    adapters = Mock()
    trace = run(tmp_path, pd.DataFrame(), adapters=adapters)
    assert trace["decision"]["status"] == "blocked"
    assert trace["stages"][0]["status"] == "error"
    assert not adapters.mock_calls


@pytest.mark.parametrize("kwargs", [{"network_mode": "auto"}, {"max_age_hours": -1},
                                    {"max_model_calls": 1000}, {"max_evidence": 0},
                                    {"max_evidence": 21}, {"asset": "AAPL"}])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        HybridConfig(**kwargs)


def test_research_agent_delegates_to_same_runtime(tmp_path, frame):
    from research_agent import ResearchAgent
    agent = ResearchAgent(tmp_path, runtime=Runtime(), policy=Policy())
    trace = agent.run_hybrid_cycle(factors=frame, now=NOW, persist=False,
                                   evidence_loader=lambda **_: [evidence()])
    assert trace["schema_version"] == "f_agent.hybrid.v1"
    assert trace["decision"]["execution_authorized"] is False


def test_readonly_event_store_does_not_initialize_or_allow_writes(tmp_path):
    import sqlite3
    from intelligence_events import IntelligenceEventStore
    with pytest.raises(FileNotFoundError):
        IntelligenceEventStore(tmp_path, read_only=True)
    assert list(tmp_path.iterdir()) == []
    writer = IntelligenceEventStore(tmp_path)
    reader = IntelligenceEventStore(tmp_path, read_only=True)
    assert reader.query_assessments(as_of=NOW.isoformat(), assets=["TL"]) == []
    assert reader.get_document("unknown") is None
    with reader._connect() as connection:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("DELETE FROM documents")


def test_snapshot_ignores_dataframe_index_labels(tmp_path, frame):
    expected = run(tmp_path, frame, as_of=frame.date.iloc[60])
    frame.index = [0] * len(frame)
    actual = run(tmp_path, frame, as_of=frame.date.iloc[60])
    assert actual["snapshot"] == expected["snapshot"]
    assert actual["baseline"] == expected["baseline"]
    assert frame.index.tolist() == [0] * len(frame)


def test_partial_unavailable_envelope_becomes_diagnosable_fallback(tmp_path, frame):
    adapters = Adapters()
    adapters.route = Mock(return_value={"status": "unavailable"})
    result = run(tmp_path, frame, adapters=adapters, config=HybridConfig(network_mode="online"))
    assert result["route"]["fallback"] is True
    assert len(result["stages"]) == 10
    assert result["decision"]["execution_authorized"] is False


@pytest.mark.parametrize("case", ["low_confidence", "low_evidence", "evidence_conflict", "router_abstain", "not_ready"])
def test_risk_guards_produce_zero_weight_without_replacing_baseline(tmp_path, frame, case):
    adapters = Adapters()
    records = [evidence()]
    policy = Policy()
    if case in {"low_confidence", "low_evidence"}:
        output = adapters.semantic({})["output"]
        output["confidence" if case == "low_confidence" else "evidence_sufficient"] = .2
        adapters.semantic = Mock(return_value=reply(output))
    elif case == "evidence_conflict":
        records.append(evidence(id="event-2", bond_impact="bearish"))
    elif case == "router_abstain":
        adapters.route = Mock(return_value=reply({"route": "abstain", "confidence": .9}))
    else:
        baseline = policy.latest(frame, model_release_id="release-test").to_dict()
        baseline.update(decision_ready=False, missing_reason="warmup")
        policy.latest = Mock(return_value=SimpleNamespace(to_dict=lambda: baseline))
    result = run(tmp_path, frame, adapters=adapters, policy=policy,
                 config=HybridConfig(network_mode="online"), evidence_loader=lambda **_: records)
    assert result["decision"]["recommendation"] == "abstain"
    assert result["decision"]["proposed_weight"] == 0
    assert result["baseline"]["suggested_weight"] == .8


def test_observer_failure_does_not_change_decision(tmp_path, frame):
    observer = Mock(side_effect=RuntimeError("render failed"))
    result = run(tmp_path, frame, on_event=observer)
    assert observer.call_count == 10
    assert result["decision"]["baseline_unchanged"] is True


def test_saved_audit_cannot_be_overwritten_or_authorize_execution(tmp_path, frame):
    import json
    result = run(tmp_path, frame, persist=True)
    with pytest.raises(FileExistsError):
        hybrid.save_hybrid_run(result, tmp_path)
    assert load_hybrid_run(tmp_path, result["run_id"])["baseline"] == result["baseline"]
    path = hybrid._run_path(tmp_path, result["run_id"])
    result["decision"]["execution_authorized"] = True
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="never authorize"):
        load_hybrid_run(tmp_path, result["run_id"])
    assert list_hybrid_runs(tmp_path) == []


def test_audit_symlink_cannot_escape_project(tmp_path, frame):
    project = tmp_path / "project"
    (project / "outputs").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (project / "outputs" / "hybrid_runs").symlink_to(outside, target_is_directory=True)
    result = run(project, frame, persist=True)
    assert result["persistence"]["status"] == "error"
    assert list(outside.iterdir()) == []


def test_real_event_store_keeps_only_revision_known_at_market_time(tmp_path, frame):
    from intelligence_events import DocumentRecord, EventAssessment, IntelligenceEventStore
    store = IntelligenceEventStore(tmp_path)
    for stamp, content in [("2026-09-30T04:00:00Z", "original policy"),
                           ("2026-09-30T08:00:00Z", "future revision")]:
        [document] = store.commit_provider_batch("official", [DocumentRecord(
            source_name="official", source_document_id="same-announcement", title="政策公告",
            content=content, published_at=stamp, received_at=stamp, available_at=stamp,
        )], next_cursor=stamp, fetched_at=stamp)
        store.add_assessment(EventAssessment(
            document_id=document.document_id, event_cluster_id=document.event_cluster_id,
            event_type="monetary_policy", severity="medium", bond_impact="bullish", confidence=.8,
            evidence=[content], affected_assets=["TL"], assessed_at=stamp,
        ))
    original_bytes = store.db_path.read_bytes()
    result = run(tmp_path, frame, evidence_loader=None)
    assert len(result["evidence"]) == 1
    assert result["evidence"][0]["summary"] == "original policy"
    assert result["evidence"][0]["document_id"] != document.document_id
    assert store.db_path.read_bytes() == original_bytes


def test_real_lightgbm_runtime_and_signal_policy_integrate_without_publishing(tmp_path, frame):
    from lightgbm import LGBMRegressor
    from sklearn.preprocessing import StandardScaler
    from strategy_runtime import ModelRuntime, SignalPolicy, get_strategy_spec
    # Tiny synthetic training is test-only, does not access production data or
    # save/promote a model, and compares the same canonical decision end to end.
    scaler = StandardScaler().fit(frame[["signal"]])
    values = scaler.transform(frame[["signal"]])
    model = LGBMRegressor(n_estimators=5, num_leaves=4, min_child_samples=5, n_jobs=1, verbosity=-1)
    model.fit(values, frame["signal"].to_numpy() * .01)
    runtime = ModelRuntime({"artifact_schema_version": 3, "model_release_id": "synthetic-test-only",
                            "model_base": model, "scaler": scaler, "features": ["signal"],
                            "selected_model_kind": "flat_combined"})
    policy = SignalPolicy(get_strategy_spec())
    expected = policy.latest(runtime.predict_frame(frame), model_release_id="synthetic-test-only").to_dict()
    result = run(tmp_path, frame, runtime=runtime, policy=policy)
    assert result["opinions"]["quant"]["status"] == "ok"
    assert result["baseline"] == expected
    assert result["decision"]["execution_authorized"] is False
    assert list(tmp_path.iterdir()) == []


def test_factor_candidate_path_is_bound_to_trace(tmp_path, frame):
    from test_research_readiness import candidate
    path, _ = candidate(tmp_path, frame)
    trace = run(tmp_path, None, factor_path=path)
    assert trace["snapshot"]["factor_path"] == str(path)
    assert len(trace["snapshot"]["factor_sha256"]) == 64
    assert trace["snapshot"]["factor_provenance"]["verified"] is True
    assert trace["baseline"]["direction"] == 1


def test_unverified_disk_factors_skip_models_even_when_runtime_injected(tmp_path, frame):
    (tmp_path / "outputs").mkdir()
    frame.to_pickle(tmp_path / "outputs/df_factors.pkl")
    runtime = Mock()
    trace = run(tmp_path, None, runtime=runtime)
    runtime.predict_frame.assert_not_called()
    assert "unverified_factor_provenance" in trace["decision"]["blocked_reasons"]


def test_factor_path_and_injected_frame_are_mutually_exclusive(tmp_path, frame):
    trace = run(tmp_path, frame, factor_path="outputs/df_factors.pkl")
    assert trace["decision"]["status"] == "blocked"
    assert "mutually exclusive" in trace["stages"][0]["reason"]


def test_trusted_test_overrides_are_explicit_in_saved_trace(tmp_path, frame):
    trace = run(tmp_path, frame)
    assert trace["integration"]["production_inputs"] is False
    assert set(trace["integration"]["injected_components"]) == {"factors", "runtime", "policy", "evidence_loader", "clock"}


def test_standard_disk_entry_has_no_trusted_test_overrides(tmp_path):
    trace = run_hybrid_research(tmp_path, persist=False)
    assert trace["integration"]["production_inputs"] is True
    assert trace["integration"]["injected_components"] == []


def test_canonical_policy_from_research_agent_is_not_a_test_override(tmp_path):
    from strategy_runtime import SignalPolicy, get_strategy_spec
    trace = run_hybrid_research(tmp_path, persist=False, policy=SignalPolicy(get_strategy_spec()))
    assert trace["integration"]["production_inputs"] is True
