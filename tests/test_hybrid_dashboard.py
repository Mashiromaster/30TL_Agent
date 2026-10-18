"""Workbench safety and rerun behavior; no model or network calls."""

from copy import deepcopy
from unittest.mock import Mock

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import hybrid_dashboard


def report(run_id="run_test"):
    return {
        "schema_version": "f_agent.hybrid.v1", "run_id": run_id,
        "generated_at": "2026-10-03T08:00:00+00:00", "asset": "TL",
        "question": "当前国债期货信号有哪些支持和反对证据？", "network_mode": "offline",
        "snapshot": {"snapshot_id": "snapshot-1", "as_of": "2026-10-02T15:15:00+08:00",
                     "rows": 2, "source_rows": 2, "age_seconds": 3600, "stale": False,
                     "excluded_future_rows": 0},
        "baseline": {"direction": 1, "suggested_weight": .4, "decision_ready": True,
                     "predicted_return": .12, "predicted_return_smooth": .10,
                     "pred_rank_pct": .73, "model_used": "flat_combined",
                     "strategy_id": "tl", "model_release_id": "release-1", "decision_id": "decision-1"},
        "stages": [{"stage_id": "quant", "label": "量化基线", "kind": "model", "status": "ok",
                    "started_at": None, "finished_at": None, "elapsed_ms": None,
                    "dependencies": [], "model": "LightGBM", "source": "local", "reason": "",
                    "input_refs": ["snapshot-1"], "output": {"direction": 1}}],
        "opinions": {"quant": {"status": "ok", "model": "LightGBM", "source": "local", "output": {"direction": 1}},
                     "jev": {"status": "unavailable", "reason": "offline_no_cache", "output": {}},
                     "llm": {"status": "skipped", "reason": "offline", "output": {}}},
        "evidence": [{"id": "E1", "title": "央行公开信息", "source": "official", "url": "https://example.com/news",
                      "published_at": "2026-10-01", "available_at": "2026-10-01", "summary": "公开证据摘要"}],
        "route": {}, "decision": {"status": "review_required", "recommendation": "reduce",
              "proposed_direction": 1, "proposed_weight": .2, "execution_authorized": False,
              "baseline_unchanged": True, "blocked_reasons": [], "conflicts": ["证据分歧"], "reasons": ["待人工研究"]},
        "budget": {"call_limit": 4, "attempted_calls": 0}, "persistence": {"status": "saved", "path": "outputs/hybrid/run_test.json"},
    }


def app_script():
    import os
    import pandas as pd
    from hybrid_dashboard import render_hybrid_workbench
    class Context:
        base_dir = os.environ["TEST_HYBRID_BASE"]
        @property
        def df_factors(self):
            return pd.read_pickle(os.environ["TEST_HYBRID_FRAME"])
        @property
        def signal(self):
            raise AssertionError("Rendering must never call ctx.signal")
    render_hybrid_workbench(Context(), api_key="secret-not-for-display")


@pytest.fixture
def app(tmp_path, monkeypatch):
    path = tmp_path / "factors.pkl"
    pd.DataFrame({"date": pd.date_range("2026-10-02 15:14", periods=2, freq="min"),
                  "ticker": "TL2612", "adj_close": [110., 111.]}).to_pickle(path)
    monkeypatch.setenv("TEST_HYBRID_BASE", str(tmp_path))
    monkeypatch.setenv("TEST_HYBRID_FRAME", str(path))
    monkeypatch.setattr(hybrid_dashboard, "list_hybrid_runs", Mock(return_value=[]))
    return AppTest.from_function(app_script)


def visible_text(app):
    return "\n".join(str(element.value) for kind in (
        "markdown", "caption", "info", "warning", "error", "success", "text")
        for element in getattr(app, kind))


def test_initial_render_is_offline_read_only(app, monkeypatch):
    run = Mock(return_value=report())
    monkeypatch.setattr(hybrid_dashboard, "run_hybrid_research", run)
    app.run()
    assert not app.exception
    run.assert_not_called()
    assert app.selectbox(key="hybrid_network").value == "offline"
    assert "未运行" in visible_text(app)
    assert "secret-not-for-display" not in visible_text(app)


def test_dependency_graph_is_vertical_for_readable_stage_labels():
    source = hybrid_dashboard._dag_source(report()["stages"])
    assert "rankdir=TB" in source
    assert "fontsize=14" in source


def test_run_renders_one_snapshot_and_survives_rerun(app, monkeypatch):
    run = Mock(return_value=report())
    monkeypatch.setattr(hybrid_dashboard, "run_hybrid_research", run)
    app.run().button(key="hybrid_run").click().run()
    assert not app.exception
    assert "run_test" in visible_text(app)
    assert "未记录" in visible_text(app)
    assert "不下单" in visible_text(app)
    run.assert_called_once()
    assert run.call_args.kwargs["config"].network_mode == "offline"
    app.text_area(key="hybrid_question").set_value("新的研究问题").run()
    run.assert_called_once()
    assert "尚未应用" in visible_text(app)
    assert "run_test" in visible_text(app)


def test_production_research_reads_disk_instead_of_injecting_preview(app, monkeypatch):
    run = Mock(return_value=report())
    monkeypatch.setattr(hybrid_dashboard, "run_hybrid_research", run)
    app.run().button(key="hybrid_run").click().run()
    assert not app.exception
    assert run.call_args.kwargs["factors"] is None
    assert run.call_args.kwargs.get("factor_path") is None


def test_online_requires_explicit_confirmation(app, monkeypatch):
    run = Mock(return_value=report())
    monkeypatch.setattr(hybrid_dashboard, "run_hybrid_research", run)
    app.run().selectbox(key="hybrid_network").select("online").run()
    app.button(key="hybrid_run").click().run()
    run.assert_not_called()
    assert "确认" in visible_text(app)
    app.checkbox(key="hybrid_online_consent").check().run()
    app.button(key="hybrid_run").click().run()
    assert not app.exception
    run.assert_called_once()
    assert run.call_args.kwargs["config"].network_mode == "online"


def test_failed_run_removes_old_result(app, monkeypatch):
    run = Mock(side_effect=[report(), RuntimeError("source unavailable")])
    monkeypatch.setattr(hybrid_dashboard, "run_hybrid_research", run)
    app.run().button(key="hybrid_run").click().run()
    app.button(key="hybrid_run").click().run()
    assert not app.exception
    assert "source unavailable" in visible_text(app)
    assert "run_test" not in visible_text(app)


def test_history_loading_does_not_execute_models(app, monkeypatch):
    monkeypatch.setattr(hybrid_dashboard, "list_hybrid_runs", Mock(return_value=[{
        "run_id": "old_run", "generated_at": "2026-10-01", "market_as_of": "2026-09-30", "status": "blocked"}]))
    load = Mock(return_value=report("old_run"))
    run = Mock()
    monkeypatch.setattr(hybrid_dashboard, "load_hybrid_run", load)
    monkeypatch.setattr(hybrid_dashboard, "run_hybrid_research", run)
    app.run().button(key="hybrid_scan_history").click().run()
    app.selectbox(key="hybrid_history").select("old_run").run()
    app.button(key="hybrid_load_history").click().run()
    assert not app.exception
    load.assert_called_once()
    run.assert_not_called()
    assert "历史快照" in visible_text(app)
    assert "old_run" in visible_text(app)


def test_missing_data_keeps_entry_and_explains_block(app, monkeypatch):
    import os
    pd.DataFrame().to_pickle(os.environ["TEST_HYBRID_FRAME"])
    blocked = report()
    blocked["baseline"] = None
    blocked["decision"].update(status="blocked", blocked_reasons=["missing_factor_data"])
    run = Mock(return_value=blocked)
    monkeypatch.setattr(hybrid_dashboard, "run_hybrid_research", run)
    app.run()
    assert not app.button(key="hybrid_run").disabled
    assert "因子" in visible_text(app)
    app.button(key="hybrid_run").click().run()
    assert not app.exception
    assert "missing_factor_data" in visible_text(app)


def test_cache_mode_explains_missing_llm_cache(app, monkeypatch):
    monkeypatch.setattr(hybrid_dashboard, "run_hybrid_research", Mock(return_value=report()))
    app.run().selectbox(key="hybrid_network").select("cache").run()
    assert "LLM" in visible_text(app) and "缓存" in visible_text(app)


def test_private_reasoning_and_secrets_are_removed_recursively():
    value = {"summary": "public", "chain_of_thought": "private-one", "api_key": "private-two",
             "nested": {"reasoning_content": "private-three", "evidence_ids": ["E1"]}}
    safe = hybrid_dashboard.public_report(value)
    assert safe == {"summary": "public", "nested": {"evidence_ids": ["E1"]}}
    assert value["api_key"] == "private-two"


@pytest.mark.parametrize("url,expected", [("https://example.com/a", True),
    ("http://example.com/a", True), ("javascript:alert(1)", False),
    ("file:///secret", False), ("https://user:pass@example.com", False), ("https://", False)])
def test_evidence_url_safety(url, expected):
    assert hybrid_dashboard.safe_source_url(url) is expected


def test_unknown_stage_times_are_not_fabricated_and_reasoning_not_rendered(app, monkeypatch):
    value = report()
    value["opinions"]["llm"] = {"status": "ok", "model": "llm-test", "source": "test",
        "output": {"summary": "公开结论", "citations": ["E1"], "reasoning_content": "PRIVATE_CHAIN"}}
    value["evidence"].append({"id": "E2", "title": "unsafe", "url": "javascript:alert(1)"})
    original = deepcopy(value)
    monkeypatch.setattr(hybrid_dashboard, "run_hybrid_research", Mock(return_value=value))
    app.run().button(key="hybrid_run").click().run()
    assert not app.exception
    assert "PRIVATE_CHAIN" not in visible_text(app)
    assert "公开结论" in visible_text(app)
    assert "未记录" in visible_text(app)
    assert value == original


def test_history_failure_clears_previously_selected_report(app, monkeypatch):
    monkeypatch.setattr(hybrid_dashboard, "list_hybrid_runs", Mock(return_value=[{"run_id": "broken"}]))
    monkeypatch.setattr(hybrid_dashboard, "run_hybrid_research", Mock(return_value=report()))
    monkeypatch.setattr(hybrid_dashboard, "load_hybrid_run", Mock(side_effect=ValueError("invalid history")))
    app.run().button(key="hybrid_run").click().run()
    app.button(key="hybrid_scan_history").click().run()
    app.selectbox(key="hybrid_history").select("broken").run()
    app.button(key="hybrid_load_history").click().run()
    assert not app.exception
    assert "invalid history" in visible_text(app)
    assert "run_test" not in visible_text(app)


def test_backend_wrappers_and_configuration_are_compatible(monkeypatch, tmp_path):
    import hybrid_runtime
    run = Mock(return_value={"run_id": "x"})
    listing = Mock(return_value=[])
    loading = Mock(return_value={"run_id": "x"})
    monkeypatch.setattr(hybrid_runtime, "run_hybrid_research", run)
    monkeypatch.setattr(hybrid_runtime, "list_hybrid_runs", listing)
    monkeypatch.setattr(hybrid_runtime, "load_hybrid_run", loading)
    config = hybrid_dashboard._make_config(network_mode="offline", asset="TL", max_model_calls=4)
    assert hybrid_dashboard.run_hybrid_research(tmp_path, config=config) == {"run_id": "x"}
    assert hybrid_dashboard.list_hybrid_runs(tmp_path) == []
    assert hybrid_dashboard.load_hybrid_run(tmp_path, "x") == {"run_id": "x"}


def test_real_offline_runtime_returns_diagnostic_with_missing_input(app):
    import os
    pd.DataFrame().to_pickle(os.environ["TEST_HYBRID_FRAME"])
    app.run().button(key="hybrid_run").click().run()
    assert not app.exception
    assert "研究阻断" in visible_text(app)
    # No production artifact exists; the UI must not substitute its preview.
    assert "invalid_snapshot" in visible_text(app)


def test_blocked_reasons_are_explained_without_losing_the_raw_code(app, monkeypatch):
    value = report()
    value["decision"].update(status="blocked", blocked_reasons=[
        "canonical_model_unavailable", "canonical_not_ready:insufficient_policy_history",
        "brand_new_runtime_code"])
    monkeypatch.setattr(hybrid_dashboard, "run_hybrid_research", Mock(return_value=value))
    app.run().button(key="hybrid_run").click().run()
    assert not app.exception
    text = visible_text(app)
    # One grouped block, not one red box per reason.
    assert "研究阻断（3 项）" in text
    assert "基线模型不可用" in text and "canonical_model_unavailable" in text
    # A prefixed code keeps its detail suffix.
    assert "insufficient_policy_history" in text
    # An unmapped code must survive verbatim rather than be swallowed.
    assert "brand_new_runtime_code" in text


def test_rich_blocked_report_discloses_staleness_boundaries_and_components(app, monkeypatch):
    value = report()
    value["snapshot"].update(stale=True, excluded_future_rows=42)
    value["baseline"].update(decision_ready=False, missing_reason="insufficient_policy_history")
    value["decision"].update(execution_authorized=True, conflicts=[{"kind": "weight", "summary": "disagree"}])
    value["opinions"]["quant"]["output"]["components"] = {"micro_alpha": .02, "macro_state": .01}
    value["opinions"]["jev"].update(status="ok", output={"probabilities": {"reduce": .7, "abstain": .3}})
    value["opinions"]["advisor"] = {"status": "ok", "output": {"summary": "公开建议"}}
    value["route"] = {"status": "ok", "reason": "route supported"}
    value["evidence"] = []
    value["stages"][0].update(reason="阶段真实理由", elapsed_ms=3)
    monkeypatch.setattr(hybrid_dashboard, "run_hybrid_research", Mock(return_value=value))
    app.run().button(key="hybrid_run").click().run()
    assert not app.exception
    assert "超过" in visible_text(app) and "42" in visible_text(app)
    assert "安全边界" in visible_text(app)
    assert "insufficient_policy_history" in visible_text(app)
    assert "没有符合" in visible_text(app)
    assert any("宏观 / 微观" in item.label for item in app.expander)


@pytest.mark.parametrize("value", [None, "bad", float("nan"), float("inf")])
def test_unavailable_numbers_never_look_like_zero(value):
    assert hybrid_dashboard._number(value) == "未提供"


@pytest.mark.parametrize("url", [None, "https://example.com/\nsecret", "https://[invalid-host"])
def test_malformed_source_links_are_not_clickable(url):
    assert not hybrid_dashboard.safe_source_url(url)


def test_dag_only_uses_recorded_edges_and_escapes_labels():
    graph = hybrid_dashboard._dag_source([
        {"stage_id": "a", "label": 'untrusted "label"', "status": "error"},
        {"stage_id": "b", "dependencies": ["a", "does-not-exist"], "status": "skipped"},
    ])
    assert "n0 -> n1" in graph
    assert "does-not-exist" not in graph
    assert '\\"label\\"' in graph


def test_bad_time_and_empty_question_do_not_start_runtime(app, monkeypatch):
    run = Mock()
    monkeypatch.setattr(hybrid_dashboard, "run_hybrid_research", run)
    app.run().text_area(key="hybrid_question").set_value(" ").run()
    app.button(key="hybrid_run").click().run()
    run.assert_not_called()
    assert "填写" in visible_text(app)
    app.text_area(key="hybrid_question").set_value("question").run()
    app.text_input(key="hybrid_as_of").set_value("NaT").run()
    app.button(key="hybrid_run").click().run()
    run.assert_not_called()
    assert "时间无效" in visible_text(app)


def test_unsupported_result_schema_is_not_displayed(app, monkeypatch):
    monkeypatch.setattr(hybrid_dashboard, "run_hybrid_research", Mock(return_value={"run_id": "wrong"}))
    app.run().button(key="hybrid_run").click().run()
    assert not app.exception
    assert "不支持" in visible_text(app)
    assert "运行编号" not in visible_text(app)


def test_history_index_failure_keeps_diagnostic_entry(app, monkeypatch):
    monkeypatch.setattr(hybrid_dashboard, "list_hybrid_runs", Mock(side_effect=OSError("index missing")))
    app.run().button(key="hybrid_scan_history").click().run()
    assert not app.exception
    assert "index missing" in visible_text(app)
    assert not app.button(key="hybrid_run").disabled


def test_report_without_stage_details_does_not_invent_timing(app, monkeypatch):
    value = report()
    value["stages"] = []
    monkeypatch.setattr(hybrid_dashboard, "run_hybrid_research", Mock(return_value=value))
    app.run().button(key="hybrid_run").click().run()
    assert not app.exception
    assert "没有阶段记录" in visible_text(app)


def test_initial_data_state_reports_invalid_timestamps(app, monkeypatch):
    import os
    pd.DataFrame({"date": ["bad-timestamp"]}).to_pickle(os.environ["TEST_HYBRID_FRAME"])
    app.run()
    assert not app.exception
    assert "无有效时间戳" in visible_text(app)


@pytest.mark.parametrize("kind,expected", [
    ("missing", "缺少 date"), ("future", "未来时间"),
    ("fresh", "数据时效"), ("partial", "无效时间戳"),
])
def test_input_date_conditions_are_visible(app, kind, expected):
    import os
    if kind == "missing":
        frame = pd.DataFrame({"close": [1.]})
    elif kind == "future":
        frame = pd.DataFrame({"date": [pd.Timestamp.now(tz="Asia/Shanghai") + pd.Timedelta(days=1)]})
    elif kind == "fresh":
        frame = pd.DataFrame({"date": [pd.Timestamp.now(tz="Asia/Shanghai") - pd.Timedelta(minutes=1)]})
    else:
        frame = pd.DataFrame({"date": ["2020-01-01", "invalid"]})
    frame.to_pickle(os.environ["TEST_HYBRID_FRAME"])
    app.run()
    assert not app.exception
    assert expected in visible_text(app)


def test_factor_read_failure_preserves_diagnostic_entry(app):
    import os
    from pathlib import Path
    Path(os.environ["TEST_HYBRID_FRAME"]).write_bytes(b"broken pickle")
    app.run()
    assert not app.exception
    assert "数据读取失败" in visible_text(app)
    assert not app.button(key="hybrid_run").disabled


def test_history_schema_mismatch_has_no_fake_result(app, monkeypatch):
    monkeypatch.setattr(hybrid_dashboard, "list_hybrid_runs", Mock(return_value=[{"run_id": "old-schema"}]))
    monkeypatch.setattr(hybrid_dashboard, "load_hybrid_run", Mock(return_value={"schema_version": "old"}))
    app.run().button(key="hybrid_scan_history").click().run()
    app.selectbox(key="hybrid_history").select("old-schema").run()
    app.button(key="hybrid_load_history").click().run()
    assert not app.exception
    assert "不支持的历史报告格式" in visible_text(app)
    assert "运行编号" not in visible_text(app)


def readiness(eligible=True, sha="sha-1"):
    return {"schema_version": "f_agent.readiness.v1", "status": "ready" if eligible else "blocked",
        "factors": {"path": "factor.pkl", "sha256": sha, "rows": 1000, "end": "2026-10-01", "pit_verified": True},
        "training": {"eligible": eligible, "valid_rows": 900, "valid_days": 300, "days_without_valid_labels": 2,
                     "partitions": {"train": {"rows": 500}, "validation": {"rows": 200}, "sealed_holdout": {"rows": 200}}},
        "model": {"status": "missing", "reason": "no release"},
        "checks": [{"id": "sample", "status": "pass" if eligible else "blocked", "detail": "enough samples" if eligible else "insufficient samples", "action": "inspect inputs"}],
        "next_actions": ["训练独立候选"]}


@pytest.fixture
def preparation(app, monkeypatch, tmp_path):
    inspect = Mock(return_value=readiness())
    path = tmp_path / "outputs" / "factor_candidates" / "one" / "factors.pkl"
    build = Mock(return_value={"status": "built", "candidate_id": "one", "factor_path": str(path),
                 "manifest_path": str(path.with_suffix(".json")), "coverage": {"rows": 1000}, "warnings": ["历史数据未补齐"]})
    train = Mock(return_value={"status": "candidate_ready", "published": False, "run_id": "train-one",
                 "candidate_manifest": "candidate.json", "gates": {"validation": "passed"}, "errors": []})
    monkeypatch.setattr(hybrid_dashboard, "inspect_research_readiness", inspect)
    monkeypatch.setattr(hybrid_dashboard, "build_local_factor_candidate", build)
    monkeypatch.setattr(hybrid_dashboard, "train_candidate", train)
    return inspect, build, train, path


def test_preparation_controls_do_not_run_automatically(app, preparation):
    inspect, build, train, _ = preparation
    app.run()
    assert not app.exception
    inspect.assert_not_called()
    build.assert_not_called()
    train.assert_not_called()
    assert "不联网" in visible_text(app)
    assert app.selectbox(key="hybrid_factor_source").value == ""


def test_selected_minute_snapshot_only_affects_explicit_factor_build(app, preparation, monkeypatch):
    import minute_dashboard
    inspect, build, train, _ = preparation
    selector = Mock(return_value="minset-selected")
    monkeypatch.setattr(minute_dashboard, "render_minute_preparation", selector)
    app.run()
    build.assert_not_called()
    app.button(key="hybrid_build_candidate").click().run()
    assert not app.exception
    assert build.call_args.kwargs == {"minute_snapshot_id": "minset-selected"}
    assert app.selectbox(key="hybrid_factor_source").value == ""
    selector.return_value = "minset-another"
    app.run()
    build.assert_called_once()
    inspect.assert_not_called()
    train.assert_not_called()


def test_readiness_is_explicit_and_survives_rerun(app, preparation):
    inspect, _, train, _ = preparation
    app.run().button(key="hybrid_check_readiness").click().run()
    assert not app.exception
    inspect.assert_called_once()
    train.assert_not_called()
    app.text_area(key="hybrid_question").set_value("changed").run()
    inspect.assert_called_once()
    assert "有效标签" in visible_text(app)


def test_built_candidate_is_separate_and_passed_by_path(app, preparation, monkeypatch):
    inspect, build, train, path = preparation
    run = Mock(return_value=report())
    monkeypatch.setattr(hybrid_dashboard, "run_hybrid_research", run)
    app.run().button(key="hybrid_build_candidate").click().run()
    assert not app.exception
    build.assert_called_once()
    inspect.assert_not_called()
    train.assert_not_called()
    assert app.selectbox(key="hybrid_factor_source").value == ""
    app.selectbox(key="hybrid_factor_source").select(str(path)).run()
    app.button(key="hybrid_run").click().run()
    assert not app.exception
    assert run.call_args.kwargs["factors"] is None
    assert run.call_args.kwargs["factor_path"] == str(path)
    assert "候选因子" in visible_text(app)


def test_training_needs_readiness_and_local_compute_consent(app, preparation):
    inspect, _, train, _ = preparation
    app.run().button(key="hybrid_train_candidate").click().run()
    train.assert_not_called()
    assert "确认" in visible_text(app)
    app.checkbox(key="hybrid_local_training_consent").check().run()
    app.button(key="hybrid_train_candidate").click().run()
    train.assert_not_called()
    inspect.assert_not_called()
    app.button(key="hybrid_check_readiness").click().run()
    app.button(key="hybrid_train_candidate").click().run()
    assert not app.exception
    assert inspect.call_count == 2
    train.assert_called_once()
    assert "未发布" in visible_text(app)
    assert "train-one" in visible_text(app)


def test_ineligible_readiness_blocks_training(app, preparation):
    inspect, _, train, _ = preparation
    inspect.return_value = readiness(False)
    app.run().button(key="hybrid_check_readiness").click().run()
    app.checkbox(key="hybrid_local_training_consent").check().run()
    app.button(key="hybrid_train_candidate").click().run()
    train.assert_not_called()
    assert "insufficient samples" in visible_text(app)


def test_changed_factor_fingerprint_blocks_training_until_rechecked(app, preparation):
    inspect, _, train, _ = preparation
    inspect.side_effect = [readiness(), readiness(sha="changed")]
    app.run().button(key="hybrid_check_readiness").click().run()
    app.checkbox(key="hybrid_local_training_consent").check().run()
    app.button(key="hybrid_train_candidate").click().run()
    train.assert_not_called()
    assert "变化" in visible_text(app)


@pytest.mark.parametrize("operation,key", [("inspect", "hybrid_check_readiness"), ("build", "hybrid_build_candidate")])
def test_preparation_failure_clears_previous_success(app, preparation, operation, key):
    inspect, build, _, _ = preparation
    target = inspect if operation == "inspect" else build
    target.side_effect = [target.return_value, RuntimeError("preparation failure")]
    app.run().button(key=key).click().run()
    app.button(key=key).click().run()
    assert not app.exception
    assert "preparation failure" in visible_text(app)
    assert not app.success


def test_training_failure_clears_prior_training_success(app, preparation):
    _, _, train, _ = preparation
    train.side_effect = [train.return_value, RuntimeError("training failed")]
    app.run().button(key="hybrid_check_readiness").click().run()
    app.checkbox(key="hybrid_local_training_consent").check().run()
    app.button(key="hybrid_train_candidate").click().run()
    app.button(key="hybrid_train_candidate").click().run()
    assert not app.exception
    assert "training failed" in visible_text(app)
    assert "train-one" not in visible_text(app)


def test_switching_source_invalidates_training_readiness(app, preparation):
    _, _, train, path = preparation
    app.run().button(key="hybrid_check_readiness").click().run()
    app.button(key="hybrid_build_candidate").click().run()
    app.selectbox(key="hybrid_factor_source").select(str(path)).run()
    app.checkbox(key="hybrid_local_training_consent").check().run()
    app.button(key="hybrid_train_candidate").click().run()
    train.assert_not_called()
    assert "重新检查" in visible_text(app)


def test_candidate_training_uses_selected_factor_path(app, preparation):
    inspect, _, train, path = preparation
    app.run().button(key="hybrid_build_candidate").click().run()
    app.selectbox(key="hybrid_factor_source").select(str(path)).run()
    app.button(key="hybrid_check_readiness").click().run()
    app.checkbox(key="hybrid_local_training_consent").check().run()
    app.button(key="hybrid_train_candidate").click().run()
    assert not app.exception
    assert inspect.call_args.kwargs["factor_path"] == str(path)
    assert train.call_args.kwargs["factor_path"] == str(path)
    assert "顶部仅为生产输入预览" in visible_text(app)


@pytest.mark.parametrize("status", ["candidate_rejected", "failed"])
def test_rejected_or_failed_training_is_not_reported_as_published(app, preparation, status):
    _, _, train, _ = preparation
    train.return_value.update(status=status, errors=["gate not passed"])
    app.run().button(key="hybrid_check_readiness").click().run()
    app.checkbox(key="hybrid_local_training_consent").check().run()
    app.button(key="hybrid_train_candidate").click().run()
    assert not app.exception
    assert "未发布" in visible_text(app)
    assert "gate not passed" in visible_text(app)
    assert not app.success


def test_publication_contract_violation_is_an_error(app, preparation):
    _, _, train, _ = preparation
    train.return_value["published"] = True
    app.run().button(key="hybrid_check_readiness").click().run()
    app.checkbox(key="hybrid_local_training_consent").check().run()
    app.button(key="hybrid_train_candidate").click().run()
    assert not app.exception
    assert "不符合未发布候选契约" in visible_text(app)
    assert not app.success


def test_changed_readiness_eligibility_prevents_training(app, preparation):
    inspect, _, train, _ = preparation
    inspect.side_effect = [readiness(), readiness(False)]
    app.run().button(key="hybrid_check_readiness").click().run()
    app.checkbox(key="hybrid_local_training_consent").check().run()
    app.button(key="hybrid_train_candidate").click().run()
    assert not app.exception
    train.assert_not_called()
    assert "复核未通过" in visible_text(app)


@pytest.mark.parametrize("value,expected", [
    ({"status": "failed", "warnings": ["bad input"]}, "未完成"),
    ({"status": "built"}, "缺少候选因子路径"),
    ({"status": "built", "factor_path": "/outside-project/candidate.pkl"}, "不在当前项目"),
    ({"status": "unexpected"}, "不支持的因子候选"),
])
def test_bad_candidate_build_never_appears_in_source_choices(app, preparation, value, expected):
    _, build, _, _ = preparation
    build.return_value = value
    app.run().button(key="hybrid_build_candidate").click().run()
    assert not app.exception
    assert expected in visible_text(app)
    assert app.selectbox(key="hybrid_factor_source").options == ["生产因子 · outputs/df_factors.pkl"]
    assert not app.success


def test_relative_candidate_path_is_resolved_within_project(app, preparation, tmp_path):
    _, build, _, _ = preparation
    build.return_value["factor_path"] = "outputs/factor_candidates/two/factors.pkl"
    app.run().button(key="hybrid_build_candidate").click().run()
    assert not app.exception
    app.selectbox(key="hybrid_factor_source").select(str(tmp_path / build.return_value["factor_path"])).run()
    assert "候选因子" in visible_text(app)


def test_stale_candidate_selection_resets_without_reading_or_training(app, preparation):
    inspect, build, train, _ = preparation
    app.run().button(key="hybrid_build_candidate").click().run()
    app.selectbox(key="hybrid_factor_source").select(build.return_value["factor_path"]).run()
    app.session_state["hybrid_factor_candidates"] = {}
    app.run()
    assert not app.exception
    assert app.selectbox(key="hybrid_factor_source").value == ""
    inspect.assert_not_called()
    build.assert_called_once()
    train.assert_not_called()


def test_bad_readiness_schema_is_visible(app, preparation):
    inspect, _, _, _ = preparation
    inspect.return_value = {"schema_version": "old"}
    app.run().button(key="hybrid_check_readiness").click().run()
    assert not app.exception
    assert "不支持的就绪检查" in visible_text(app)


def test_readiness_warning_check_is_visible(app, preparation):
    inspect, _, _, _ = preparation
    inspect.return_value["checks"] = [{"status": "warning", "detail": "stale factor data"}]
    app.run().button(key="hybrid_check_readiness").click().run()
    assert not app.exception
    assert "stale factor data" in visible_text(app)


def test_readiness_limits_verification_claim_and_discloses_coverage_sources(app, preparation):
    inspect, _, _, _ = preparation
    inspect.return_value["factors"]["provenance"] = {
        "verified": True, "coverage": {"largest_observed_day_gap_days": 207},
        "sources": {"CNN": {"status": "excluded", "reason": "requires fold-local training"}}}
    app.run().button(key="hybrid_check_readiness").click().run()
    assert not app.exception
    text = visible_text(app)
    assert "本地构建契约与哈希：已核验" in text
    assert "不代表行情完整性或 PIT 全认证" in text
    assert "207 天" in text
    assert "CNN" in text and "requires fold-local training" in text
    assert "时点可用性核验：已核验" not in text
    assert "构建来源与覆盖限制" in [item.label for item in app.expander]


def test_preparation_wrappers_delegate_without_reinterpreting_results(monkeypatch):
    import sys
    from types import SimpleNamespace
    inspect, build, train = Mock(return_value={"kind": "inspect"}), Mock(return_value={"kind": "build"}), Mock(return_value={"kind": "train"})
    monkeypatch.setitem(sys.modules, "research_readiness", SimpleNamespace(inspect_research_readiness=inspect))
    monkeypatch.setitem(sys.modules, "factor_candidates", SimpleNamespace(build_local_factor_candidate=build))
    monkeypatch.setitem(sys.modules, "LightGBM_model", SimpleNamespace(train_candidate=train))
    assert hybrid_dashboard.inspect_research_readiness("base", factor_path="candidate") == {"kind": "inspect"}
    assert hybrid_dashboard.build_local_factor_candidate("base") == {"kind": "build"}
    assert hybrid_dashboard.train_candidate("base", factor_path="candidate") == {"kind": "train"}
    inspect.assert_called_once_with("base", factor_path="candidate")
    build.assert_called_once_with("base")
    train.assert_called_once_with("base", factor_path="candidate")


@pytest.mark.parametrize("snapshot_field", [{"factor_path": "outputs/factor_candidates/one/factors.pkl"}, {"source": "injected"}])
def test_report_declares_recorded_input_source(app, preparation, monkeypatch, snapshot_field):
    value = report()
    value["snapshot"].update(snapshot_field)
    monkeypatch.setattr(hybrid_dashboard, "run_hybrid_research", Mock(return_value=value))
    app.run().button(key="hybrid_run").click().run()
    assert not app.exception
    assert "因子来源" in visible_text(app) or "内存因子快照" in visible_text(app)


def saved_candidate(base_dir, number=1, **updates):
    import json
    candidate_id = f"local-{number:032x}"
    directory = base_dir / "artifacts" / "factor_candidates" / candidate_id
    directory.mkdir(parents=True)
    (directory / "factors.pkl").write_bytes(b"must not be unpickled")
    manifest = {"schema_version": 1, "status": "built", "candidate_id": candidate_id,
                "factor_file": "factors.pkl", "factor_sha256": "a" * 64,
                "coverage": {"rows": 1200}, "warnings": ["local history only"]}
    manifest.update(updates)
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def test_saved_candidate_scan_is_explicit_and_selection_keeps_preview_read_only(app, preparation, monkeypatch, tmp_path):
    import os
    from pathlib import Path
    directory = saved_candidate(tmp_path)
    scan = Mock(wraps=hybrid_dashboard.list_saved_factor_candidates)
    monkeypatch.setattr(hybrid_dashboard, "list_saved_factor_candidates", scan)
    run = Mock(return_value=report())
    monkeypatch.setattr(hybrid_dashboard, "run_hybrid_research", run)
    preview = Path(os.environ["TEST_HYBRID_FRAME"])
    before = preview.read_bytes()
    app.run()
    scan.assert_not_called()
    app.button(key="hybrid_scan_candidates").click().run()
    assert not app.exception
    scan.assert_called_once()
    assert app.selectbox(key="hybrid_factor_source").value == ""
    app.selectbox(key="hybrid_factor_source").select(str(directory / "factors.pkl")).run()
    assert not app.exception
    scan.assert_called_once()
    run.assert_not_called()
    inspect, build, train, _ = preparation
    inspect.assert_not_called()
    build.assert_not_called()
    train.assert_not_called()
    assert preview.read_bytes() == before
    app.button(key="hybrid_run").click().run()
    assert run.call_args.kwargs["factors"] is None
    assert run.call_args.kwargs["factor_path"] == str(directory / "factors.pkl")
    assert "不加载" in visible_text(app)


def test_saved_candidate_scan_reads_manifests_only_and_is_bounded(tmp_path, monkeypatch):
    from pathlib import Path
    for number in range(1, 36):
        saved_candidate(tmp_path, number)
    opened = []
    original_open = Path.open
    def track_open(path, *args, **kwargs):
        opened.append(path)
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", track_open)
    monkeypatch.setattr(pd, "read_pickle", Mock(side_effect=AssertionError("must not unpickle")))
    result = hybrid_dashboard.list_saved_factor_candidates(tmp_path)
    assert result["scanned"] == 30
    assert result["limited"] is True
    assert len(result["candidates"]) == 30
    assert len(opened) == 30
    assert all(path.name == "manifest.json" for path in opened)


@pytest.mark.parametrize("updates", [
    {"schema_version": "1"}, {"schema_version": True}, {"status": "failed"},
    {"candidate_id": "another"}, {"factor_file": "../factors.pkl"},
    {"factor_file": "/tmp/factors.pkl"}, {"factor_file": "other.pkl"},
])
def test_saved_candidate_scan_skips_contract_mismatch(tmp_path, updates):
    saved_candidate(tmp_path, **updates)
    result = hybrid_dashboard.list_saved_factor_candidates(tmp_path)
    assert result["candidates"] == []
    assert result["skipped"] == 1


@pytest.mark.parametrize("content", [b"invalid json", b"[]", b"\xff", b" " * (1024 * 1024 + 1)],
                         ids=["bad-json", "not-object", "bad-encoding", "too-large"])
def test_saved_candidate_scan_skips_bad_or_oversized_manifest(tmp_path, content):
    directory = saved_candidate(tmp_path)
    (directory / "manifest.json").write_bytes(content)
    result = hybrid_dashboard.list_saved_factor_candidates(tmp_path)
    assert result["candidates"] == []
    assert result["skipped"] == 1


@pytest.mark.parametrize("target", ["candidate", "manifest", "factor", "root", "artifacts"])
def test_saved_candidate_scan_rejects_symlinks(tmp_path, target):
    directory = saved_candidate(tmp_path)
    path = {"candidate": directory, "manifest": directory / "manifest.json", "factor": directory / "factors.pkl",
            "root": directory.parent, "artifacts": directory.parent.parent}[target]
    moved = tmp_path / "moved"
    path.rename(moved)
    path.symlink_to(moved, target_is_directory=moved.is_dir())
    assert hybrid_dashboard.list_saved_factor_candidates(tmp_path)["candidates"] == []


def test_saved_candidate_scan_missing_factor_or_root_is_empty(tmp_path):
    assert hybrid_dashboard.list_saved_factor_candidates(tmp_path)["candidates"] == []
    (tmp_path / "artifacts" / "factor_candidates" / "not-a-candidate").mkdir(parents=True)
    assert hybrid_dashboard.list_saved_factor_candidates(tmp_path)["scanned"] == 0
    directory = saved_candidate(tmp_path)
    (directory / "factors.pkl").unlink()
    assert hybrid_dashboard.list_saved_factor_candidates(tmp_path)["candidates"] == []


def test_saved_candidate_scan_failure_clears_old_scan_status(app, preparation, monkeypatch):
    scan = Mock(side_effect=[{"candidates": [], "scanned": 0, "skipped": 0, "limited": False}, RuntimeError("scan unavailable")])
    monkeypatch.setattr(hybrid_dashboard, "list_saved_factor_candidates", scan)
    app.run().button(key="hybrid_scan_candidates").click().run()
    app.button(key="hybrid_scan_candidates").click().run()
    assert not app.exception
    assert "scan unavailable" in visible_text(app)
    assert "扫描完成" not in visible_text(app)


def test_saved_candidate_manifest_growth_is_bounded(tmp_path, monkeypatch):
    import io
    from pathlib import Path
    saved_candidate(tmp_path)
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: io.BytesIO(b" " * (1024 * 1024 + 1)))
    result = hybrid_dashboard.list_saved_factor_candidates(tmp_path)
    assert result["candidates"] == []
    assert result["skipped"] == 1


def test_saved_candidate_scan_limit_is_disclosed(app, preparation, monkeypatch):
    monkeypatch.setattr(hybrid_dashboard, "list_saved_factor_candidates", Mock(return_value={
        "candidates": [], "scanned": 30, "skipped": 30, "limited": True}))
    app.run().button(key="hybrid_scan_candidates").click().run()
    assert not app.exception
    assert "不是所有历史候选的完整清单" in visible_text(app)


def split_app_script():
    import os
    import streamlit as st
    from hybrid_dashboard import render_hybrid_workbench, render_preparation_workbench
    class Context:
        base_dir = os.environ["TEST_HYBRID_BASE"]
        @property
        def df_factors(self):
            raise AssertionError("Opening a split view must not deserialize preview factors")
        @property
        def signal(self):
            raise AssertionError("Opening a split view must not compute signals")
    page = st.radio("Section", ["research", "preparation", "other"], key="test_section")
    if page == "research":
        render_hybrid_workbench(Context(), include_preparation=False)
    elif page == "preparation":
        render_preparation_workbench(Context())
    else:
        st.info("Other workspace")


def test_split_research_hides_maintenance_and_never_scans_on_open(app, preparation, monkeypatch):
    inspect, build, train, _ = preparation
    scan = Mock()
    history = Mock()
    run = Mock()
    monkeypatch.setattr(hybrid_dashboard, "list_saved_factor_candidates", scan)
    monkeypatch.setattr(hybrid_dashboard, "list_hybrid_runs", history)
    monkeypatch.setattr(hybrid_dashboard, "run_hybrid_research", run)
    split = AppTest.from_function(split_app_script).run()
    assert not split.exception
    assert "hybrid_run" in [item.key for item in split.button]
    assert "hybrid_build_candidate" not in [item.key for item in split.button]
    assert "hybrid_train_candidate" not in [item.key for item in split.button]
    assert "minute_format" not in [item.key for item in split.selectbox]
    for operation in (scan, history, inspect, build, train, run):
        operation.assert_not_called()
    assert "读取失败" not in visible_text(split)


def test_dedicated_preparation_is_expanded_and_does_not_render_research(app, preparation):
    inspect, build, train, _ = preparation
    split = AppTest.from_function(split_app_script).run()
    split.radio(key="test_section").set_value("preparation").run()
    assert not split.exception
    assert "hybrid_build_candidate" in [item.key for item in split.button]
    assert "hybrid_run" not in [item.key for item in split.button]
    # The dedicated page owns the heading, so the steps are not nested in an
    # outer collapsible: they must be reachable without expanding anything.
    assert "数据与模型准备" not in [item.label for item in split.expander]
    assert "不联网" in visible_text(split)
    for operation in (inspect, build, train):
        operation.assert_not_called()


def test_factor_selection_and_research_parameters_survive_unrendered_page(app, preparation, monkeypatch):
    _, _, _, path = preparation
    run = Mock(return_value=report())
    monkeypatch.setattr(hybrid_dashboard, "run_hybrid_research", run)
    split = AppTest.from_function(split_app_script).run()
    split.radio(key="test_section").set_value("preparation").run()
    split.button(key="hybrid_build_candidate").click().run()
    split.selectbox(key="hybrid_factor_source").select(str(path)).run()
    split.radio(key="test_section").set_value("research").run()
    assert split.selectbox(key="hybrid_factor_source").value == str(path)
    split.text_area(key="hybrid_question").set_value("复核政策变化的证据").run()
    split.selectbox(key="hybrid_network").select("cache").run()
    split.number_input(key="hybrid_max_calls").set_value(2).run()
    split.number_input(key="hybrid_max_age").set_value(48).run()
    split.radio(key="test_section").set_value("other").run()
    split.run()
    split.radio(key="test_section").set_value("research").run()
    assert not split.exception
    assert split.selectbox(key="hybrid_factor_source").value == str(path)
    assert split.text_area(key="hybrid_question").value == "复核政策变化的证据"
    assert split.selectbox(key="hybrid_network").value == "cache"
    assert split.number_input(key="hybrid_max_calls").value == 2
    assert split.number_input(key="hybrid_max_age").value == 48
    run.assert_not_called()
    split.button(key="hybrid_run").click().run()
    assert run.call_args.kwargs["factor_path"] == str(path)
    assert run.call_args.kwargs["config"].network_mode == "cache"


def test_page_navigation_does_not_preserve_online_authorization(app, preparation, monkeypatch):
    run = Mock(return_value=report())
    monkeypatch.setattr(hybrid_dashboard, "run_hybrid_research", run)
    split = AppTest.from_function(split_app_script).run()
    split.selectbox(key="hybrid_network").select("online").run()
    split.checkbox(key="hybrid_online_consent").check().run()
    split.radio(key="test_section").set_value("preparation").run()
    split.run()
    split.radio(key="test_section").set_value("research").run()
    assert not split.exception
    assert split.selectbox(key="hybrid_network").value == "online"
    assert split.checkbox(key="hybrid_online_consent").value is False
    split.button(key="hybrid_run").click().run()
    run.assert_not_called()
    assert "请先确认在线" in visible_text(split)


def test_preparation_uses_retained_research_age_without_restoring_training_consent(app, preparation):
    inspect, _, train, _ = preparation
    split = AppTest.from_function(split_app_script).run()
    split.number_input(key="hybrid_max_age").set_value(72).run()
    split.radio(key="test_section").set_value("preparation").run()
    split.button(key="hybrid_check_readiness").click().run()
    assert inspect.call_args.kwargs["max_age_hours"] == 72
    split.checkbox(key="hybrid_local_training_consent").check().run()
    split.radio(key="test_section").set_value("other").run()
    split.radio(key="test_section").set_value("preparation").run()
    assert not split.exception
    assert split.checkbox(key="hybrid_local_training_consent").value is False
    train.assert_not_called()


def test_report_survives_page_navigation_without_rerunning(app, preparation, monkeypatch):
    run = Mock(return_value=report())
    monkeypatch.setattr(hybrid_dashboard, "run_hybrid_research", run)
    split = AppTest.from_function(split_app_script).run()
    split.button(key="hybrid_run").click().run()
    split.radio(key="test_section").set_value("other").run()
    split.radio(key="test_section").set_value("research").run()
    assert not split.exception
    run.assert_called_once()
    assert "run_test" in visible_text(split)
    assert [item.label for item in split.tabs] == ["决策链路", "模型证据", "原始情报"]


def test_new_workspace_discards_retained_source_and_report(app, preparation, monkeypatch, tmp_path):
    _, _, _, path = preparation
    run = Mock(return_value=report())
    monkeypatch.setattr(hybrid_dashboard, "run_hybrid_research", run)
    split = AppTest.from_function(split_app_script).run()
    split.radio(key="test_section").set_value("preparation").run()
    split.button(key="hybrid_build_candidate").click().run()
    split.selectbox(key="hybrid_factor_source").select(str(path)).run()
    split.radio(key="test_section").set_value("research").run()
    split.button(key="hybrid_run").click().run()
    monkeypatch.setenv("TEST_HYBRID_BASE", str(tmp_path / "other-workspace"))
    split.run()
    assert not split.exception
    assert split.selectbox(key="hybrid_factor_source").value == ""
    assert "run_test" not in visible_text(split)
    assert "hybrid_factor_candidates" not in split.session_state
    assert str(path) not in str(split.session_state["hybrid_preferences"])
    run.assert_called_once()
