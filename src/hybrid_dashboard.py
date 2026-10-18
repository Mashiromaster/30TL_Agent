"""Read-only rendering and explicit-run controls for the hybrid research runtime.

The selected report is the only source for result cards. Page reruns and history
inspection never regenerate model opinions or access ``ctx.signal``.
"""

from __future__ import annotations

import json
import math
import re
import stat
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import streamlit as st

from intelligence_events import redact_sensitive_text


DEFAULT_QUESTION = "当前国债期货信号有哪些支持和反对证据？"
MODES = {"offline": "离线 · 不访问模型服务", "cache": "缓存 · 仅复用已有结果", "online": "在线 · 允许受预算约束的模型调用"}
STATUS = {"ok": "已完成", "skipped": "已跳过", "unavailable": "不可用", "error": "失败",
          "blocked": "已阻断", "review_required": "待人工研究", "research_only": "仅研究", "not_run": "未运行"}
DIRECTIONS = {1: "做多", -1: "做空", 0: "观望"}
PRIVATE_FIELDS = frozenset({"chain_of_thought", "reasoning_content", "internal_reasoning", "thinking", "thinking_content", "reasoning", "thoughts",
    "scratchpad", "llm_reasoning", "api_key", "apikey", "access_token", "refresh_token",
    "authorization", "password", "secret", "token", "credentials"})


def run_hybrid_research(*args, **kwargs):
    from hybrid_runtime import run_hybrid_research as run
    return run(*args, **kwargs)


def list_hybrid_runs(*args, **kwargs):
    from hybrid_runtime import list_hybrid_runs as listing
    return listing(*args, **kwargs)


def load_hybrid_run(*args, **kwargs):
    from hybrid_runtime import load_hybrid_run as load
    return load(*args, **kwargs)


def _make_config(**kwargs):
    from hybrid_runtime import HybridConfig
    return HybridConfig(**kwargs)


def inspect_research_readiness(*args, **kwargs):
    from research_readiness import inspect_research_readiness as inspect
    return inspect(*args, **kwargs)


def build_local_factor_candidate(*args, **kwargs):
    from factor_candidates import build_local_factor_candidate as build
    return build(*args, **kwargs)


def train_candidate(*args, **kwargs):
    from LightGBM_model import train_candidate as train
    return train(*args, **kwargs)


def list_saved_factor_candidates(base_dir):
    """Read at most 30 small candidate manifests, never deserialize factors.

    Discovery is not provenance verification. The runtime/readiness inspector
    still verifies the actual snapshot before it can be used.
    """
    base = Path(base_dir).resolve()
    artifacts = base / "artifacts"
    root = artifacts / "factor_candidates"
    result = {"candidates": [], "scanned": 0, "skipped": 0, "limited": False}
    if artifacts.is_symlink() or root.is_symlink() or not root.is_dir():
        return result
    for directory in root.iterdir():
        if not re.fullmatch(r"local-[0-9a-f]{32}", directory.name):
            continue
        if result["scanned"] >= 30:
            result["limited"] = True
            break
        result["scanned"] += 1
        try:
            if directory.is_symlink() or not directory.is_dir() or directory.resolve().parent != root:
                raise ValueError("candidate directory is not a local directory")
            manifest_path = directory / "manifest.json"
            info = manifest_path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_size > 1024 * 1024:
                raise ValueError("manifest is not a bounded regular file")
            with manifest_path.open("rb") as handle:
                payload = handle.read(1024 * 1024 + 1)
            if len(payload) > 1024 * 1024:
                raise ValueError("manifest exceeds the size bound")
            value = json.loads(payload.decode("utf-8"))
            if (not isinstance(value, dict) or type(value.get("schema_version")) is not int
                    or value["schema_version"] != 1 or value.get("status") != "built"
                    or value.get("candidate_id") != directory.name or value.get("factor_file") != "factors.pkl"):
                raise ValueError("unsupported candidate manifest contract")
            factor_path = directory / "factors.pkl"
            if not stat.S_ISREG(factor_path.lstat().st_mode) or factor_path.resolve().parent != directory:
                raise ValueError("candidate factor is not a local regular file")
            result["candidates"].append({"status": "built", "candidate_id": directory.name,
                "factor_path": str(factor_path), "manifest_path": str(manifest_path),
                "factor_sha256": value.get("factor_sha256"), "coverage": value.get("coverage", {}),
                "warnings": value.get("warnings", []), "discovered_from_manifest": True})
        except (OSError, ValueError, TypeError, RecursionError):
            result["skipped"] += 1
    return result


def _text(value, secrets=()):
    text = redact_sensitive_text(str(value))
    for secret in secrets:
        if secret:
            text = text.replace(str(secret), "[REDACTED]")
    return text


def public_report(value, *, secrets=()):
    """Copy a public report without private reasoning or credential fields."""
    if isinstance(value, dict):
        return {str(key): public_report(item, secrets=secrets) for key, item in value.items()
                if str(key).lower() not in PRIVATE_FIELDS
                and not str(key).lower().endswith(("_api_key", "_access_token", "_password", "_secret"))}
    if isinstance(value, (list, tuple)):
        return [public_report(item, secrets=secrets) for item in value]
    if isinstance(value, str):
        return _text(value, secrets)
    return value


def safe_source_url(value):
    """Only link to explicit HTTP(S) URLs without embedded credentials."""
    if not isinstance(value, str) or any(ord(char) < 32 for char in value):
        return False
    try:
        parts = urlsplit(value)
        return bool(parts.scheme in {"http", "https"} and parts.hostname
                    and not parts.username and not parts.password)
    except ValueError:
        return False


def _number(value, *, percent=False, digits=2):
    try:
        number = float(value)
        if not math.isfinite(number):
            return "未提供"
        return f"{number:.{digits}%}" if percent else f"{number:.{digits}f}"
    except (TypeError, ValueError):
        return "未提供"


def _clear_selected():
    for key in ("hybrid_report", "hybrid_report_origin", "hybrid_run_request"):
        st.session_state.pop(key, None)


def _dag_source(stages):
    """Render only recorded dependency edges; labels are escaped DOT strings."""
    palette = {"ok": "#D8F3E5", "error": "#FADBD8", "blocked": "#FADBD8",
               "unavailable": "#FFF0C2", "skipped": "#EDF0F4", "not_run": "#EDF0F4"}
    indexes = {str(stage.get("stage_id", index)): f"n{index}" for index, stage in enumerate(stages)}
    lines = ['digraph run { rankdir=TB; graph [bgcolor="transparent",nodesep=0.35,ranksep=0.35];',
             'node [shape=box,style="rounded,filled",fontname="sans-serif",fontsize=14,margin="0.18,0.12",color="#B6C5D4"];',
             'edge [color="#8796A5",arrowsize=0.65];']
    for index, stage in enumerate(stages):
        status = str(stage.get("status", "not_run"))
        label = str(stage.get("label") or stage.get("stage_id") or "阶段")
        label = label.replace(" · ", "\n", 1)
        label += "\n" + STATUS.get(status, status)
        lines.append(f'n{index} [label={json.dumps(label, ensure_ascii=False)},fillcolor="{palette.get(status, "#EDF0F4")}"];')
        for dependency in stage.get("dependencies", []):
            if str(dependency) in indexes:
                lines.append(f'{indexes[str(dependency)]} -> n{index};')
    return "\n".join(lines + ["}"])


def _render_stages(report):
    st.subheader("运行路径与阶段记录")
    stages = report.get("stages") or []
    if not stages:
        st.info("这份报告没有阶段记录，无法还原执行路径或耗时。")
        return
    st.graphviz_chart(_dag_source(stages), width="stretch")
    st.caption("节点和依赖来自本次报告；成功仅表示该阶段完成，不表示交易获批。缺失时间显示未记录。")
    for stage in stages:
        status = str(stage.get("status", "unknown"))
        elapsed = stage.get("elapsed_ms")
        timing = "耗时未记录" if elapsed is None else f"耗时 {_number(elapsed, digits=0)} ms"
        with st.expander(f"{stage.get('label', stage.get('stage_id', '阶段'))} · {STATUS.get(status, status)} · {timing}"):
            st.caption(f"开始：{stage.get('started_at') or '未记录'} ｜ 结束：{stage.get('finished_at') or '未记录'}")
            if stage.get("reason"):
                st.text(stage["reason"])
            st.json({key: stage[key] for key in ("stage_id", "kind", "model", "source", "dependencies", "input_refs", "output") if key in stage})


def _render_models(report):
    st.subheader("模型证据 · 职责分离")
    opinions = report.get("opinions") or {}
    baseline = report.get("baseline") or {}
    columns = st.columns(3)
    for column, name, title in zip(columns, ("quant", "jev", "llm"),
                                   ("量化模型 / LightGBM", "Jev · 语义概率", "LLM · 引用与研究摘要")):
        with column:
            st.markdown(f"**{title}**")
            opinion = opinions.get(name) or {}
            status = str(opinion.get("status", "unavailable"))
            st.caption(f"状态：{STATUS.get(status, status)} ｜ 模型：{opinion.get('model') or '未提供'}")
            st.caption(f"来源：{opinion.get('source') or '未提供'}")
            if opinion.get("reason"):
                st.text(opinion["reason"])
            output = opinion.get("output") or {}
            if name == "quant" and baseline:
                st.metric("预测收益（%）", _number(baseline.get("predicted_return"), digits=4))
                st.metric("预测排名分位", _number(baseline.get("pred_rank_pct"), percent=True))
                st.caption(f"实际模型类型：{baseline.get('model_used') or opinion.get('model') or '未提供'}")
            if isinstance(output, dict):
                for key in ("summary", "public_summary", "assessment"):
                    if isinstance(output.get(key), str) and output[key]:
                        st.text(output[key])
                probabilities = output.get("probabilities") or output.get("gate_probabilities")
                if isinstance(probabilities, dict) and probabilities:
                    st.dataframe(pd.DataFrame([{"判断": key, "概率": value} for key, value in probabilities.items()]), hide_index=True, width="stretch")
                refs = output.get("evidence_ids") or output.get("citations") or output.get("evidence_refs")
                if refs:
                    st.text("引用证据：" + json.dumps(refs, ensure_ascii=False, default=str))
            with st.expander("公开结构化输出"):
                st.json(output if output else {"说明": "没有记录输出，不补造模型判断"})
    components = ((opinions.get("quant") or {}).get("output") or {}).get("components")
    if components:
        with st.expander("量化宏观 / 微观分解（模型真实输出）"):
            st.json(components)
    advisor = opinions.get("advisor")
    if advisor:
        with st.expander("研究裁决 / 宏微观证据（仅已记录内容）"):
            st.json(advisor)
    st.caption("预测分位、策略评分和语义概率含义不同，不做同权相加；仅展示公开输出，不展示内部思维链。")


def _render_evidence(report):
    st.subheader("可追溯证据")
    evidence = report.get("evidence") or []
    if not evidence:
        st.info("本轮没有符合时点约束的公开证据。缺少证据不是支持交易的理由。")
        return
    columns = ("id", "title", "source", "published_at", "available_at", "assessed_at", "bond_impact", "severity", "confidence")
    st.dataframe(pd.DataFrame([{key: item.get(key) for key in columns} for item in evidence]), hide_index=True, width="stretch")
    for index, item in enumerate(evidence):
        with st.expander(f"{item.get('id', index)} · {item.get('title') or '未命名证据'}"):
            if item.get("summary"):
                st.text(item["summary"])
            url = item.get("url")
            if safe_source_url(url):
                st.link_button("查看原始出处", url)
            elif url:
                st.caption("来源链接不是安全的 HTTP(S) 地址，未生成可点击链接。")


def _render_report(report):
    st.divider()
    st.subheader("本次研究结果")
    st.text(f"运行编号：{report.get('run_id', '未知')}")
    st.caption(f"生成时间：{report.get('generated_at') or '未记录'} ｜ 标的：{report.get('asset', '未提供')} ｜ 模式：{report.get('network_mode', '未提供')}")
    st.text("研究问题：" + str(report.get("question", "未记录")))
    snapshot = report.get("snapshot") or {}
    st.caption(f"数据截至：{snapshot.get('as_of') or '未记录'} ｜ 快照：{snapshot.get('snapshot_id') or '未记录'} ｜ 可用行数：{snapshot.get('rows', '未记录')}")
    source_path = snapshot.get("factor_path") or snapshot.get("source_path")
    if source_path:
        st.text("本报告因子来源：" + str(source_path))
    elif snapshot.get("source") == "injected":
        st.caption("本报告输入为传入的内存因子快照；未伪造磁盘路径。")
    if snapshot.get("stale"):
        st.warning("输入数据超过本次时效阈值。以下为历史快照研究，不是当前可执行交易信号。")
    if snapshot.get("excluded_future_rows"):
        st.caption(f"已排除快照时点之后的数据：{snapshot['excluded_future_rows']} 行。")

    baseline = report.get("baseline") or {}
    decision = report.get("decision") or {}
    left, right = st.columns(2)
    with left:
        st.markdown("**基础策略意图 · 不由研究模型改写**")
        if baseline:
            st.metric("基础方向", DIRECTIONS.get(baseline.get("direction"), "未提供"))
            st.metric("基础目标权重", _number(baseline.get("suggested_weight"), percent=True))
            if baseline.get("decision_ready") is False:
                st.warning("基础策略未就绪：" + str(baseline.get("missing_reason", "未记录具体原因")))
            st.caption(f"模型 release：{baseline.get('model_release_id') or '未记录'} ｜ 决策编号：{baseline.get('decision_id') or '未记录'}")
        else:
            st.warning("基础信号不可用，见下方研究阻断。缺失不等于观望信号。")
    with right:
        st.markdown("**研究建议 · 非执行指令**")
        st.metric("建议", {"maintain": "维持", "reduce": "降低风险", "abstain": "不交易 / 暂缓"}.get(decision.get("recommendation"), "未提供"))
        st.metric("建议权重", _number(decision.get("proposed_weight"), percent=True))
        st.caption("建议方向：" + DIRECTIONS.get(decision.get("proposed_direction"), "未提供"))
    status = str(decision.get("status", "unknown"))
    st.info("研究状态：" + STATUS.get(status, status) + "。本工作台不下单、不授权实盘。")
    if decision.get("execution_authorized") is not False or decision.get("baseline_unchanged") is not True:
        st.error("报告的安全边界声明缺失或异常，不能作为执行依据。")
    blocked = decision.get("blocked_reasons") or []
    if blocked:
        st.error("研究阻断（%d 项）\n\n" % len(blocked)
                 + "\n".join("- " + _blocked_reason_text(reason) for reason in blocked))
    for conflict in decision.get("conflicts") or []:
        st.warning("分歧：" + (json.dumps(conflict, ensure_ascii=False) if isinstance(conflict, dict) else str(conflict)))
    reasons = decision.get("reasons") or []
    if reasons:
        with st.expander("各环节状态（%d 条）" % len(reasons)):
            for reason in reasons:
                st.text(str(reason))
    budget = report.get("budget") or {}
    st.caption(f"模型调用预算：尝试 {budget.get('attempted_calls', '未记录')} / 上限 {budget.get('call_limit', '未记录')}。离线和缓存模式不发出新的模型请求。")
    path_tab, models_tab, evidence_tab = st.tabs(["决策链路", "模型证据", "原始情报"])
    with path_tab:
        _render_stages(report)
    with models_tab:
        _render_models(report)
    with evidence_tab:
        _render_evidence(report)
    if report.get("route"):
        with st.expander("路由依据"):
            st.json(report["route"])
    persistence = report.get("persistence") or {}
    st.caption(f"持久化状态：{persistence.get('status') or '未记录'}" + (f" ｜ {persistence['reason']}" if persistence.get("reason") else ""))
    with st.expander("原始报告 JSON（公开字段）"):
        st.json(report)
    st.download_button("下载研究报告 JSON（已排除密钥和内部推理）", json.dumps(report, ensure_ascii=False, indent=2, default=str),
                       file_name="hybrid_research_report.json", mime="application/json", key="hybrid_download")


def _readiness_result(value, api_key):
    if not isinstance(value, dict) or value.get("schema_version") != "f_agent.readiness.v1":
        raise ValueError("不支持的就绪检查报告格式")
    return public_report(value, secrets=(api_key,))


BLOCKED_REASONS = {
    "canonical_model_unavailable": "基线模型不可用：未找到已发布的 LightGBM 模型，需先训练或发布",
    "canonical_not_ready": "基线策略未就绪",
    "invalid_snapshot": "输入快照无效：因子数据缺失或为空",
    "stale_market_snapshot": "行情快照过期：超过本次设定的时效阈值，需更新行情",
    "unverified_factor_provenance": "因子来源未核验：构建契约或哈希未通过，需重新构建候选",
}


def _blocked_reason_text(reason):
    """Explain a block in Chinese, keeping the raw code for debugging.

    Runtime codes may carry a ``code:detail`` suffix (canonical_not_ready).
    """
    raw = str(reason)
    code, _, detail = raw.partition(":")
    text = BLOCKED_REASONS.get(code)
    if not text:
        return raw
    if detail:
        text += "（" + detail + "）"
    return f"{text} · `{code}`"


def _render_readiness(value):
    factors = value.get("factors") or {}
    training = value.get("training") or {}
    st.caption("这是显式检查时的只读快照，不会自动训练、刷新数据或发布模型。")
    st.text(f"检查文件：{factors.get('path', '未记录')} ｜ SHA256：{factors.get('sha256', '未记录')}")
    st.caption(f"检查数据范围：{factors.get('start', '未记录')} ～ {factors.get('end', '未记录')} ｜ {factors.get('rows', '未记录')} 行。")
    st.caption(f"有效标签：{training.get('valid_rows', '未记录')} 行 / {training.get('valid_days', '未记录')} 日；无有效标签日期：{training.get('days_without_valid_labels', '未记录')}。")
    st.caption("本地构建契约与哈希：" + ("已核验" if factors.get("pit_verified") else "未通过 / 未提供") + "；不代表行情完整性或 PIT 全认证。")
    provenance = factors.get("provenance") or {}
    if isinstance(provenance, dict) and provenance:
        coverage = provenance.get("coverage") or {}
        gap = coverage.get("largest_observed_day_gap_days") if isinstance(coverage, dict) else None
        if isinstance(gap, (int, float)) and gap > 1:
            st.warning(f"覆盖限制：相邻已观测日期的最大间隔为 {gap:g} 天；不等同于已核验交易日缺失数量，交易日历与行情完整性尚未认证。")
        sources = provenance.get("sources") or {}
        if isinstance(sources, dict):
            for name, source in sources.items():
                if isinstance(source, dict) and source.get("status") == "excluded":
                    st.text(f"构建排除来源 {name}：{source.get('reason') or '未记录原因'}")
        with st.expander("构建来源与覆盖限制"):
            st.json(provenance)
    model = value.get("model") or {}
    st.caption(f"生产模型状态：{model.get('status', '未记录')} ｜ release：{model.get('release_id') or '未记录'} ｜ {model.get('reason') or ''}")
    for check in value.get("checks") or []:
        text = f"{check.get('id', '检查项')}：{check.get('detail', '未记录')}"
        if check.get("action"):
            text += " ｜ 建议：" + str(check["action"])
        if check.get("status") == "blocked":
            st.error(text)
        elif check.get("status") == "warning":
            st.warning(text)
        else:
            st.text(text)
    if training.get("partitions"):
        with st.expander("标签分区 / 生产模型状态"):
            st.json({"partitions": training["partitions"], "model": value.get("model", {})})
    for action in value.get("next_actions") or []:
        st.text("下一步：" + str(action))


def _bind_workspace(ctx):
    """Keep state scoped to one checkout without triggering any data access."""
    base_dir = str(Path(ctx.base_dir).resolve())
    if st.session_state.get("hybrid_base_dir") != base_dir:
        for key in list(st.session_state):
            if key.startswith("hybrid_"):
                st.session_state.pop(key, None)
        st.session_state["hybrid_base_dir"] = base_dir
    return base_dir


def _restore_preferences(*keys):
    # Streamlit removes widget state when its page is not rendered. Store only
    # non-sensitive choices separately; never restore compute/network consent.
    saved = st.session_state.get("hybrid_preferences", {})
    for key in keys:
        if key not in st.session_state and key in saved:
            st.session_state[key] = saved[key]


def _remember_preferences(**values):
    st.session_state["hybrid_preferences"] = {
        **st.session_state.get("hybrid_preferences", {}), **values}


def _render_candidate_scan(base_dir, *, api_key=None):
    if st.button("读取已保存因子候选", key="hybrid_scan_candidates"):
        st.session_state.pop("hybrid_scan_result", None)
        try:
            listing = public_report(list_saved_factor_candidates(base_dir), secrets=(api_key,))
            candidates = dict(st.session_state.get("hybrid_factor_candidates", {}))
            for item in listing["candidates"]:
                candidates[item["factor_path"]] = item
            st.session_state["hybrid_factor_candidates"] = candidates
            st.session_state["hybrid_scan_result"] = listing
        except Exception as exc:
            st.error("已保存候选读取失败：" + _text(exc, (api_key,)))
    listing = st.session_state.get("hybrid_scan_result")
    if listing:
        st.caption(f"扫描完成：{listing['scanned']} 个目录，读取 {len(listing['candidates'])} 个清单，跳过 {listing['skipped']} 个异常项。只读 1MB 以内的清单，不加载 pickle。")
        st.caption("清单可读不代表因子已通过来源或训练资格验证，需选择后检查。")
        if listing.get("limited"):
            st.warning("本次已达到最多 30 个清单的扫描上限；列表不是所有历史候选的完整清单。")


def _render_factor_source():
    candidates = st.session_state.get("hybrid_factor_candidates", {})
    choices = [""] + list(candidates)
    _restore_preferences("hybrid_factor_source")
    if st.session_state.get("hybrid_factor_source", "") not in choices:
        st.session_state["hybrid_factor_source"] = ""
    selected = st.selectbox("研究因子来源", choices,
        format_func=lambda value: "生产因子 · outputs/df_factors.pkl" if not value else f"候选因子 · {candidates[value].get('candidate_id', Path(value).parent.name)}",
        key="hybrid_factor_source")
    _remember_preferences(hybrid_factor_source=selected)
    if selected:
        st.caption("候选因子只作为本次研究输入，不替换生产因子；文件未自动读取，日期范围与训练资格需在下方检查。顶部仅为生产输入预览。")
    return selected or None


@contextmanager
def _preparation_frame(*, expanded, framed):
    """Wrap the steps in an expander only when embedded in another page."""
    if not framed:
        yield
        return
    with st.expander("数据与模型准备", expanded=expanded):
        st.caption("就绪检查只读。构建仅使用已有本地行情，不联网、不补新行情；训练只写入独立候选目录，不发布、不替换生产模型或原始因子。")
        yield


def _render_preparation(base_dir, *, api_key=None, expanded=False, framed=True):
    """Separate explicit preparation actions from research execution.

    ``framed=False`` drops the outer expander and the policy caption for the
    dedicated page, which already carries both in its own header.
    """
    max_age = st.session_state.get("hybrid_max_age",
        st.session_state.get("hybrid_preferences", {}).get("hybrid_max_age", 24))
    with _preparation_frame(expanded=expanded, framed=framed):
        st.markdown("**1 · 行情接入与覆盖**")
        from minute_dashboard import render_minute_preparation
        minute_snapshot = render_minute_preparation(base_dir)
        st.markdown("**2 · 因子候选与研究来源**")
        _render_candidate_scan(base_dir, api_key=api_key)
        if st.button("构建本地因果因子候选", key="hybrid_build_candidate"):
            st.session_state.pop("hybrid_build_result", None)
            try:
                with st.spinner("从本地已有数据构建独立因子候选…"):
                    build_options = {"minute_snapshot_id": minute_snapshot} if minute_snapshot else {}
                    result = public_report(build_local_factor_candidate(base_dir, **build_options), secrets=(api_key,))
                if not isinstance(result, dict) or result.get("status") not in {"built", "failed"}:
                    raise ValueError("不支持的因子候选构建结果")
                if result["status"] == "built":
                    if not result.get("factor_path"):
                        raise ValueError("构建结果缺少候选因子路径")
                    path = Path(result["factor_path"])
                    if not path.is_absolute():
                        path = Path(base_dir) / path
                    path = path.resolve()
                    if not path.is_relative_to(Path(base_dir).resolve()):
                        raise ValueError("候选因子路径不在当前项目内")
                    result = {**result, "factor_path": str(path)}
                    candidates = dict(st.session_state.get("hybrid_factor_candidates", {}))
                    candidates[str(path)] = result
                    st.session_state["hybrid_factor_candidates"] = candidates
                st.session_state["hybrid_build_result"] = result
            except Exception as exc:
                st.error("候选构建失败：" + _text(exc, (api_key,)))
        built = st.session_state.get("hybrid_build_result")
        if built:
            if built.get("status") == "built":
                st.success("因子候选已构建，请在下方选择该快照。")
                st.text(str(built.get("factor_path", "")))
            else:
                st.error("候选构建未完成，不会自动回退成一次成功构建。")
            for warning in built.get("warnings") or []:
                st.warning(str(warning))
            with st.expander("因子候选覆盖范围与构建记录"):
                st.json(built)

        factor_path = _render_factor_source()
        selected = factor_path or ""
        st.markdown("**3 · 就绪检查与独立模型候选**")
        if st.button("检查数据与模型就绪状态（只读）", key="hybrid_check_readiness"):
            st.session_state.pop("hybrid_readiness", None)
            st.session_state.pop("hybrid_readiness_source", None)
            try:
                with st.spinner("检查标签可用性、数据时点与模型状态…"):
                    value = _readiness_result(inspect_research_readiness(base_dir, factor_path=factor_path,
                        max_age_hours=max_age), api_key)
                st.session_state["hybrid_readiness"] = value
                st.session_state["hybrid_readiness_source"] = selected
            except Exception as exc:
                st.error("就绪检查失败：" + _text(exc, (api_key,)))

        ready = st.session_state.get("hybrid_readiness")
        same_source = st.session_state.get("hybrid_readiness_source") == selected
        if ready and same_source:
            _render_readiness(ready)
        elif ready:
            st.warning("因子来源已切换；之前的就绪结论不适用于当前快照，请重新检查。")
        consent = st.checkbox("我确认启动本地训练（占用 CPU 与内存，不发布模型）", key="hybrid_local_training_consent")
        if st.button("训练独立模型候选（不发布）", key="hybrid_train_candidate"):
            st.session_state.pop("hybrid_train_result", None)
            st.session_state.pop("hybrid_train_source", None)
            if not consent:
                st.error("请先确认本地计算范围，再启动候选训练。")
            elif not ready or not same_source:
                st.error("请先对当前因子来源重新检查就绪状态，再训练候选。")
            elif (ready.get("training") or {}).get("eligible") is not True:
                st.error("当前输入未通过训练资格检查；请先处理上面的阻断项，不会启动训练。")
            else:
                try:
                    refreshed = _readiness_result(inspect_research_readiness(base_dir, factor_path=factor_path,
                        max_age_hours=max_age), api_key)
                    st.session_state["hybrid_readiness"] = refreshed
                    if (refreshed.get("training") or {}).get("eligible") is not True:
                        raise ValueError("训练前复核未通过；请重新检查数据与阻断项")
                    old_sha = (ready.get("factors") or {}).get("sha256")
                    new_sha = (refreshed.get("factors") or {}).get("sha256")
                    if not old_sha or not new_sha or old_sha != new_sha:
                        raise ValueError("因子内容指纹缺失或已发生变化，请重新检查并确认当前快照")
                    with st.spinner("本地候选训练与验证中；不会发布生产模型…"):
                        result = public_report(train_candidate(base_dir, factor_path=factor_path), secrets=(api_key,))
                    if not isinstance(result, dict) or result.get("published") is not False or result.get("status") not in {"candidate_ready", "candidate_rejected", "failed"}:
                        raise ValueError("训练返回了不符合未发布候选契约的结果")
                    st.session_state["hybrid_train_result"] = result
                    st.session_state["hybrid_train_source"] = factor_path or "生产因子 · outputs/df_factors.pkl"
                except Exception as exc:
                    st.error("候选训练未完成：" + _text(exc, (api_key,)))
        trained = st.session_state.get("hybrid_train_result")
        if trained:
            if trained.get("status") == "candidate_ready":
                st.success("模型候选已生成且未发布；生产模型保持不变，需要独立发布审核。")
            elif trained.get("status") == "candidate_rejected":
                st.warning("模型候选未通过验收，未发布，不会替换生产模型。")
            else:
                st.error("本次候选训练失败，未发布。")
            st.text("训练使用的因子来源：" + str(st.session_state.get("hybrid_train_source", "未记录")))
            st.text("候选训练编号：" + str(trained.get("run_id", "未记录")))
            for error in trained.get("errors") or []:
                st.error(str(error))
            with st.expander("候选模型验收与分区报告"):
                st.json(trained)
        return factor_path


def render_preparation_workbench(ctx, *, api_key=None):
    """Standalone maintenance view; opening it never runs research or training."""
    base_dir = _bind_workspace(ctx)
    st.caption("行情 → 因子快照 → 就绪检查 → 独立模型候选。构建只用已有本地行情，不联网；训练写入独立候选目录，不替换生产模型。")
    return _render_preparation(base_dir, api_key=api_key, expanded=True, framed=False)


def _render_production_preview(ctx, *, api_key=None):
    """Legacy embedded workbench preview; split views do not load this input."""
    try:
        st.caption("生产输入预览 · 下方若选择候选快照，其数据范围需单独检查，不代表顶部预览已切换。")
        frame = ctx.df_factors
        if frame is None or frame.empty:
            st.warning("当前没有因子数据。仍可运行一次诊断，查看缺输入或未发布模型的具体阻断原因。")
        elif "date" not in frame:
            st.warning("因子数据缺少 date 列，无法确认输入时点。运行诊断将返回具体阻断信息。")
        else:
            dates = pd.to_datetime(frame["date"], errors="coerce")
            if dates.notna().any():
                st.caption(f"当前本地因子（生产）：{len(frame):,} 行 ｜ {dates.min()} ～ {dates.max()}。打开页面不刷新行情、不调用模型。")
                latest = pd.Timestamp(dates.max())
                latest = latest.tz_localize("Asia/Shanghai") if latest.tzinfo is None else latest
                age_hours = (pd.Timestamp.now(tz="UTC") - latest.tz_convert("UTC")).total_seconds() / 3600
                age_limit = st.session_state.get("hybrid_max_age", 24)
                if age_hours < 0:
                    st.warning("当前因子包含未来时间；运行时会按研究截至时间过滤，不把未来信息当作当前证据。")
                elif age_hours > age_limit:
                    st.warning(f"当前数据是历史快照：距最新数据约 {age_hours:,.1f} 小时，超过 {age_limit} 小时时效阈值。可做研究诊断，不能冒充实时交易依据。")
                else:
                    st.caption(f"数据时效：距最新数据约 {age_hours:.1f} 小时（尚未验证模型、完整性或交易资格）。")
                if dates.isna().any():
                    st.warning(f"另有 {int(dates.isna().sum())} 行无效时间戳，运行时会进行输入校验。")
            else:
                st.warning("因子数据无有效时间戳，无法确认数据日期；请运行诊断查看阻断详情。")
    except Exception as exc:
        frame = None
        st.warning("因子数据读取失败：" + _text(exc, (api_key,)))


def render_hybrid_workbench(ctx, *, api_key=None, include_preparation=True):
    """Render explicit controls and one immutable selected run, never auto-run.

    ``include_preparation`` retains the legacy embedded view. The routed
    research view uses False to leave ingestion and training on their own page.
    """
    base_dir = _bind_workspace(ctx)
    _restore_preferences("hybrid_question", "hybrid_network", "hybrid_max_calls",
                         "hybrid_max_age", "hybrid_max_evidence", "hybrid_as_of")
    st.subheader("混合决策工作台")
    st.caption("TL 国债期货 · 量化基线 × Jev 语义概率 × LLM 证据审议 · 只研究，不下单")
    if include_preparation:
        _render_production_preview(ctx, api_key=api_key)
        factor_path = _render_preparation(base_dir, api_key=api_key)
    else:
        with st.expander("研究输入 · 因子快照"):
            _render_candidate_scan(base_dir, api_key=api_key)
            factor_path = _render_factor_source()
            st.caption("行情接入、覆盖审计、因子构建和候选训练已移至「数据与模型」。打开本页不读取因子文件、不刷新行情。")
        st.caption("本轮输入：" + ("生产因子 · outputs/df_factors.pkl" if not factor_path
                                  else "候选因子 · " + Path(factor_path).parent.name))

    question = st.text_area("研究问题", value=DEFAULT_QUESTION, key="hybrid_question", height=80)
    first, second = st.columns([2, 1])
    with first:
        mode = st.selectbox("数据与模型访问模式", list(MODES), format_func=MODES.get, key="hybrid_network")
    with second:
        max_calls = st.number_input("单次模型调用上限", min_value=0, max_value=4, value=4, key="hybrid_max_calls")
    consent = False
    if mode == "online":
        consent = st.checkbox("我确认：将有限研究快照发送到已配置的模型服务，可能产生费用；密钥不写入研究快照，不会下单。", key="hybrid_online_consent")
    elif mode == "cache":
        st.info("缓存模式不发起新请求。Jev 仅可复用符合本次输入的缓存；LLM 未实现可复用缓存时明确显示不可用，不伪造判断。")
    else:
        st.caption("默认离线：只读取本地快照与可用证据；未调用的远端模型显示跳过或不可用。")
    with st.expander("研究约束"):
        max_age = st.number_input("数据时效阈值（小时）", min_value=1, max_value=720, value=24, key="hybrid_max_age")
        max_evidence = st.number_input("证据数量上限", min_value=1, max_value=20, value=20, key="hybrid_max_evidence")
        as_of = st.text_input("研究截至时间（可选，留空使用运行时点）", value="", placeholder="2026-10-03T15:15:00+08:00", key="hybrid_as_of")
    _remember_preferences(hybrid_question=question, hybrid_network=mode,
        hybrid_max_calls=max_calls, hybrid_max_age=max_age,
        hybrid_max_evidence=max_evidence, hybrid_as_of=as_of)
    request = {"question": question.strip(), "network_mode": mode, "as_of": as_of.strip(),
               "max_age_hours": max_age, "max_model_calls": max_calls, "max_evidence": max_evidence,
               "factor_path": factor_path}
    if st.button("运行混合研究 / 诊断", type="primary", width="stretch", key="hybrid_run"):
        _clear_selected()
        if mode == "online" and not consent:
            st.error("请先确认在线数据发送范围，再启动本轮研究。")
        elif not question.strip():
            st.error("请填写研究问题。")
        else:
            with st.spinner("按本次快照运行研究与安全检查…"):
                try:
                    if as_of.strip():
                        parsed = pd.Timestamp(as_of.strip())
                        if pd.isna(parsed):
                            raise ValueError("研究截至时间无效")
                    config = _make_config(network_mode=mode, asset="TL", max_age_hours=max_age,
                                          max_model_calls=max_calls, max_evidence=max_evidence)
                    # Preview data is never injected: runtime must read and
                    # verify the selected on-disk snapshot's provenance.
                    result = run_hybrid_research(base_dir, factors=None, factor_path=factor_path, question=question.strip(),
                        as_of=as_of.strip() or None, config=config, api_key=api_key, persist=True)
                    if not isinstance(result, dict) or result.get("schema_version") != "f_agent.hybrid.v1":
                        raise ValueError("研究运行返回了不支持的报告格式")
                    st.session_state["hybrid_report"] = public_report(result, secrets=(api_key,))
                    st.session_state["hybrid_report_origin"] = "run"
                    st.session_state["hybrid_run_request"] = request
                except Exception as exc:
                    st.error("本次运行失败：" + _text(exc, (api_key,)))

    with st.expander("历史运行 · 只读追溯"):
        if st.button("读取历史运行列表", key="hybrid_scan_history"):
            st.session_state.pop("hybrid_history_index", None)
            try:
                st.session_state["hybrid_history_index"] = list_hybrid_runs(base_dir, limit=30)
            except Exception as exc:
                st.warning("历史索引不可用：" + _text(exc, (api_key,)))
        try:
            history = st.session_state.get("hybrid_history_index", [])
            by_id = {item["run_id"]: item for item in history if isinstance(item, dict) and item.get("run_id")}
            choices = [""] + list(by_id)
            if st.session_state.get("hybrid_history") not in choices:
                st.session_state["hybrid_history"] = ""
            def history_label(run_id):
                item = by_id.get(run_id, {})
                return "选择已保存运行" if not run_id else f"{run_id} · {item.get('generated_at', '时间未记录')} · {item.get('status', '状态未记录')}"
            selected = st.selectbox("历史运行", choices, format_func=history_label, key="hybrid_history")
            if st.button("读取历史快照", key="hybrid_load_history", disabled=not selected):
                _clear_selected()
                try:
                    result = load_hybrid_run(base_dir, selected)
                    if not isinstance(result, dict) or result.get("schema_version") != "f_agent.hybrid.v1":
                        raise ValueError("不支持的历史报告格式")
                    st.session_state["hybrid_report"] = public_report(result, secrets=(api_key,))
                    st.session_state["hybrid_report_origin"] = "history"
                except Exception as exc:
                    st.error("历史读取失败：" + _text(exc, (api_key,)))
        except Exception as exc:
            st.warning("历史索引不可用：" + _text(exc, (api_key,)))

    selected_report = st.session_state.get("hybrid_report")
    if selected_report:
        if st.session_state.get("hybrid_report_origin") == "history":
            st.info("正在查看历史快照；使用该次保存的模型、参数、证据和数据时点，与当前控件或当前因子数据无关。未重新运行模型。")
        elif st.session_state.get("hybrid_run_request") != request:
            st.warning("上方参数变更尚未应用。下方仍是原运行快照，需要重新点击运行才会生成新结论。")
        st.caption("以下所有卡片来自同一运行编号。当前输入数据的变化不会回写这份报告；要研究新数据，请显式运行。")
        _render_report(selected_report)
    else:
        st.info("未运行：准备好研究问题后启动一次研究。没有结果时不会生成演示判断或假成功状态。")
