"""Navigation and read-only overview; dispatch exactly one active tool.

This module owns presentation only. It never collects data, loads models,
creates reports or changes the research/trading policies.
"""
from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from intelligence_events import redact_sensitive_text


SECTIONS = {
    "overview": {"label": "研究总览", "hint": "先确认数据，再进入研究。", "views": ()},
    "monitor": {"label": "市场与舆情", "hint": "行情、宏观与事件证据，按需查看。", "views": (
        ("market", "行情监控"), ("sentiment", "舆情情报"), ("macro", "宏观环境"), ("signal", "策略信号"))},
    "research": {"label": "混合研究", "hint": "LightGBM × Jev × LLM · 同一快照，可追溯审议。", "views": (("hybrid", "决策工作台"),)},
    "factors": {"label": "因子实验室", "hint": "从因子观察到验证与挖掘。", "views": (
        ("factor_analysis", "因子分析"), ("factor_eval", "因子评估"), ("genetic", "遗传挖掘"))},
    "data": {"label": "数据与模型", "hint": "独立数据批次、候选训练与模型检查。", "views": (
        ("preparation", "数据准备"), ("model_eval", "模型评估"), ("evolution", "迭代诊断"))},
    "review": {"label": "回测与复盘", "hint": "区分历史表现、交易记忆与实时能力。", "views": (
        ("backtest", "回测表现"), ("memory", "交易记忆"))},
    "settings": {"label": "系统设置", "hint": "连接配置、缓存维护与任务调度。", "views": (
        ("api", "连接与缓存"), ("scheduler", "定时任务"))},
}


STYLE = """
<style>
:root { --fa-ink:#173044; --fa-muted:#526578; --fa-line:#dce4e9;
        --fa-accent:#087e83; --fa-soft:#edf5f5; --fa-paper:#f5f7f9; }
.stApp { background:var(--fa-paper); color:var(--fa-ink); }
[data-testid="stMainBlockContainer"] { max-width:1440px; padding-top:4.5rem; padding-bottom:3rem; }
[data-testid="stSidebar"] { background:#edf1f4; border-right:1px solid var(--fa-line); }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding:8px 2px; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background:#dcebed; border-radius:6px; color:#075d61; font-weight:650; }
h1 { font-size:1.85rem !important; font-weight:650 !important; letter-spacing:-.035em; }
h2 { font-size:1.3rem !important; } h3 { font-size:1.12rem !important; }
[data-testid="stMetric"] { background:white; border:1px solid var(--fa-line); border-radius:8px; padding:16px 18px; }
[data-testid="stMetricValue"] { font-size:1.65rem; font-variant-numeric:tabular-nums; }
[data-testid="stCaptionContainer"] { color:var(--fa-muted); }
[data-testid="stExpander"] { background:rgba(255,255,255,.6); border-radius:8px; }
button[kind="primary"] { background:#087e83; border-color:#087e83; }
button:focus-visible, input:focus-visible, [role="radio"]:focus-visible { outline:3px solid #087e83 !important; outline-offset:3px; }
.fa-brand { display:flex; align-items:center; gap:12px; margin:0 0 28px; }
.fa-monogram { display:grid; place-items:center; width:38px; height:38px; border-radius:8px; background:#173044; color:white; font-size:22px; font-weight:700; }
.fa-brand strong { letter-spacing:.1em; font-size:17px; } .fa-brand small { display:block; color:#526578; font-size:11px; margin-top:3px; }
.fa-eyebrow { color:#526578; font-size:11px; font-weight:650; letter-spacing:.13em; margin-bottom:10px; }
.fa-status { display:inline-block; color:#075d61; background:#deeeed; border:1px solid #c2dedd; border-radius:4px; font-size:12px; padding:5px 9px; }
.fa-task { border-top:2px solid #087e83; padding:15px 0 4px; margin-top:8px; }
.fa-task h3 { margin:0 0 7px; padding:0; } .fa-task p { color:#526578; font-size:14px; line-height:1.65; min-height:46px; }
.fa-flow { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:18px; border-top:1px solid #dce4e9; padding-top:18px; }
.fa-step { border-left:2px solid #ccd9df; padding-left:13px; }
.fa-step span { display:block; color:#087e83; font-size:11px; font-weight:700; letter-spacing:.1em; }
.fa-step strong { display:block; margin:6px 0; font-size:15px; }
.fa-step small { color:#526578; line-height:1.65; font-size:13px; }
@media(max-width:768px) {
 [data-testid="stMainBlockContainer"] { padding:4.5rem 1rem 2rem; }
 .fa-flow { grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px 10px; }
 h1 { font-size:1.6rem !important; }
 [data-testid="stMetricValue"] { font-size:1.4rem; }
 [data-testid="stRadio"] [role="radiogroup"] { flex-wrap:wrap; }
 .fa-task p { min-height:0; }
}
@media(prefers-reduced-motion:reduce) { * { scroll-behavior:auto !important; } }
</style>
"""


def summarize_factors(frame, *, now=None):
    result = {"rows": 0, "latest": None, "status": "missing", "invalid_rows": 0}
    if frame is None or frame.empty:
        return result
    result["rows"] = len(frame)
    if "date" not in frame:
        return {**result, "status": "invalid"}
    try:
        dates = pd.to_datetime(frame["date"], errors="coerce", format="mixed")
        result["invalid_rows"] = int(dates.isna().sum())
        latest = pd.Timestamp(dates.max())
        if pd.isna(latest):
            return {**result, "status": "invalid"}
        latest = latest.tz_localize("Asia/Shanghai") if latest.tzinfo is None else latest.tz_convert("Asia/Shanghai")
        current = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="Asia/Shanghai")
        current = current.tz_localize("Asia/Shanghai") if current.tzinfo is None else current
        hours = (current - latest).total_seconds() / 3600
        return {**result, "latest": latest.strftime("%Y-%m-%d %H:%M"), "age_hours": hours,
                "status": "future" if hours < 0 else "stale" if hours > 24 else "recent"}
    except (ValueError, TypeError, OverflowError):
        return {**result, "status": "invalid"}


def _navigate(section):
    st.session_state["workspace_section"] = section


def _remember_view(section, key):
    saved = dict(st.session_state.get("workspace_views", {}))
    saved[section] = st.session_state[key]
    st.session_state["workspace_views"] = saved


def render_overview(ctx):
    st.title("研究总览")
    st.caption("TL · 30 年期国债期货 / 本地研究工作区")
    try:
        summary = summarize_factors(ctx.df_factors)
    except Exception as exc:
        summary = summarize_factors(None)
        st.error("本地因子读取失败：" + redact_sensitive_text(str(exc))[:300])
    report = st.session_state.get("hybrid_report") or {}
    columns = st.columns(4)
    columns[0].metric("本地因子记录", f"{summary['rows']:,}" if summary["rows"] else "暂无")
    columns[1].metric("因子数据截至", (summary["latest"] or "未确认").split(" ")[0])
    columns[2].metric("本会话研究", "已载入快照" if report else "尚未运行")
    columns[3].metric("执行边界", "仅研究")
    status = summary["status"]
    if status == "stale":
        st.warning(f"数据待更新 · 因子快照截至 {summary['latest']}（上海时间）。当前只能做历史研究，不能视为实时交易依据。")
    elif status in {"missing", "invalid", "future"}:
        messages = {"missing": "尚无本地因子快照", "invalid": "因子时间字段无效", "future": "因子包含未来时间"}
        st.warning(messages[status] + " · 请先到「数据与模型」检查输入，再运行研究。")
    else:
        st.info(f"因子时点 {summary['latest']} 在 24 小时内；完整性、来源和模型资格仍需单独检查，不代表可交易。")
    if summary["invalid_rows"]:
        st.caption(f"另有 {summary['invalid_rows']:,} 行时间戳无效，不能据此确认完整覆盖。")

    st.subheader("工作入口")
    entries = [
        ("01", "观察市场", "查看行情、宏观变化和债券事件，区分数据时间与页面时间。", "进入市场与舆情", "monitor", "workspace_open_monitor"),
        ("02", "形成研究", "让量化基线与模型证据对照，逐阶段检查结论和分歧。", "开始混合研究", "research", "workspace_start_research"),
        ("03", "准备与验证", "审计分钟覆盖、构建因子候选，再检查模型训练资格。", "检查数据与模型", "data", "workspace_open_data"),
    ]
    for column, (number, title, description, label, section, key) in zip(st.columns(3), entries):
        with column:
            st.markdown(f'<div class="fa-task"><div class="fa-eyebrow">WORKFLOW / {number}</div><h3>{title}</h3><p>{description}</p></div>', unsafe_allow_html=True)
            st.button(label, key=key, on_click=_navigate, args=(section,), type="primary" if section == "research" else "secondary", width="stretch")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.subheader("研究链路")
    st.markdown('''<div class="fa-flow" aria-label="研究处理顺序">
      <div class="fa-step"><span>01 / INPUT</span><strong>数据与事件</strong><small>分钟行情 · 宏观 · 舆情<br>先核对来源与时点</small></div>
      <div class="fa-step"><span>02 / BASELINE</span><strong>量化基线</strong><small>因子 → LightGBM<br>版本化策略提供基础信号</small></div>
      <div class="fa-step"><span>03 / EVIDENCE</span><strong>混合审议</strong><small>Jev + LLM<br>证据引用、分歧与约束</small></div>
      <div class="fa-step"><span>04 / REVIEW</span><strong>人工复核</strong><small>可追溯研究快照<br>不自动改仓或下单</small></div>
    </div>''', unsafe_allow_html=True)
    st.caption("流程说明，不代表本轮已经执行。模型可用性和各阶段状态以实际运行报告为准。")
    if report:
        with st.expander("本会话已载入的研究记录"):
            st.text("运行编号：" + redact_sensitive_text(str(report.get("run_id", "未记录"))))
            st.text("生成时间：" + redact_sensitive_text(str(report.get("generated_at", "未记录"))))
            st.caption("这是已有快照，不随页面切换重新生成。完整证据与审计请进入混合研究。")


def render_workspace(ctx, renderers):
    """Render a single destination. Callbacks must be lazy, zero-argument functions."""
    st.markdown(STYLE, unsafe_allow_html=True)
    if st.session_state.get("workspace_section") not in SECTIONS:
        st.session_state["workspace_section"] = "overview"
    with st.sidebar:
        st.markdown('<div class="fa-brand"><div class="fa-monogram">F</div><div><strong>F_AGENT</strong><small>FINANCIAL RESEARCH WORKSPACE</small></div></div>', unsafe_allow_html=True)
        section = st.radio("工作区导航", list(SECTIONS), format_func=lambda key: SECTIONS[key]["label"], key="workspace_section", label_visibility="collapsed")
        st.divider()
        st.markdown('<span class="fa-status">TL · 研究模式</span>', unsafe_allow_html=True)
        st.caption("页面切换不运行研究。\n\n模型连接和缓存维护已移至系统设置。")
    config = SECTIONS[section]
    st.markdown(f'<div class="fa-eyebrow">F_AGENT / {escape(config["label"])}</div>', unsafe_allow_html=True)
    if section == "overview":
        render_overview(ctx)
        return
    st.title(config["label"])
    st.caption(config["hint"])
    choices = dict(config["views"])
    if len(choices) > 1:
        key = "workspace_view_" + section
        saved = st.session_state.get("workspace_views", {}).get(section)
        if st.session_state.get(key) not in choices:
            st.session_state[key] = saved if saved in choices else next(iter(choices))
        view = st.radio("当前功能", list(choices), format_func=choices.get, key=key, horizontal=True,
                        label_visibility="collapsed", on_change=_remember_view, args=(section, key))
    else:
        view = next(iter(choices))
    st.divider()
    renderers[view]()
