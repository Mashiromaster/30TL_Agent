# -*- coding: utf-8 -*-
# dashboard_v2.py — 升级版Dashboard
# 信号看板(真实信号) + 交易记忆(完整复盘) + RAG增强 + 自我迭代
# 运行: python -m streamlit run dashboard_v2.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys
import io
import json
import hashlib
import subprocess
from datetime import datetime

# ═══ Windows Streamlit (MSYS/git-bash) 全局修复：stdout/stderr → StringIO ═══
# 必须在所有 import 之前执行。os.write(b'') 在 pipe 上不报错，
# 所以无条件重定向——Streamlit 自身不依赖 stdout 来捕获日志。
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding='utf-8', write_through=True)
    sys.stderr = io.TextIOWrapper(io.BytesIO(), encoding='utf-8', write_through=True)

sys.path.insert(0, os.path.dirname(__file__))
from config import BASE_DIR
import chart_theme
from strategy_agent import StrategyContext
from rag_tool import RAGAnalyzer
from memory import TradingMemory
from signal_dashboard import SignalDashboard
from self_iteration import SelfIterationEngine
from intelligence_events import redact_sensitive_text

st.set_page_config(
    page_title="F_Agent TL策略",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════
# API Key 管理 — 服务器密钥只读；用户密钥仅保存在当前 Streamlit session
# ═══════════════════════════════════════════════════════════════

def _load_server_api_key(base_dir: str) -> str:
    """Read only the server DeepSeek secret without exporting it globally."""
    configured = os.environ.get("DEEPSEEK_API_KEY", "")
    if configured:
        return configured
    env_path = os.path.join(base_dir, ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        key, _, value = line.partition('=')
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key.strip() == "DEEPSEEK_API_KEY" and value:
                            return value
        except Exception:
            pass
    return ""


_SERVER_DEEPSEEK_API_KEY = _load_server_api_key(BASE_DIR)


def _effective_api_key():
    return st.session_state.get("user_deepseek_api_key", "") or (
        _SERVER_DEEPSEEK_API_KEY
    )


@st.cache_data(ttl=300, show_spinner="加载数据中...")
def load_ctx():
    return StrategyContext(BASE_DIR)


@st.cache_resource
def _load_signal_dash_cached(model_signature, strategy_fingerprint):
    return SignalDashboard(BASE_DIR)


def load_signal_dash():
    """Cache the runtime by model bytes and StrategySpec, not process lifetime."""
    from model_release import ReleaseManager
    from strategy_runtime import get_strategy_spec

    manager = ReleaseManager(
        os.path.join(BASE_DIR, "artifacts", "models"),
        legacy_model_path=os.path.join(BASE_DIR, "models", "trained_model.pkl"),
    )
    signature = manager.current_signature()
    model_signature = (
        (
            signature.source,
            signature.release_id,
            signature.manifest_sha256,
            signature.artifact_sha256,
        )
        if signature is not None
        else ("missing", None, None, None)
    )
    strategy_fingerprint = get_strategy_spec().fingerprint()
    return _load_signal_dash_cached(model_signature, strategy_fingerprint)


@st.cache_data(ttl=30, show_spinner=False)
def load_sentiment_snapshot():
    from dataclasses import asdict
    from intelligence_events import IntelligenceEventStore

    db_path = os.path.join(BASE_DIR, "data", "intelligence", "events.sqlite")
    if not os.path.exists(db_path):
        return {"source_health": [], "assessments": [], "alerts": [], "deliveries": []}
    store = IntelligenceEventStore(BASE_DIR, read_only=True)
    return {
        "source_health": [asdict(item) for item in store.list_source_health()],
        "assessments": store.recent_event_records(limit=50),
        "alerts": [asdict(item) for item in store.list_alerts()],
        "deliveries": [{key: value for key, value in item.items() if key not in {"payload", "payload_json"}}
                       for item in store.list_deliveries(limit=30)],
    }


def load_rag():
    key = _effective_api_key()
    key_fingerprint = hashlib.sha256(key.encode("utf-8")).hexdigest() if key else ""
    if st.session_state.get("_rag_key_fingerprint") != key_fingerprint:
        st.session_state["_rag_instance"] = RAGAnalyzer(
            BASE_DIR, api_key=key
        )
        st.session_state["_rag_key_fingerprint"] = key_fingerprint
    if "_rag_instance" not in st.session_state:
        st.session_state["_rag_instance"] = RAGAnalyzer(
            BASE_DIR, api_key=key
        )
    return st.session_state["_rag_instance"]


@st.cache_resource
def load_iterator():
    return SelfIterationEngine(BASE_DIR)


def main():
    from dashboard_workspace import render_workspace
    try:
        ctx = load_ctx()
    except Exception as e:
        st.error(f"数据加载失败: {redact_sensitive_text(e)}")
        return
    render_workspace(ctx, {
        "market": lambda: render_market_from_v1(ctx),
        "signal": lambda: render_signal_v2(ctx),
        "sentiment": render_sentiment_monitor,
        "macro": lambda: render_macro_from_v1(ctx),
        "hybrid": lambda: render_ai_fusion(ctx),
        "factor_analysis": lambda: render_factor_from_v1(ctx),
        "factor_eval": lambda: render_factor_eval(ctx),
        "genetic": lambda: render_genetic_mining(ctx),
        "preparation": lambda: render_data_preparation(ctx),
        "model_eval": lambda: render_eval_tab(ctx),
        "evolution": lambda: render_evolution_combined(ctx),
        "backtest": lambda: render_backtest_from_v1(ctx),
        "memory": lambda: render_memory_v2(ctx),
        "api": render_connection_settings,
        "scheduler": lambda: render_cron_tab(ctx),
    })
    # Legacy view imports set their own title; keep this entrypoint consistent.
    st.set_page_config(page_title="F_Agent 研究工作台", page_icon="📈", layout="wide")


def render_connection_settings():
    """Keep session secrets out of the global navigation and preserve across pages."""
    st.subheader("模型连接")
    def save_session_key():
        st.session_state["user_deepseek_api_key"] = st.session_state.get("sidebar_api_key", "").strip()
    st.text_input("本会话 DeepSeek API Key（可选）", type="password",
        value=st.session_state.get("user_deepseek_api_key", ""), placeholder="sk-...",
        help="仅当前浏览器会话，不写入环境变量或文件；留空使用服务器配置。",
        key="sidebar_api_key", on_change=save_session_key)
    if st.session_state.get("user_deepseek_api_key"):
        st.caption("当前会话密钥已配置；切换页面不会清除。")
    elif _SERVER_DEEPSEEK_API_KEY:
        st.caption("服务器密钥已配置；具体值不会显示。")
    else:
        st.info("尚未配置模型连接。离线研究和数据检查仍可使用。")
    st.caption("配置密钥不会发起模型调用；在线研究仍需在运行页面单独确认。")
    st.divider()
    st.subheader("本地缓存维护")
    st.caption("以下操作只清理进程内缓存，不下载最新行情，不删除原始数据，不启动模型训练。")
    left, right = st.columns(2)
    with left:
        if st.button("重新读取本地数据", key="workspace_reload_local", width="stretch"):
            st.cache_data.clear()
            st.rerun()
    with right:
        if st.button("清除数据与模型内存缓存", key="workspace_clear_memory", width="stretch"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()


def render_data_preparation(ctx):
    from hybrid_dashboard import render_preparation_workbench
    render_preparation_workbench(ctx, api_key=_effective_api_key())


# ================================================================
# Tab 0: 信号看板 V2 — 真实排名信号
# ================================================================
def render_signal_v2(ctx):
    sd = load_signal_dash()

    if not sd.is_ready:
        st.warning("⚠ 模型未找到。请先运行训练: python main.py --mode train")
        return

    if ctx.df_factors is None:
        st.warning("无因子数据")
        return

    signal = sd.generate_live_signal(ctx.df_factors)
    if not signal:
        st.warning("信号生成失败 — 数据不足")
        return
    if not signal.get('decision_ready', True):
        st.warning(
            "策略历史不足，尚不可决策: "
            + signal.get('missing_reason', 'insufficient_history')
        )
        return

    st.subheader("📡 TL 统一策略信号")

    # Top metric cards
    col1, col2, col3, col4, col5 = st.columns(5)
    d = signal['direction']
    d_color = {1: '#00E676', -1: '#FF5252', 0: '#9E9E9E'}[d]

    with col1:
        st.metric("信号方向", signal['direction_name'], delta=None)
    with col2:
        rank = signal.get('pred_rank_pct')
        if rank is None:
            st.metric("策略置信度", f"{signal['confidence']:.1%}")
        else:
            st.metric("预测排名分位", f"{rank:.1%}",
                      delta=f"{rank - 0.5:+.1%}")
    with col3:
        st.metric("置信度", f"{signal['confidence']:.1%}")
    with col4:
        st.metric("市场价格", f"{signal['close']:.3f}")
    with col5:
        st.metric("市场状态", signal['regime_name'])

    # 2x2 layout
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("**信号详情**")
        detail_data = {
            '指标': ['预测收益(原始)', '预测收益(平滑)', '排名分位', '上阈值', '下阈值',
                    '使用模型', '目标仓位', '策略版本'],
            '值': [
                f"{signal['predicted_return']:+.6f}%",
                f"{signal['predicted_return_smooth']:+.6f}%",
                f"{rank:.3f}" if rank is not None else 'N/A',
                f"{signal['upper_threshold']:.6f}" if signal.get('upper_threshold') is not None else 'N/A',
                f"{signal['lower_threshold']:.6f}" if signal.get('lower_threshold') is not None else 'N/A',
                signal['model_used'],
                f"{signal['suggested_weight']:.1%}",
                signal.get('strategy_id', 'N/A'),
            ]
        }
        st.dataframe(pd.DataFrame(detail_data), hide_index=True, width='stretch')

    with col_b:
        st.markdown("**信号强度仪表**")
        rank = signal.get('pred_rank_pct')
        gauge_value = (rank if rank is not None else signal.get('confidence', 0.0)) * 100
        gauge_title = '预测排名分位 (%)' if rank is not None else '策略置信度 (%)'
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=gauge_value,
            number={'suffix': '%', 'font': {'size': 40}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': d_color},
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(255,82,82,0.3)'},
                    {'range': [30, 50], 'color': 'rgba(255,183,77,0.15)'},
                    {'range': [50, 70], 'color': 'rgba(255,183,77,0.15)'},
                    {'range': [70, 100], 'color': 'rgba(0,230,118,0.3)'},
                ],
                'threshold': {
                    'line': {'color': d_color, 'width': 4},
                    'thickness': 0.8, 'value': gauge_value,
                }
            },
            title={'text': gauge_title},
        ))
        fig.update_layout(height=250, margin=dict(l=30, r=30, t=40, b=20))
        st.plotly_chart(fig, width='stretch')

    # Performance snapshot
    df_pred = signal.get('df_full')
    if df_pred is not None:
        perf = sd.compute_performance_snapshot(df_pred)
        if perf:
            st.divider()
            st.markdown("**近期预测表现 (近5000根bar)**")
            c1, c2, c3 = st.columns(3)
            c1.metric("滚动IC", f"{perf['ic']:.4f}")
            c2.metric("信号命中率", f"{perf['hit_rate']:.1%}")
            c3.metric("预测波动率(年化)", f"{perf['pred_vol']:.2%}")

    # Prediction chart
    st.divider()
    st.markdown("**预测走势与信号分布**")
    with st.spinner("渲染图表..."):
        fig = sd.build_prediction_chart(df_pred, days=7)
        if fig:
            st.plotly_chart(fig, width='stretch')


# ================================================================
# Tab 6: 研究RAG V2 — 增强版
# ================================================================
def render_rag_v2(ctx):
    st.subheader("📚 研究知识库检索 (RAG增强)")

    rag = load_rag()

    col1, col2 = st.columns([3, 1])

    with col1:
        query = st.text_input("🔍 输入研究问题",
                              placeholder="例: 当前央行货币政策对30年期国债的影响是什么?",
                              key="rag_query_input")

    with col2:
        doc_filter = st.selectbox("文档类型过滤", [
            "全部", "货币政策报告", "中金所月报", "研报", "新闻", "宏观快照", "本地资料",
        ], key="rag_filter")

    filter_map = {
        "货币政策报告": {'doc_type': 'monetary_policy_report'},
        "中金所月报": {'doc_type': 'cffex_monthly'},
        "研报": {'doc_type': 'research_report'},
        "新闻": {'doc_type': 'news'},
        "宏观快照": {'doc_type': 'macro_snapshot'},
        "本地资料": {'doc_type': 'agentic_rag_local'},
    }

    if st.button("🔎 检索", width='stretch', key="rag_search_btn") and query:
        with st.spinner("正在检索研究知识库..."):
            fd = filter_map.get(doc_filter) if doc_filter != "全部" else None
            result = rag.query(query, top_k=8, filter_dict=fd)

        st.divider()
        st.markdown("### 📝 AI回答")
        st.markdown(result.get('answer', '无法生成回答'))

        st.divider()
        st.markdown("### 📖 引用来源")
        sources = result.get('sources', [])
        if sources:
            src_df = pd.DataFrame(sources)
            st.dataframe(src_df, width='stretch', hide_index=True)

        stats = result.get('stats', {})
        st.caption(f"知识库: {stats.get('total_chunks', 0)} 个文本块")

    # Index management
    st.divider()
    st.markdown("**知识库管理**")

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        if st.button("📥 更新索引 (增量)", width='stretch', key="rag_update"):
            with st.spinner("更新索引中..."):
                stats = rag.build_index(force_refresh=False)
            st.success(f"索引已更新: {stats.get('total_chunks', 0)} 个文本块")

    with col_m2:
        if st.button("🔄 重建索引 (全量)", width='stretch', key="rag_rebuild"):
            with st.spinner("重建索引中..."):
                stats = rag.build_index(force_refresh=True)
            st.success(f"索引已重建: {stats.get('total_chunks', 0)} 个文本块")

    with col_m3:
        stats = rag.vector_store.get_stats()
        st.metric("当前索引", f"{stats.get('total_chunks', 0)} chunks",
                  delta=f"doc:{stats.get('document_chunks', 0)} passage:{stats.get('passage_chunks', 0)}")

    # Research briefing
    st.divider()
    st.markdown("**研究简报生成**")
    briefing_topic = st.selectbox("简报主题", [
        "当前市场宏观环境分析",
        "央行货币政策走向",
        "30年期国债期货市场结构",
        "中美利差与跨境资本流动",
        "国债期货机构持仓分析",
    ], key="briefing_topic")

    if st.button("📋 生成研究简报", key="gen_briefing"):
        with st.spinner("正在生成研究简报..."):
            result = rag.query(briefing_topic, top_k=10)
        st.markdown("---")
        st.markdown(result.get('answer', '生成失败'))
        st.caption(f"基于 {len(result.get('sources', []))} 个来源")



# ================================================================
# Tab: AI 融合决策 (合并 AI情报 + RAG — signal_fusion 交叉决策)
# ================================================================
def render_sentiment_monitor():
    st.subheader("🧠 Research Agent / 舆情监控")

    st.markdown("""
    **权限边界**: 模型与版本化策略产生唯一基础信号；舆情、RAG、Jev 与 LLM
    只产生可追溯的研究证据和影子建议，不修改方向、仓位，也不下单。
    """)

    with st.expander("📡 舆情监控 Agent", expanded=True):
        from bond_monitor_config import MonitorConfig
        from dataclasses import asdict
        try:
            monitor_config = MonitorConfig.load(BASE_DIR)
        except Exception as exc:
            monitor_config = None
            st.error(f"监控配置无效：{redact_sensitive_text(exc)}")
        with st.expander("配置订阅主题 / 发行人 / 数据源"):
            st.caption("entities 为发行人名称或债券代码；keywords 与 entities 分别匹配任一词，组间为 AND。飞书密钥只从服务器环境变量读取。")
            config_text = st.text_area("监控配置 JSON", value=json.dumps(
                asdict(monitor_config or MonitorConfig()), ensure_ascii=False, indent=2), height=300,
                key="bond_monitor_config_editor")
            if st.button("校验并保存监控配置", key="save_bond_monitor_config"):
                try:
                    monitor_config = MonitorConfig.from_dict(json.loads(config_text))
                    monitor_config.save(BASE_DIR)
                    st.success("已保存，下次抓取生效；已运行的命令行监控需重启。")
                except Exception as exc:
                    st.error(f"未保存：{redact_sensitive_text(exc)}")
        st.caption("仅覆盖已配置公开数据源，并非全网/社交平台全量监控；页面打开不会自动启动轮询。")
        allow_notify = st.checkbox("本次允许发送已配置的飞书通知", value=False, key="bond_allow_notifications")
        mc1, mc2 = st.columns([1, 3])
        with mc1:
            if st.button("立即抓取一次", width='stretch', key="sentiment_run", disabled=monitor_config is None):
                with st.spinner("抓取、去重和事件分析中..."):
                    try:
                        from jev_client import JevClient
                        from sentiment_monitor import SentimentMonitor
                        run = SentimentMonitor(
                            BASE_DIR, jev_adapter=JevClient(BASE_DIR), config=monitor_config,
                            allow_notifications=allow_notify,
                        ).run_once()
                        load_sentiment_snapshot.clear()
                        message = (
                            f"监控完成: {run.status}; 新文档 {run.documents_inserted}; "
                            f"新评估 {run.assessments_created}; 预警 {run.alerts_opened}"
                        )
                        if run.status == 'failed':
                            st.error(message)
                        elif run.status == 'partial_failure':
                            st.warning(message)
                        else:
                            st.success(message)
                        from pathlib import Path
                        digest_path = Path(run.report_path).with_suffix(".md")
                        if digest_path.exists():
                            st.session_state["bond_monitor_digest"] = digest_path.read_text(encoding="utf-8")
                    except Exception as e:
                        st.error(
                            f"舆情监控失败: {redact_sensitive_text(e)}"
                        )
        snapshot = load_sentiment_snapshot()
        health = snapshot.get('source_health', [])
        with mc2:
            if not health:
                st.info("尚无舆情数据；可手动运行，或启用 10 分钟调度任务。")
            else:
                stale = [item for item in health if item.get('source_stale')]
                if stale:
                    st.warning("存在失效数据源：" + "、".join(
                        item.get('source_name', '?') for item in stale
                    ))
                st.dataframe(pd.DataFrame(health), hide_index=True, width='stretch')

        alerts = [item for item in snapshot.get('alerts', []) if item.get('status') == 'open']
        assessments = snapshot.get('assessments', [])
        a1, a2 = st.columns(2)
        with a1:
            st.markdown("**未处理预警**")
            if alerts:
                st.dataframe(pd.DataFrame(alerts), hide_index=True, width='stretch')
            else:
                st.caption("无未处理预警")
        with a2:
            st.markdown("**最近事件评估**")
            if assessments:
                view_cols = [
                    name for name in (
                        'assessed_at', 'event_type', 'severity', 'policy_stance',
                        'bond_impact', 'confidence', 'prompt_injection_flag'
                    ) if name in assessments[0]
                ]
                st.dataframe(
                    pd.DataFrame(assessments)[view_cols], hide_index=True,
                    width='stretch'
                )
            else:
                st.caption("暂无事件评估")

        task_choices = [task.id for task in (monitor_config.tasks if monitor_config else [])]
        selected_task = st.selectbox("按订阅筛选证据", ["全部"] + task_choices, key="bond_evidence_task")
        st.caption("confidence 是规则评分，不是预测正确率。官方原文只说明出处，不代表市场影响已经验证。")
        for item in assessments:
            raw = item.get("raw_response", {})
            if selected_task != "全部" and not next(task for task in monitor_config.tasks
                if task.id == selected_task).matches(item['title'] + "\n" + item['content']):
                continue
            with st.expander(f"{item.get('title', '未命名')} · {raw.get('fact_status', 'unknown')} · {item['severity']}"):
                st.write({"来源": item['source_name'], "出处核验": raw.get('verification', 'unknown'),
                          "发布时间": item['published_at'], "首次可用时间": item['available_at'],
                          "事件编号": item['event_cluster_id'], "期限": item['tenors'],
                          "实体": raw.get('entities', []), "修订自": item['document_metadata'].get('revision_of')})
                from urllib.parse import urlsplit
                if urlsplit(item['url']).scheme in {'http', 'https'}:
                    st.link_button("查看原始出处", item['url'])
                if item['prompt_injection_flag']:
                    st.warning("内容已隔离，不发送外部通知，不作为模型输入。")
                else:
                    for quote in item['evidence']:
                        st.text(quote)
                    if raw.get('facts'):
                        st.dataframe(pd.DataFrame(raw['facts']), hide_index=True)
                    semantic = raw.get('semantic', {})
                    if semantic.get('status') == 'validated':
                        st.text("模型解读（未独立核实）：" + semantic['summary'])
        if snapshot.get('deliveries'):
            with st.expander("通知投递记录（local 不等于已外发）"):
                st.dataframe(pd.DataFrame(snapshot['deliveries']), hide_index=True, width='stretch')
        if st.session_state.get("bond_monitor_digest"):
            st.download_button("下载本次证据简报", st.session_state['bond_monitor_digest'],
                               file_name="bond_monitor_digest.md", mime="text/markdown")

def render_ai_fusion(ctx):
    from hybrid_dashboard import render_hybrid_workbench
    render_hybrid_workbench(ctx, api_key=_effective_api_key(), include_preparation=False)
def render_evolution_combined(ctx):
    st.subheader("🔄 自我进化引擎 (诊断+微调+适配)")

    tab_e1, tab_e2, tab_e3 = st.tabs(["📊 诊断报告", "🔧 微调迭代", "📈 进化历史"])

    # ── Tab 1: 诊断报告 (原自我迭代) ──
    with tab_e1:
        st.markdown("**迭代诊断**: 从交易记忆分析失败模式, 检测制度漂移, 建议优化方向")

        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("🔍 运行完整诊断", width='stretch', type="primary", key="evo_diag"):
                with st.spinner("运行诊断中..."):
                    iterator = load_iterator()
                    report = iterator.run_diagnostic()
                    st.session_state['iter_report'] = report
                    st.session_state['iter_text'] = iterator.generate_report_text(report)

        with col2:
            if 'iter_report' in st.session_state:
                report = st.session_state['iter_report']
                drift = report.get('drift', {})
                if drift.get('detected'):
                    st.error("⚠ 制度漂移检测到!")
                else:
                    st.success("✅ 无制度漂移")

        if 'iter_report' in st.session_state:
            report = st.session_state['iter_report']
            perf = report.get('performance', {})
            st.divider()
            cols = st.columns(4)
            with cols[0]: st.metric("整体IC", perf.get('overall_ic', 'N/A'))
            with cols[1]: st.metric("滚动IC(30天)", perf.get('rolling_ic_latest', 'N/A'))
            with cols[2]: st.metric("记忆准确率", f"{perf.get('memory_accuracy', 0):.1%}" if perf.get('memory_accuracy') else 'N/A')
            with cols[3]: st.metric("近期准确率", f"{perf.get('recent_20_accuracy', 0):.1%}" if perf.get('recent_20_accuracy') else 'N/A')

            recs = report.get('recommendations', [])
            if recs:
                st.divider()
                st.markdown("**💡 优化建议**")
                for r in recs:
                    icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(r['priority'], '⚪')
                    st.markdown(f"{icon} **[{r['priority'].upper()}] {r['action']}**: {r['detail']}")

            if 'iter_text' in st.session_state:
                with st.expander("完整诊断报告", expanded=False):
                    st.code(st.session_state['iter_text'][:3000], language='text')

    # ── Tab 2: 微调迭代 (原自我进化 - LoRA式微调) ──
    with tab_e2:
        st.warning(
            "旧版 LoRA 式 AdapterStack 已停用：它没有统一 release manifest、"
            "哈希绑定和回测门禁。模型更新统一走“训练候选 → 门禁 → ReleaseManager”。"
        )

        # Load evolution status
        engine = None
        try:
            from self_evolution import SelfEvolutionEngine
            engine = SelfEvolutionEngine(BASE_DIR)
            status = engine.get_status()
        except Exception as e:
            status = _build_fallback_status(BASE_DIR)
            st.warning(
                f"引擎初始化降级: {redact_sensitive_text(e)[:80]}"
            )

        col_e1, col_e2, col_e3, col_e4 = st.columns(4)
        with col_e1: st.metric("基模型", "✓" if status['base_model_loaded'] else "✗")
        with col_e2: st.metric("适配器数", status['adapter_stack_size'])
        with col_e3: st.metric("进化周期", status['total_cycles'])
        with col_e4:
            latest = status.get('latest_report', {})
            st.metric("最新IC变化", f"{latest.get('ic_improvement', 'N/A')}")

        # Adapter table
        adapters = status.get('active_adapters', [])
        if adapters:
            st.divider()
            df_a = pd.DataFrame(adapters)
            df_a.columns = ['ID', '时间', '衰减', '验证IC', '焦点', '方向']
            st.dataframe(df_a, hide_index=True, width='stretch')

        # Actions
        st.divider()
        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            st.button(
                "🔄 每周适配（已停用）",
                width='stretch',
                key="evo_weekly2",
                disabled=True,
            )
        with col_a2:
            st.button(
                "🔧 旧双月重训练（已停用）",
                width='stretch',
                key="evo_bim2",
                disabled=True,
            )
        with col_a3:
            st.button(
                "🔮 适配器预测（已停用）",
                width='stretch',
                key="evo_pred2",
                disabled=True,
            )
        st.caption("请在训练页执行受控训练；生产推理只读取当前 managed release。")

        # CNN status
        st.divider()
        st.markdown("**🧠 CNN嵌入状态**")
        cnn_status = {}
        try:
            from micro_cnn import MicroCNNPipeline
            cnn = MicroCNNPipeline(BASE_DIR)
            cnn_status = cnn.get_status()
        except: pass
        cc1, cc2, cc3 = st.columns(3)
        with cc1: st.metric("CNN模型", "✓" if cnn_status.get('model_exists') else "✗")
        with cc2: st.metric("嵌入提取", "✓" if cnn_status.get('embeddings_exist') else "✗")
        with cc3: st.metric("因子注入", "✓" if cnn_status.get('injected') else "✗")

    # ── Tab 3: 进化历史 ──
    with tab_e3:
        history = status.get('history', []) if 'status' in dir() else []
        if history:
            rows = []
            for h in history[-15:]:
                rows.append({'时间': h.get('timestamp', '')[:19], '类型': h.get('cycle_type', ''),
                           '基IC': h.get('base_ic'), '组合IC': h.get('combined_ic'),
                           '提升': h.get('ic_improvement'), '适配器': h.get('adapter_stack_size', 0)})
            df_hist = pd.DataFrame(rows)
            st.dataframe(df_hist, hide_index=True, width='stretch')
        else:
            st.info("暂无进化历史")
# ================================================================
# Tab 7: 交易记忆 V2 — 完整复盘+可视化
# ================================================================
_MIN_ACCURACY_SAMPLES = 10


def _accuracy_trend(mem, window=20):
    """Rolling hit rate over evaluated records, or None when too few."""
    records = [
        r for r in mem.get_recent(10_000)
        if r.get('is_correct') is not None and r.get('trade_dt')
    ]
    if len(records) < window:
        return None
    frame = pd.DataFrame({
        'trade_dt': pd.to_datetime([r['trade_dt'] for r in records], errors='coerce'),
        'hit': [1.0 if r['is_correct'] else 0.0 for r in records],
    }).dropna(subset=['trade_dt']).sort_values('trade_dt')
    if len(frame) < window:
        return None
    frame['rolling'] = frame['hit'].rolling(window).mean() * 100
    return frame.dropna(subset=['rolling'])


def _render_accuracy_breakdown(buckets):
    """One measure, one hue. Sample size rides each bar so a thin
    category cannot read as a result."""
    if not buckets:
        st.caption("暂无分组记录。")
        return
    names = list(buckets)
    accs = [buckets[n]['accuracy'] * 100 for n in names]
    counts = [buckets[n]['count'] for n in names]
    thin = [n for n, c in zip(names, counts) if c < _MIN_ACCURACY_SAMPLES]
    width = chart_theme.bar_width(len(names))
    fig = go.Figure(go.Bar(
        x=names, y=accs,
        marker_color=chart_theme.SERIES,
        marker_line=dict(color=chart_theme.SURFACE, width=2),
        marker_cornerradius=4,
        width=[width] * len(names),
        text=[f"{a:.1f}%<br>n={c}" for a, c in zip(accs, counts)],
        textposition='outside',
        textfont=dict(color=chart_theme.INK, size=12),
        cliponaxis=False,
        customdata=counts,
        hovertemplate='%{x}<br>准确率 %{y:.1f}%<br>样本 %{customdata} 笔<extra></extra>',
    ))
    # The rule sits behind the bars, so an inline label would collide; the
    # caption below carries its meaning instead.
    chart_theme.reference_line(fig, 50)
    chart_theme.apply_theme(fig, height=230, y_title='准确率 %')
    fig.update_yaxes(range=[0, 118], ticksuffix='%')
    st.plotly_chart(fig, width='stretch')
    notes = ["横线为 50% 随机水平。"]
    if thin:
        notes.append(
            "样本不足 %d 笔，不能据此判断分组优劣：%s。"
            % (_MIN_ACCURACY_SAMPLES, "、".join(thin))
        )
    st.caption(" ".join(notes))


def render_memory_v2(ctx):
    st.subheader("🧠 交易记忆系统")

    mem = TradingMemory(BASE_DIR)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📥 从预测回填记忆", width='stretch'):
            with st.spinner("回填中..."):
                n = mem.backfill_from_predictions()
            st.success(f"已回填 {n} 条记录")

    with col2:
        if st.button("📊 更新实际结果", width='stretch'):
            with st.spinner("更新中..."):
                mem.update_actuals()
            st.success("已更新")

    with col3:
        if st.button("🤖 LLM反思分析", width='stretch'):
            with st.spinner("AI分析中..."):
                reflection = mem.llm_reflection(
                    api_key=_effective_api_key()
                )
            st.session_state['reflection_text'] = reflection

    # Statistics
    st.divider()
    stats = mem.reflection_stats()
    if 'error' in stats:
        st.info("暂无交易记忆数据，请先回填")
        return

    # Top stats row
    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
    with col_s1:
        st.metric("总记录", stats['total_records'])
    with col_s2:
        st.metric("已评估", stats['evaluated'])
    with col_s3:
        st.metric("整体准确率", f"{stats['overall_accuracy']:.1%}")
    with col_s4:
        st.metric("最近10天", f"{stats['recent_10_accuracy']:.1%}")
    with col_s5:
        st.metric("连胜", f"{stats['current_win_streak']} (max:{stats['max_win_streak']})")

    # Accuracy over time — the change-over-time question comes first
    st.divider()
    st.markdown("**滚动准确率走势（20 笔窗口）**")
    trend = _accuracy_trend(mem)
    if trend is None:
        st.caption("已评估记录不足 20 笔，滚动准确率尚不可读。")
    else:
        fig = go.Figure(go.Scatter(
            x=trend['trade_dt'], y=trend['rolling'], mode='lines',
            line=dict(color=chart_theme.SERIES, width=2, shape='spline',
                      smoothing=0.4),
            hovertemplate='%{x|%Y-%m-%d}<br>滚动准确率 %{y:.1f}%<extra></extra>',
        ))
        last = trend.iloc[-1]
        fig.add_trace(go.Scatter(
            x=[last['trade_dt']], y=[last['rolling']], mode='markers+text',
            marker=dict(color=chart_theme.SERIES, size=9,
                        line=dict(color=chart_theme.SURFACE, width=2)),
            text=[f"{last['rolling']:.1f}%"], textposition='middle right',
            textfont=dict(color=chart_theme.INK), hoverinfo='skip',
        ))
        chart_theme.reference_line(fig, 50, '随机水平 50%')
        chart_theme.apply_theme(fig, height=280, y_title='准确率 %')
        fig.update_yaxes(range=[0, 100], ticksuffix='%')
        st.plotly_chart(fig, width='stretch')

    # Breakdowns — identity, not magnitude: one measure, one hue
    st.divider()
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("**按市场状态**")
        _render_accuracy_breakdown(stats.get('by_regime', {}))
    with col_c2:
        st.markdown("**按交易方向**")
        _render_accuracy_breakdown(stats.get('by_direction', {}))

    # Cross table
    st.divider()
    st.markdown("**状态×方向交叉准确率**")
    cross = stats.get('by_regime_direction', {})
    if cross:
        rows = []
        for regime, dirs in cross.items():
            for dname, info in dirs.items():
                rows.append({
                    '市场状态': regime,
                    '交易方向': dname,
                    '准确率': f"{info['accuracy']:.1%}",
                    '样本数': info['count'],
                })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

    # Recent records
    st.divider()
    st.markdown("**最近20条交易记录**")
    recent = mem.get_recent(20)
    if recent:
        recs = []
        for r in reversed(recent):
            actual = r.get('actual_return')
            actual_str = f"{actual:+.4f}%" if actual is not None else "—"
            correct = r.get('is_correct')
            icon = '✅' if correct else ('❌' if correct is False else '—')
            recs.append({
                '日期': r.get('trade_dt', ''),
                '状态': r.get('regime_name', ''),
                '方向': r.get('direction_name', ''),
                '预测(平滑)': f"{r.get('predicted_return_smooth', 0):+.4f}%",
                '实际结果': actual_str,
                '判定': icon,
                '置信度': f"{r.get('confidence', 0):.1%}",
            })
        st.dataframe(pd.DataFrame(recs), hide_index=True, width='stretch',
                     height=400)

    # LLM Reflection
    if 'reflection_text' in st.session_state:
        st.divider()
        st.markdown("### 🤖 AI归因反思")
        st.info(st.session_state['reflection_text'])


# ================================================================
# Tab 8: 自我迭代
# ================================================================
def render_iteration(ctx):
    st.subheader("🔄 自我迭代引擎")

    iterator = load_iterator()

    if st.button("🔍 运行完整诊断", width='stretch', type="primary"):
        with st.spinner("运行诊断中..."):
            report = iterator.run_diagnostic()
        st.session_state['iter_report'] = report
        st.session_state['iter_text'] = iterator.generate_report_text(report)

    if 'iter_report' in st.session_state:
        report = st.session_state['iter_report']

        # Performance
        perf = report.get('performance', {})
        st.divider()
        st.markdown("### 📊 性能概览")

        cols = st.columns(4)
        with cols[0]:
            st.metric("整体IC", perf.get('overall_ic', 'N/A'))
        with cols[1]:
            st.metric("滚动IC(30天)", perf.get('rolling_ic_latest', 'N/A'))
        with cols[2]:
            st.metric("记忆准确率", f"{perf.get('memory_accuracy', 0):.1%}" if perf.get('memory_accuracy') else 'N/A')
        with cols[3]:
            st.metric("近期准确率", f"{perf.get('recent_20_accuracy', 0):.1%}" if perf.get('recent_20_accuracy') else 'N/A')

        # Drift warning
        drift = report.get('drift', {})
        if drift.get('detected'):
            st.error(f"⚠️ **制度漂移检测到!** 滚动IC={drift['info']['recent_ic_mean']:.4f}, "
                    f"建议用{drift['info']['suggested_window_months']}月窗口重新训练")
        else:
            st.success("✅ 未检测到制度漂移")

        # Failure patterns
        patterns = report.get('patterns', [])
        if patterns:
            st.divider()
            st.markdown("### 🔍 失败模式")
            for p in patterns:
                with st.expander(f"[{p['type']}] {p.get('suggestion', '')[:60]}...", expanded=False):
                    st.json(p)

        # Recommendations
        recs = report.get('recommendations', [])
        if recs:
            st.divider()
            st.markdown("### 💡 优化建议")
            for r in recs:
                icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(r['priority'], '⚪')
                st.markdown(f"{icon} **[{r['priority'].upper()}] {r['action']}**: {r['detail']}")

        # Auto-adjust
        st.divider()
        st.markdown("### ⚙️ 参数自动调优")
        if st.button("📐 分析最优信号参数", key="auto_tune"):
            with st.spinner("分析中..."):
                adj = iterator.auto_adjust_signal_params()
            if adj:
                st.success(f"建议置信度阈值: {adj['suggested_confidence_threshold']}, "
                          f"预期准确率: {adj['expected_accuracy']:.1%}")

                # Show all thresholds
                all_r = adj.get('all_results', {})
                if all_r:
                    data = [{'阈值': k.replace('conf_threshold_', ''), '准确率': f"{v['accuracy']:.1%}", '样本': v['samples']}
                            for k, v in sorted(all_r.items())]
                    st.dataframe(pd.DataFrame(data), hide_index=True)

        # Full report
        if 'iter_text' in st.session_state:
            st.divider()
            st.markdown("### 📋 完整诊断报告")
            st.code(st.session_state['iter_text'], language='text')

        # Action buttons
        st.divider()
        st.markdown("### 🚀 执行操作")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            if st.button("🔄 运行重训练 (12月+90天)", width='stretch'):
                st.info("请手动执行: cd src && python retrain_optimized.py")
        with col_a2:
            if st.button("💾 保存诊断报告", width='stretch'):
                path = os.path.join(BASE_DIR, "outputs", f"iteration_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2, default=str)
                st.success(f"已保存: {path}")


# ================================================================
# Delegated tabs (from original dashboard — pass-through to keep existing)
# ================================================================
def render_market_from_v1(ctx):
    from market_dashboard import render_market_tab
    render_market_tab(ctx)


def render_factor_from_v1(ctx):
    from dashboard import render_factor_tab
    render_factor_tab(ctx)


def render_backtest_from_v1(ctx):
    from dashboard import render_backtest_tab
    render_backtest_tab(ctx)


def render_macro_from_v1(ctx):
    from dashboard import render_macro_tab
    render_macro_tab(ctx)


def render_intelligence_from_v1(ctx):
    from dashboard import render_intelligence_tab
    render_intelligence_tab(ctx)


# ================================================================
# Tab 10: 模型评估 (借鉴 Dexter eval)
# ================================================================
def render_eval_tab(ctx):
    st.subheader("📋 模型评估 (借鉴 Dexter eval system)")

    col1, col2, col3 = st.columns(3)
    with col1:
        window = st.selectbox("评估窗口", [7, 14, 30, 60], index=2, key="eval_window")
    with col2:
        step = st.selectbox("步长", [5, 10, 15, 30], index=1, key="eval_step")
    with col3:
        if st.button("🔍 运行评估", width='stretch', key="run_eval"):
            with st.spinner("运行评估中..."):
                import subprocess, sys
                result = subprocess.run(
                    [sys.executable, "eval_runner.py", "--window", str(window), "--step", str(step)],
                    cwd=os.path.join(BASE_DIR, "src"),
                    capture_output=True, text=True, timeout=120
                )
            st.session_state['eval_output'] = redact_sensitive_text(result.stdout)

    if 'eval_output' in st.session_state:
        st.divider()
        st.code(st.session_state['eval_output'], language='text')

    # Quick IC check from predictions
    st.divider()
    st.markdown("**快速IC检查**")
    pred_path = os.path.join(BASE_DIR, "outputs", "df_predictions.pkl")
    if os.path.exists(pred_path):
        df = pd.read_pickle(pred_path)
        if 'Target_Ret' in df.columns and 'Pred_Ret' in df.columns:
            mask = df['Target_Ret'].notna() & df['Pred_Ret'].notna()
            if mask.sum() > 100:
                ic = np.corrcoef(df.loc[mask, 'Pred_Ret'], df.loc[mask, 'Target_Ret'])[0, 1]
                st.metric("当前整体IC", f"{ic:.4f}")

                # Daily IC trend
                df['trade_date'] = pd.to_datetime(df['date']).dt.date
                daily = df.groupby('trade_date').agg({'Pred_Ret': 'mean', 'Target_Ret': 'sum'}).dropna()
                if len(daily) > 30:
                    rolling_ic = daily['Pred_Ret'].rolling(30).corr(daily['Target_Ret']).dropna()
                    if len(rolling_ic) > 0:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=rolling_ic.index, y=rolling_ic.values,
                            mode='lines', name='30天滚动IC',
                            line=dict(color='#FFB74D', width=2),
                        ))
                        fig.add_hline(y=0, line_dash="dot", line_color="gray")
                        fig.add_hline(y=0.03, line_dash="dash", line_color="green", opacity=0.3)
                        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                                         template='plotly_dark')
                        st.plotly_chart(fig, width='stretch')


# ================================================================
# Tab: 因子评估 (Alphalens风格)
# ================================================================
def render_factor_eval(ctx):
    st.subheader("🔍 因子评估 (Alphalens 风格)")

    imp_path = os.path.join(BASE_DIR, "outputs", "feature_importance.csv")
    factor_path = os.path.join(BASE_DIR, "outputs", "df_factors.pkl")

    if not os.path.exists(factor_path):
        st.warning("无因子数据")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        top_n = st.slider("评估Top N因子", 5, 40, 10, key="feval_topn")
    with col2:
        if st.button("🔍 快速IC评估", width='stretch', key="feval_run"):
            with st.spinner(f"评估Top {top_n}个因子..."):
                from factor_evaluator import FactorEvaluator
                from training_data import LabelSpec, build_forward_labels

                df = build_forward_labels(
                    pd.read_pickle(factor_path), LabelSpec(horizon_bars=30)
                )
                df = df.loc[df['label_valid']].copy()
                evaluator = FactorEvaluator(df)

                # Use feature importance to pick top factors
                if os.path.exists(imp_path):
                    imp = pd.read_csv(imp_path)
                    candidates = imp[imp['importance'] > 0].head(top_n)['feature'].tolist()
                else:
                    candidates = evaluator.factor_cols[:top_n]

                scores = []
                for col in candidates:
                    if col not in evaluator.factor_cols:
                        continue
                    try:
                        ic_info = evaluator.compute_ic(col)
                        quantiles = evaluator.quantile_returns(col)
                        turnover = evaluator.factor_turnover(col)
                        ic = ic_info.get('ic', 0.0)
                        ls = quantiles['long_short'] if quantiles else 0.0
                        ac1 = turnover['autocorr_1'] if turnover else 0.0
                        scores.append({'factor': col, 'ic': round(ic,4),
                                      'long_short': round(ls,6), 'stability': round(ac1 if not np.isnan(ac1) else 0, 3)})
                    except (TypeError, ValueError):
                        continue

                st.session_state['feval_scores'] = pd.DataFrame(scores).sort_values('ic', key=abs, ascending=False)

    if 'feval_scores' in st.session_state:
        df_scores = st.session_state['feval_scores']
        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**|IC| 排名**")
            fig = px.bar(df_scores.head(15).iloc[::-1], x='ic', y='factor', orientation='h',
                         color='ic', color_continuous_scale='RdBu', color_continuous_midpoint=0)
            fig.update_layout(height=400, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, width='stretch')

        with col_b:
            st.markdown("**因子稳定性 (AC1)**")
            fig = px.scatter(df_scores.head(15), x='stability', y='ic', text='factor',
                            color='ic', color_continuous_scale='RdBu', color_continuous_midpoint=0)
            fig.add_hline(y=0, line_dash="dot", line_color="gray")
            fig.add_vline(x=0.5, line_dash="dash", line_color="green", opacity=0.3)
            fig.update_layout(height=400, margin=dict(l=10,r=10,t=10,b=10))
            fig.update_traces(textposition='top center', textfont=dict(size=9))
            st.plotly_chart(fig, width='stretch')

        st.markdown("**详细数据**")
        st.dataframe(df_scores.head(20).style.format({'ic': '{:.4f}', 'long_short': '{:.6f}', 'stability': '{:.3f}'}),
                    hide_index=True, width='stretch')

        # Redundancy check
        if len(df_scores) >= 8:
            st.divider()
            st.markdown("**因子相关性 (Top8)**")
            df = pd.read_pickle(factor_path)
            top8 = df_scores.head(8)['factor'].tolist()
            corr = df[top8].corr(method='spearman')
            fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                           zmin=-1, zmax=1, aspect='auto')
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')


# ================================================================
# Tab: 超参优化 (Optuna + Ensemble)
# ================================================================
def render_hyperopt_tab(ctx):
    st.subheader("⚙️ 超参数优化 (Optuna + 集成)")

    st.markdown("""
    **Optuna 贝叶斯超参搜索** 可用于自动寻找最优 LightGBM 参数，
    替代手动窗口扫描。同时也支持 **多模型集成** (LightGBM + XGBoost + CatBoost)
    和 **多时域堆叠** (30/60/120min) 预测。
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        trials = st.slider("Optuna Trials", 10, 100, 30, 10, key="hyper_trials")
    with col2:
        mode = st.selectbox("优化模式", ["optuna", "ensemble", "multihorizon", "all"],
                           key="hyper_mode")
    with col3:
        if st.button("🚀 运行优化", width='stretch', type="primary", key="hyper_run"):
            with st.spinner(f"运行中 ({mode}, {trials} trials)..."):
                import subprocess, sys
                result = subprocess.run(
                    [sys.executable, "optuna_optimizer.py", "--mode", mode, "--trials", str(trials)],
                    cwd=os.path.join(BASE_DIR, "src"),
                    capture_output=True, text=True, timeout=600
                )
            st.session_state['hyper_output'] = redact_sensitive_text(result.stdout)
            if result.returncode != 0:
                st.session_state['hyper_error'] = redact_sensitive_text(
                    result.stderr
                )

    if 'hyper_output' in st.session_state:
        st.divider()
        st.markdown("**优化输出**")
        st.code(st.session_state['hyper_output'][-3000:], language='text')
    if 'hyper_error' in st.session_state:
        st.error(st.session_state['hyper_error'][:500])

    # Show signed/hashed release metadata without deserializing a pickle in the
    # long-lived web process.
    st.divider()
    try:
        from model_release import ReleaseManager, load_manifest

        manager = ReleaseManager(
            os.path.join(BASE_DIR, "artifacts", "models"),
            legacy_model_path=os.path.join(
                BASE_DIR, "models", "trained_model.pkl"
            ),
        )
        signature = manager.current_signature()
        if signature is not None and signature.source == "release":
            manifest = load_manifest(signature.manifest_path)
            evaluation_path = manager.active_artifact_path("evaluation")
            evaluation = (
                json.loads(evaluation_path.read_text(encoding="utf-8"))
                if evaluation_path is not None else {}
            )
            st.markdown("**当前托管模型 release**")
            st.json({
                "release_id": signature.release_id,
                "model_sha256": signature.artifact_sha256,
                "strategy_id": manifest.get("metadata", {}).get("strategy_id"),
                "strategy_fingerprint": manifest.get("metadata", {}).get(
                    "strategy_fingerprint"
                ),
                "selected_model": evaluation.get("selected_model"),
                "release_gate_passed": evaluation.get("release_gate_passed"),
            })
        elif signature is not None:
            st.warning("检测到未托管 legacy 模型；生产推理已禁用该 pickle。")
    except Exception as exc:
        st.error(
            f"模型 release 元数据不可用: {redact_sensitive_text(exc)}"
        )

    # Multi-horizon model status
    mh_path = os.path.join(BASE_DIR, "models", "multi_horizon_model.pkl")
    if os.path.exists(mh_path):
        st.caption("检测到未托管多时域 pickle；Dashboard 不会反序列化它。")


# ================================================================
# Tab: 定时调度 (Cron Scheduler)
# ================================================================
def render_cron_tab(ctx):
    st.subheader("⏰ 定时任务调度")

    cron_path = os.path.join(BASE_DIR, "outputs", "cron_jobs.json")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📋 刷新任务列表", width='stretch', key="cron_refresh"):
            st.cache_data.clear()
            st.rerun()

    with col2:
        if st.button("▶️ 运行全部任务", width='stretch', key="cron_runall"):
            with st.spinner("运行中..."):
                import subprocess, sys
                result = subprocess.run(
                    [sys.executable, "cron_scheduler.py", "run-all"],
                    cwd=os.path.join(BASE_DIR, "src"),
                    capture_output=True, text=True, timeout=600
                )
            st.session_state['cron_all_output'] = redact_sensitive_text(
                result.stdout
            )

    with col3:
        try:
            from cron_scheduler import daemon_is_running

            daemon_running = daemon_is_running(BASE_DIR)
        except Exception:
            daemon_running = False
        st.metric("守护进程", "🟢 运行中" if daemon_running else "⚪ 未运行")

    # Job list
    if os.path.exists(cron_path):
        with open(cron_path, 'r', encoding='utf-8') as f:
            jobs = json.load(f)
    else:
        import copy
        from cron_scheduler import DEFAULT_JOBS

        jobs = copy.deepcopy(DEFAULT_JOBS)

    st.divider()
    st.markdown("**任务列表**")

    for j in jobs:
        status_icon = "✅" if j.get('enabled', True) else "⏸️"
        last_run = j.get('last_run', '从未运行')
        last_error = j.get('last_error', '')

        col_j1, col_j2, col_j3 = st.columns([3, 1, 1])
        with col_j1:
            st.markdown(f"{status_icon} **{j['name']}** — `{j['kind']} {j.get('expr','')}`  "
                       f"_上次: {last_run[:16]}_")
            if last_error:
                st.caption(f"⚠️ {last_error[:100]}")
        with col_j2:
            if st.button("▶ 运行", key=f"cron_run_{j['id']}"):
                with st.spinner(f"执行 {j['name']}..."):
                    import subprocess, sys
                    result = subprocess.run(
                        [sys.executable, "cron_scheduler.py", "run", j['id']],
                        cwd=os.path.join(BASE_DIR, "src"),
                        capture_output=True, text=True, timeout=300
                    )
                job_key = 'cron_' + j['id'] + '_output'
                st.session_state[job_key] = redact_sensitive_text(
                    result.stdout
                )
                st.rerun()
        with col_j3:
            new_state = not j.get('enabled', True)
            if st.button("⏸ 暂停" if j.get('enabled', True) else "▶ 启用", key=f"cron_toggle_{j['id']}"):
                j['enabled'] = new_state
                with open(cron_path, 'w', encoding='utf-8') as f:
                    json.dump(jobs, f, ensure_ascii=False, indent=2)
                st.rerun()

        # Show per-job output
        output_key = f'cron_{j["id"]}_output'
        if output_key in st.session_state:
            with st.expander(f"输出: {j['name']}", expanded=False):
                st.code(st.session_state[output_key][-1000:], language='text')

    if 'cron_all_output' in st.session_state:
        st.divider()
        st.markdown("**全量运行输出**")
        st.code(st.session_state['cron_all_output'][-2000:], language='text')


def _build_fallback_status(base_dir):
    """当进化引擎 pickle 损坏时，从 JSON 报告构建最小状态（只读）"""
    import os as _os, json as _json
    report = {}
    history = []
    managed_release_loaded = False

    try:
        from model_release import ReleaseManager

        manager = ReleaseManager(
            _os.path.join(base_dir, "artifacts", "models"),
            legacy_model_path=_os.path.join(
                base_dir, "models", "trained_model.pkl"
            ),
        )
        signature = manager.current_signature()
        managed_release_loaded = bool(
            signature is not None and signature.source == "release"
        )
    except Exception:
        # A corrupt pointer or artifact is not a loaded model.  The detailed
        # release panel reports the integrity error separately.
        managed_release_loaded = False
    
    report_path = _os.path.join(base_dir, "outputs", "evolution_report.json")
    if _os.path.exists(report_path):
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                report = _json.load(f)
        except: pass
    
    hist_path = _os.path.join(base_dir, "outputs", "evolution_history.jsonl")
    if _os.path.exists(hist_path):
        try:
            with open(hist_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            history.append(_json.loads(line))
                        except: pass
        except: pass
    
    return {
        'base_model_loaded': managed_release_loaded,
        'adapter_stack_size': 0,
        'active_adapters': [],
        'latest_report': report,
        'history': history[-20:],
        'total_cycles': len(history),
    }


# ================================================================
# Tab: 遗传因子挖掘 (Genetic Programming Factor Mining)
# ================================================================
def render_genetic_mining(ctx):
    from genetic_dashboard import render_genetic_mining as render_research_mining
    render_research_mining(ctx)

# ================================================================
# Tab: 自我进化 (LoRA-inspired Self-Evolution)
# ================================================================
def render_evolution(ctx):
    st.subheader("🔧 模型迭代状态")

    st.markdown("""
    旧版 AdapterStack 已退役，不再加载 `adapter_stack.pkl`，也不能写入生产预测。
    当前唯一模型更新路径是：**训练候选 → 同策略成本后回测 → release gate →
    ReleaseManager 原子切换**。漂移诊断仍可只读运行。
    """)

    # Read-only compatibility status. The engine never deserializes the legacy
    # adapter pickle.
    engine = None
    engine_error = None
    try:
        from self_evolution import SelfEvolutionEngine
        engine = SelfEvolutionEngine(BASE_DIR)
        status = engine.get_status()
    except Exception as e:
        engine_error = redact_sensitive_text(e)
        status = _build_fallback_status(BASE_DIR)
        st.warning(
            f"进化状态读取失败（已降级到只读模式）: {engine_error[:100]}"
        )

    # ─── Status cards ───
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("基模型", "✓ 已加载" if status['base_model_loaded'] else "✗ 未加载")
    with col2:
        st.metric("适配器堆栈", f"{status['adapter_stack_size']} 个")
    with col3:
        st.metric("进化周期", status['total_cycles'])
    with col4:
        latest = status.get('latest_report', {})
        ic_imp = latest.get('ic_improvement', 'N/A')
        st.metric("最新IC提升", f"{ic_imp:+.4f}" if isinstance(ic_imp, float) else ic_imp)

    st.divider()

    # ─── Active Adapters ───
    st.markdown("**🔗 活跃适配器**")

    adapters = status.get('active_adapters', [])
    if adapters:
        import pandas as pd
        df_adapters = pd.DataFrame(adapters)
        df_adapters.columns = ['适配器ID', '训练时间', '衰减权重', '验证IC', '焦点市场', '焦点方向']
        df_adapters['衰减权重'] = df_adapters['衰减权重'].apply(lambda x: f"{x:.3f}")
        df_adapters['验证IC'] = df_adapters['验证IC'].apply(lambda x: f"{x:.4f}")
        st.dataframe(df_adapters, width='stretch', hide_index=True)
    else:
        st.info("旧版适配器已停用；当前不会加载或生成适配器。")

    st.divider()

    # ─── Evolution History ───
    st.markdown("**📈 进化历史**")

    history = status.get('history', [])
    if history:
        import pandas as pd
        rows = []
        for h in history[-15:]:
            rows.append({
                '时间': h.get('timestamp', '')[:19],
                '类型': h.get('cycle_type', ''),
                '基模型IC': h.get('base_ic'),
                '组合IC': h.get('combined_ic'),
                'IC提升': h.get('ic_improvement'),
                '适配器数': h.get('adapter_stack_size', 0),
            })
        df_hist = pd.DataFrame(rows)
        
        # Plot IC trend
        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_hist['时间'], y=df_hist['基模型IC'],
                mode='lines+markers', name='基模型IC',
                line=dict(color='gray', dash='dot')
            ))
            fig.add_trace(go.Scatter(
                x=df_hist['时间'], y=df_hist['组合IC'],
                mode='lines+markers', name='组合IC (Base+Adapters)',
                line=dict(color='#00B388')
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.3)
            fig.update_layout(
                title="IC 进化轨迹",
                xaxis_title="",
                yaxis_title="IC",
                height=350,
                margin=dict(l=0, r=0, t=40, b=0),
                legend=dict(orientation='h', yanchor='bottom', y=1.02),
            )
            st.plotly_chart(fig, width='stretch')
        except Exception:
            st.dataframe(df_hist, width='stretch', hide_index=True)
    else:
        st.info("暂无进化历史。")

    st.divider()

    # ─── Actions ───
    st.markdown("**⚡ 旧入口状态**")

    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        st.button(
            "🔄 每周适配（已停用）",
            width='stretch',
            key="evo_weekly",
            disabled=True,
        )

    with col_a2:
        st.button(
            "🔧 旧双月重训练（已停用）",
            width='stretch',
            key="evo_bimonthly",
            disabled=True,
        )

    with col_a3:
        st.button(
            "🔮 适配器预测（已停用）",
            width='stretch',
            key="evo_predict",
            disabled=True,
        )

    st.caption("请使用“训练候选”入口；只有通过统一门禁的 release 才能用于推理。")

    # Output sections
    for key, title in [
        ('evo_weekly_output', '每周适配输出'),
        ('evo_bimonthly_output', '双月重训练输出'),
        ('evo_predict_output', '组合预测输出'),
    ]:
        if key in st.session_state:
            with st.expander(title, expanded=False):
                st.code(st.session_state[key][-2000:], language='text')

    st.divider()

    # ─── Latest Report ───
    latest = status.get('latest_report', {})
    if latest:
        st.markdown("**📋 最新进化报告**")
        
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.metric("周期类型", latest.get('cycle_type', '?'))
            st.metric("基模型IC", f"{latest.get('base_ic', 'N/A')}")
            st.metric("适配器堆栈", latest.get('adapter_stack_size', 0))
        with col_r2:
            st.metric("组合IC", f"{latest.get('combined_ic', 'N/A')}")
            st.metric("IC变化", f"{latest.get('ic_improvement', 'N/A')}")
            drift = latest.get('feedback', {}).get('drift_severity', 0)
            st.metric("制度漂移", f"{drift:.2f}", 
                     delta="⚠ 严重" if drift > 0.5 else "正常",
                     delta_color="off" if drift > 0.5 else "normal")

        recs = latest.get('recommendations', [])
        if recs:
            st.markdown("**优化建议**")
            for r in recs:
                st.markdown(f"- {r}")

    # ─── Architecture Diagram ───
    st.divider()
    st.markdown("**🏗️ 受控模型迭代架构**")
    st.markdown("""
    ```
    ┌──────────────────────────────────────────────────────────┐
    │  Point-in-time 特征 / 可选 CNN 研究特征                 │
    │                         ↓                                │
    │  候选训练 → 同一 SignalPolicy 成本后回测 → Gate Attestation │
    │                         ↓                                │
    │  ReleaseManager 校验哈希与门禁 → 原子切换 current.json   │
    │                         ↓                                │
    │  ModelRuntime 只加载 managed release → 信号/回测/记忆    │
    │                                                          │
    │  舆情/Jev/研究 Agent：仅 shadow evidence，不改交易仓位    │
    └──────────────────────────────────────────────────────────┘
    ```
    """)

    # ─── MicroCNN 微观结构嵌入 ───
    st.divider()
    st.markdown("**🧠 微观结构CNN嵌入 (Bottleneck Embedding)**")

    st.markdown("""
    *时空卷积网络从微观结构特征的时序窗口中提取64维稠密嵌入，作为"学习到的微观因子"注入LightGBM。*
    """)

    # Load CNN status
    cnn_status = {}
    try:
        from micro_cnn import MicroCNNPipeline
        cnn = MicroCNNPipeline(BASE_DIR)
        cnn_status = cnn.get_status()
    except Exception as e:
        cnn_status['error'] = redact_sensitive_text(e)[:100]
        cnn_status['has_torch'] = False

    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    with col_c1:
        st.metric("PyTorch", "✓" if cnn_status.get('has_torch') else "✗ 未安装")
    with col_c2:
        st.metric("CNN模型", "✓ 已训练" if cnn_status.get('model_exists') else "✗ 未训练")
    with col_c3:
        st.metric("嵌入提取", "✓ 已提取" if cnn_status.get('embeddings_exist') else "✗ 未提取")
    with col_c4:
        st.metric("因子注入", "✓ 已注入" if cnn_status.get('injected') else "✗ 未注入")

    if cnn_status.get('train_stats'):
        stats = cnn_status['train_stats']
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("验证IC", f"{stats.get('val_ic', 0):.4f}")
        with col_s2:
            st.metric("参数量", f"{stats.get('n_params', 0):,}")
        with col_s3:
            st.metric("瓶颈维度", cnn_status.get('bottleneck_dim', 64))

    col_ca1, col_ca2, col_ca3 = st.columns(3)
    with col_ca1:
        if st.button("🔬 训练CNN+注入", width='stretch', key="cnn_pipeline",
                     help="训练微观CNN → 提取嵌入 → 注入因子文件 (需PyTorch)"):
            with st.spinner("CNN训练中... (可能需要5-15分钟)"):
                import subprocess, sys
                result = subprocess.run(
                    [sys.executable, "micro_cnn.py", "pipeline", BASE_DIR],
                    cwd=os.path.join(BASE_DIR, "src"),
                    capture_output=True, text=True, timeout=1800
                )
            st.session_state['cnn_pipeline_output'] = redact_sensitive_text(
                result.stdout
            )
            if result.returncode == 0:
                st.success("CNN嵌入已注入! 下一轮LightGBM训练将自动使用CNN特征。")
            else:
                st.error(
                    f"失败: {redact_sensitive_text(result.stderr)[:300]}"
                )
            st.rerun()

    with col_ca2:
        if st.button("📊 查看状态", width='stretch', key="cnn_status_btn"):
            import subprocess, sys
            result = subprocess.run(
                [sys.executable, "micro_cnn.py", "status", BASE_DIR],
                cwd=os.path.join(BASE_DIR, "src"),
                capture_output=True, text=True, timeout=30
            )
            st.session_state['cnn_status_output'] = redact_sensitive_text(
                result.stdout
            )
            st.rerun()

    with col_ca3:
        if st.button("💉 重新注入", width='stretch', key="cnn_inject",
                     help="重新注入CNN嵌入到因子文件 (需已训练模型)"):
            import subprocess, sys
            result = subprocess.run(
                [sys.executable, "micro_cnn.py", "inject", "--force", BASE_DIR],
                cwd=os.path.join(BASE_DIR, "src"),
                capture_output=True, text=True, timeout=60
            )
            st.session_state['cnn_inject_output'] = redact_sensitive_text(
                result.stdout
            )
            if result.returncode == 0:
                st.success("注入完成!")
            else:
                st.error(
                    f"失败: {redact_sensitive_text(result.stderr)[:200]}"
                )
            st.rerun()

    for key, title in [
        ('cnn_pipeline_output', 'CNN管道输出'),
        ('cnn_status_output', 'CNN状态'),
        ('cnn_inject_output', '注入输出'),
    ]:
        if key in st.session_state:
            with st.expander(title, expanded=False):
                st.code(st.session_state[key][-2000:], language='text')

    st.markdown("""
    **模型迭代边界**

    | 阶段 | 技术 | 状态 | 作用 |
    |:--|:--|:--|:--|
    | ① CNN嵌入 | 时空1D-CNN → 64维瓶颈 | 研究特征 | 生成候选特征，不能直接发布模型 |
    | ② AdapterStack | 旧残差LGBM pickle | 已停用 | 不再读取、写入或参与预测 |
    | ③ 受控训练 | 候选 + 回测 + attestation | 唯一入口 | 通过 ReleaseManager 原子发布 |

    ```
    微观特征 (25×30bar窗口)
        ↓ 1D Temporal CNN
    64维瓶颈嵌入（候选特征）
        ↓
    统一训练 / 回测 / release gate
        ↓
    managed release → ModelRuntime
    ```
    """)


if __name__ == '__main__':
    main()
