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
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from strategy_agent import StrategyContext
from rag_tool import RAGAnalyzer
from memory import TradingMemory
from signal_dashboard import SignalDashboard
from self_iteration import SelfIterationEngine

st.set_page_config(
    page_title="F_Agent TL策略",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = r"D:\桌面\F_Agent"


@st.cache_data(ttl=300, show_spinner="加载数据中...")
def load_ctx():
    return StrategyContext(BASE_DIR)


@st.cache_resource
def load_signal_dash():
    return SignalDashboard(BASE_DIR)


@st.cache_resource
def load_rag():
    return RAGAnalyzer(BASE_DIR)


@st.cache_resource
def load_iterator():
    return SelfIterationEngine(BASE_DIR)


def main():
    st.title("F_Agent — TL 30Y国债期货智能策略系统")
    st.caption("LightGBM双模型 · 排名信号 · 自我迭代 · RAG增强")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ 控制面板")
        if st.button("🔄 刷新全部数据", width='stretch'):
            st.cache_data.clear()
            st.rerun()

        if st.button("🧹 清除缓存", width='stretch'):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

        st.divider()

        try:
            ctx = load_ctx()
            sd = load_signal_dash()
            if sd.is_ready and ctx.df_factors is not None:
                signal = sd.generate_live_signal(ctx.df_factors)
            else:
                signal = ctx.signal

            if signal:
                st.subheader("📡 实时信号")
                d = signal.get('direction', 0)
                d_label = {1: '🟢 LONG', -1: '🔴 SHORT', 0: '⚪ FLAT'}[d]
                st.metric("方向", d_label)
                st.metric("置信度", f"{signal.get('confidence', 0):.1%}")
                st.metric("排名分位", f"{signal.get('pred_rank_pct', 0):.1%}")
                st.metric("状态", signal.get('regime_name', 'N/A'))
            else:
                st.warning("无信号数据")
        except Exception as e:
            st.warning(f"数据加载: {e}")

        st.divider()
        st.caption(f"更新: {datetime.now().strftime('%H:%M:%S')}")

    # Tabs
    tabs = st.tabs([
        "📡 信号看板",
        "📊 市场监控",
        "🔬 因子分析",
        "🔍 因子评估",
        "💰 回测表现",
        "🌍 宏观环境",
        "🤖 AI情报",
        "📚 研究RAG",
        "🧠 交易记忆",
        "🔄 自我迭代",
        "📋 模型评估",
        "⚙️ 超参优化",
        "⏰ 定时调度",
    ])

    try:
        ctx = load_ctx()
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        return

    with tabs[0]: render_signal_v2(ctx)
    with tabs[1]: render_market_from_v1(ctx)
    with tabs[2]: render_factor_from_v1(ctx)
    with tabs[3]: render_factor_eval(ctx)
    with tabs[4]: render_backtest_from_v1(ctx)
    with tabs[5]: render_macro_from_v1(ctx)
    with tabs[6]: render_intelligence_from_v1(ctx)
    with tabs[7]: render_rag_v2(ctx)
    with tabs[8]: render_memory_v2(ctx)
    with tabs[9]: render_iteration(ctx)
    with tabs[10]: render_eval_tab(ctx)
    with tabs[11]: render_hyperopt_tab(ctx)
    with tabs[12]: render_cron_tab(ctx)


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

    st.subheader("📡 实时交易信号 (排名信号系统)")

    # Top metric cards
    col1, col2, col3, col4, col5 = st.columns(5)
    d = signal['direction']
    d_color = {1: '#00E676', -1: '#FF5252', 0: '#9E9E9E'}[d]

    with col1:
        st.metric("信号方向", signal['direction_name'], delta=None)
    with col2:
        st.metric("预测排名分位", f"{signal['pred_rank_pct']:.1%}",
                  delta=f"{signal['pred_rank_pct'] - 0.5:+.1%}" if signal['pred_rank_pct'] else None)
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
            '指标': ['预测收益(原始)', '预测收益(平滑)', '排名分位', '上阈值(70%)', '下阈值(30%)',
                    '使用模型', '建议仓位', '信号确认'],
            '值': [
                f"{signal['predicted_return']:+.6f}%",
                f"{signal['predicted_return_smooth']:+.6f}%",
                f"{signal['pred_rank_pct']:.3f}",
                '0.70', '0.30',
                signal['model_used'],
                f"{signal['suggested_weight']:.1%}",
                '10根bar连续同向',
            ]
        }
        st.dataframe(pd.DataFrame(detail_data), hide_index=True, width='stretch')

    with col_b:
        st.markdown("**信号强度仪表**")
        rank = signal.get('pred_rank_pct', 0.5)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=rank * 100,
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
                    'thickness': 0.8, 'value': rank * 100,
                }
            },
            title={'text': '预测排名分位 (%)'},
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
# Tab 7: 交易记忆 V2 — 完整复盘+可视化
# ================================================================
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
                reflection = mem.llm_reflection()
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

    # Accuracy charts
    st.divider()
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown("**按市场状态准确率**")
        by_regime = stats.get('by_regime', {})
        if by_regime:
            regimes = list(by_regime.keys())
            accs = [by_regime[r]['accuracy'] * 100 for r in regimes]
            counts = [by_regime[r]['count'] for r in regimes]
            fig = go.Figure(data=[
                go.Bar(name='准确率%', x=regimes, y=accs,
                       text=[f'{a:.1f}%' for a in accs], textposition='auto',
                       marker_color=['#4CAF50', '#FF9800', '#9C27B0']),
            ])
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
            fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5)
            st.plotly_chart(fig, width='stretch')

    with col_c2:
        st.markdown("**按交易方向准确率**")
        by_dir = stats.get('by_direction', {})
        if by_dir:
            dirs = list(by_dir.keys())
            daccs = [by_dir[d]['accuracy'] * 100 for d in dirs]
            dcounts = [by_dir[d]['count'] for d in dirs]
            fig = go.Figure(data=[
                go.Bar(name='准确率%', x=dirs, y=daccs,
                       text=[f'{a:.1f}%' for a in daccs], textposition='auto',
                       marker_color=['#00E676', '#FF5252']),
            ])
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
            fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5)
            st.plotly_chart(fig, width='stretch')

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
    from dashboard import render_market_tab
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
            st.session_state['eval_output'] = result.stdout

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
                df = pd.read_pickle(factor_path)
                df['next_close'] = df['close'].shift(-1)
                df['future_close'] = df['close'].shift(-31)
                df['Target_Ret'] = (df['future_close'] / df['next_close'] - 1) * 100

                # Use feature importance to pick top factors
                if os.path.exists(imp_path):
                    imp = pd.read_csv(imp_path)
                    candidates = imp[imp['importance'] > 0].head(top_n)['feature'].tolist()
                else:
                    exclude = {'date','trade_dt','ticker','close','open','high','low',
                              'volume','money','oi','time','next_close','future_close',
                              'Target_Ret','Hour','Minute','Minute_of_Day'}
                    candidates = [c for c in df.columns if c not in exclude][:top_n]

                from scipy import stats as sp_stats
                scores = []
                for col in candidates:
                    if col not in df.columns: continue
                    s = df[col].shift(1)
                    t = df['Target_Ret']
                    mask = s.notna() & t.notna()
                    if mask.sum() < 100: continue
                    try:
                        ic = sp_stats.spearmanr(s[mask].astype(float), t[mask].astype(float))[0]
                        if np.isnan(ic): ic = 0.0
                        # Quantile long-short
                        q = pd.qcut(s[mask], 5, labels=False, duplicates='drop')
                        ls_ret = t[mask].groupby(q).mean()
                        ls = ls_ret.iloc[-1] - ls_ret.iloc[0] if len(ls_ret) >= 2 else 0
                        # Stability
                        ac1 = s.dropna().autocorr(lag=1)
                        scores.append({'factor': col, 'ic': round(ic,4),
                                      'long_short': round(ls,6), 'stability': round(ac1 if not np.isnan(ac1) else 0, 3)})
                    except: pass

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
            st.session_state['hyper_output'] = result.stdout
            if result.returncode != 0:
                st.session_state['hyper_error'] = result.stderr

    if 'hyper_output' in st.session_state:
        st.divider()
        st.markdown("**优化输出**")
        st.code(st.session_state['hyper_output'][-3000:], language='text')
    if 'hyper_error' in st.session_state:
        st.error(st.session_state['hyper_error'][:500])

    # Show current model params
    st.divider()
    model_path = os.path.join(BASE_DIR, "models", "trained_model.pkl")
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            m = pickle.load(f)
        st.markdown("**当前模型**")
        st.json({
            'config': m.get('config', 'unknown'),
            'features': len(m.get('features', [])),
            'has_xgb': 'model_xgb' in m,
            'has_weights': 'weights' in m,
        })

    # Multi-horizon model status
    mh_path = os.path.join(BASE_DIR, "models", "multi_horizon_model.pkl")
    if os.path.exists(mh_path):
        with open(mh_path, 'rb') as f:
            mh = pickle.load(f)
        st.markdown("**多时域模型**")
        for h, r in mh.get('results', {}).items():
            st.metric(f"{h}min", f"IC={r['ic']:.4f}")


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
            st.session_state['cron_all_output'] = result.stdout

    with col3:
        daemon_running = False
        try:
            import subprocess
            check = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                                   capture_output=True, text=True, timeout=5)
            daemon_running = 'cron_scheduler' in check.stdout
        except: pass
        st.metric("守护进程", "🟢 运行中" if daemon_running else "⚪ 未运行")

    # Job list
    if os.path.exists(cron_path):
        with open(cron_path, 'r', encoding='utf-8') as f:
            jobs = json.load(f)
    else:
        jobs = [
            {"id": "daily_update", "name": "每日行情更新", "kind": "cron", "expr": "0 16 * * 1-5", "enabled": True, "command": "update"},
            {"id": "daily_signal", "name": "每日信号生成", "kind": "cron", "expr": "0 16 * * 1-5", "enabled": True, "command": "inference"},
            {"id": "weekly_eval", "name": "每周模型评估", "kind": "cron", "expr": "0 17 * * 5", "enabled": True, "command": "eval"},
            {"id": "weekly_memory", "name": "每周记忆回填", "kind": "cron", "expr": "0 17 * * 5", "enabled": True, "command": "memory"},
            {"id": "monthly_retrain", "name": "月度重训练", "kind": "cron", "expr": "0 9 1 * *", "enabled": True, "command": "retrain"},
            {"id": "monthly_iteration", "name": "月度自迭代", "kind": "cron", "expr": "0 10 1 * *", "enabled": True, "command": "iterate"},
        ]

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
                st.session_state[job_key] = result.stdout
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


if __name__ == '__main__':
    main()
