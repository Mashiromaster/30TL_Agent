# -*- coding: utf-8 -*-
# dashboard.py — TL国债期货策略 可视化交互界面
# python -m streamlit run dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))
from strategy_agent import StrategyContext
from llm_intelligence import LLMAnalyzer
from rag_tool import RAGAnalyzer
from inference import SignalGenerator
from memory import TradingMemory

st.set_page_config(
    page_title="TL策略 Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = r"D:\桌面\F_Agent"


@st.cache_data(ttl=300, show_spinner="加载数据中...")
def load_data(base_dir):
    return StrategyContext(base_dir)


@st.cache_resource
def load_signal_generator(base_dir):
    model_path = os.path.join(base_dir, "models", "trained_model.pkl")
    if not os.path.exists(model_path):
        return None
    return SignalGenerator(model_path)


@st.cache_resource
def load_rag_analyzer(base_dir):
    return RAGAnalyzer(base_dir)


def get_recent_predictions(base_dir, df_factors, days=7):
    model = load_signal_generator(base_dir)
    if model is None:
        return None
    if df_factors is None or len(df_factors) == 0:
        return None
    df = df_factors.copy()
    df['date'] = pd.to_datetime(df['date'])
    cutoff = df['date'].max() - pd.Timedelta(days=days)
    recent = df[df['date'] >= cutoff].tail(2000).copy()
    if len(recent) < 10:
        return None
    result = model.predict(recent)
    result['date'] = recent['date'].values
    result['close'] = recent['close'].values if 'close' in recent.columns else np.nan
    return result


def main():
    st.title("TL 国债期货量化策略 Dashboard")
    st.caption("30年期国债期货 LightGBM 双模型策略 · 实时监控与回测分析")

    # ---- Sidebar ----
    with st.sidebar:
        st.header("⚙️ 控制面板")
        base_dir = st.text_input("项目路径", BASE_DIR)
        st.divider()

        if st.button("🔄 刷新数据", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        # Key metrics in sidebar
        try:
            ctx = load_data(base_dir)
            signal = ctx.signal
            if signal:
                st.divider()
                st.subheader("📡 最新信号")
                d = signal.get('direction', 0)
                d_label = {1: '🟢 LONG', -1: '🔴 SHORT', 0: '⚪ FLAT'}[d]
                st.metric("信号方向", d_label)
                st.metric("置信度", f"{signal.get('confidence', 0):.1%}")
                st.metric("市场状态", signal.get('regime_name', 'N/A'))
                st.metric("合约", str(signal.get('close', '')))
                st.caption(f"更新时间: {signal.get('timestamp', 'N/A')}")
        except Exception as e:
            st.warning(f"数据加载失败: {e}")

    # ---- Main Tabs ----
    try:
        ctx = load_data(base_dir)
    except Exception as e:
        st.error(f"无法加载数据: {e}")
        return

    tabs = st.tabs([
        "📡 信号看板",
        "📊 市场监控",
        "🔬 因子分析",
        "💰 回测表现",
        "🌍 宏观环境",
        "🤖 AI情报分析",
        "📚 研究RAG",
        "🧠 交易记忆",
    ])

    with tabs[0]:
        render_signal_tab(ctx)
    with tabs[1]:
        render_market_tab(ctx)
    with tabs[2]:
        render_factor_tab(ctx)
    with tabs[3]:
        render_backtest_tab(ctx)
    with tabs[4]:
        render_macro_tab(ctx)
    with tabs[5]:
        render_intelligence_tab(ctx)
    with tabs[6]:
        render_rag_tab(ctx)
    with tabs[7]:
        render_memory_tab(ctx)


# ================================================================
# Tab 1: 信号看板
# ================================================================
def render_signal_tab(ctx):
    signal = ctx.signal
    if not signal:
        st.warning("暂无信号数据，请先运行推理模式")
        return

    st.subheader("当前交易信号")

    # Top row: 4 big metric cards
    col1, col2, col3, col4 = st.columns(4)

    direction = signal.get('direction', 0)
    d_color = {1: '#00C853', -1: '#FF1744', 0: '#9E9E9E'}[direction]
    d_label = {1: '做多 LONG', -1: '做空 SHORT', 0: '观望 FLAT'}[direction]
    d_icon = {1: '🟢', -1: '🔴', 0: '⚪'}[direction]

    with col1:
        st.markdown(f"""
        <div style="background:{d_color}22; border-left:4px solid {d_color};
                    padding:16px; border-radius:8px;">
            <h3 style="margin:0; color:{d_color};">{d_icon} {d_label}</h3>
            <p style="margin:4px 0; color:#888;">信号方向</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        conf = signal.get('confidence', 0)
        st.markdown(f"""
        <div style="background:#1E88E522; border-left:4px solid #1E88E5;
                    padding:16px; border-radius:8px;">
            <h3 style="margin:0;">{conf:.1%}</h3>
            <p style="margin:4px 0; color:#888;">置信度</p>
        </div>
        """, unsafe_allow_html=True)
        st.progress(conf)

    with col3:
        regime = signal.get('regime_name', 'N/A')
        regime_id = signal.get('market_regime', -1)
        r_color = {0: '#4CAF50', 1: '#FF9800', 2: '#9C27B0'}.get(regime_id, '#888')
        st.markdown(f"""
        <div style="background:{r_color}22; border-left:4px solid {r_color};
                    padding:16px; border-radius:8px;">
            <h3 style="margin:0;">{regime}</h3>
            <p style="margin:4px 0; color:#888;">市场状态</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        weight = signal.get('suggested_weight', 0)
        st.markdown(f"""
        <div style="background:#FFC10722; border-left:4px solid #FFC107;
                    padding:16px; border-radius:8px;">
            <h3 style="margin:0;">{weight:.1%}</h3>
            <p style="margin:4px 0; color:#888;">建议仓位</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Signal detail
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**预测详情**")
        detail_df = pd.DataFrame({
            '指标': ['原始预测收益', '平滑预测收益', '上阈值', '下阈值', '使用模型', '当前价格'],
            '值': [
                f"{signal.get('predicted_return', 0):+.4f}%",
                f"{signal.get('predicted_return_smooth', 0):+.4f}%",
                f"{signal.get('upper_threshold', 0):+.4f}%",
                f"{signal.get('lower_threshold', 0):+.4f}%",
                signal.get('model_used', 'N/A'),
                f"{signal.get('close', 'N/A')}",
            ]
        })
        st.dataframe(detail_df, hide_index=True, use_container_width=True)

    with col_b:
        st.markdown("**阈值位置示意**")
        pred_smooth = signal.get('predicted_return_smooth', 0)
        upper = signal.get('upper_threshold', 0)
        lower = signal.get('lower_threshold', 0)
        all_vals = [lower, pred_smooth, upper]
        vmin, vmax = min(all_vals) - 0.002, max(all_vals) + 0.002

        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+delta",
            value=pred_smooth,
            delta={'reference': (upper + lower) / 2, 'position': "top"},
            gauge={
                'axis': {'range': [vmin, vmax], 'tickformat': '.4f'},
                'bar': {'color': d_color},
                'steps': [
                    {'range': [vmin, lower], 'color': 'rgba(255,23,68,0.2)'},
                    {'range': [lower, upper], 'color': 'rgba(158,158,158,0.2)'},
                    {'range': [upper, vmax], 'color': 'rgba(0,200,83,0.2)'},
                ],
                'threshold': {
                    'line': {'color': d_color, 'width': 3},
                    'thickness': 0.8,
                    'value': pred_smooth,
                }
            },
            title={'text': '平滑预测收益 (%)'},
        ))
        fig.update_layout(height=280, margin=dict(l=30, r=30, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Recent prediction history (last week, model inference on recent data)
    st.divider()
    st.markdown("**近期预测走势 (近一周)**")
    df_factors = ctx.df_factors
    with st.spinner("正在运行模型推理..."):
        df_plot = get_recent_predictions(BASE_DIR, df_factors, days=7)

    if df_plot is not None and len(df_plot) > 0:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Scatter(
            x=df_plot['date'], y=df_plot['Pred_Ret'],
            mode='lines', name='Pred_Ret',
            line=dict(color='#1E88E5', width=1.5),
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=df_plot['date'],
            y=df_plot['Pred_Ret'].ewm(span=min(160, len(df_plot)//3), adjust=False).mean(),
            mode='lines', name='Pred_Smooth',
            line=dict(color='#FF9800', width=2),
        ), secondary_y=False)

        if 'close' in df_plot.columns:
            fig.add_trace(go.Scatter(
                x=df_plot['date'], y=df_plot['close'],
                mode='lines', name='Close',
                line=dict(color='#90A4AE', width=0.8),
                opacity=0.5,
            ), secondary_y=True)

        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, secondary_y=False)
        fig.update_yaxes(title_text="Predicted Return (%)", secondary_y=False, gridcolor='#333')
        fig.update_yaxes(title_text="Price", secondary_y=True, gridcolor='#333')
        fig.update_layout(
            height=350, margin=dict(l=10, r=10, t=10, b=10),
            hovermode='x', legend=dict(orientation='h', yanchor='top', y=-0.2),
            template='plotly_dark',
        )
        fig.update_xaxes(rangebreaks=[
            dict(bounds=["sat", "mon"]),
            dict(bounds=[15.25, 9.5], pattern="hour"),
            dict(bounds=[11.5, 13], pattern="hour"),
        ])
        st.plotly_chart(fig, use_container_width=True)
    elif df_factors is not None and len(df_factors) > 0:
        st.info("模型文件未找到，无法生成近期预测。请先运行训练模式生成模型。")
    else:
        st.info("无因子数据")


# ================================================================
# Tab 2: 市场监控
# ================================================================
def _resample_ohlc(df, freq='5min'):
    """Resample 1-min data to coarser OHLC bars. Drops non-trading periods (NaN OHLC)."""
    df = df.set_index('date')
    ohlc = df['close'].resample(freq).ohlc()
    if hasattr(ohlc.columns, 'levels'):
        ohlc.columns = ohlc.columns.droplevel(0)
    vol = df['volume'].resample(freq).sum()
    result = ohlc.join(vol.rename('volume'))
    for col in ['Market_Regime']:
        if col in df.columns:
            result[col] = df[col].resample(freq).last()
    result = result.dropna(subset=['open', 'high', 'low', 'close']).reset_index()
    return result


def render_market_tab(ctx):
    df = ctx.df_factors
    if df is None or len(df) == 0:
        st.warning("无因子数据")
        return

    st.subheader("主力合约行情")

    df['date'] = pd.to_datetime(df['date'])
    date_min = df['date'].min().date()
    date_max = df['date'].max().date()

    col1, col2, col3 = st.columns(3)
    with col1:
        lookback = st.selectbox(
            "回看周期",
            ["最近1周", "最近1月", "最近2月", "最近3月", "最近半年", "全部"],
            index=2
        )
    with col2:
        resolution = st.selectbox(
            "K线分辨率",
            ["1天", "1小时", "30分钟", "15分钟", "5分钟", "1分钟"],
            index=0
        )
    with col3:
        ticker_filter = st.selectbox(
            "合约筛选",
            ["全部"] + sorted(df['ticker'].dropna().unique().tolist())
        )

    resample_map = {"1天": "1D", "1小时": "1h", "30分钟": "30min", "15分钟": "15min", "5分钟": "5min", "1分钟": None}
    days_map = {"最近1周": 7, "最近1月": 30, "最近2月": 60, "最近3月": 90, "最近半年": 180, "全部": 9999}
    cutoff = pd.Timestamp(date_max) - pd.Timedelta(days=days_map[lookback])

    df_plot = df[df['date'] >= cutoff].copy()
    if ticker_filter != "全部":
        df_plot = df_plot[df_plot['ticker'] == ticker_filter]

    # Resample if needed
    freq = resample_map[resolution]
    if freq:
        df_plot = _resample_ohlc(df_plot, freq)

    # Build adaptive rangebreaks based on resolution
    if freq == '1D':
        rangebreaks = [dict(bounds=["sat", "mon"])]
        gap_threshold = pd.Timedelta(days=2)
    else:
        rangebreaks = [
            dict(bounds=["sat", "mon"]),
            dict(bounds=[15.25, 9.5], pattern="hour"),
            dict(bounds=[11.5, 13], pattern="hour"),
        ]
        gap_threshold = pd.Timedelta(hours=4)

    # Auto-detect large gaps (holidays) and hide them
    dates = df_plot['date'].sort_values().values
    gaps = np.diff(dates)
    gap_mask = gaps > gap_threshold
    gap_starts = dates[:-1][gap_mask]
    gap_ends = dates[1:][gap_mask]
    for s, e in zip(gap_starts, gap_ends):
        rangebreaks.append(dict(
            bounds=[s + pd.Timedelta(minutes=1), e - pd.Timedelta(minutes=1)]
        ))

    with st.spinner("渲染图表中..."):
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.65, 0.35],
            vertical_spacing=0.03,
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df_plot['date'], open=df_plot['open'], high=df_plot['high'],
            low=df_plot['low'], close=df_plot['close'],
            name='OHLC',
        ), row=1, col=1)

        # Regime shading
        if 'Market_Regime' in df_plot.columns:
            colors = {0: 'rgba(76,175,80,0.05)', 1: 'rgba(255,152,0,0.1)', 2: 'rgba(156,39,176,0.1)'}
            for rid, rcolor in colors.items():
                mask = df_plot['Market_Regime'] == rid
                if mask.any():
                    blocks = _find_continuous_blocks(df_plot, mask)
                    for start, end in blocks:
                        fig.add_vrect(
                            x0=df_plot['date'].iloc[start], x1=df_plot['date'].iloc[end],
                            fillcolor=rcolor, layer="below", line_width=0, row=1, col=1,
                        )

        # Volume
        vol_colors = ['#00C853' if c >= o else '#FF1744'
                      for c, o in zip(df_plot['close'], df_plot['open'])]
        fig.add_trace(go.Bar(
            x=df_plot['date'], y=df_plot['volume'], name='成交量',
            marker_color=vol_colors, opacity=0.6,
        ), row=2, col=1)

    fig.update_layout(
        height=500, margin=dict(l=10, r=10, t=10, b=10),
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        legend=dict(orientation='h', yanchor='top', y=-0.05),
    )
    fig.update_xaxes(rangebreaks=rangebreaks, row=1, col=1)
    fig.update_xaxes(rangebreaks=rangebreaks, row=2, col=1)
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Regime distribution
    if 'Market_Regime' in df.columns:
        st.divider()
        st.markdown("**市场状态分布**")
        regime_counts = df['Market_Regime'].value_counts().sort_index()
        regime_labels = {0: '正常', 1: '高波动', 2: '趋势'}
        pie_data = pd.DataFrame({
            '状态': [regime_labels.get(i, str(i)) for i in regime_counts.index],
            '样本数': regime_counts.values,
            '占比': (regime_counts.values / len(df) * 100).round(1),
        })
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.dataframe(pie_data, hide_index=True, use_container_width=True)
        with col_b:
            fig = px.pie(pie_data, values='样本数', names='状态',
                         color='状态',
                         color_discrete_map={'正常': '#4CAF50', '高波动': '#FF9800', '趋势': '#9C27B0'})
            fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)


def _find_continuous_blocks(df, mask):
    """Find contiguous True blocks in a boolean mask."""
    blocks = []
    in_block = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_block:
            start = i
            in_block = True
        elif not v and in_block:
            blocks.append((start, i - 1))
            in_block = False
    if in_block:
        blocks.append((start, len(mask) - 1))
    return blocks


# ================================================================
# Tab 3: 因子分析
# ================================================================
def render_factor_tab(ctx):
    st.subheader("因子分析")

    imp = ctx.importance
    df = ctx.df_factors

    if imp is None or df is None:
        st.warning("无因子数据")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**特征重要性 (Top 15)**")
        top15 = imp[imp['importance'] > 0].head(15)
        fig = px.bar(
            top15.iloc[::-1], x='importance', y='feature',
            orientation='h',
            color='importance',
            color_continuous_scale='Blues',
        )
        fig.update_layout(
            height=450, margin=dict(l=10, r=10, t=10, b=10),
            yaxis={'categoryorder': 'total ascending'},
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**当前因子值 vs 历史分位**")
        # Select top 10 important available factors
        top10 = imp[imp['importance'] > 0].head(10)['feature'].tolist()
        available_features = [f for f in top10 if f in df.columns]

        pct_data = []
        latest = df.iloc[-1]
        for feat in available_features:
            series = df[feat].dropna()
            if len(series) < 100:
                continue
            cur = latest[feat]
            pct = (series < cur).mean() * 100
            mean_val = series.mean()
            pct_data.append({
                '因子': feat,
                '当前值': round(cur, 4),
                '分位': round(pct, 1),
                '均值': round(mean_val, 4),
            })

        if pct_data:
            pct_df = pd.DataFrame(pct_data)
            # Color code extreme percentiles
            fig = go.Figure()
            colors = [
                '#FF1744' if p > 90 or p < 10 else
                '#FF9800' if p > 75 or p < 25 else '#4CAF50'
                for p in pct_df['分位']
            ]
            fig.add_trace(go.Bar(
                y=pct_df['因子'], x=pct_df['分位'],
                orientation='h', marker_color=colors,
                text=pct_df['分位'].apply(lambda x: f'{x:.0f}%'),
                textposition='outside',
            ))
            fig.add_vline(x=50, line_dash="dot", line_color="gray", opacity=0.3)
            fig.add_vline(x=25, line_dash="dash", line_color="orange", opacity=0.2)
            fig.add_vline(x=75, line_dash="dash", line_color="orange", opacity=0.2)
            fig.update_layout(
                height=450, margin=dict(l=10, r=40, t=10, b=10),
                xaxis=dict(title='历史分位 (%)', range=[-5, 105]),
                yaxis={'categoryorder': 'total ascending'},
            )
            st.plotly_chart(fig, use_container_width=True)

    # Anomaly flags
    st.divider()
    st.markdown("**异常因子预警**")
    all_alerts = []
    for feat in available_features:
        series = df[feat].dropna()
        if len(series) < 100:
            continue
        cur = latest[feat]
        pct = (series < cur).mean() * 100
        if pct > 90:
            all_alerts.append((feat, '极高位', pct, '#FF1744'))
        elif pct > 75:
            all_alerts.append((feat, '偏高', pct, '#FF9800'))
        elif pct < 10:
            all_alerts.append((feat, '极低位', pct, '#FF1744'))
        elif pct < 25:
            all_alerts.append((feat, '偏低', pct, '#FF9800'))

    if all_alerts:
        cols = st.columns(min(len(all_alerts), 4))
        for i, (feat, status, pct, color) in enumerate(all_alerts):
            with cols[i % 4]:
                st.markdown(f"""
                <div style="background:{color}22; border-left:4px solid {color};
                            padding:10px; border-radius:6px; margin:4px 0;">
                    <b>{feat}</b><br/>
                    <span style="color:{color};">{status} ({pct:.0f}%分位)</span>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.success("所有核心因子均在正常范围")


# ================================================================
# Tab 4: 回测表现
# ================================================================
def render_backtest_tab(ctx):
    st.subheader("策略回测表现")

    metrics = ctx.metrics
    df_pred = ctx.df_pred

    if not metrics:
        st.warning("无回测数据，请先运行训练模式")
        return

    # Metrics cards row
    key_metrics = [
        ('累计收益', '累计收益', ''),
        ('年化收益', '年化收益', ''),
        ('夏普比率', '夏普比率', ''),
        ('Calmar比率', 'Calmar比率', ''),
        ('最大回撤', '最大回撤', ''),
        ('日胜率', '日胜率', ''),
        ('持仓比例', '持仓比例', ''),
        ('年化换手', '年化换手', ''),
    ]
    cols = st.columns(len(key_metrics))
    for i, (label, key, _) in enumerate(key_metrics):
        val = metrics.get(key, 'N/A')
        with cols[i]:
            st.metric(label=label, value=val)

    st.divider()

    # NAV curve
    if df_pred is not None and len(df_pred) > 0:
        col_l, col_r = st.columns([2, 1])

        with col_l:
            st.markdown("**累计净值 & 回撤**")

            # Build daily returns from predictions (simplified NAV)
            df_daily = df_pred.groupby(pd.to_datetime(df_pred['date']).dt.date).agg({
                'close': 'last',
                'Target_Ret': 'sum',
                'Pred_Ret': 'sum',
            }).reset_index()
            df_daily.columns = ['date', 'close', 'target_ret', 'pred_ret']

            # Simple P&L: sign(Pred_Ret) * Target_Ret
            df_daily['strategy_ret'] = np.sign(df_daily['pred_ret']) * df_daily['target_ret']
            df_daily['nav'] = (1 + df_daily['strategy_ret'] / 100).cumprod()
            df_daily['peak'] = df_daily['nav'].cummax()
            df_daily['dd'] = (df_daily['nav'] - df_daily['peak']) / df_daily['peak'] * 100

            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                row_heights=[0.6, 0.4], vertical_spacing=0.05,
            )
            fig.add_trace(go.Scatter(
                x=df_daily['date'], y=df_daily['nav'],
                mode='lines', name='NAV',
                line=dict(color='#1E5D3A', width=2),
                fill='tozeroy', fillcolor='rgba(30,93,58,0.1)',
            ), row=1, col=1)
            fig.add_hline(y=1, line_dash="dot", line_color="gray", row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df_daily['date'], y=df_daily['dd'],
                mode='lines', name='回撤',
                line=dict(color='#FF1744', width=1.5),
                fill='tozeroy', fillcolor='rgba(255,23,68,0.15)',
            ), row=2, col=1)
            fig.update_layout(
                height=450, margin=dict(l=10, r=10, t=10, b=10),
                hovermode='x unified', showlegend=False,
            )
            fig.update_yaxes(title_text="NAV", row=1, col=1)
            fig.update_yaxes(title_text="回撤 %", row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown("**日收益分布**")
            pos_ret = df_daily[df_daily['strategy_ret'] > 0]['strategy_ret']
            neg_ret = df_daily[df_daily['strategy_ret'] < 0]['strategy_ret']

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=pos_ret, nbinsx=30, name='盈利',
                marker_color='#00C853', opacity=0.7,
            ))
            fig.add_trace(go.Histogram(
                x=neg_ret, nbinsx=30, name='亏损',
                marker_color='#FF1744', opacity=0.7,
            ))
            fig.update_layout(
                height=450, margin=dict(l=10, r=10, t=10, b=10),
                barmode='overlay', legend=dict(orientation='h'),
                xaxis_title='日收益 (%)',
            )
            st.plotly_chart(fig, use_container_width=True)


# ================================================================
# Tab 5: 宏观环境
# ================================================================
def render_macro_tab(ctx):
    st.subheader("宏观环境分析")

    df = ctx.df_factors
    mf = ctx.macro_factors

    if df is None:
        st.warning("无因子数据")
        return

    latest = df.iloc[-1]

    # Row 1: Yield curve + CN-US spread
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**收益率曲线形态**")
        maturity_labels = ['1Y', '3Y', '5Y', '7Y', '10Y', '30Y']
        curve_cols = ['Yield_1Y', 'Yield_3Y', 'Yield_5Y', 'Yield_7Y', 'Yield_10Y', 'Yield_30Y']
        available_curve = [(lab, col) for lab, col in zip(maturity_labels, curve_cols) if col in df.columns]

        if available_curve:
            # Current yield curve
            current_yields = [latest[col] for _, col in available_curve]
            labs = [lab for lab, _ in available_curve]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=labs, y=current_yields,
                mode='lines+markers', name='当前',
                line=dict(color='#1E88E5', width=3),
                marker=dict(size=10),
            ))

            # 1 month ago (if available)
            if len(df) > 5000:
                past = df.iloc[-5000]
                past_yields = [past[col] for _, col in available_curve]
                fig.add_trace(go.Scatter(
                    x=labs, y=past_yields,
                    mode='lines+markers', name='约1月前',
                    line=dict(color='#9E9E9E', width=1.5, dash='dash'),
                    marker=dict(size=6),
                ))

            fig.update_layout(
                height=350, margin=dict(l=10, r=10, t=10, b=10),
                hovermode='x', xaxis_title='期限',
                yaxis_title='收益率 (%)',
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("无收益率曲线数据")

    with col2:
        st.markdown("**宏观指标趋势**")
        macro_cols = ['PMI_ZScore', 'CPI_Momentum', 'M2_Surprise', 'Macro_Surprise_Composite']
        available_macro = [c for c in macro_cols if c in df.columns]

        if available_macro:
            # Sample daily (take last value per day)
            df_daily_sample = df.set_index('date').resample('D')[available_macro].last().dropna(how='all')
            df_tail = df_daily_sample.tail(180)

            fig = go.Figure()
            colors = ['#1E88E5', '#FF9800', '#4CAF50', '#9C27B0']
            for col, color in zip(available_macro, colors):
                fig.add_trace(go.Scatter(
                    x=df_tail.index, y=df_tail[col],
                    mode='lines', name=col,
                    line=dict(color=color, width=1.5),
                ))
            fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
            fig.update_layout(
                height=350, margin=dict(l=10, r=10, t=10, b=10),
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='top', y=-0.15),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("无宏观指标数据")

    # Row 2: CN-US spread timeline + Key macro cards
    col3, col4 = st.columns([2, 1])

    with col3:
        st.markdown("**中美利差历史走势**")
        if 'CN_US_10Y_Spread' in df.columns:
            df_daily_spread = df.set_index('date').resample('D')['CN_US_10Y_Spread'].last().dropna()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_daily_spread.index, y=df_daily_spread.values,
                mode='lines', name='中美10Y利差',
                line=dict(color='#1E88E5', width=2),
                fill='tozeroy', fillcolor='rgba(30,136,229,0.1)',
            ))
            # Color positive/negative regions
            fig.add_hrect(y0=0, y1=df_daily_spread.max() * 1.1,
                          fillcolor='rgba(0,200,83,0.05)', line_width=0)
            fig.add_hrect(y0=df_daily_spread.min() * 1.1, y1=0,
                          fillcolor='rgba(255,23,68,0.05)', line_width=0)
            fig.add_hline(y=0, line_color="gray", line_dash="dot")
            fig.update_layout(
                height=350, margin=dict(l=10, r=10, t=10, b=10),
                hovermode='x', yaxis_title='利差 (%)',
            )
            st.plotly_chart(fig, use_container_width=True)
        elif mf is not None and len(mf) > 0:
            st.info("因子中无中美利差数据，请检查宏观因子提取")

    with col4:
        st.markdown("**宏观快照**")
        macro_items = [
            ('PMI_ZScore', 'PMI Z-Score', '+ 扩张, - 收缩'),
            ('CPI_Momentum', 'CPI 动量', '+ 通胀, - 通缩'),
            ('M2_Surprise', 'M2 超预期', '+ 宽松, - 收紧'),
            ('Macro_Surprise_Composite', '宏观综合', '+ 向好, - 偏弱'),
        ]
        for col_name, label, hint in macro_items:
            if col_name in df.columns:
                cur = latest[col_name]
                series = df[col_name].dropna()
                pct = (series < cur).mean() * 100 if len(series) > 0 else 50

                if cur > 0.5:
                    color = '#00C853'
                elif cur < -0.5:
                    color = '#FF1744'
                else:
                    color = '#FFC107'

                st.markdown(f"""
                <div style="background:{color}11; border-left:4px solid {color};
                            padding:10px; border-radius:6px; margin:6px 0;">
                    <b>{label}</b>: {cur:+.4f} (分位 {pct:.0f}%)<br/>
                    <small style="color:#888;">{hint}</small>
                </div>
                """, unsafe_allow_html=True)

    # Macro factor data coverage
    if mf is not None and len(mf) > 0:
        st.divider()
        st.caption(f"宏观因子历史: {mf['available_date'].min()} ~ {mf['available_date'].max()}  "
                   f"({len(mf):,} 行)")


# ================================================================
# Tab 6: AI 市场情报分析
# ================================================================
def render_intelligence_tab(ctx):
    st.subheader("AI 市场情报分析")
    st.caption("DeepSeek V4 结合量化数据与市场新闻，生成结构化情报报告")

    col1, col2 = st.columns([1, 3])
    with col1:
        run_analysis = st.button(
            "生成AI分析报告",
            type="primary",
            use_container_width=True,
        )

    if not run_analysis:
        st.info("点击上方按钮，AI将综合量化数据与最新市场新闻生成分析报告")
        with st.expander("将被纳入分析的数据"):
            signal = ctx.signal
            if signal:
                st.markdown(f"- **当前信号**: {signal.get('direction_name', 'N/A')}, 置信度 {signal.get('confidence', 0):.1%}")
                st.markdown(f"- **市场状态**: {signal.get('regime_name', 'N/A')}")
            st.markdown(f"- **因子数据**: {len(ctx.df_factors) if ctx.df_factors is not None else 0} 行")
            st.markdown(f"- **回测指标**: {'可用' if ctx.metrics else '不可用'}")
        return

    # Check API key
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        st.error("""
        **未设置 `DEEPSEEK_API_KEY` 环境变量**

        **设置方法：**
        ```bash
        set DEEPSEEK_API_KEY=sk-your-key-here
        ```
        或在系统环境变量中添加后重启 Dashboard。
        """)
        return

    with st.spinner("AI正在分析市场数据..."):
        try:
            cache_dir = os.path.join(BASE_DIR, "data", "macro")
            llm = LLMAnalyzer(ctx, cache_dir=cache_dir)

            with st.status("分析进度", expanded=True) as status:
                st.write("获取最新债券市场新闻...")
                news_context = llm._fetch_news_context()
                st.write("   - 新闻数据已获取")

                st.write("整理量化数据上下文...")
                quant_context = llm._build_quantitative_context()
                st.write("   - 量化数据已整理")

                st.write("调用 DeepSeek V4 模型...")
                report = llm.generate_intelligence_report(save=True)
                st.write("   - 分析完成")

                status.update(label="分析完成", state="complete", expanded=False)

            st.success("AI 分析报告已生成")
            st.markdown(report)

        except Exception as e:
            import traceback
            st.error(f"AI 分析失败: {e}")
            with st.expander("错误详情"):
                st.code(traceback.format_exc())
            st.info("请检查网络连接和 API Key 配置后重试")


# ================================================================
# Tab 7: 研究 RAG — 研报知识库检索问答
# ================================================================
def render_rag_tab(ctx):
    st.subheader("研究知识库检索 (RAG)")
    st.caption("检索央行货政报告、中金所月报、券商研报、债券新闻 → AI 生成答案")

    rags = load_rag_analyzer(BASE_DIR)

    # ---- Status bar ----
    stats = rags.vector_store.get_stats()
    has_key = bool(rags.api_key)

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.metric("索引文档块", stats['total_chunks'])
    with col_s2:
        st.metric("LLM 模式", "DeepSeek V4" if has_key else "离线检索")
    with col_s3:
        st.metric("Embedding", rags.vector_store.embedding_model_name.split('/')[-1])

    st.divider()

    # ---- Build index section ----
    with st.expander("索引管理", expanded=(stats['total_chunks'] == 0)):
        st.markdown("**数据源**: 央行货政报告 (季度) · 中金所月报 · 新浪债券研报 · 本地债券新闻")

        col_b1, col_b2 = st.columns([1, 3])
        with col_b1:
            do_rebuild = st.button("重建索引", type="secondary",
                                   help="重新爬取所有研究报告并重建向量索引")
        with col_b2:
            if do_rebuild:
                with st.spinner("正在爬取研究报告并构建索引... 约需 1-2 分钟"):
                    new_stats = rags.build_index(force_refresh=True)
                st.success(f"索引构建完成: {new_stats['total_chunks']} 个文本块")
                st.rerun()

        # Show available filter options
        st.caption("文档过滤选项 (问题框下方可选)")

    st.divider()

    # ---- Query interface ----
    st.markdown("### 提出问题")

    # Preset questions
    preset_questions = [
        "当前货币政策立场和未来走向如何？降准降息空间还有多大？",
        "国债期货市场近期运行情况如何？多头还是空头占优？",
        "当前债券市场面临的主要风险是什么？",
        "中美利差处于什么水平，对国内债市有什么影响？",
        "央行对长端利率的态度如何？",
    ]

    # Use separate keys: rag_current_q = authoritative value, rag_q_input = widget binding
    if 'rag_current_q' not in st.session_state:
        st.session_state.rag_current_q = ''

    def _sync_rag_q():
        st.session_state.rag_current_q = st.session_state.rag_q_input

    # Input area
    col_q1, col_q2 = st.columns([3, 1])
    with col_q1:
        st.text_input(
            "输入你的研究问题",
            value=st.session_state.rag_current_q,
            placeholder="例如: 当前货币政策立场如何？降息空间还有多大？",
            key="rag_q_input",
            on_change=_sync_rag_q,
        )
    with col_q2:
        top_k = st.selectbox("检索数量", [3, 5, 8, 10], index=1)

    # Preset question chips
    st.caption("快速提问:")
    q_cols = st.columns(len(preset_questions))
    for i, q in enumerate(preset_questions):
        with q_cols[i]:
            if st.button(q[:20] + "...", key=f"preset_{i}", help=q, use_container_width=True):
                st.session_state.rag_current_q = q
                st.rerun()

    question = st.session_state.rag_current_q

    # Filter by document type
    filter_options = {
        "全部": None,
        "货币政策报告": 'monetary_policy_report',
        "中金所月报": 'cffex_monthly',
        "券商研报": 'research_report',
        "债券新闻": 'news',
    }
    filter_choice = st.selectbox("文档类型过滤", list(filter_options.keys()), index=0)

    st.divider()

    if question:
        with st.spinner("正在检索研究报告并生成回答..."):
            filter_dict = {'doc_type': filter_options[filter_choice]} if filter_options[filter_choice] else None
            result = rags.query(question, top_k=top_k, filter_dict=filter_dict)

        # Answer display
        st.markdown("### 回答")
        st.markdown(result['answer'])

        # Sources
        st.divider()
        st.markdown("**参考来源**")
        if result['sources']:
            for i, src in enumerate(result['sources']):
                with st.expander(f"{src['source']} — {src['title'][:60]}"):
                    st.markdown(f"- **类型**: {src['doc_type']}")
                    st.markdown(f"- **日期**: {src['date']}")
        else:
            st.caption("(无参考来源)")

        # Show raw context in expander (debug)
        with st.expander("查看检索上下文"):
            st.text(result.get('context', '(无上下文)')[:3000])


# ================================================================
# Tab 8: 交易记忆
# ================================================================
def render_memory_tab(ctx):
    st.subheader("交易记忆系统")
    st.caption("记录每日交易决策、预测 vs 实际、归因反思")

    mem = TradingMemory(BASE_DIR)
    records = mem._load_all()

    if not records:
        st.warning("记忆系统尚未初始化，点击下方按钮从历史数据回填。")
        if st.button("从历史数据初始化记忆", use_container_width=True):
            with st.spinner("从 df_predictions.pkl 重建记忆记录..."):
                n = mem.backfill_from_predictions()
                st.success(f"已初始化 {n} 条历史记录")
                st.rerun()
        return

    stats = mem.reflection_stats()
    if 'error' in stats:
        st.warning(f"统计不可用: {stats['error']}")
        return

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("总记录数", stats['total_records'])
    with col2:
        st.metric("整体准确率", f"{stats['overall_accuracy']:.1%}")
    with col3:
        st.metric("最近10天", f"{stats['recent_10_accuracy']:.1%}")
    with col4:
        st.metric("当前连胜", f"{stats['current_win_streak']} 天")
    with col5:
        st.metric("最大连胜", f"{stats['max_win_streak']} 天")

    st.divider()

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("准确率 × 市场状态")
        by_regime = stats.get('by_regime', {})
        if by_regime:
            regimes = list(by_regime.keys())
            accs = [by_regime[r]['accuracy'] for r in regimes]
            fig = go.Figure(data=[
                go.Bar(x=regimes, y=accs, text=[f"{a:.1%}" for a in accs],
                       textposition='auto', marker_color=['#636efa', '#ef553b', '#00cc96'])
            ])
            fig.update_layout(yaxis_tickformat='.0%', yaxis_title='准确率',
                              height=300, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        st.subheader("准确率 × 方向")
        by_dir = stats.get('by_direction', {})
        if by_dir:
            dirs = list(by_dir.keys())
            accs_d = [by_dir[d]['accuracy'] for d in dirs]
            fig = go.Figure(data=[
                go.Bar(x=dirs, y=accs_d, text=[f"{a:.1%}" for a in accs_d],
                       textposition='auto', marker_color=['#00cc96', '#ef553b'])
            ])
            fig.update_layout(yaxis_tickformat='.0%', yaxis_title='准确率',
                              height=300, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("状态 × 方向 准确率矩阵")
    cross = stats.get('by_regime_direction', {})
    if cross:
        rows = []
        for regime, dirs in cross.items():
            row = {'市场状态': regime}
            for dname in ['做多', '做空']:
                if dname in dirs:
                    row[dname] = f"{dirs[dname]['accuracy']:.1%} (n={dirs[dname]['count']})"
                else:
                    row[dname] = '-'
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("预测 vs 实际收益 (最近30天)")
    recent_30 = records[-30:]
    if recent_30:
        dates = [r['trade_dt'] for r in recent_30]
        preds = [r['predicted_return_smooth'] for r in recent_30]
        actuals = [r.get('actual_return') for r in recent_30]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=dates, y=preds, mode='lines+markers',
                                  name='预测收益', line=dict(color='#636efa')))
        fig2.add_trace(go.Scatter(x=dates, y=actuals, mode='lines+markers',
                                  name='实际收益', line=dict(color='#ef553b')))
        fig2.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5)
        fig2.update_layout(height=350, margin=dict(t=10, b=10),
                           yaxis_title='收益率(%)', hovermode='x unified')
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("最近 20 条记忆记录")
    recent_20 = records[-20:][::-1]
    table_data = []
    for r in recent_20:
        is_correct = r.get('is_correct')
        if is_correct is True:
            status = '正确'
        elif is_correct is False:
            status = '错误'
        else:
            status = '—'
        table_data.append({
            '日期': r['trade_dt'],
            '状态': r['regime_name'],
            '方向': r['direction_name'],
            '置信度': f"{r['confidence']:.1%}",
            '预测收益': f"{r['predicted_return_smooth']:.4f}%",
            '实际收益': f"{r.get('actual_return', 'N/A')}",
            '结果': status,
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("LLM 归因反思")
    st.caption("基于最近 20 条交易记录，DeepSeek 分析失败模式和改善建议")

    if st.button("执行归因反思", use_container_width=True, type="primary"):
        with st.spinner("DeepSeek 分析中..."):
            reflection = mem.llm_reflection()
            st.markdown(reflection)


if __name__ == '__main__':
    main()
