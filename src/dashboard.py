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

sys.path.insert(0, os.path.dirname(__file__))
from strategy_agent import StrategyContext
from llm_intelligence import LLMAnalyzer

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

    # Recent prediction history
    st.divider()
    st.markdown("**近期预测走势 (测试集)**")
    df_pred = ctx.df_pred
    if df_pred is not None and len(df_pred) > 0 and 'Pred_Ret' in df_pred.columns:
        df_plot = df_pred.tail(500).copy()
        df_plot['date'] = pd.to_datetime(df_plot['date'])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_plot['date'], y=df_plot['Pred_Ret'],
            mode='lines', name='Pred_Ret',
            line=dict(color='#1E88E5', width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=df_plot['date'],
            y=df_plot['Pred_Ret'].ewm(span=160, adjust=False).mean(),
            mode='lines', name='Pred_Smooth',
            line=dict(color='#FF9800', width=2),
        ))
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
        fig.update_layout(
            height=300, margin=dict(l=10, r=10, t=10, b=10),
            hovermode='x', legend=dict(orientation='h', yanchor='top', y=-0.1),
        )
        fig.update_xaxes(rangebreaks=[
            dict(bounds=["sat", "mon"]),
            dict(bounds=[15.25, 9.5], pattern="hour"),
            dict(bounds=[11.5, 13], pattern="hour"),
        ])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("无预测历史数据")


# ================================================================
# Tab 2: 市场监控
# ================================================================
def render_market_tab(ctx):
    df = ctx.df_factors
    if df is None or len(df) == 0:
        st.warning("无因子数据")
        return

    st.subheader("主力合约行情")

    # Date range selector
    df['date'] = pd.to_datetime(df['date'])
    date_min = df['date'].min().date()
    date_max = df['date'].max().date()

    col1, col2 = st.columns(2)
    with col1:
        lookback = st.selectbox(
            "回看周期",
            ["最近1周", "最近1月", "最近3月", "最近半年", "全部"],
            index=2
        )
    with col2:
        ticker_filter = st.selectbox(
            "合约筛选",
            ["全部"] + sorted(df['ticker'].dropna().unique().tolist())
        )

    days_map = {"最近1周": 7, "最近1月": 30, "最近3月": 90, "最近半年": 180, "全部": 9999}
    cutoff = pd.Timestamp(date_max) - pd.Timedelta(days=days_map[lookback])

    df_plot = df[df['date'] >= cutoff].copy()
    if ticker_filter != "全部":
        df_plot = df_plot[df_plot['ticker'] == ticker_filter]

    # Price chart with regime background
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],
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

    # RV_30 (realized volatility)
    if 'RV_30' in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_plot['date'], y=df_plot['RV_30'].clip(upper=30),
            mode='lines', name='RV_30 (波动率)',
            line=dict(color='#FF9800', width=1.5),
        ), row=3, col=1)

    # 交易时段 rangebreaks (隐藏非交易时间和周末)
    TRADING_RANGEBREAKS = [
        dict(bounds=["sat", "mon"]),               # 隐藏周末
        dict(bounds=[15.25, 9.5], pattern="hour"), # 隐藏 15:15-09:30
        dict(bounds=[11.5, 13], pattern="hour"),   # 隐藏午休 11:30-13:00
    ]

    fig.update_layout(
        height=600, margin=dict(l=10, r=10, t=10, b=10),
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        legend=dict(orientation='h', yanchor='top', y=-0.05),
    )
    fig.update_xaxes(rangebreaks=TRADING_RANGEBREAKS, row=1, col=1)
    fig.update_xaxes(rangebreaks=TRADING_RANGEBREAKS, row=2, col=1)
    fig.update_xaxes(rangebreaks=TRADING_RANGEBREAKS, row=3, col=1)
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    fig.update_yaxes(title_text="RV_30", row=3, col=1)

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


if __name__ == '__main__':
    main()
