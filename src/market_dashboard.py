"""Market UI with explicit series identity and independent daily/minute sources."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from market_data import (
    VARIETY_CONFIG,
    contract_candidates,
    load_quote_series,
    resample_ohlc,
)


@st.cache_data(ttl=60, show_spinner=False)
def _load_snapshot(base_dir, symbol, frequency, refresh=True):
    return load_quote_series(base_dir, symbol, frequency, refresh=refresh)


def minute_quality_warnings(minutes, daily):
    """Compare observed windows, not a guessed weekday/holiday calendar."""
    if daily is None or daily.empty:
        return ["没有同一合约的日线参考，无法核对分钟数据的日期覆盖和成交量。"]
    expected = daily.copy()
    expected["day"] = pd.to_datetime(expected["date"]).dt.normalize()
    expected = expected.drop_duplicates("day", keep="last").set_index("day")
    if minutes is None or minutes.empty:
        return [f"所选日线区间有 {len(expected)} 个有行情日期，但没有对应分钟记录。"]
    actual = minutes.copy()
    actual["day"] = pd.to_datetime(actual["date"]).dt.normalize()
    observed = actual.groupby("day").agg(
        volume=("volume", "sum"), high=("high", "max"), low=("low", "min")
    )
    messages = []
    missing = expected.index.difference(observed.index)
    if len(missing):
        examples = "、".join(day.strftime("%Y-%m-%d") for day in missing[:4])
        messages.append(
            f"分钟覆盖不足：同合约日线存在、分钟数据未返回的日期共 {len(missing)} 个"
            f"（如 {examples}）。近期分钟接口不能补齐这些历史，未用其他合约填充。"
        )
    unreferenced = observed.index.difference(expected.index)
    if len(unreferenced):
        examples = "、".join(day.strftime("%Y-%m-%d") for day in unreferenced[:4])
        messages.append(
            f"日线参考缺少 {len(unreferenced)} 个已返回分钟行情的日期（如 {examples}），"
            "可能参考数据滞后；保留这些分钟行情，但无法核对完整性。"
        )
    common = expected.index.intersection(observed.index)
    if len(common):
        wrong_volume = ~np.isclose(
            observed.loc[common, "volume"], expected.loc[common, "volume"],
            rtol=0, atol=1e-6,
        )
        if wrong_volume.any():
            messages.append(
                f"有 {int(wrong_volume.sum())} 日的分钟成交量合计与同源日线不一致："
                "可能是日内未完成、分钟缺失或上游口径差异，不能当作完整日线。"
            )
        wrong_range = (
            ~np.isclose(observed.loc[common, "high"], expected.loc[common, "high"], rtol=0, atol=1e-6)
            | ~np.isclose(observed.loc[common, "low"], expected.loc[common, "low"], rtol=0, atol=1e-6)
        )
        if wrong_range.any():
            messages.append(f"有 {int(wrong_range.sum())} 日的分钟高低点未与日线对齐，分钟窗口可能不完整。")
    return messages


def _window_bounds(frame, lookback, specific_date):
    days = pd.DatetimeIndex(pd.to_datetime(frame["date"]).dt.normalize().unique()).sort_values()
    if specific_date is not None:
        target = pd.Timestamp(specific_date)
        idx = int(abs(days - target).argmin())
        if target not in days:
            st.info(
                f"所选日期 {specific_date} 无返回记录（可能休市或数据缺失），"
                f"定位到最近有记录日期 {days[idx].date()}。"
            )
        count = st.slider("前后有记录日期数", 3, 60, 15, key="market_window_days")
        return days[max(0, idx - count)], days[min(len(days) - 1, idx + count)]
    length = {"最近1周": 7, "最近1月": 30, "最近2月": 60, "最近3月": 90, "最近半年": 180}
    return (days[0] if lookback == "全部" else days[-1] - pd.Timedelta(days=length[lookback] - 1)), days[-1]


def _in_window(frame, start, end):
    if frame is None or frame.empty:
        return pd.DataFrame()
    dates = pd.to_datetime(frame["date"]).dt.normalize()
    return frame.loc[(dates >= start) & (dates <= end)].copy()


def render_market_tab(ctx):
    variety = st.session_state.get("selected_variety", "TL")
    cfg = VARIETY_CONFIG.get(variety, VARIETY_CONFIG["TL"])
    st.subheader(f"{cfg['label']} 行情 · 主力连续 / 具体合约")
    cols = st.columns(4)
    with cols[0]:
        lookback = st.selectbox("回看周期", ["最近1周", "最近1月", "最近2月", "最近3月", "最近半年", "全部"], index=3, key="market_lookback")
    with cols[1]:
        resolution = st.selectbox("K线分辨率", ["1天", "1小时", "30分钟", "15分钟", "5分钟", "1分钟"], key="market_resolution")
    with cols[2]:
        symbol = st.selectbox(
            "行情合约", [cfg["main_symbol"]] + contract_candidates(variety),
            format_func=lambda value: f"{value} · 主力连续" if value.endswith("0") else value,
            key=f"market_symbol_{variety}",
        )
    with cols[3]:
        specific_date = st.date_input("跳转到特定日期", value=None, format="YYYY-MM-DD", key="market_specific_date")
    with st.expander("数据来源与历史合约"):
        custom = st.text_input("历史合约代码（可选，填写后覆盖上方选择）", placeholder=f"例如 {variety}2609", key=f"market_custom_{variety}").strip().upper()
        st.caption("合约列表是按季月生成的候选，是否上市及有数据以接口返回为准。日线和分钟线分别获取、分别缓存；不拼接旧文件中的混合合约数据。")
        st.caption("公开新浪行情仅供研究，可能延迟或修订；分钟缺口不会用前值、零成交量或其他合约补造。")
    if custom:
        if not re.fullmatch(rf"{re.escape(variety)}\d{{2}}(?:03|06|09|12)", custom):
            st.error(f"请输入同品种季月合约代码（03/06/09/12），例如 {variety}2609。")
            return
        symbol = custom
    controls = st.columns([3, 1])
    with controls[0]:
        refresh = st.toggle("从接口获取最新快照", value=True, key="market_network")
    with controls[1]:
        if st.button("刷新行情", key="market_refresh"):
            _load_snapshot.clear()
    if symbol.endswith("0"):
        st.info(f"{symbol} 是供应商主力连续序列，不是可下单的具体合约。换月价差不能直接视为持仓收益。")
    frequency = "1D" if resolution == "1天" else "1min"
    with st.spinner(f"读取 {symbol} {resolution} 行情…"):
        snapshot = _load_snapshot(str(ctx.base_dir), symbol, frequency, refresh)
    if snapshot.error:
        st.warning(f"行情读取提示：{snapshot.error}")
    data = snapshot.bars
    if data is None or data.empty:
        st.error("没有可验证的该合约数据；不会用旧合约或分钟片段冒充完整日 K。")
        return
    data = data.sort_values("date")
    source_label = "本地缓存（非本轮实时获取）" if snapshot.from_cache else "新浪接口快照"
    st.caption(
        f"序列：{symbol} | 来源：{source_label} | 已返回范围："
        f"{data['date'].min()} ～ {data['date'].max()} | "
        f"缓存采集时间：{snapshot.fetched_at or '未知'}"
    )
    st.caption("已返回范围只表示首末记录，不保证中间完整。页面交互或点击刷新时更新，接口结果缓存 60 秒，并非逐笔推送。")
    age = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None).normalize() - pd.Timestamp(data["date"].max()).normalize()
    if age.days > 3:
        st.warning(f"末条行情距上海当前日期已有 {age.days} 天，可能休市、合约已到期或数据源滞后；请核实后使用。")

    # Daily records are a coverage reference, never a replacement for minutes.
    # A lagging reference must not hide newer minute bars.
    daily = data if frequency == "1D" else pd.DataFrame()
    if frequency == "1min":
        reference = _load_snapshot(str(ctx.base_dir), symbol, "1D", refresh)
        daily = reference.bars
        if reference.error:
            st.warning(f"日线参考读取提示：{reference.error}")
    bounds_source = (
        pd.concat([data[["date"]], daily[["date"]]], ignore_index=True)
        if frequency == "1min" and daily is not None and not daily.empty else data
    )
    start, end = _window_bounds(bounds_source, lookback, specific_date)
    window = _in_window(data, start, end)
    if frequency == "1min":
        st.caption("分钟图仅展示实际返回的窗口，不能从最近的分钟片段重建完整历史日线。")
        for message in minute_quality_warnings(window, _in_window(daily, start, end)):
            st.warning(message)
    else:
        st.caption("日 K 使用该序列的独立日线接口，未从短窗口分钟数据聚合；当日记录可能继续修订，尚未与交易所日历及行情独立对账。")
    if window.empty:
        st.info("所选区间没有该合约数据，请调整区间；缺口未被隐藏或补值。")
        return
    freq = {"1小时": "1h", "30分钟": "30min", "15分钟": "15min", "5分钟": "5min"}.get(resolution)
    plot = resample_ohlc(window, freq) if freq else window
    st.caption(f"所选日期区间：{start.date()} ～ {end.date()} | 显示 {len(plot)} 根 K 线")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=plot["date"], open=plot["open"], high=plot["high"], low=plot["low"], close=plot["close"],
        name=f"{symbol} · {resolution}", increasing_line_color="#ef5350", decreasing_line_color="#26a69a",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=plot["date"], y=plot["volume"], name="成交量",
        marker_color=["#ef5350" if close >= opening else "#26a69a" for close, opening in zip(plot["close"], plot["open"])],
    ), row=2, col=1)
    breaks = [dict(bounds=["sat", "mon"])]
    if frequency == "1min":
        breaks += [dict(bounds=[15.26, 9.5], pattern="hour"), dict(bounds=[11.51, 13], pattern="hour")]
    fig.update_xaxes(rangebreaks=breaks)
    fig.update_layout(height=550, hovermode="x unified", xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=15, b=10),
                      legend=dict(orientation="h", y=-0.08))
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    st.plotly_chart(fig, width="stretch", key=f"market_chart_{variety}")
    with st.expander("查看行情原始字段"):
        fields = [name for name in ["date", "ticker", "open", "high", "low", "close", "volume", "oi", "source_symbol", "series_kind"] if name in plot]
        st.dataframe(plot[fields].tail(500), hide_index=True, width="stretch")
        st.caption("仅展示末 500 条；不会改变图表计算范围。")
    factors = getattr(ctx, "df_factors", None)
    if variety == "TL" and factors is not None and not factors.empty and "Market_Regime" in factors:
        with st.expander("市场状态分布（历史模型样本）"):
            st.caption("此统计来自独立历史因子文件，不代表当前实时合约状态，也不用于填充上方行情。")
            if "date" in factors:
                factor_dates = pd.to_datetime(factors["date"], errors="coerce")
                st.caption(f"因子样本范围：{factor_dates.min()} ～ {factor_dates.max()}")
            counts = factors["Market_Regime"].value_counts().sort_index()
            st.dataframe(pd.DataFrame({
                "状态": [{0: "正常", 1: "高波动", 2: "趋势"}.get(value, str(value)) for value in counts.index],
                "样本数": counts.values,
            }), hide_index=True, width="stretch")
