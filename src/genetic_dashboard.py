"""Research-only genetic discovery UI using the current train/validation API."""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from genetic_discovery import run_discovery, save_discovery
from intelligence_events import redact_sensitive_text


def render_genetic_mining(ctx):
    st.subheader("🧬 遗传规划因子挖掘")
    st.markdown("""
    **遗传规划 + 符号回归**：只在训练段搜索表达式，在独立验证段按 Rank IC 筛选。

    - 使用与训练链路一致的前向收益标签：下一根入场、持有 30 根，不跨合约，禁止标签持有期跨夜。
    - 按有效标签所在日期划分：至少 120 个训练日、60 个验证日、60 个封存留出日，并清除跨分区标签。
    - 只按验证集选择因子；相关系数绝对值 ≥ 0.90 的重复因子剔除。封存留出集不参与搜索、筛选或本页报告评估。
    """)
    st.caption("这是研究候选流程，不会修改生产模型、训练输入或交易信号。反复尝试参数也会过拟合验证集；验证 IC 不是独立最终样本外收益证明。")
    columns = st.columns(4)
    with columns[0]:
        population = st.slider("种群大小", 500, 3000, 1500, 250, key="gp_pop")
    with columns[1]:
        generations = st.slider("进化代数", 5, 50, 15, 5, key="gp_gen")
    with columns[2]:
        threshold = st.slider("最低|IC|门槛", 0.005, 0.05, 0.01, 0.005, key="gp_minic")
    with columns[3]:
        max_factors = st.slider("最多保留因子", 5, 50, 20, 5, key="gp_maxf")

    base_dir = str(ctx.base_dir)
    frame = ctx.df_factors
    if frame is not None and not frame.empty and "date" in frame:
        dates = pd.to_datetime(frame["date"], errors="coerce")
        st.caption(f"当前因子数据：{len(frame):,} 行，{dates.min()} ～ {dates.max()}。挖掘使用这份历史数据，不自动刷新为最新行情。")
    if st.button("🚀 启动遗传因子挖掘", width="stretch", type="primary", key="gp_run"):
        # A failed new attempt must not leave a prior result looking successful.
        st.session_state.pop("gp_research_result", None)
        st.session_state.pop("gp_research_source", None)
        st.session_state.pop("gp_saved_bundle", None)
        if frame is None or frame.empty:
            st.error("没有可用因子数据，请先运行因子构建流程。")
            return
        with st.spinner("构建标签与时间切分，在训练段挖掘、验证段筛选中…"):
            try:
                result = run_discovery(
                    frame, population_size=population, generations=generations,
                    max_new_factors=max_factors, min_ic_threshold=threshold,
                    n_jobs=1,
                )
                st.session_state["gp_research_result"] = result
                st.session_state["gp_research_source"] = base_dir
            except Exception as exc:
                st.error(f"挖掘失败：{redact_sensitive_text(exc)}")
                return

    result = st.session_state.get("gp_research_result")
    if result is not None and st.session_state.get("gp_research_source") == base_dir:
        report = result.report
        programs = report.get("programs", [])
        st.caption(f"研究结果生成时间：{report.get('timestamp', '未知')}。修改上方参数不会重算这份结果，需重新点击启动。")
        mining_config = report.get("mining_config", {})
        if mining_config:
            st.caption(
                f"本次实际参数：种群 {mining_config['population_size']}，进化 {mining_config['generations']} 代，"
                f"最低 |IC| {mining_config['min_ic_threshold']}，最多保留 {mining_config['max_new_factors']} 个因子。"
            )
        quality = report.get("data_quality", {})
        if quality.get("input_days", 0) > quality.get("labeled_days", 0):
            st.warning(
                f"原始数据覆盖 {quality['input_days']} 个日期，仅 {quality['labeled_days']} 个日期有有效标签。"
                "稀疏采样或不足的日内数据可能无法覆盖 30 根持有期；分区按有效标签日期计算，不代表连续的近期交易日。"
            )
        partitions = report.get("partitions", {})
        if partitions:
            labels = {"train": "训练（搜索）", "validation": "验证（筛选）", "sealed_holdout": "封存留出（未评估）"}
            st.dataframe(pd.DataFrame([
                {"分区": labels.get(name, name), "有效行数": metadata.get("rows"), "有效日期数": metadata.get("days"),
                 "起始": metadata.get("start"), "结束": metadata.get("end")}
                for name, metadata in partitions.items()
            ]), hide_index=True, width="stretch")
        if not programs:
            st.warning("未发现通过验证门槛的新因子。本次结果有效，但不建议为了得到因子而反复放宽门槛。")
        else:
            st.success(f"发现 {len(programs)} 个通过验证筛选的研究候选因子；尚未完成封存留出评估。")
            ranking = pd.DataFrame([
                {"因子名": item["name"], "验证 Rank IC": item["validation_ic"],
                 "|验证 IC|": abs(item["validation_ic"]), "表达式": item["expression"],
                 "复杂度": item["complexity"]}
                for item in programs
            ]).sort_values("|验证 IC|", ascending=False)
            st.dataframe(ranking, hide_index=True, width="stretch")
            figure = go.Figure(go.Bar(x=ranking["因子名"], y=ranking["|验证 IC|"], name="|验证 Rank IC|"))
            figure.update_layout(title="验证集 IC（非最终样本外评估）", height=300, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(figure, width="stretch", key="gp_validation_ic")
            with st.expander("查看生成因子（仅训练 / 验证段）"):
                st.dataframe(result.factors.tail(50), hide_index=True, width="stretch")
        st.caption("研究结果包含冻结的训练标准化器、表达式及验证集评分；不会自动加入训练，也不会替换正式模型。")
        actions = st.columns(2)
        with actions[0]:
            if st.button("💾 保存研究结果（因子 / 挖掘器 / 报告）", width="stretch", key="gp_save_df"):
                st.session_state.pop("gp_saved_bundle", None)
                try:
                    saved = save_discovery(result, base_dir)
                    st.session_state["gp_saved_bundle"] = str(saved)
                except Exception as exc:
                    st.error(f"保存失败：{redact_sensitive_text(exc)}")
            if st.session_state.get("gp_saved_bundle"):
                st.success(f"研究结果已保存到：{st.session_state['gp_saved_bundle']}")
        with actions[1]:
            st.download_button(
                "📄 下载因子报告 JSON", json.dumps(report, ensure_ascii=False, indent=2, default=str),
                file_name="genetic_discovery_report.json", mime="application/json", key="gp_save_report",
                width="stretch",
            )
    if frame is not None:
        existing = [str(name) for name in frame if str(name).startswith("GP_")]
        if existing:
            st.caption(f"输入中已有 {len(existing)} 个 GP 列；本次搜索不将既有 GP 因子再作为输入。")
