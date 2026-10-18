"""Thin, explicit local CSV/coverage workflow for the research workbench."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from intelligence_events import redact_sensitive_text
from prepare_market_data import audit_local_minutes, build_minute_snapshot, import_minute_csv


def list_sources(base_dir):
    from minute_store import list_minute_batches, list_minute_snapshots
    return {"batches": list_minute_batches(base_dir), "snapshots": list_minute_snapshots(base_dir)}


def _upload():
    uploaded = st.file_uploader("本地 TL 分钟 CSV（格式以上方选择为准，最多 25 MiB）", type=["csv"], max_upload_size=25, key="minute_csv_upload")
    return uploaded.getvalue() if uploaded is not None else None


def _error(prefix, exc):
    st.error(prefix + redact_sensitive_text(str(exc))[:1500])


def _show_quality(report):
    coverage = report.get("coverage") or {}
    st.caption("只验证文件结构与观测覆盖；不认证供应商、授权真实性、历史修订或交易所日历。")
    if coverage:
        columns = st.columns(3)
        for column, label, value in zip(columns, ["观测记录", "历史不完整合约日", "待核对范围"], [
            coverage.get("rows"), coverage.get("incomplete_completed_contract_days"),
            (report.get("repair_summary") or {}).get("range_count"),
        ]):
            column.metric(label, f"{value:,}" if isinstance(value, int) else "未报告")
        st.caption(f"观测范围：{coverage.get('start') or '未报告'} ～ {coverage.get('end') or '未报告'}。缺口按本地常规模板统计，不等同于交易所认证缺数。")
        with st.expander("原始覆盖统计"):
            st.json(coverage)
    if report.get("repair_ranges"):
        st.markdown("**待核对 / 补数范围（不触发下载）**")
        st.dataframe(report["repair_ranges"], hide_index=True, use_container_width=True)
    st.download_button("下载分钟覆盖审计 JSON", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                       file_name="minute_coverage_audit.json", mime="application/json", key="minute_audit_download")
    with st.expander("覆盖审计详情与限制"):
        st.json(report)


def render_minute_preparation(base_dir):
    """Return a user-selected snapshot id or None (legacy), with no autorun."""
    base = str(Path(base_dir).resolve())
    if st.session_state.get("minute_base") != base:
        for key in list(st.session_state):
            if key.startswith("minute_"):
                del st.session_state[key]
        st.session_state["minute_base"] = base
    with st.expander("分钟行情接入与覆盖审计", expanded=False):
        st.caption("默认不联网。CSV 操作只在本地执行；下方公开近期入口须单独确认联网。导入仅保存独立批次，组装不覆盖旧行情。")
        from recent_minutes_dashboard import render_recent_minutes
        render_recent_minutes(base)
        formats = {"canonical": "标准结束标签 CSV", "tqsdk": "天勤 DataDownloader · 单合约导出"}
        input_format = st.selectbox("本地 CSV 格式", list(formats), format_func=formats.get, key="minute_input_format")
        interval_confirmed = False
        if input_format == "tqsdk":
            st.caption("只接受单一 CFFEX.TL季度合约的官方9列格式；核对北京时间与epoch纳秒一致，再将开始标签加60秒。保留原始source.csv及转换清单。")
            interval_confirmed = st.checkbox("确认该导出使用 dur_sec=60；CSV 本身不能独立证明周期", key="minute_tq_interval")
            st.code("datetime,datetime_nano," + ",".join(
                "CFFEX.TL2606." + name for name in ("open", "high", "low", "close", "volume", "open_oi", "close_oi")
            ), language=None)
            st.caption("以上 TL2606 仅为列名示例；请上传实际授权导出，不要将标准模板改名为天勤文件。")
        else:
            st.code("date,ticker,open,high,low,close,volume,oi", language=None)
            st.download_button("下载 CSV 表头模板", "date,ticker,open,high,low,close,volume,oi\n",
                               file_name="tl_minute_template.csv", mime="text/csv", key="minute_template_download")
        st.caption("标准格式约定：Asia/Shanghai，60秒K线的结束时间；volume为单bar成交量，oi为bar结束持仓量。可额外提供trade_dt。天勤格式仅执行上述明确转换，不猜测其它格式。")
        payload = _upload()
        provider = st.text_input("声明的数据来源（不是来源认证）", value="local-export", key="minute_provider")
        license_note = st.text_input("本地使用授权说明（不要填写账号或密钥）", key="minute_license")
        consent = st.checkbox("我有权使用该数据，并确认其符合上述分钟、成交量和时间戳约定；仅保存本地独立批次。", key="minute_import_consent")
        inspect_clicked = st.button("校验 CSV（不保存）", key="minute_inspect")
        import_clicked = st.button("导入为独立批次", key="minute_import")
        if inspect_clicked or import_clicked:
            st.session_state.pop("minute_import_result", None)
            try:
                if payload is None or not provider.strip() or not license_note.strip():
                    raise ValueError("请提供 CSV、来源及本地使用说明")
                if import_clicked and not consent:
                    raise ValueError("请先确认数据使用权限与字段语义")
                if input_format == "tqsdk" and not interval_confirmed:
                    raise ValueError("请先确认天勤导出周期为60秒")
                options = {"input_format": "tqsdk", "bar_seconds": 60} if input_format == "tqsdk" else {}
                result = import_minute_csv(base, payload, provider=provider, license_note=license_note, persist=import_clicked, **options)
                if result.get("status") not in {"inspected", "imported"}:
                    raise ValueError("CSV未通过校验，未生成可用批次")
                st.session_state["minute_import_result"] = result
                if result["status"] == "imported":
                    batches = dict(st.session_state.get("minute_batch_options", {}))
                    batches[result["batch_id"]] = result
                    st.session_state["minute_batch_options"] = batches
            except Exception as exc:
                _error("分钟导入/校验失败：", exc)
        if st.session_state.get("minute_import_result"):
            result = st.session_state["minute_import_result"]
            st.info("上次显式操作结果：" + result["status"] + "。修改上传文件或声明不会自动重新验证。")
            with st.expander("上次 CSV 操作记录"):
                st.json(result)

        if st.button("读取已保存分钟批次与快照", key="minute_list"):
            try:
                listing = list_sources(base)
                st.session_state["minute_batch_options"] = {entry["batch_id"]: entry for entry in listing["batches"]}
                st.session_state["minute_snapshot_options"] = {entry["snapshot_id"]: entry for entry in listing["snapshots"]}
                st.caption("每类最多读取30个清单；列表读取不加载模型、不认证数据、不自动选择。")
            except Exception as exc:
                _error("读取分钟清单失败：", exc)
        batches = st.session_state.get("minute_batch_options", {})
        preferences = st.session_state.get("minute_preferences", {})
        for key in ("minute_batches", "minute_source"):
            if key not in st.session_state and key in preferences:
                st.session_state[key] = preferences[key]
        old_selection = st.session_state.get("minute_batches", [])
        if any(item not in batches for item in old_selection):
            st.session_state["minute_batches"] = []
        selected_batches = st.multiselect("组装快照使用的批次", list(batches), key="minute_batches")
        include_legacy = st.checkbox("同时纳入旧本地原始行情（来源未认证，不覆盖旧文件）", key="minute_include_legacy")
        if st.button("组装独立分钟快照", key="minute_assemble"):
            st.session_state.pop("minute_build_result", None)
            try:
                if not selected_batches and not include_legacy:
                    raise ValueError("请显式选择批次，或确认纳入旧行情")
                result = build_minute_snapshot(base, selected_batches, include_legacy=include_legacy)
                if result.get("status") != "built" or not result.get("snapshot_id"):
                    raise ValueError("未生成有效分钟快照")
                snapshots = dict(st.session_state.get("minute_snapshot_options", {}))
                snapshots[result["snapshot_id"]] = result
                st.session_state["minute_snapshot_options"] = snapshots
                st.session_state["minute_build_result"] = result
            except Exception as exc:
                _error("分钟快照组装失败：", exc)
        if st.session_state.get("minute_build_result"):
            st.info("独立快照已生成，但未替换旧行情或自动切换来源。请在下方显式选择。")
            with st.expander("上次分钟快照记录"):
                st.json(st.session_state["minute_build_result"])

        snapshots = st.session_state.get("minute_snapshot_options", {})
        choices = [""] + list(snapshots)
        if st.session_state.get("minute_source", "") not in choices:
            st.session_state["minute_source"] = ""
        selected = st.selectbox("构建新因子时的分钟来源", choices, key="minute_source",
                                format_func=lambda item: item or "旧本地原始行情（未认证）")
        st.session_state["minute_preferences"] = {"minute_batches": selected_batches, "minute_source": selected}
        st.caption("当前构建来源：" + (selected or "旧本地原始行情") + "。只影响下一次显式构建因子，不改变已保存因子、研究结果或生产模型。")
        if st.button("审计当前分钟来源（只读）", key="minute_audit"):
            st.session_state.pop("minute_quality", None)
            try:
                st.session_state["minute_quality"] = audit_local_minutes(base, snapshot_id=selected or None)
                st.session_state["minute_quality_source"] = selected
            except Exception as exc:
                _error("分钟覆盖审计失败：", exc)
        if st.session_state.get("minute_quality") and st.session_state.get("minute_quality_source") == selected:
            _show_quality(st.session_state["minute_quality"])
        return selected or None
