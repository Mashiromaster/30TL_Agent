"""Explicit network preview, separate admission of the exact captured bytes."""
import streamlit as st
from pathlib import Path

from intelligence_events import redact_sensitive_text
from recent_minutes import fetch_recent_minutes, commit_recent_capture


def render_recent_minutes(base_dir):
    base = str(Path(base_dir).resolve())
    if st.session_state.get("minute_recent_base") != base:
        for key in list(st.session_state):
            if key.startswith("minute_recent_"):
                del st.session_state[key]
        st.session_state["minute_recent_base"] = base
    with st.expander("公开近期行情 · 显式联网获取", expanded=False):
        st.caption("新浪近期滚动窗口，不是历史补数服务；不定时轮询、不回退旧缓存、不自动入库。默认未知时间标签，只查看返回范围。")
        symbol = st.text_input("具体 TL 合约（不是主力连续）", value="", placeholder="例如 TL2612", key="minute_recent_symbol").strip().upper()
        labels = {"unverified": "未核验（仅查看）", "end": "已自行核验：结束时间", "start": "已自行核验：开始时间（显式加60秒）"}
        label = st.selectbox("提供方原始分钟时间标签", list(labels), format_func=labels.get, key="minute_recent_label")
        context = (base, symbol, label)
        if st.session_state.get("minute_recent_context") != context:
            st.session_state["minute_recent_consent"] = False
            st.session_state["minute_recent_context"] = context
            st.session_state.pop("minute_recent_saved", None)
        network = st.checkbox("允许本次向新浪发送该合约代码，获取一次近期分钟窗口", key="minute_recent_network")
        if st.button("获取近期窗口（不保存）", key="minute_recent_fetch"):
            st.session_state.pop("minute_recent_capture", None)
            st.session_state.pop("minute_recent_saved", None)
            st.session_state["minute_recent_consent"] = False
            try:
                if not network:
                    raise ValueError("请先确认本次联网请求")
                with st.spinner("正在获取一个合约的近期窗口…"):
                    item = fetch_recent_minutes(base_dir, symbol, allow_network=True, bar_label=label)
                st.session_state["minute_recent_capture"] = item
                st.session_state["minute_recent_request"] = context
            except Exception as exc:
                st.error("近期获取失败：" + redact_sensitive_text(str(exc))[:1500])
        item = st.session_state.get("minute_recent_capture")
        if item is not None:
            matches = st.session_state.get("minute_recent_request") == context
            st.info("上次显式获取结果；刷新页面控件不代表重新采集。" + ("" if matches else " 当前请求已变化，不能保存旧预览。"))
            cols = st.columns(3)
            cols[0].metric("返回记录", str(item.report.get("received_rows", "未报告")))
            cols[1].metric("规范校验通过", str(item.report.get("accepted_rows", 0)))
            cols[2].metric("入库状态", "未保存")
            st.caption(f"提供方范围：{item.report.get('provider_start', '未报告')} ～ {item.report.get('provider_end', '未报告')}")
            if item.report.get("status") == "review_required":
                st.warning("时间标签/成交量与持仓量口径尚未核验：本次仅预览，不允许直接入库，不能作为实时交易依据。")
            else:
                st.caption("时间语义仅为你的声明；未认证交易日历、来源许可或历史完整性。")
            with st.expander("本次请求审计与限制"):
                st.json(item.report)
            if item.csv_bytes is not None and matches:
                st.download_button("下载本次规范 CSV（仅研究，语义仍为声明）", item.csv_bytes,
                    file_name=f"{symbol}_recent.csv", mime="text/csv", key="minute_recent_download")
        license_note = st.text_input("近期行情本地使用授权说明（不填账号密钥）", key="minute_recent_license")
        consent = st.checkbox("我已核对该来源的时间标签、单bar成交量和结束持仓量口径，并有权保存本次数据", key="minute_recent_consent")
        if st.button("保存本次已校验窗口为独立批次", key="minute_recent_commit"):
            st.session_state.pop("minute_recent_saved", None)
            try:
                if not consent or not license_note.strip():
                    raise ValueError("请先核对语义并填写本地使用授权说明")
                if item is None or st.session_state.get("minute_recent_request") != context:
                    raise ValueError("请先获取与当前请求一致的窗口")
                result = commit_recent_capture(base_dir, item, license_note=license_note, confirmed=True)
                if result.get("status") != "imported" or not result.get("batch_id"):
                    raise ValueError("未生成有效独立批次")
                batches = dict(st.session_state.get("minute_batch_options", {}))
                batches[result["batch_id"]] = result
                st.session_state["minute_batch_options"] = batches
                st.session_state["minute_recent_saved"] = result
                st.session_state.pop("minute_recent_capture", None)
            except Exception as exc:
                st.error("近期窗口保存失败：" + redact_sensitive_text(str(exc))[:1500])
        if st.session_state.get("minute_recent_saved"):
            st.success("已保存独立批次；未自动组装快照、切换来源、构建因子或补齐历史。")
            with st.expander("上次近期窗口保存记录"):
                st.json(st.session_state["minute_recent_saved"])
