@echo off
REM 启动 F_Agent Dashboard V2 (升级版)
REM 包含: 真实信号看板 + 交易记忆 + RAG增强 + 自我迭代
pushd "%~dp0\src"
python -m streamlit run dashboard_v2.py
popd
