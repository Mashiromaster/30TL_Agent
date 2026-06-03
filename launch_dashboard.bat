@echo off
title F_Agent TL 策略
echo ============================================
echo   F_Agent - TL 30Y 国债期货 智能策略
echo   端口: 8503
echo ============================================
echo.
start http://localhost:8503
echo.
cd /d "D:\桌面\F_Agent\src"
python -m streamlit run dashboard_v2.py --server.port 8503 --server.headless true
pause
