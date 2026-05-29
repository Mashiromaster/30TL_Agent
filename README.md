# 30TL Agent — 国债期货量化策略系统

[中文](#) | TL 30年期国债期货 · 量化投机策略 · AI驱动分析

![Demo](docs/demo.png)

## 项目概述

基于 **LightGBM 双模型**（基础模型 + 高波动/趋势市模型）的 TL（30年期国债期货）量化投机策略。整合量价、微观结构、基差和宏观因子，预测 30 分钟 forward return，分市场状态执行多空交易，搭配 Streamlit 可视化 Dashboard 和 DeepSeek V4 AI 市场情报分析。

## 核心特性

- **双模型架构** — 基础模型 + 高波动/趋势市专用模型，自适应市场状态
- **117 维因子体系** — 动量、波动率、微观结构、量价、技术、基差、宏观 7 大类
- **实时信号生成** — 每日更新行情 → 因子计算 → 模型推理 → 交易信号
- **AI 市场情报** — DeepSeek V4 自动抓取债市新闻 + 量化数据交叉验证 → 结构化分析报告
- **可视化 Dashboard** — Streamlit + Plotly 6 面板交互式看板
- **Claude Agent 交互** — 自然语言查询市场快照、因子诊断、宏观环境

## 回测表现

| 指标 | 数值 | 指标 | 数值 |
|------|------|------|------|
| 累计收益 | 8.55% | 年化收益 | 25.19% |
| 夏普比率 | **3.20** | 最大回撤 | -1.50% |
| Calmar 比率 | **16.76** | 年化波动 | 7.25% |
| 日胜率 | 50.0% | 持仓比例 | 33.7% |

## 系统架构

```
┌──────────────────────────────────────────────────────┐
│                    Data Pipeline                       │
│  AKShare API → 分钟行情 · Tick快照 · 宏观数据 · 新闻    │
└────────────────────────┬─────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────┐
│                  Factor Engine                         │
│  117 Features: 动量 · 波动率 · 微观结构 · 宏观 · 基差    │
└────────────────────────┬─────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────┐
│              LightGBM Dual Model                       │
│  Base Model + High-Vol Regime Model → Signal Output   │
└────────────────────────┬─────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────┐
│                  Output Layer                          │
│  Dashboard · AI分析 · Agent交互 · Signal.json          │
└──────────────────────────────────────────────────────┘
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置 API Key (AI情报分析功能)
set DEEPSEEK_API_KEY=your_deepseek_api_key

# 3. 训练模型
cd src
python main.py --mode train

# 4. 生成实时信号
python main.py --mode inference

# 5. 启动 Dashboard
python -m streamlit run dashboard.py

# 6. Agent 交互查询
python strategy_agent.py --query briefing
```

## 目录结构

```
30TL_Agent/
├── data/                  # 原始数据（行情/宏观/Tick）
├── models/                # 训练好的模型
├── outputs/               # 因子/信号/回测结果
├── src/                   # 源代码
│   ├── main.py            # 入口：train / inference
│   ├── LightGBM_model.py  # LightGBM 双模型训练
│   ├── inference.py       # 实时信号生成
│   ├── factor_extraction.py  # 因子构建
│   ├── macro_factors.py   # 宏观因子计算
│   ├── data_fetcher.py    # AKShare 宏观数据采集
│   ├── dashboard.py       # Streamlit 可视化
│   ├── llm_intelligence.py   # DeepSeek AI 情报分析
│   ├── strategy_agent.py  # Claude Agent 交互层
│   └── backtest.py        # 分状态回测
└── tools/                 # 工具脚本
```

## 技术栈

`Python 3.12` `LightGBM` `scikit-learn` `AKShare` `Streamlit` `Plotly` `DeepSeek V4` `Claude Agent`

## 免责声明

本项目仅供学习和研究使用，不构成任何投资建议。量化策略基于历史数据回测，过往表现不代表未来收益。投资有风险，入市需谨慎。
