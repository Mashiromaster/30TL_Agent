# F_Agent — TL 国债期货量化策略系统

30年期国债期货（TL）量化投机策略，基于 LightGBM 双模型，整合量价、微观结构、基差和宏观因子，预测 30 分钟 forward return，分市场状态执行多空交易。

## 核心能力

- **量化交易** — 117 维因子 → LightGBM 双模型 → 分状态动态信号
- **AI 情报** — DeepSeek V4 债市新闻分析 + 量化数据交叉验证
- **研究 RAG** — 央行报告/中金所月报/券商研报语义检索 + 生成式问答
- **交易记忆** — 每日决策记录、预测 vs 实际追踪、LLM 归因反思
- **可视化** — Streamlit + Plotly 8 面板交互式 Dashboard

## 回测表现

> 测试集: 2025-02-06 ~ 2025-05-27，9 个月滚动窗口 + 60 天半衰衰减

| 指标 | 数值 | 指标 | 数值 |
|------|------|------|------|
| 累计收益 | 9.67% | 年化收益 | 28.06% |
| 夏普比率 | **2.01** | 最大回撤 | -2.56% |
| Calmar | **10.94** | 年化波动 | 12.95% |
| 日胜率 | 45.9% | 持仓比例 | 39.4% |

### 分市场状态

| 状态 | 占比 | IC | 收益贡献 |
|------|:--:|:--:|:--:|
| 正常市 | 83.1% | +0.034 | +10.40% |
| 高波动市 | 15.7% | -0.021 | -0.02% |
| 趋势市 | 1.2% | +0.096 | -0.85% |

## 系统架构

```
┌────────────────────────────────────────────────────┐
│                   Data Pipeline                     │
│  AKShare → 分钟行情 · Tick快照 · 宏观数据 · 新闻     │
└──────────────────────┬─────────────────────────────┘
                       ↓
┌────────────────────────────────────────────────────┐
│                 Factor Engine                       │
│  117 Features: 动量 · 波动率 · 微观结构 · 宏观 · 基差 │
└──────────────────────┬─────────────────────────────┘
                       ↓
┌────────────────────────────────────────────────────┐
│             LightGBM Dual Model                     │
│  Base Model + High-Vol/Trend Model → 30min Forecast │
│  9-Month Window + 60d Time Decay                    │
└──────────────────────┬─────────────────────────────┘
                       ↓
┌────────────────────────────────────────────────────┐
│                  Output Layer                       │
│  Dashboard(8 Tabs) · AI情报 · RAG检索 · 交易记忆     │
│  Agent CLI · Signal.json · LLM预测对比               │
└────────────────────────────────────────────────────┘
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置 API Key (AI情报 & RAG 问答)
set DEEPSEEK_API_KEY=your_deepseek_api_key

# 3. 每日更新行情数据
cd src
python update_market_data.py

# 4. 生成实时信号
python main.py --mode inference

# 5. 启动 Dashboard
streamlit run dashboard.py

# 6. CLI 工具
python strategy_agent.py --query briefing   # 完整晨报
python strategy_agent.py --query signal     # 信号解读
python strategy_agent.py --query factors    # 因子诊断
python strategy_agent.py --query macro      # 宏观环境
```

## Dashboard 面板

| Tab | 功能 |
|-----|------|
| 信号看板 | 当前交易方向/置信度/仓位 + 近一周预测走势 |
| 市场监控 | K线图(日/时/分可切换) + 成交量 + 市场状态分布 |
| 因子分析 | 特征重要性 Top15 + 因子分位图 + 异常预警 |
| 回测表现 | NAV曲线 + 回撤 + 日收益分布 + 风险指标 |
| 宏观环境 | 收益率曲线形态 + 中美利差 + 宏观指标趋势 |
| AI情报分析 | DeepSeek V4 量化数据 + 新闻 → 结构化报告 |
| 研究RAG | 研报语义检索 + AI生成式问答 + 文档类型过滤 |
| 交易记忆 | 预测 vs 实际追踪 + 准确率矩阵 + LLM归因反思 |

## 因子体系（117 个特征）

| 类别 | 数量 | 示例 |
|------|------|------|
| 动量 | 7 | Short_Momentum_1D/3D/5D, Mid_Momentum_1M/2M, TSMOM |
| 波动率 | 6 | RV_30, RV_120, Vol_Surge, ATR_14, Vol_Regime |
| 微观结构 | 41 | Spread, Imbalance, Signed_Vol, VPIN, HF_RV, Cum_Net_Open |
| 量价 | 4 | OI_Volume_Flow, Smart_Money, Large_Trade_Direction |
| 技术 | 4 | MACD_Hist, RSI, BB_Position |
| 市场状态 | 4 | Market_Regime, Is_High_Vol, Trend_Consistency |
| 基差 | 3 | Basis_ZScore_20, Basis_ZScore_10, Basis_Trend |
| **宏观** | **15** | **CN_US_10Y_Spread, YC_Slope_30Y_10Y, M2_Surprise, PMI_ZScore, CPI_Momentum** |

### 宏观数据源（AKShare）

| 数据 | API | 状态 |
|------|-----|:--:|
| 国债收益率曲线 | `bond_china_yield()` | 正常 |
| PMI / CPI / M2 | `macro_china_*` | 正常 |
| 国债期货分钟行情 | `futures_zh_minute_sina()` | 正常 (近5天) |
| 中美利差 | `bond_zh_us_rate()` | 正常 |
| SHIBOR / 回购利率 | — | 编码问题待修复 |
| 社融 / OMO | — | 暂不可用 |

## 目录结构

```
F_Agent/
├── data/                         # 原始数据
│   ├── TL分钟级量价数据.pkl       # 原始分钟行情 (470K+ 行)
│   ├── TL合约价差日频数据.pkl     # 日频基差数据
│   ├── main_contract_spliced.pkl  # 主力合约拼接
│   ├── tick/                     # 每日tick快照 (.pkl)
│   ├── macro/                    # AKShare 宏观数据缓存
│   └── rag/                      # RAG 索引 (ChromaDB + 研报PDF)
├── models/                       # 训练好的模型
│   └── trained_model.pkl
├── outputs/                      # 输出文件
│   ├── df_factors.pkl            # 因子集 (155K行 × 117特征)
│   ├── df_predictions.pkl        # 模型预测结果
│   ├── tick_minute_features.pkl  # tick → 分钟特征
│   ├── macro_factors.pkl         # 宏观因子缓存
│   ├── signal.json               # 最新交易信号
│   ├── signal_history.csv        # 历史信号记录
│   ├── backtest_metrics.csv      # 回测指标
│   ├── strategy_report.png       # 回测净值图
│   ├── feature_importance.csv    # 特征重要性
│   ├── trade_memory.jsonl        # 交易记忆记录
│   ├── llm_predictions.csv       # LLM 预测对比
│   └── ai_intelligence*.json     # AI 情报报告
├── src/                          # 源代码
│   ├── main.py                   # 入口: --mode train|inference
│   ├── LightGBM_model.py         # LightGBM 双模型训练
│   ├── inference.py              # 实时信号生成
│   ├── backtest.py               # 分状态策略回测
│   ├── factor_extraction.py      # 因子构建主流程
│   ├── macro_factors.py          # 宏观/资金面因子
│   ├── data_fetcher.py           # AKShare 宏观数据采集
│   ├── update_market_data.py     # 行情数据增量更新
│   ├── mc_ex.py                  # 主力合约拼接+复权
│   ├── tick_data_processor.py    # tick → 分钟微观结构
│   ├── strategy_agent.py         # CLI Agent 交互层
│   ├── dashboard.py              # Streamlit 8-Tab Dashboard
│   ├── llm_intelligence.py       # DeepSeek AI 情报分析
│   ├── llm_predictor.py          # LLM 预测对比模块
│   ├── rag_tool.py               # RAG 研究工具
│   ├── memory.py                 # 交易记忆系统
│   └── check_timestamp_format.py # 时间戳格式检查
└── requirements.txt              # Python 依赖
```

## RAG 研究工具

基于 ChromaDB + BGE-small-zh Embedding 的研报语义检索引擎，覆盖央行货政报告、中金所月度报告、券商研报、债券新闻四类数据源。

```bash
python rag_tool.py build              # 构建/重建索引
python rag_tool.py search <问题>      # CLI 检索
python rag_tool.py briefing           # 生成周度研究简报
```

## 模型要点

详见 [imp.md](imp.md)，关键决策：

1. **MAE 优于 MSE** — 30 分钟收益有大量离群值，MAE 对尾部更稳健
2. **极度正则化** — max_depth=3, lambda=15, min_child_samples=350
3. **9 个月训练窗口** — 覆盖一个市场周期又不包含旧制度噪声
4. **60 天半衰衰减** — 近期数据更高权重，捕捉结构微变

## 技术栈

`Python 3.12` `LightGBM` `scikit-learn` `AKShare` `Streamlit` `Plotly` `DeepSeek V4` `ChromaDB` `BGE-small-zh` `sentence-transformers`

## 免责声明

本项目仅供学习和研究使用，不构成任何投资建议。量化策略基于历史数据回测，过往表现不代表未来收益。
