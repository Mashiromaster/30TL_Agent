# F_Agent — 固收量化交易系统

## 项目概述
TL（30年期国债期货）量化投机策略。基于 LightGBM 双模型（基础模型 + 高波动/趋势市模型），
整合量价、微观结构、基差和宏观因子，预测30分钟 forward return，分市场状态执行多空交易。

## 目录结构
```
F_Agent/
├── data/                         # 原始数据输入
│   ├── TL分钟级量价数据.pkl       # 原始分钟行情
│   ├── TL合约价差日频数据.pkl     # 日频基差数据（CTD券）
│   ├── main_contract_spliced.pkl  # 主力合约拼接中间文件
│   ├── TL_新数据_2026.pkl         # 2026年新数据
│   ├── tick/                     # 每日tick快照数据（.pkl）
│   └── macro/                    # AKShare原始数据缓存
├── models/                       # 训练好的模型
│   └── trained_model.pkl
├── outputs/                      # 模型输出/结果
│   ├── df_factors.pkl            # 最终因子集（117个特征）
│   ├── df_predictions.pkl        # 模型预测结果
│   ├── tick_minute_features.pkl  # tick快照聚合为分钟特征
│   ├── macro_factors.pkl         # 宏观因子缓存
│   ├── signal.json               # 最新交易信号
│   ├── signal_history.csv        # 历史信号记录
│   ├── backtest_metrics.csv      # 回测指标
│   ├── strategy_report.png       # 回测报告图
│   └── feature_importance.csv    # 特征重要性
├── src/                          # 源代码
│   ├── main.py                   # ★ 入口：--mode train|inference
│   ├── inference.py              # ★ 实时信号生成（SignalGenerator）
│   ├── strategy_agent.py         # ★ Claude Agent交互层（市场解读）
│   ├── update_market_data.py     # ★ 行情数据更新（AKShare分钟线）
│   ├── dashboard.py              # ★ Streamlit可视化UI
│   ├── mc_ex.py                  # 主力合约拼接+复权
│   ├── tick_data_processor.py    # tick->分钟微观结构特征
│   ├── data_fetcher.py           # ★ AKShare宏观数据采集+缓存
│   ├── macro_factors.py          # ★ 宏观/资金面因子计算
│   ├── factor_extraction.py      # 因子构建主流程（整合所有因子）
│   ├── LightGBM_model.py         # LightGBM训练（双模型+自检测宏观因子）
│   └── backtest.py               # 分状态策略回测+杠杆控制
├── tools/                        # 工具脚本
│   └── read_all_pkl.py
├── CLAUDE.md
├── .gitignore
└── README.md
```

## 运行方式

### 训练模式（因子构建 → 训练 → 回测）
```bash
cd "D:\桌面\F_Agent\src"
python main.py --mode train
```

### 推理模式（因子更新 → 加载模型 → 实时信号）
```bash
python main.py --mode inference
```
输出：控制台信号 + `signal.json` + `signal_history.csv`

### 行情数据更新（每日收盘后运行）
```bash
python update_market_data.py
```
从 AKShare 拉取 TL 主力合约 1分钟K线，追加到原始数据文件。

### Agent 交互层（市场解读、自然语言查询）
```bash
python strategy_agent.py --query snapshot     # 市场快照
python strategy_agent.py --query signal       # 信号解读
python strategy_agent.py --query factors      # 因子诊断
python strategy_agent.py --query macro        # 宏观环境
python strategy_agent.py --query performance  # 策略表现
python strategy_agent.py --query briefing     # 完整晨报
```

### Dashboard 可视化UI
```bash
python -m streamlit run dashboard.py
```
浏览器访问 `http://localhost:8501`，5个Tab：信号看板、市场监控、因子分析、回测表现、宏观环境。

### 日常操作流程
```bash
# 1. 更新行情数据
python update_market_data.py

# 2. 生成今日信号
python main.py --mode inference

# 3. 查看分析
python strategy_agent.py --query briefing

# 4. 打开Dashboard
python -m streamlit run dashboard.py
```

## 因子体系（117个特征）

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

## 宏观数据源（AKShare 1.18.63）

| 数据 | API | 状态 |
|------|-----|------|
| 国债收益率曲线 | `ak.bond_china_yield()` | ✅ |
| PMI | `ak.macro_china_pmi()` | ✅ |
| CPI | `ak.macro_china_cpi()` | ✅ |
| M2 | `ak.macro_china_money_supply()` | ✅ |
| 国债期货日行情 | `ak.futures_main_sina()` | ✅ |
| 国债期货分钟行情 | `ak.futures_zh_minute_sina()` | ✅ (最近~5天) |
| 中美利差 | `ak.bond_zh_us_rate()` | ✅ |
| SHIBOR | `ak.macro_china_shibor_all()` | ⚠️ 编码问题待修复 |
| 回购利率 | `ak.rate_interbank()` | ⚠️ 编码问题待修复 |
| 社融 | `ak.macro_china_shrzgm()` | ⚠️ SSL错误 |
| 央行OMO | — | ❌ 无稳定API |

## 模型架构
- LightGBM 双模型：基础模型 + 高波动/趋势市专用模型
- 预测目标：30分钟 forward return
- 训练/验证/测试：70%/15%/15% 时序切分
- RobustScaler 标准化

## 数据状态 (更新于 2026-05-29)

### TL分钟行情数据
- 数据范围: 2023-04-21 ~ 2026-05-28 (昨日)，共 470,227 行
- 覆盖 14 个合约: TL2306 ~ TL2609
- 最新合约 TL2609 从 2026-05-22 开始累积，TL2606 从 2025-09-15
- AKShare `futures_zh_minute_sina` 提供**近实时**分钟数据（交易时段延迟数秒）
- **必须每日运行 `update_market_data.py`** 累积数据，单合约仅返回最近 ~5 天

### 已知问题与限制

### 模型期限
- 当前模型训练数据截止 2025-10-31，距今约7个月
- 2025-11 至今有约7个月未参与训练的新数据（TL2606 + TL2609）
- **可重新训练**：数据已连续，因子计算可覆盖全区间
- 用间断数据训练会导致因子计算错误（已验证夏普从3.20降至-3.08）

### Tick数据
- tick快照特征文件（`tick_minute_features.pkl`）仅覆盖 2023-04 ~ 2025-10
- 新数据（2025-11 之后）无 tick 快照，微观结构因子值缺失（填0）
- 如需更新需重新获取 tick 快照数据

### AKShare 接口稳定性
- `futures_zh_minute_sina` 返回最近 ~5 天数据，需每日运行累积
- 国债收益率曲线 API 需分批拉取（每次约1年），全量拉取约需30秒
- SHIBOR、回购利率等接口偶有编码或 SSL 问题

## 回测表现
- 夏普比率：3.20
- 最大回撤：-1.50%
- Calmar：16.76
- 年化收益：25.19%

## 环境
- Windows 11, Python 3.12
- 核心依赖：pandas 2.2.2, numpy 1.26.4, lightgbm 4.6.0, akshare 1.18.63
- Dashboard: streamlit 1.52, plotly
- 注意：AKShare 数据拉取需要网络，离线时使用本地缓存

## 当前进度
- [x] Phase 1: 宏观数据基础设施（已完成 2026-05-27）
- [x] Phase 2: 实时信号生成（训练/推理模式分离，已完成 2026-05-28）
- [ ] Phase 3: 数据库持久化
- [x] Phase 4: Claude Agent 交互层（市场解读、自然语言查询，已完成 2026-05-28）
- [x] Phase 5: 可视化Dashboard + 交互UI（已完成 2026-05-28）
- [ ] Phase 6: 模型重训练（累积了 2025-11 ~ 2026-05 新数据，可重新训练）
