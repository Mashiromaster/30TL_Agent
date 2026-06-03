# F_Agent — TL 30年期国债期货智能量化策略系统

## 项目基本信息

- **仓库**: https://github.com/Mashiromaster/30TL_Agent
- **本地路径**: `D:\桌面\F_Agent`
- **Dashboard端口**: `8503`
- **桌面快捷方式**: `F_Agent_TL策略.lnk`
- **Python**: 3.12 (Anaconda, `D:\Anaconda`)
- **Git**: `D:\Git\bin\bash.exe` (非标准位置, 需 `HERMES_GIT_BASH_PATH`)

---

## 当前系统状态 (2026-06-03)

### 模型
- **架构**: LightGBM 双模型 (Base Model + HighVol/Trend Model)
- **训练配置**: 12个月滚动窗口 + 90天半衰时间衰减
- **特征数**: 91 → 实测最优 (40特征筛选后IC从0.039跌至0, XGBoost/Ensemble均无法超越)
- **Test IC**: 0.0388 (Normal市 0.0536, HighVol市 -0.0006, Trend市 -0.1198)
- **提升**: 原始模型IC 0.015 → 0.039 (Normal IC从 **-0.200** 扭正为 **+0.0536**)
- **损失函数**: MAE (regression_l1) — MSE在极低信噪比下过拟合严重

### 信号系统
- **方式**: 甜点区排名信号 (Sweet Spot Ranking)
- **参数**: smooth_span=60, lookback=480, confirm=10
- **做空甜点**: rank 15%-30% (准确率 72.7%)
- **做多甜点**: rank 70%-85% (准确率 63.6%)
- **跳过区间**: rank 0-15% 和 85-100% (极端排名=噪声, 准确率仅55.6%/46.9%)
- **配置**: `outputs/params.json`

### 数据
- **分钟行情**: 2023-04-21 ~ 2026-06-02 (488K行, 14个合约)
- **因子集**: 159K行 × 151特征
- **预测集**: 23,916行 (测试期 2025-07-07 ~ 2026-06-02, 219天)
- **Tick数据**: 仅到 2025-10-31, 之后微观结构因子填0
- **交易记忆**: 219笔, 甜点区信号准确率 59.3%

---

## 核心技术架构

```
Data Pipeline (AKShare) → Factor Engine (151维)
→ LightGBM Dual Model (12月+90d衰减)
→ Sweet Spot Ranking (15-30% short / 70-85% long)
→ Dashboard 13-Tab (Streamlit)
```

### 因子体系 151维

| 类别 | 数量 | 关键因子 |
|------|------|----------|
| 动量 | 7 | Short_Momentum_1D/3D/5D, Mid_Momentum_1M/2M, TSMOM |
| 波动率 | 6 | RV_30, RV_120, Vol_Surge, ATR_14, Vol_Regime |
| 微观结构 | 44 | Spread, Imbalance, Signed_Vol, VPIN, HF_RV, Cum_Net_Open |
| 量价 | 4 | OI_Volume_Flow, Smart_Money, Large_Trade_Direction |
| 技术 | 4 | MACD_Hist, RSI, BB_Position |
| 市场状态 | 4 | Market_Regime, Is_High_Vol, Trend_Consistency |
| 基差 | 3 | Basis_ZScore_20, Basis_ZScore_10, Basis_Trend |
| 宏观 | 14 | CN_US_10Y_Spread, PMI_ZScore, M2_Surprise, YC_* |
| 增强因子 | 35 | OI_Growth_5D/20D, Vol_Breakout, ADX, Spread_Stress, Vol_of_Vol |

### Top 10 特征重要性
1. Macro_Surprise_Composite (81) — 宏观综合意外
2. CN_US_10Y_Spread (51) — 中美利差
3. Basis_ZScore_20 (39) — 基差偏离
4. Short_Momentum_5D (38)
5. CN_US_10Y_Spread_Z (33)
6. Mid_Momentum_2M (31)
7. Mid_Momentum_1M (30)
8. YC_Slope_30Y_10Y (29)
9. YC_Curvature (29)
10. Basis_ZScore_10 (26)

---

## 关键发现与经验教训

### 1. 极度正则化是这个领域唯一有效策略
金融时间序列信噪比极低 (最优单特征IC仅 0.02-0.03):
- LightGBM: max_depth=3, lambda_l1/l2=15, feature_fraction=0.3, min_child=350 — 必须
- 任何放宽正则化的尝试都导致严重过拟合
- MSE loss 追逐离群值, MAE 预测条件中位数更稳健

### 2. 训练窗口不是越大越好
- 9个月: Sharpe 2.01 (最优在当时)
- 12个月: IC 0.039 (当前最优, 覆盖更多制度转换)
- 18个月: IC 0.030 (旧制度数据成为噪声)
- 金融数据关键是"相关性"而非"数量"

### 3. 模型自信 ≠ 准确
甜点区发现的根本原因: 模型在极端排名时(rank<15%或>85%)反而错误率最高:
- 极端做空 (0-15%): 55.6% (仅比随机略好)
- 做空甜点 (15-30%): 72.7% (真正有效)
- 做多甜点 (70-85%): 63.6%
- 极端做多 (85-100%): 46.9% (比扔硬币还差)

### 4. 特征筛选砍掉交互效应
将91特征降到40个(按|IC|排序取Top40): IC从 0.039 跌到 0.000 → LightGBM的树结构依赖弱特征间的交互

### 5. 集成学习未能超越
- XGBoost单独: IC 0.016 (正则化不足)
- CatBoost: 未安装, 未测试
- LightGBM+XGBoost加权集成: IC 0.015 (拉低LGBM)
- 多时域共识: IC 0.003 (120min预测衰减太快)

### 6. 30分钟是最优预测时域
- 30min: IC 0.039 (最佳)
- 60min: IC -0.009 (信息衰减)
- 120min: IC -0.037 (几乎无预测力)
- 与imp.md早期结论一致 (120min有更好per-regime IC但不适合回测框架)

### 7. 宏观因子主导
Macro_Surprise_Composite 始终#1, 说明2025H2后市场由宏观政策驱动, 技术和微观因子降级

---

## 所有模块清单 (src/)

### 核心流程
| 文件 | 功能 |
|------|------|
| `main.py` | 入口: `--mode train\|inference\|iterate` |
| `factor_extraction.py` | 因子构建主流程 (数据加载→基差→Tick→宏观→核心→组合→增强) |
| `LightGBM_model.py` | LightGBM双模型训练 (9月窗口+60d衰减, 基础参数) |
| `inference.py` | SignalGenerator: 加载模型→预测 (旧版绝对阈值逻辑) |
| `backtest.py` | 分状态策略回测 (绝对阈值+杠杆控制) |
| `retrain_optimized.py` | 全量重训练+9窗口自动扫描 |

### Dashboard & 信号
| 文件 | 功能 |
|------|------|
| `dashboard_v2.py` | **13-Tab全功能面板** (端口8503) |
| `signal_dashboard.py` | 甜点区排名信号引擎 + 预测可视化 |
| `dashboard.py` | 旧版8-Tab (部分Tab通过pass-through复用) |

### 自我迭代 & 评估 (借鉴开源项目)
| 文件 | 功能 | 灵感来源 |
|------|------|----------|
| `self_iteration.py` | 自我迭代引擎 (IC监控/漂移检测/失败分析/时间衰减/参数调优) | Dexter temporal-decay |
| `eval_runner.py` | 滚动窗口模型评估 (IC序列/基线对比/退化检测/MD报告) | Dexter eval system |
| `factor_evaluator.py` | Alphalens风格因子评估 (IC衰减/分位数收益/换手率/相关性/综合评分) | Alphalens ⭐4,293 |
| `optuna_optimizer.py` | Optuna贝叶斯超参搜索 + 多模型集成 + 多时域预测 | RektGBM + Freqtrade |
| `cron_scheduler.py` | 定时任务调度 (6任务 + daemon守护进程) | Dexter cron tool |

### 增强因子
| 文件 | 功能 |
|------|------|
| `enhanced_factors.py` | 34个制度自适应因子 (Shock/ADX/Vol_Breakout/OI_Growth等) |

### 数据 & 工具
| 文件 | 功能 |
|------|------|
| `data_fetcher.py` | AKShare宏观数据采集+缓存 (10个数据源) |
| `tick_data_processor.py` | 半秒快照→分钟微观结构特征 |
| `mc_ex.py` | 主力合约拼接+复权 |
| `update_market_data.py` | 行情数据增量更新 (每日运行) |
| `macro_factors.py` | 宏观/资金面因子计算 (5类22因子) |
| `strategy_agent.py` | CLI Agent 交互层 (StrategyContext + 市场解读) |
| `llm_intelligence.py` | DeepSeek AI情报分析 |
| `llm_predictor.py` | LLM预测对比 |
| `rag_tool.py` | RAG研究工具 (爬虫+ChromaDB+BGE-small-zh+LLM生成) |
| `memory.py` | 交易记忆系统 (记录/回填/统计/LLM反思) |
| `backtest_v2.py` | 排名信号回测+参数扫描 |
| `retrain_v2.py` | V2训练 (MSE loss尝试, 失败) |
| `train_final.py` | 最终重训练管道 (Optuna+XGBoost集成, 未超越baseline) |
| `check_timestamp_format.py` | 时间戳格式检测 |

---

## 13-Tab Dashboard 结构

```
📡 信号看板   — signal_dashboard.py (甜点区排名信号+预测图+绩效)
📊 市场监控   — dashboard.py (K线+成交量+状态)
🔬 因子分析   — dashboard.py (重要性+分位+异常)
🔍 因子评估   — factor_evaluator.py (Alphalens: IC排名+稳定性+相关性)
💰 回测表现   — dashboard.py (NAV+回撤+收益分布)
🌍 宏观环境   — dashboard.py (曲线+利差+指标)
🤖 AI情报     — dashboard.py (DeepSeek分析)
📚 研究RAG    — rag_tool.py (检索+过滤+简报)
🧠 交易记忆   — memory.py (准确率矩阵+LLM反思)
🔄 自我迭代   — self_iteration.py (诊断+漂移+参数调优)
📋 模型评估   — eval_runner.py (滚动IC+基线对比+退化检测)
⚙️ 超参优化   — optuna_optimizer.py (Optuna+集成+多时域)
⏰ 定时调度   — cron_scheduler.py (6任务运行/暂停)
```

---

## 定时任务 (推荐启用)

```bash
python cron_scheduler.py daemon  # 后台守护进程
```

| ID | 频率 | 功能 |
|----|------|------|
| daily_update | 工作日 16:00 | AKShare拉取最新分钟行情 |
| daily_signal | 工作日 16:00 | 生成交易信号 |
| weekly_eval | 周五 17:00 | 滚动窗口IC评估 |
| weekly_memory | 周五 17:00 | 同步交易记忆 |
| monthly_retrain | 每月1号 09:00 | 全量重训练 |
| monthly_iteration | 每月1号 10:00 | 自我迭代诊断 |

---

## 未来优化方向

### 高优先级
1. **滚动重训练自动化**: 每月运行 `retrain_optimized.py`, 自动部署新模型
2. **Tick数据补充**: 2025-11之后的Tick快照缺失, 微观结构因子填0 → 重新获取后IC可能再提升
3. **Optuna + 全特征**: 用91特征(不做筛选)做更大规模贝叶斯搜索(200+ trials), 可能找到更优参数
4. **Normal市优化**: Normal市占82%样本, IC=0.054已不错但仍可提升 → 专项模型

### 中优先级
5. **CatBoost集成**: 安装测试CatBoost, 可能与LGBM互补 (不同树结构)
6. **多时域级联确认**: "仅30min+60min同时指向同一方向才执行" → 信号数减少但准确率可能更高
7. **宏观数据实时化**: SHIBOR/Repo接口修复 → 实时资金面因子
8. **跨品种套利**: TL/T/TF/TS 四个国债期货品种的跨期/跨品种信号

### 低优先级
9. **DeepLOB/LSTM**: 借鉴 `jessgess/deep-learning-for-order-book` (⭐137) 的LSTM订单簿模型
10. **RL交易**: 借鉴 `FinRL-DeepSeek` (⭐328) 做强化学习仓位管理
11. **基因编程因子挖掘**: 借鉴 `GP-Alpha-Miner` (⭐7) 自动发现新因子公式
12. **Alphalens完整集成**: IC衰减曲线 + 因子收益t-test

---

## 已知限制

- **Tick数据截止**: 2025-10-31, 之后微观结构因子全0 → 模型可能低估了近期微观结构的作用
- **SHIBOR/Repo**: AKShare接口偶有编码/SSL问题, 资金面因子可能缺失
- **社融/PBOC OMO**: 无稳定API, 依赖代理变量
- **Port 8501**: 被秋招助手Agent占用 → F_Agent固定用8503
- **中文路径**: `D:\桌面\F_Agent` 含中文字符, 某些工具可能有兼容性问题
- **模型文件**: 未纳入git (`.gitignore`), 需本地训练

---

## 日常操作速查

```bash
# 启动Dashboard
双击 D:\桌面\F_Agent_TL策略.lnk

# CLI操作
cd D:\桌面\F_Agent\src
python main.py --mode train          # 全量重训练
python main.py --mode inference      # 生成信号
python main.py --mode iterate        # 自我迭代
python update_market_data.py         # 更新行情
python cron_scheduler.py list        # 查看定时任务
python cron_scheduler.py daemon      # 启动守护进程

# 评估
python eval_runner.py --window 30    # 30天滚动评估
python factor_evaluator.py           # 因子评估
python optuna_optimizer.py --mode all --trials 50  # 超参搜索
```
