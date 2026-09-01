# F_Agent — TL 30年期国债期货智能量化策略系统

## 项目基本信息

- **仓库**: https://github.com/Mashiromaster/30TL_Agent
- **本地路径**: `D:\桌面\F_Agent`
- **Dashboard端口**: `8503` (8501被秋招助手占用)
- **Python**: 3.12 (Anaconda, `D:\Anaconda`)
- **Git**: `D:\Git\bin\bash.exe` (非标准位置, 需 `HERMES_GIT_BASH_PATH`)

---

## 系统架构

```
Data Pipeline (AKShare) → Factor Engine (151维)
→ CNN 时序卷积 (64维 Bottleneck Embedding)
→ LightGBM 双模型 (155特征, 12月+90d衰减)
→ Sweet Spot Ranking (15-30% short / 70-85% long)
→ AdapterStack (LoRA式每周残差修正)
→ Dashboard 14-Tab (Streamlit, Port 8503)
```

### 三阶段微调架构

```
① CNN 瓶颈嵌入 (按需训练)
   微观特征(25×30bar窗口) → 1D Temporal CNN → 64维稠密嵌入 → 注入 df_factors.pkl

② LoRA 增量适配 (每周)
   冻结基模型 + 残差适配器(20树,d=2)  |  反馈分析 → 训练 → 衰减堆栈
   f'(x) = f_base(x) + Σ g_adapter_i(x) × 0.85^weeks

③ 全量重训练 (每两月)
   吸收所有适配器经验 + CNN特征 → 全量训练 LightGBM (200树,d=3) → 清空堆栈
```

### LoRA 数学类比

| LoRA (神经网络) | F_Agent (树模型) |
|:--|:--|
| W' = W + A×B | f'(x) = f_base(x) + Σ g_adapter_i(x) × w_i |
| A×B: 低秩适配 (~1%参数) | g_adapter: 极小LGBM (20树,depth=2, ~1/20参数) |
| 全量微调 | 双月全量重训练 |

---

## 当前模型状态 (2026-06-04)

### 模型
- **架构**: LightGBM 双模型 (Base Model + HighVol/Trend Model)
- **训练配置**: 12个月滚动窗口 + 90天半衰时间衰减
- **特征数**: 91 → 实测最优 (40特征筛选IC从0.039跌至0, XGBoost/Ensemble均无法超越)
- **Test IC**: 0.0388 (Normal市 0.0536, HighVol市 -0.0006, Trend市 -0.1198)
- **提升**: 原始模型IC 0.015 → 0.039 (Normal IC从 **-0.200** 扭正为 **+0.0536**)
- **损失函数**: MAE (regression_l1) — MSE在极低信噪比下过拟合严重
- **LightGBM 参数**: max_depth=3, lambda_l1/l2=15, feature_fraction=0.3, min_child_samples=350

### 信号系统
- **方式**: 甜点区排名信号 (Sweet Spot Ranking)
- **参数**: smooth_span=60, lookback=480, confirm=10
- **做空甜点**: rank 15%-30% (准确率 72.7%)
- **做多甜点**: rank 70%-85% (准确率 63.6%)
- **跳过区间**: rank 0-15% 和 85-100% (极端排名=噪声, 准确率仅55.6%/46.9%)

### 数据
- **分钟行情**: 2023-04-21 ~ 2026-06-02 (488K行, 14个合约)
- **因子集**: 159K行 × 151特征 (+ 64 CNN嵌入后 = 215列)
- **预测集**: 23,916行 (测试期 2025-07-07 ~ 2026-06-02, 219天)
- **Tick数据**: 仅到 2025-10-31, 之后微观结构因子填0
- **交易记忆**: 219笔, 甜点区信号准确率 59.3%

### 自我进化状态
- **适配器堆栈**: 1个 (adapter_2026W23, IC=0.0113, weight=1.0)
- **最新进化**: Base IC 0.0228 → Combined IC 0.0232 (+0.0005)
- **CNN模型**: 已实现未训练 (PyTorch可用)

---

## 核心发现与经验教训 (CRITICAL)

### 1. 极度正则化是金融ML唯一有效策略
金融时间序列信噪比极低 (最优单特征IC仅 0.02-0.03):
- LightGBM: max_depth=3, lambda_l1=15, lambda_l2=15, feature_fraction=0.3, min_child_samples=350 — **必须严格执行**
- **任何放宽正则化的尝试都导致严重过拟合**
- MSE loss 追逐离群值, MAE 预测条件中位数更稳健
- **强正则化LGBM优于Optuna调参/XGBoost/Ensemble等复杂方案** — 低信噪比数据验证集IC是噪声，追逐它会过拟合

### 2. 模型自信 ≠ 准确
甜点区发现的根本原因: 模型在极端排名时(rank<15%或>85%)反而错误率最高:
- 极端做空 (0-15%): 55.6% (仅比随机略好)
- 做空甜点 (15-30%): **72.7%** (真正有效)
- 做多甜点 (70-85%): **63.6%**
- 极端做多 (85-100%): 46.9% (比扔硬币还差, 做多方向大概率错误)

### 3. 特征筛选砍掉交互效应
将91特征降到40个(按|IC|排序取Top40): IC从 0.039 跌到 0.000 → LightGBM的树结构依赖弱特征间的交互

### 4. 30分钟是最优预测时域
- 30min: IC 0.039 (最佳)
- 60min: IC -0.009 (信息衰减)
- 120min: IC -0.037 (几乎无预测力)

### 5. 训练窗口存在"甜点区"
- 9个月: Sharpe 2.01
- **12个月: IC 0.039** (当前最优, 覆盖更多制度转换)
- 18个月: IC 0.030 (旧制度数据成为噪声)
- 金融数据关键是"相关性"而非"数量"

### 6. 集成学习未能超越
- XGBoost单独: IC 0.016
- LightGBM+XGBoost加权集成: IC 0.015 (拉低LGBM)
- 多时域共识: IC 0.003

### 7. 宏观因子主导
Macro_Surprise_Composite 始终#1, 说明2025H2后市场由宏观政策驱动, 技术和微观因子降级

### 8. LoRA适配的可靠性验证
第一周适配器仅训练出1棵树(early stopping自动终止) — 低容量约束防止在弱信号上过拟合，验证了LoRA设计思路

### 9. CNN时序嵌入的合理性
受限于L1盘口数据，用**时间维度补偿空间维度不足**：学习微观特征在30bar内的动态演化模式优于静态快照

### 10. Windows Streamlit stdout 兼容性
MSYS/git-bash 伪终端下 stdout/stderr 是 pipe handle，所有 `print()` 触发 `OSError(22)`。四个入口文件全部在模块顶部无条件重定向:
```python
if sys.platform == 'win32':
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
```
API Key 管理: `.env` 文件(自动加载) > Dashboard侧边栏输入(type=password) > 系统环境变量。Key **永远不写入代码文件**。

---

## 因子体系 151维 + 64维 CNN

| 类别 | 数量 | 关键因子 |
|------|------|----------|
| 动量 | 7 | Short_Momentum_1D/3D/5D, Mid_Momentum_1M/2M, TSMOM, Momentum_Alignment |
| 波动率 | 6 | RV_30, RV_120, Vol_Surge, ATR_14, Vol_Regime |
| 微观结构 | 44 | Spread, Imbalance, Signed_Vol, VPIN, HF_RV, Cum_Net_Open, Close_Pressure, Open_Price_Push, Trade_Intensity, Vol_Disconnect |
| 量价 | 4 | OI_Volume_Flow, Smart_Money, Large_Trade_Direction |
| 技术 | 4 | MACD_Hist, RSI, BB_Position |
| 市场状态 | 4 | Market_Regime, Is_High_Vol, Trend_Consistency |
| 基差 | 3 | Basis_ZScore_20, Basis_ZScore_10, Basis_Trend |
| 宏观 | 14 | CN_US_10Y_Spread, PMI_ZScore, M2_Surprise, YC_*, Macro_Surprise_Composite |
| 增强因子 | 35 | OI_Growth_5D/20D, Vol_Breakout, ADX, Spread_Stress, Vol_of_Vol |
| **CNN嵌入** | **64** | **CNN_Emb_00 ~ CNN_Emb_63 (时空CNN瓶颈输出)** |

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

## 所有模块清单 (src/)

### 核心流程
| 文件 | 功能 |
|------|------|
| `main.py` | 入口: `--mode train\|inference\|iterate\|evolve` |
| `factor_extraction.py` | 因子构建主流程 (数据加载→基差→Tick→宏观→核心→组合→增强) |
| `LightGBM_model.py` | LightGBM双模型训练 (自动识别CNN特征+宏观特征, 9月窗口+60d衰减基础) |
| `inference.py` | SignalGenerator: 加载模型→预测 (旧版绝对阈值逻辑) |
| `backtest.py` | 分状态策略回测 (绝对阈值+杠杆控制) |
| `backtest_v2.py` | 排名信号回测+参数扫描 |
| `retrain_optimized.py` | 全量重训练+9窗口自动扫描 |

### CNN & 进化 (新增)
| 文件 | 功能 |
|------|------|
| `micro_cnn.py` | **MicroCNN 时空1D-CNN**: 25特征×30bar→64维瓶颈嵌入→注入因子文件 (~35K参数) |
| `self_evolution.py` | **LoRA进化引擎**: FrozenBaseModel + ResidualAdapter(20树,d=2) + AdapterStack + FeedbackAnalyzer |

### Dashboard & 信号
| 文件 | 功能 |
|------|------|
| `dashboard_v2.py` | **14-Tab全功能面板** (端口8503, 分两行7+7, 含API Key侧边栏) |
| `signal_dashboard.py` | 甜点区排名信号引擎 + 预测可视化 |
| `dashboard.py` | 旧版8-Tab (部分Tab通过pass-through复用) |

### 自我迭代 & 评估
| 文件 | 功能 |
|------|------|
| `self_iteration.py` | 自我迭代引擎 (IC监控/漂移检测/失败分析/时间衰减30d/14d/参数调优) |
| `eval_runner.py` | 滚动窗口模型评估 (IC序列/基线对比/退化检测/MD报告) |
| `factor_evaluator.py` | Alphalens风格因子评估 (IC衰减/分位数收益/换手率/相关性/综合评分) |
| `optuna_optimizer.py` | Optuna贝叶斯超参搜索 + 多模型集成 + 多时域预测 |
| `cron_scheduler.py` | 定时任务调度 (8个任务 + daemon守护进程) |

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
| `llm_intelligence.py` | DeepSeek AI情报分析 (Windows兼容: _log→os.write(2,...) ) |
| `llm_predictor.py` | LLM预测对比 |
| `rag_tool.py` | RAG研究工具 (爬虫+ChromaDB+BGE-small-zh+LLM生成, Windows兼容) |
| `memory.py` | 交易记忆系统 (记录/回填/统计/LLM反思) |
| `check_timestamp_format.py` | 时间戳格式检测 |

---

## 14-Tab Dashboard 结构

### 第一行 (7个)
```
📡 信号看板   — signal_dashboard.py (甜点区排名信号+预测图+绩效)
📊 市场监控   — dashboard.py (K线+成交量+状态)
🔬 因子分析   — dashboard.py (重要性+分位+异常)
🔍 因子评估   — factor_evaluator.py (Alphalens: IC排名+稳定性+相关性)
💰 回测表现   — dashboard.py (NAV+回撤+收益分布)
🌍 宏观环境   — dashboard.py (曲线+利差+指标)
🤖 AI情报     — dashboard.py (DeepSeek分析+新闻交叉验证)
```

### 第二行 (7个)
```
📚 研究RAG    — rag_tool.py (检索+过滤+简报)
🧠 交易记忆   — memory.py (准确率矩阵+LLM反思)
🔄 自我迭代   — self_iteration.py (诊断+漂移+参数调优)
📋 模型评估   — eval_runner.py (滚动IC+基线对比+退化检测)
⚙️ 超参优化   — optuna_optimizer.py (Optuna+集成+多时域)
⏰ 定时调度   — cron_scheduler.py (8个任务运行/暂停)
🔧 微调迭代   — self_evolution.py + micro_cnn.py (LoRA适配器+CNN嵌入)
```

---

## 自动化调度 (8个任务)

| ID | 频率 | 功能 |
|----|------|------|
| daily_update | 工作日 16:00 | AKShare拉取最新分钟行情 |
| daily_signal | 工作日 16:00 | 生成交易信号 |
| weekly_eval | 周五 17:00 | 滚动窗口IC评估 |
| weekly_memory | 周五 17:00 | 同步交易记忆 |
| **weekly_adapt** | **周六 08:00** | **LoRA适配器训练** |
| monthly_retrain | 每月1号 09:00 | 全量重训练 |
| monthly_iteration | 每月1号 10:00 | 自我迭代诊断 |
| **bimonthly_retrain** | **双月1号 09:00** | **全量重训练+清空适配器堆栈** |

```bash
python cron_scheduler.py daemon  # 后台守护进程
```

---

## API Key 管理

三种配置方式 (优先级从高到低):
1. **Dashboard 侧边栏输入** — type=password输入框，即时生效，仅当前会话
2. **`.env` 文件** — 项目根目录，启动时自动加载 → `os.environ`
3. **系统环境变量** — 传统方式 `set DEEPSEEK_API_KEY=***`

模板: `.env.example` → 复制为 `.env` 填入真实key (`.env` 已加入 `.gitignore`)

---

## 量化ML管道规则 (CRITICAL)

1. 每次重训练后必须立即运行 `TradingMemory.backfill_from_predictions()` 重建交易记忆，否则Dashboard显示过期数据
2. 新建的CLI工具必须立即集成到Dashboard tab中，不要等用户提醒
3. 金融ML中强正则化(LGBM max_depth=3, lambda=15, min_child_samples=350)优于Optuna调参/XGBoost/Ensemble等复杂方案 — 低信噪比数据验证集IC是噪声，追逐它会过拟合
4. 端口8501被秋招助手占用，F_Agent Dashboard固定使用8503端口
5. F_Agent量化ML管道规则: 修改任何模型/特征后必须先验证IC不倒退再上线

---

## 日常操作速查

```bash
# 启动Dashboard
双击 D:\桌面\F_Agent_TL策略.lnk → http://localhost:8503

# === 核心流程 ===
cd D:\桌面\F_Agent\src
python main.py --mode train            # 全量重训练 (含CNN特征)
python main.py --mode inference        # 生成信号
python main.py --mode iterate          # 自我迭代
python main.py --mode evolve           # 微调迭代状态

# === CNN 微观嵌入 ===
python micro_cnn.py pipeline           # 一键训练CNN+提取+注入
python micro_cnn.py status             # 查看状态

# === LoRA 增量适配 ===
python self_evolution.py weekly        # 每周适配
python self_evolution.py bimonthly     # 双月重训练
python self_evolution.py status        # 状态
python self_evolution.py predict       # 组合预测

# === 评估 ===
python eval_runner.py --window 30      # 30天滚动评估
python factor_evaluator.py             # Alphalens因子评估
python optuna_optimizer.py --mode all --trials 50  # 超参搜索

# === 数据维护 ===
python update_market_data.py           # 更新行情
python cron_scheduler.py list          # 查看定时任务
python cron_scheduler.py daemon        # 守护进程
```

---

## 已知限制

- **Tick数据截止**: 2025-10-31，之后微观结构因子全0 → CNN仅在tick覆盖期训练
- **SHIBOR/Repo**: AKShare接口偶有编码/SSL问题, 资金面因子可能缺失
- **社融/PBOC OMO**: 无稳定API, 依赖代理变量
- **Port 8501**: 被秋招助手Agent占用 → F_Agent固定用8503
- **中文路径**: `D:\桌面\F_Agent` 含中文字符, 某些工具可能有兼容性问题
- **模型文件**: 未纳入git (`.gitignore`), 需本地训练
- **CNN需PyTorch**: 若未安装, LightGBM正常工作(只是没有CNN特征)

---

## 未来优化方向

### 高优先级
1. **CNN训练+评估**: 运行 `micro_cnn.py pipeline` → 重训LightGBM → 对比IC提升
2. **Tick数据补充**: 2025-11之后的Tick快照缺失, 重新获取后IC可能再提升
3. **CatBoost集成**: 安装测试CatBoost, 可能与LGBM互补
4. **Normal市专项优化**: Normal市占82%样本, IC=0.054仍可提升

### 中优先级
5. **多时域级联确认**: "30min+60min同时指向同一方向才执行" → 减少假信号
6. **宏观数据实时化**: SHIBOR/Repo接口修复 → 实时资金面因子
7. **跨品种套利**: TL/T/TF/TS 四个国债期货品种

### 低优先级
8. **DeepLOB/LSTM**: LSTM订单簿模型
9. **RL交易**: 强化学习仓位管理
10. **基因编程因子挖掘**: 自动发现新因子公式
