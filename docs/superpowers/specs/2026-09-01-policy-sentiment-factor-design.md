# 研报文本情绪因子设计 (Policy Sentiment Factor)

- **日期**: 2026-09-01
- **状态**: 已批准，待实现
- **目标**: 把宏观研报（非结构化文本）转成日频文本情绪因子，前向填充到分钟 bar 进入 LightGBM 训练，兑现简历"宏观信息检索与微观交易决策**联合建模**"——研报信号从决策层下沉到因子层。

---

## 1. 背景与粒度错配

研报是「宏观+低频」（一份报告覆盖数周），模型吃的是「微观+30min bar」（159K 行）。因此研报变不成真正的微观因子，而是**低频文本情绪因子前向填充到每个 bar**。

现状：`rag_tool.py` 已爬研报进 ChromaDB；`signal_fusion.py` 在**决策层**用关键词匹配调仓。本设计把它下沉到**因子层**——成为 `df_factors.pkl` 的一列，直接进模型训练。

## 2. 核心难点：防前视偏差

研报文档多数只有 `fetched_at`（爬取日），部分源（新浪研报，`rag_tool.py:334`）有真实发布日 `date`。**只用有真实发布日的研报**。三重防前视保险：

1. 无真实发布日的研报 → 丢弃
2. `available_date = 发布日 + 1天`（T+1 生效，当天研报次日才影响 bar）
3. merge 后 `ffill`——只向未来填充，绝不回填过去

（与现有 `merge_macro_to_minute` 完全一致的时间对齐范式。）

## 3. 打分：关键词词典（零成本、可复现）

复用 `signal_fusion.py` 已有词典：
- bull = [降准, 降息, 宽松, 利多, 下行, 回落, 流动性充裕]
- bear = [加息, 紧缩, 利空, 上行, 通胀, 收紧, 流动性紧张]

`Policy_Sentiment = (bull_hits − bear_hits) / (bull_hits + bear_hits + 1)` ∈ [−1, 1]

## 4. 数据流

```
① 取文档   已存研报中仅保留有真实发布日 date 的
② 打分     关键词词频 → Policy_Sentiment ∈ [-1,1]
③ 日频聚合 同一发布日多份 → 均值; 生成 date→sentiment
④ 防前视   available_date = 发布日 + 1天
⑤ 注入     复用 merge_macro_to_minute(df, sentiment_df) → ffill → 缺失填 0
           产出 2 列: Policy_Sentiment, Policy_Sentiment_MA5 (5日均值降噪)
```

## 5. 因子体系登记

`LightGBM_model.py` 的 `macro_patterns` 加 `'Policy_Sentiment'` → 现有 auto-detect 机制自动纳入训练特征，**无需改训练逻辑**。

## 6. 错误处理

- 无研报 / 无真实发布日 → 因子全 0（模型 auto-detect 到全 0 列不崩，仅无贡献），打印告警。
- 不阻断因子构建主流程（try/except 包裹，非致命，与 genetic mining 一致的降级风格）。

## 7. 落地文件

| 文件 | 改动 |
|------|------|
| `src/text_factor.py` | 新增 `build_sentiment_factor(base_dir) → DataFrame[available_date, Policy_Sentiment, Policy_Sentiment_MA5]` |
| `src/factor_extraction.py` | `run_process` 在 `calculate_enhanced_factors` 后、保存前，插入文本情绪 merge 步（复用 `merge_macro_to_minute`） |
| `src/LightGBM_model.py` | `macro_patterns` 加 `'Policy_Sentiment'` |
| `docs/superpowers/specs/` | 本文档 |

## 8. 非目标（YAGNI）

- 不接 LLM 打分、不做事件抽取、不改甜点区/MoE、不新增 Dashboard tab（因子自动出现在现有「因子分析」tab）。

## 9. 验证

- 合成研报数据自检：打分公式 [−1,1]、T+1 对齐、无研报时全 0 降级。
- 真实 IC 增量需在生产环境（有爬取研报 + df_factors）跑 `--mode train` 前后对比（遵循 rule #5 不倒退）。
