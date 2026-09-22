# Jev 宏观语义因子与影子门控

## 决策合同

- 目标：把带真实发布时间的政策、研报和新闻文本转换为可回测的概率因子。
- 下游动作：Jev 因子可以参与下一次 LightGBM 候选训练；在线 Jev 只记录影子判断。
- 当前基线：`Policy_Sentiment` 关键词多空计数。
- 不可接受错误：未来信息泄漏、移动模型版本、API 失败生成伪零因子、Jev 直接反转方向或下单。
- 回退：`F_AGENT_JEV_MODE=off` 且 `F_AGENT_JEV_GATE_MODE=off`，保留关键词因子。
- 生产门槛：语义密封集通过质量门槛，并且现有 walk-forward 与 sealed holdout 证明增量 IC/交易指标不退化。

Jev 只负责文本语义判断。日期计算、发布时间、数值运算、收益预测、仓位上限和订单执行继续由确定性代码与 LightGBM 主链负责。

## 数据合同

输入文档必须包含以下发布时间字段之一，按优先级使用：

1. `release_ts`
2. `published_at`
3. `pub_ts`
4. `date`

包含日内时刻时从该时刻生效；只有日期时，保守地从下一工作日 09:00 生效。`fetched_at` 不被视为发布时间。问题模板要求只判断文档明确表达的内容，不允许根据后续事件判断当时政策是否“正确”。

输出包括政策方向、长债方向、影响等级、是否新增信息、流动性/增长/通胀/供给渠道概率及各 Choice/Score 置信度。所有 `JEV_` 特征自动归入宏观特征组，默认 TTL 为 10 天。

每个请求按照“文档状态 + 固定问题模板 + 精确模型版本”计算 SHA-256。缓存只保存哈希、模型版本、模板版本和响应，不保存原文；缓存位置为 `data/rag/jev_cache/`，不会进入 Git。

## 运行模式

在 `.env` 中配置：

```dotenv
TYPESAFE_API_KEY=***
F_AGENT_JEV_MODEL=jev-1.13.0

# off: 不读取缓存、不请求 API
# cache: 只重放已经冻结的缓存
# online: 缓存未命中时请求 API 并原子写入缓存
F_AGENT_JEV_MODE=online

# off 或 shadow；shadow 只写审计字段，不改变方向和仓位
F_AGENT_JEV_GATE_MODE=shadow
```

禁止使用 `jev-latest` 或 `jev-preview`。模型别名变化会破坏历史因子复现。

首次启用时，先运行 `PYTHONPATH=src .venv/bin/python src/rag_tool.py build` 刷新 RAG 索引并导出 `data/rag/semantic_events.pkl`，再以 `F_AGENT_JEV_MODE=online` 运行因子构建，为其中带真实发布时间的文本生成缓存。当前语料如果没有真实发布时间，导出结果可以为空；这是防止前视偏差的预期行为，不应回退使用抓取时间或报告期。缓存播种完成后，回测建议改为 `F_AGENT_JEV_MODE=cache`，保证同一数据快照不再访问网络。API 不可用、响应结构错误或缓存缺失时，不生成该文档的 Jev 事件，已有 Jev 状态按自身时间戳和 TTL 延续，关键词基线不受影响。

## 语义质量门槛

`jev_evaluation.evaluate_choice_predictions()` 在带 `available_ts` 的密封中文标注集上计算 coverage、accuracy、macro-F1、Brier、log-loss 和 ECE。`assert_semantic_promotion_ready()` 默认要求：

- 至少 300 条完整标注；
- 概率覆盖率不低于 95%；
- macro-F1 比关键词基线至少提升 0.02；
- Brier 必须优于关键词基线；
- ECE 不高于 0.10。

通过这一层只代表 Jev 语义可进入候选实验，不代表可以上线。随后仍必须执行现有按日期切分、purge、walk-forward、宏观消融和 sealed holdout release gate。未通过时继续使用关键词基线与当前生产模型。

将密封集指标写入机器可读门槛报告：

```python
from jev_evaluation import write_semantic_gate_report

write_semantic_gate_report(
    "outputs/jev_semantic_gate.json",
    candidate_metrics,
    keyword_baseline_metrics,
    model="jev-1.13.0",
    question_schema_version="macro-document-v1",
    dataset_id="sealed-cn-macro-v1-sha256:<数据集哈希>",
)
```

训练入口会重新计算门槛，并核对模型版本、问题 Schema 和非空密封集标识。报告缺失、损坏、指标未通过或版本不匹配时，所有 `JEV_` 特征会从本次训练候选中自动剔除，关键词因子与其他宏观因子继续运行；不能只手工把 `passed` 改成 `true` 绕过检查。

## 文件入口

- `src/jev_client.py`：固定模型、问题模板、响应校验、无原文缓存、文档特征和影子门控。
- `src/text_factor.py`：真实发布时间、关键词基线、Jev 事件流与 TTL。
- `src/jev_evaluation.py`：语义校准指标、版本绑定的门槛报告与 fail-closed 校验。
- `src/signal_fusion.py`：`off/shadow` 在线审计；没有执行权限。
- `outputs/factor_artifact_audit.json`、候选模型和消融报告：记录模型、Schema、运行模式、文档覆盖率与实际 `JEV_` 特征；不记录原文或密钥。

官方接口依据：[Introduction](https://docs.typesafe.ai/introduction)、[Models](https://docs.typesafe.ai/models)、[Choice](https://docs.typesafe.ai/primitives/choice)、[Score](https://docs.typesafe.ai/primitives/score)、[Noul](https://docs.typesafe.ai/primitives/noul)、[Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13)。
