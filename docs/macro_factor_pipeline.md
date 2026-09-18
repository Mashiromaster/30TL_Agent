# 宏观因子管线

宏观因子现在以“可用时点”而不是“统计期日期”为主键。原始观测先在自身频率上计算变化、滚动统计和状态，再通过 `available_ts` 向后 as-of 合并到分钟线。禁止把月度数据先扩成日频后再滚动，也禁止在分钟表上计算名为 `5D`、`20D` 的宏观差分。

## 发布时点

优先级如下：

1. 原始数据中的 `available_ts`；
2. 原始数据中的 `release_ts`；
3. `data/macro/release_calendar.csv` 的历史发布记录；
4. 保守回退规则。

发布日历列固定为：

```csv
series,period_date,release_ts
pmi,2025-08-01,2025-08-31 09:30:00
cpi,2025-08-01,2025-09-10 09:30:00
m2,2025-08-01,2025-09-12 17:00:00
```

`period_date` 是统计期，`release_ts` 是策略首次可使用该数值的北京时间。真实日历缺失时，PMI、CPI、M2/社融分别使用月末、次月 15 日、次月 20 日的保守时点；日频市场数据在下一工作日 09:00 生效。

资金面数据中，Repo 历史接口按一年上限自动分段，保留 FR/FDR 的水平、5 日变化、20 日偏离及 FR007-FDR007 分层利差；LPR 同时兼容 AKShare 的中英文返回 schema，并作为低频政策利率生成水平、变化和期限利差。

## 新鲜度与缺失

日频因子默认 TTL 为 5 个自然日，月度因子为 50 个自然日。超过 TTL 后因子值置为 0，同时保留 `_macro_coverage_count`、`_macro_stale_count` 和 `_macro_max_age_days`，使模型能区分“真实零”“尚未发布”和“已经过期”。没有真实数据源的占位因子不会生成。

## 缓存与审计

原始缓存使用同名 `.meta.json` 保存 schema 版本、请求区间、实际覆盖区间和列信息；`force_refresh` 会下沉到原始数据层。刷新失败时仅在策略允许的情况下回退到可读旧缓存。写入使用临时文件加原子替换。

需要强制重取时可设置 `F_AGENT_MACRO_FORCE_REFRESH=1`。

因子构建完成后会生成 `outputs/factor_artifact_audit.json`，包含时间覆盖、常数/全零列、各列覆盖率以及宏观覆盖与陈旧比例。

## 模型验证

标签由 `adj_close` 按合约构造，默认不跨交易日。训练、验证和封存测试按交易日切分，并用 `label_end_ts` 清除边界重叠。模型报告同时给出：

- micro-only；
- macro-only；
- flat-combined；
- hierarchical slow-macro/fast-micro；
- gain 与 split 重要性；
- 封存测试上的按交易日宏观 block-permutation ΔIC。

模型选择只读取验证集。封存测试不参与模型、特征、GP 或路由选择，只在所有选择冻结后执行一次发布否决：封存 IC 非正时保留 `*_candidate` 候选产物，但不会覆盖生产模型、预测和重要性文件。GP 默认关闭；设置 `F_AGENT_GENETIC_MINING=1` 后，它只在训练集发现候选、在验证集筛选，并将冻结的 scaler、列顺序和程序保存到模型产物。

验证期还会切成 5 个连续日期块；只有至少 3 个块为正且中位 IC 为正，产物才会标记 `promotion_eligible=true`。该标记不读取封存测试结果。
