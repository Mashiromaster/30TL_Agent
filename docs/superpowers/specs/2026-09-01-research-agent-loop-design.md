# Research Agent 闭环编排设计

- **日期**: 2026-09-01
- **状态**: 已批准，待实现
- **目标**: 把散落的 factor/inference/fusion/memory 子系统串成一个可审计的自动化闭环，并让 fusion 的调整结果回流 memory，兑现简历第 2 条"自动化 Research Agent + Memory/RAG/反馈机制"。

---

## 1. 背景与缺口

`signal_fusion.py`（三层融合决策：Memory规则 + RAG政策 + LLM推理）**已被** `dashboard_v2.py` 的"AI融合决策"Tab 和 `strategy_agent.ai_fusion_decision()` 调用。真实缺口是：

1. **无编排入口**：要手动分别跑 `--mode train` / `--mode inference` / Dashboard 点融合，没有"一键跑完整链路"的自动化入口。
2. **反馈不回流**：`memory.record_signal` 只记录模型原始信号，fusion 的调整（方向/调仓/层级/理由）丢失，下次 fusion 无法知道"融合后到底准不准"。
3. **inference 未串 fusion/memory**：`inference.py` / `main.py` 都未串接 fusion 与 memory 回填。

## 2. 架构：ResearchAgent 编排类

新增 `src/research_agent.py`，`main.py` 新增 `--mode research`：

```
ResearchAgent.run_cycle()
  ① 数据/因子更新  → factor_extraction.run_process()      (df_factors.pkl)
  ② 信号生成       → SignalGenerator.generate_signal()     (MoE/base 自动路由)
  ③ 融合决策       → SignalFusionEngine.fuse(signal, memory, rag, alerts)
  ④ 记录到记忆     → TradingMemory.record_signal(fused)     ← 含 fusion 字段
  ⑤ 回填校准       → TradingMemory.update_actuals()         (有实际收益时)
  ⑥ 闭环报告       → outputs/research_cycle_{date}.json + 控制台摘要
```

**为何独立成编排类而非塞进 inference**：inference 职责是"模型→信号"单跳，不该知道 fusion/memory/rag。编排逻辑独立成 `ResearchAgent`，各子系统保持单一职责、可独立测试。这正是"自动化 Research Agent"该有的形态：一个明确的编排层协调各子系统。

## 3. 反馈回流（闭环核心）

- `memory.record_signal` 的 record 增加 4 个**可选**字段（默认回退到原始信号值 → 向后兼容，老记录不受影响）：
  - `fused_direction`：融合后方向
  - `fused_weight`：融合后仓位
  - `fusion_level`：none / rule / rag / llm
  - `fusion_reasons`：简短理由列表
- `memory.reflection_stats` 增加交叉统计 **fusion_level × 准确率**（回答"LLM 融合后信号比纯规则准吗"）。
- 回流路径：fusion 调整 → 记入 memory → 统计融合层准确率 → 下次 `SignalFusionEngine._apply_memory_rules` 可读到"rag/llm 层历史准确率"据此决定是否信任融合调整。

## 4. 数据流与错误处理

- `run_cycle()` 每步 try/except 独立降级：RAG 索引不存在 → 跳过 RAG 层；LLM key 缺失 → 跳过 LLM 层；**因子文件缺失 → 明确报错停止**。单步失败不中断整链，记入报告 `step_status`。
- 复用 fusion 已有的 `.env`/环境变量 key 管理。
- 报告 `outputs/research_cycle_{YYYY-MM-DD}.json`：`step_status`（每步 ok/skipped/error）+ 最终融合信号 + 是否触发方向调整。

## 5. Dashboard 衔接

`dashboard_v2.py` 现有"AI融合决策"Tab 顶部加「▶ 运行完整 Research 闭环」按钮，触发 `ResearchAgent.run_cycle()` 并展示 `step_status` + 融合层准确率统计。**不新增 Tab**，复用现有。

## 6. 落地文件

| 文件 | 改动 |
|------|------|
| `src/research_agent.py` | 新增 `ResearchAgent` 编排类 + `run_cycle()` |
| `src/main.py` | 新增 `--mode research` |
| `src/memory.py` | `record_signal` 加 4 fusion 字段；`reflection_stats` 加 fusion_level×准确率交叉统计 |
| `src/dashboard_v2.py` | "AI融合决策"Tab 加「运行闭环」按钮 + 融合层统计 |
| `docs/superpowers/specs/` | 本设计文档 |

## 7. 非目标（YAGNI）

- 不改甜点区参数、不改 MoE、不动因子体系、不新增 Dashboard tab。
- 不做 fusion 结果自动改模型权重（那是 self_evolution 职责）——本闭环仅统计融合层表现，不自动干预模型。
