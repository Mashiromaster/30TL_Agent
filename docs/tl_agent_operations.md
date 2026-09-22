# TL Agent 运行、发布与故障处理手册

更新时间：2026-09-23

适用范围：TL P0 统一决策链与 P1 舆情监控。当前系统是研究系统，**没有券商、订单、成交或账户执行能力**。

## 1. 权威链路

```text
df_factors.pkl
  -> ModelRuntime（只读模型推理）
  -> SignalPolicy（唯一策略规则）
  -> SignalDecision / Effective_Position
       -> CLI / Dashboard / Backtest / TradingMemory / ResearchAgent

NewsProvider(cursor)
  -> SQLite Document + cursor 事务
  -> TLRuleAnalyzer + 可选 Jev
  -> EventAssessment / Alert / SourceHealth
  -> ResearchAgent 研究证据
  -X-> 方向 / 仓位 / 订单

训练
  -> artifacts/experiments/<run_id>/ 候选
  -> IC/稳定性 + 封存集 veto + 同策略成本回测
  -> 哈希 manifest
  -> artifacts/models/releases/<release_id>/
  -> artifacts/models/current.json 原子切换
```

默认策略是 `tl_regime_breakout_v1`。旧推理和排名策略只能通过显式 `--strategy` 选择，不能与默认策略共享 fingerprint 或记忆统计。存在执行滞后时，系统不会用分钟中位数猜测跨午休、周末、节假日或停牌后的成交时间：`effective_from=null`，并以 `execution_lag_bars` 明示生效需等待的可交易 bar 数。回测把收盘后形成的目标记为 `Executed_Position`，只有上一 bar 已持有的 `Held_Position` 才能获得当前 close-to-close 收益。

## 2. 常用命令

在仓库根目录执行：

```bash
# 舆情抓取一次
.venv/bin/python src/main.py --mode monitor

# 每 10 分钟持续监控；失败采用有上限的指数退避
.venv/bin/python src/main.py --mode monitor --watch --interval-seconds 600

# 使用现有因子运行 Research Agent；舆情、RAG、Jev、LLM 只做研究影子层
.venv/bin/python src/main.py --mode research

# 离线排障：不抓舆情、不调用 LLM
.venv/bin/python src/main.py --mode research --no-sentiment --no-llm

# 显式重建因子后运行研究闭环
.venv/bin/python src/main.py --mode research --rebuild-factors

# 更新因子并生成统一策略信号
.venv/bin/python src/main.py --mode inference

# 训练候选、评估、通过后发布；这是唯一标准生产发布入口
.venv/bin/python src/main.py --mode train

# 启动界面
.venv/bin/python -m streamlit run src/dashboard_v2.py
```

`monitor` 和 `inference/research --rebuild-factors` 会访问外部数据源；先确认网络、数据授权和运行环境。不要用测试命令触发这些入口。

## 3. 舆情数据与状态

| 路径 | 内容 | 权威性 |
|---|---|---|
| `data/intelligence/events.sqlite` | 原文、独立 source cursor、结构化评估、告警、源健康、运行记录 | 权威事件账本 |
| `outputs/intelligence/<run_id>.json` | 单次监控摘要 | 可重建报告 |
| `outputs/research_cycle_<run_id>.json` | 一次研究闭环的因子、策略、模型与舆情 provenance | 审计报告 |
| `outputs/fusion_signal.json` | 最新研究层影子建议 | 非交易指令 |

默认 provider 使用 AkShare 的东方财富全球 7×24 资讯流，再按 TL、利率、流动性和债券关键词过滤；不再误用带默认股票代码的个股新闻接口。它仍是可替换原型，不具备已验证的覆盖率、授权状态或延迟 SLA。游标由 `published_at + source_document_id` 组成，同时间戳消息会被包容性重读并由文档主键去重。

`source_stale=true` 表示本次传输/抓取失败；`content_stale=true` 表示虽然接口可访问，但最后一篇有效文档（从未收到文档时则从第一次成功空抓取开始计时）已超过 72 小时。内容基线独立持久化，连续成功空轮询不会刷新它。二者都与“成功抓取且暂时没有新事件”区分，失败或空结果不会清空最后一份有效文档和游标。单源每轮最多接收 200 篇、2 MiB 正文批量，并限制标题、正文、URL、metadata 和待分析队列；超限会增加 `documents_dropped`，本轮状态为 `partial_failure`，不会伪装成成功。

外部正文是不可信输入。检测到常见提示注入后，文档仍保留供审计，但评估会被隔离为 `quarantined_untrusted_content`：置信度和渠道分数归零、立场/影响改为 `unclear`、不调用 Jev、也不能开启告警。正常文档的 Jev 也只允许写入数值型 `JEV_*` 字段。`trade_action`、`tool_calls`、订单、broker 和目标仓位字段不会进入事件评估契约。

## 4. 模型发布门禁

`LightGBM_model.run_process()` 只先写候选目录。候选提升前依次检查：

1. 训练/验证/封存集时点切分和 schema；
2. 仅在验证集选择 flat 或 hierarchical 模型；
3. 验证期分块稳定性；
4. 封存集 IC 一次性 veto；
5. 用同一 `SignalPolicy`、执行滞后、佣金和滑点做成本后回测；
6. 至少 20 个观察日、策略实际有持仓、指标均有限、成本后总收益和 Sharpe 严格高于配置下限、最大回撤未突破配置上限；
7. 训练端写入绑定 release、run、策略 fingerprint 和模型 SHA-256 的 `gate_attestation`，发布端重新读取并独立验证，调用方不能用布尔参数绕过 gate；
8. 候选必须位于配置的 `artifacts/experiments/` 根目录内，所有候选文件的大小和 SHA-256 与 manifest 一致。

默认要求成本后总收益 `> 0`、Sharpe `> 0`，最大回撤不低于 `-35%`。这三项和至少 20 个观察日构成不可放松的安全基线；环境变量只能进一步收紧：

```bash
F_AGENT_RELEASE_MIN_TOTAL_RETURN=0.02 \
F_AGENT_RELEASE_MIN_SHARPE=0.50 \
F_AGENT_RELEASE_MAX_DRAWDOWN=0.25 \
.venv/bin/python src/main.py --mode train
```

收益阈值只是发布安全门槛，不是统计显著性或盈利承诺。未通过时保留候选和失败原因，不改变当前 release。通过后，`ReleaseManager` 在跨进程 release lock 内先物化并 fsync 不可变 release，再以一次替换原子更新唯一权威指针 `current.json`。旧模型、预测、重要性和评估文件随后才尽力刷新，允许短暂滞后；它们只是兼容视图，绝不能成为生产运行时的回退来源。

本轮实现没有执行训练，因此没有改变当前模型。当前工作区有未托管的 `models/trained_model.pkl`，但没有 `artifacts/models/current.json`；生产 `SignalGenerator`、Dashboard 和 ResearchAgent 会拒绝直接反序列化该 pickle 并失败关闭。只有显式的离线迁移/测试入口可设置 `allow_legacy=True`。要恢复生产推理，必须显式执行训练并让候选通过上述门禁。

旧版 `AdapterStack` 同样已退役：`adapter_stack.pkl` 不再被读取、写入、移动或删除，CLI、Dashboard 和 cron 都不能触发适配器训练或组合预测。即使遗留文件损坏，系统也只报告其存在，不会反序列化或“自动修复”。这条路径不会提供环境变量逃生口；后续若重做在线适配，必须先纳入候选 manifest、同策略回测和统一发布门禁。

## 5. 查看与回滚 release

查看当前指针和已有版本：

```bash
cat artifacts/models/current.json
find artifacts/models/releases -maxdepth 2 -name manifest.json -print
```

切回一个已经存在且校验通过的不可变版本：

```bash
.venv/bin/python src/main.py --mode rollback --release-id <release_id>
```

回滚会在 release lock 内重新校验 manifest 及全部 artifact，然后原子切换权威指针；旧路径兼容投影随后尽力刷新，失败时会告警但不会把已经生效的新指针改回去。不要手工编辑 `current.json`，也不要直接覆盖 `models/trained_model.pkl`。损坏的活动 manifest 或 release artifact 会使运行时失败关闭，而不会静默退回另一个模型；兼容投影不是权威读取源。

## 6. 调度

```bash
.venv/bin/python src/cron_scheduler.py list
.venv/bin/python src/cron_scheduler.py run sentiment_monitor
.venv/bin/python src/cron_scheduler.py daemon
```

首次运行会生成 `outputs/cron_jobs.json`。已有配置会迁移：补入 10 分钟舆情任务，并强制禁用旧 `weekly_adapt` 和 `bimonthly_retrain` 生产任务。月度训练指向统一 `main.py --mode train`。cron 支持五字段的列表、范围和步长，并采用标准的 day-of-month/day-of-week 语义。

调度器仍是单进程、顺序执行的轻量实现，没有分布式锁或任务队列；同一仓库不要并行启动多个 daemon。daemon 会写 `outputs/cron_daemon.json` 心跳，Dashboard 通过心跳时效和 PID 做跨平台的 best-effort 状态判断。子任务退出非零、超时、未知任务以及 monitor 的 `failed/partial_failure` 都会向 cron CLI 传播为非零退出码。

## 7. 失败语义

| 故障 | 系统行为 |
|---|---|
| 新闻 provider 超时 | 标记 `source_stale`，保留旧文档和 cursor；watch 退避后重试 |
| 抓取成功但内容超过 72 小时 | 标记 `content_stale`，整轮为 `partial_failure` |
| 文档批量或字段超限 | 丢弃超限项、记录 `documents_dropped`，整轮为 `partial_failure` |
| 正文命中提示注入 | 保留原文但隔离评估；不调用 Jev、不发告警、不形成方向性证据 |
| Jev/LLM/RAG 不可用 | 记录研究层错误；不阻断已就绪的确定性基础 Decision |
| 因子重建失败 | ResearchAgent 默认禁止静默使用陈旧因子 |
| 策略历史不足 | `decision_ready=false`、`missing_reason=insufficient_policy_history`；不进入融合或记忆 |
| 候选门禁失败 | 保留 `artifacts/experiments/<run_id>`；当前 release 不变 |
| 发布在指针替换前中断 | 旧 `current.json` 继续权威生效 |
| 指针已替换、兼容投影失败 | 新 release 继续权威生效；旧兼容视图可暂时滞后并告警 |
| 活动 release 哈希不一致 | 推理失败关闭，不回退到未验证 legacy 模型 |
| 调用旧 AdapterStack 入口 | 明确失败；遗留 pickle 保持原样，不读取、不改名、不删除 |

## 8. 配置与安全边界

- `DEEPSEEK_API_KEY`：只供研究解释、RAG 和反思；没有 key 时确定性路径仍可运行。Dashboard 中手工输入的 key 只保存在当前 Streamlit session，不写入进程全局环境。
- `F_AGENT_JEV_MODE=off|cache|online`：Jev 数据模式；线上模式需单独验证授权和稳定性。
- `F_AGENT_JEV_GATE_MODE=off|shadow`：仅允许关闭或影子审计。
- `F_AGENT_GENETIC_MINING=1`：训练候选时启用 GP；仍必须通过统一门禁。
- `F_AGENT_RELEASE_MIN_TOTAL_RETURN`：成本后总收益严格下限，默认 `0.0`，不能为负。
- `F_AGENT_RELEASE_MIN_SHARPE`：Sharpe 严格下限，默认 `0.0`，不能为负。
- `F_AGENT_RELEASE_MAX_DRAWDOWN`：发布回撤上限，必须在 `(0, 0.35]`。

Research Agent 和舆情 monitor 的构造接口不接受 broker 或 release writer。`SignalFusionEngine` 即使收到恶意 RAG/LLM 文本，也只保存 `proposed_direction/proposed_weight`；`adjusted_direction/adjusted_weight` 始终还原为基础 Decision，传入旧的 `allow_trade_adjustment=True` 会直接报错。RAG 网络采集仅允许 HTTPS、443 端口和显式官方域名白名单；每次 DNS 解析和重定向都拒绝私网/回环/链路本地地址，禁用环境代理，并限制 MIME 与响应体大小。本地 RAG 目录和文件拒绝符号链接、路径逃逸及超大文件。

## 9. 验证

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider
```

2026-09-23 离线结果：`201 passed, 2 warnings in 1.62s`。四个关键新增模块（`strategy_runtime`、`model_release`、`intelligence_events`、`sentiment_monitor`）合计 branch coverage 为 `82%`。两条 warning 来自 AkShare 间接依赖 `py_mini_racer` 在 Python 3.14 下的弃用提示。测试未访问真实新闻、未训练模型、未连接行情 watch 或券商。
