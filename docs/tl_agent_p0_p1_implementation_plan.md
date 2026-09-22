# TL Agent P0/P1 实施计划

状态：P0/P1 已实现并通过离线回归；等待 Gate 2 提交确认

基线：`main` / `8026714`

上位设计：[architecture_research_2026-09-22.md](./architecture_research_2026-09-22.md)

实施结果（2026-09-23）：八个 Slice 均已落地；新增和既有测试合计
`201 passed`，四个关键新增模块合计 branch coverage 为 `82%`。本轮未运行真实训练、在线舆情抓取、行情 watch 或任何交易动作，
也未修改已有模型和历史输出。运行、发布、回滚和故障边界见
[tl_agent_operations.md](./tl_agent_operations.md)。

当前工作区只有未托管 legacy pickle，没有活动 `current.json`。这是刻意保留的安全迁移边界：生产推理会失败关闭，直到用户显式执行训练并通过 release gate；本轮没有把未知历史模型包装成“已验证 release”。

## 1. 本轮目标

本轮只完成架构文档中的 TL P0 与舆情 P1：

1. 将模型推理、信号策略、回测、看板、ResearchAgent 和记忆回放接到同一套可版本化契约。
2. 修正“当前 bar 决策获得当前 bar 收益”的时序问题。
3. 将训练产物改为候选，只有统一评估通过后才能由唯一发布入口提升。
4. 实现增量舆情监控 Agent：采集、游标、原文、结构化事件、告警、源健康、回放。
5. 把舆情结果接入 ResearchAgent 的研究报告，但禁止其直接改交易方向、仓位或生成订单。
6. 保留现有 CLI、JSON 字段和 Dashboard 的兼容入口，逐步撤销重复实现。

本轮不接真实券商、不实现自动下单、不训练新模型、不承诺策略收益，也不进行四页面 UI 的整体改版。

## 2. 锁定的设计选择

### 2.1 权威 TL 策略

默认策略定义为 `tl_regime_breakout_v1`，以当前主训练流程后实际运行的 `Backtester` 参数为迁移基准：

- EMA 160；
- 480 bar 滚动阈值；
- 分市场状态阈值、确认窗口、最小持有期、冷却期与权重；
- 决策至少滞后一根 bar 生效；
- 成本、滑点和执行滞后属于 `StrategySpec`，不能由回测文件另行定义。

旧的看板甜点区策略保留为 `tl_rank_sweetspot_legacy_v1`，旧推理规则保留为 `tl_inference_breakout_legacy_v1`。它们只用于行为对照和历史解释，不共享默认策略名称或历史成绩。原 `backtest_v2` 的独立参数扫描已经撤销；任何规则变化都必须注册新的 `StrategySpec` 和 fingerprint。

### 2.2 Agent 与交易的边界

舆情 Agent 只拥有以下工具：新闻 provider、事件存储、结构化分析器、告警输出和研究报告读取。它不会获得 broker、订单服务、模型发布目录或 `SignalFusionEngine` 的写权限。

外部新闻和研报均是不可信输入。命中提示注入模式的文档会隔离为非方向性事件，不调用 Jev、不创建告警；其余情况下 Jev/LLM 也只输出 `EventAssessment`。概率是语义置信度，不是收益概率。分析失败保留原文并标记失败，不产生“中性”假结果。

### 2.3 存储与依赖

- 不新增运行时依赖；事务状态使用 Python 标准库 `sqlite3`。
- SQLite 保存 provider 游标、原文索引、事件评估、告警和运行状态。
- 原文内容及哈希是事实源；Chroma 仍是可重建的检索索引。
- 监控输出只写入 `outputs/intelligence/` 和 `data/intelligence/`。
- 模型候选与 release 使用完整目录和 manifest；`models/trained_model.pkl` 仅作为兼容投影，由发布服务更新。

## 3. 目标链路

```text
FactorSnapshot
  -> ModelRuntime.predict_frame
  -> SignalPolicy.evaluate_frame
  -> Decision + EffectivePosition
       -> CLI / Dashboard / Backtest / Memory / ResearchAgent

NewsProvider.poll(cursor)
  -> Document ledger + cursor transaction
  -> deterministic/Jev EventAssessment
  -> event clustering + Alert
  -> ResearchAgent evidence context
  -X-> signal direction / target weight / order

train_candidate
  -> unified policy backtest + release report
  -> ReleaseManager.promote
  -> versioned ModelRelease
```

## 4. 有序任务清单

### Slice 1：策略契约与纯 SignalPolicy

先新增失败测试，再实现：

- `StrategySpec`：策略 ID、版本、参数、成本、执行滞后和稳定 fingerprint。
- `SignalDecision`：内容寻址的 `decision_id`、`as_of`、`effective_from`、`execution_lag_bars`、方向、目标权重、置信度、策略与模型溯源。未来可交易 bar 无法由现有数据确定时，`effective_from` 必须为空，不能用分钟中位数猜测。
- `SignalPolicy.evaluate_frame()`：纯函数，不读写文件，不读取 `Target_Ret`。
- 输出统一兼容列：`Pred_Smooth`、`Upper_Q`、`Lower_Q`、`Raw_Signal`、`Signal_Confirmed`、`Target_Position`、`Effective_Position`。
- 输入时间必须有序且不重复；历史不足明确返回不可决策状态，不能静默补零。

首批测试：fingerprint 稳定、阈值边界、分状态确认/持有/冷却、未来数据不改变历史决策、乱序/重复拒绝、执行生效时间晚于决策时间。

预计文件：

- 新增 `src/strategy_runtime.py`
- 新增 `tests/test_strategy_runtime.py`

验收：同一预测快照只由该模块产生策略列和最终 Decision。

### Slice 2：ModelRuntime 与兼容 facade

- 从 `inference.SignalGenerator` 抽出只负责模型加载和预测的 `ModelRuntime`。
- 兼容当前 flat、hierarchical、GP 和合格旧 MoE 产物。
- ReleaseManager 可以为缺少 release metadata 的旧模型生成只读盘点签名；生产 `SignalGenerator` 默认拒绝反序列化未托管 pickle，只有显式离线迁移/测试可设置 `allow_legacy=True`。
- `SignalGenerator` 保留构造函数、`predict()` 和 `generate_signal()`，内部转发给 runtime/policy。
- `signal.json` 保留旧字段，追加策略 fingerprint、release ID、decision ID 和生效时间。

首批测试：flat/hierarchical 路由、GP schema 漂移、旧产物 metadata、compat facade 输出。

预计文件：

- 修改 `src/inference.py`
- 新增 `tests/test_model_runtime.py`

验收：模型层不再拥有信号阈值，也不隐式训练或发布。

### Slice 3：五个消费者统一与回测时序修复

- `SignalDashboard` 直接展示 policy 产物，不再计算另一套信号。
- `Backtester` 区分当前收盘后的 `Executed_Position` 与赚取当前 close-to-close 收益的上一 bar `Held_Position`，成本参数来自 spec。
- `TradingMemory.backfill_from_predictions()` 使用相同 policy，记录 fingerprint，避免不同策略混算。
- `ResearchAgent` 注入 runtime/policy，保留现有调用签名并增加无副作用测试模式。
- `SignalFusion` 只生成带 provenance 的研究调整结果；基础 Decision 不被覆盖。
- Streamlit runtime 缓存键包含模型签名与策略 fingerprint。

首批测试：CLI、Dashboard、Backtest、Memory、ResearchAgent 在 fusion 前的最后一条 Decision 完全一致；t 时刻决策只能作用于 t+1 及之后收益。

预计文件：

- 修改 `src/signal_dashboard.py`
- 修改 `src/backtest.py`
- 修改 `src/memory.py`
- 修改 `src/research_agent.py`
- 修改 `src/signal_fusion.py`
- 修改 `src/dashboard_v2.py`
- 新增 `tests/test_strategy_consumer_parity.py`
- 新增 `tests/test_backtest_execution.py`
- 新增 `tests/test_memory_policy_parity.py`
- 新增 `tests/test_research_agent.py`

验收：策略参数只允许在 registry/spec 中出现，消费者不能拥有自己的阈值常量。

### Slice 4：候选训练、统一评估与唯一发布者

- 新增 `ModelRelease` manifest 与 `ReleaseManager`。
- 训练先写 `artifacts/experiments/<run_id>/`，不直接更新生产模型。
- 统一评估至少包含 schema、当前验证门槛、封存集 veto、同一 SignalPolicy 的成本后回测，以及不可被配置放松的正收益、正 Sharpe、20 日观察和 35% 最大回撤安全基线。
- 完整候选目录校验哈希和训练端 `gate_attestation` 后，发布服务在跨进程锁内切换唯一权威 release 指针，再尽力更新非权威旧路径兼容投影。
- `retrain_optimized`、`retrain_v2`、`train_final`、`self_evolution` 不再拥有生产路径写权限；调度器不再默认调用旧生产写者。
- 不改变本轮已有生产模型，除非显式执行训练且候选通过全部 gate。

首批测试：候选失败不触碰现用 release、调用方不能用布尔参数绕过 gate、attestation/fingerprint/hash 绑定、并发发布串行化、兼容投影失败不回滚权威指针、只有 ReleaseManager 能更新生产指针、成本后收益或 Sharpe 不为正时阻止提升。

预计文件：

- 新增 `src/model_release.py`
- 修改 `src/LightGBM_model.py`
- 修改 `src/main.py`
- 修改四条旧训练/进化入口和 `src/cron_scheduler.py`
- 新增 `tests/test_model_release.py`

验收：源码扫描中，生产 release 更新只有一个权威写者。

### Slice 5：舆情事件账本与增量 provider

- 定义 `DocumentRecord`、`SourceCursor`、`SourceHealth`、`EventAssessment`、`Alert` 和 `MonitorRun`。
- 实现 SQLite 事务：原文落库成功后才推进 source cursor。
- provider 独立保存 cursor/high-watermark、`last_success_at` 和错误状态。
- 首个 provider 使用东方财富全球 7×24 流并过滤 TL/利率/债券关键词；接口允许后续加入央行、交易所和授权新闻源。
- 游标同时包含 UTC 发布时间和 source document ID，包容性重读同时间戳记录后由主键去重。
- `source_stale` 表示传输失败，`content_stale` 使用独立持久化基线表示最后有效内容（或首次成功空抓取）超过 72 小时，连续空轮询不能重置基线；文档数、字段长度、批量字节和待分析数均有上限。
- 文档级去重保留不同来源；相同事实可聚为一个 event，转载次数不增加 novelty。
- 抓取失败标记 `source_stale`，不能展示为“今日无事件”，也不能清空 last-known-good。

首批测试：原文与游标原子提交、崩溃恢复 exactly-once、分页重复/乱序/迟到、空成功与失败区分、每个源独立游标、相同内容多来源保留文档但聚成一个事件。

预计文件：

- 新增 `src/intelligence_events.py`
- 新增 `src/sentiment_monitor.py`
- 新增 `tests/test_intelligence_event_store.py`
- 新增 `tests/test_sentiment_monitor.py`

验收：重启不漏不重；源不可用时仍能查询旧事件并看到 stale 状态。

### Slice 6：结构化分析、告警与 Agent 安全边界

- 先执行确定性的 TL 关键词/实体/渠道 baseline，再可选调用现有 Jev。
- Jev 保持 `off/cache/online`、固定模型、schema 校验和 fail-closed。
- 检测外部文本中的提示注入模式；命中后隔离为 `quarantined_untrusted_content`，置信度/渠道分数归零，不调用 Jev，也不能打开告警。
- EventAssessment 记录 evidence、时间、资产、期限、渠道概率、置信度、schema/model 版本与缺失原因。
- 告警支持新建、抑制、重新打开和已报告状态。
- 监控 Agent 构造函数不接受 broker/executor/release writer。

首批测试：提示注入不能改方向/权重或创建订单；Jev 超时保留原文和基础量化 Decision；额外 `trade_action/tool_calls` 字段不进入事件契约；高影响事件也不能写交易产物。

验收：舆情分析输出永远是研究证据，不是交易命令。

### Slice 7：接入 ResearchAgent、CLI、Dashboard 与调度

- `ResearchAgent.run_cycle()` 增加 monitor step，将最新事件与 source health 放入报告。
- monitor 失败只记录 step status，不阻断因子、模型和基础 Decision；但一次 monitor/cron 命令会对 `failed` 或 `partial_failure` 返回非零，避免调度层误报成功。
- CLI 新增一次运行和 watch 模式；watch 有有界轮询、退避和干净退出。
- Dashboard 现有 AI/研究区域展示最新事件、证据、告警和源健康，不新建重复分析入口。
- 修复自定义 cron 对日期、月份和单个星期字段的匹配；增加舆情 monitor 任务、子进程错误传播和 daemon 心跳。
- RAG 的历史事件查询增加强制 `as_of` 过滤入口。
- RAG 网络采集增加 HTTPS/域名/DNS/重定向/响应体边界，本地材料拒绝符号链接、路径逃逸和超大文件；Dashboard API key 保持在 session 内，不写进程全局环境。

首批测试：monitor 失败降级、事件 as-of 排除未来/过期、cron 日期匹配、CLI once、watch 单轮异常恢复、Dashboard reader 不重新分析正文。

预计文件：

- 修改 `src/research_agent.py`
- 修改 `src/main.py`
- 修改 `src/dashboard_v2.py`
- 修改 `src/cron_scheduler.py`
- 修改 `src/rag_tool.py`
- 新增对应测试

验收：一次 Research cycle 可追踪到因子快照、模型 release、StrategySpec、基础 Decision、舆情证据和研究报告。

### Slice 8：回归、安全审查与运行文档

- 运行全量测试和新增关键路径覆盖。
- 静态检查重复策略参数、旧生产写者、monitor 对交易模块的非法导入。
- 对外部输入、文件路径、SQLite 查询、密钥和网络失败做安全复核。
- 增加迁移、运行、回滚与故障排查文档。
- 更新架构文档的“已实现/仍待实现”边界，不把 paper/live 写成已完成。

验收：全量测试通过；四个关键新增模块合计 branch coverage 达到 82%；兼容输出仍可被现有 UI 读取，但生产模型必须来自托管 release；不改动用户现有模型和历史输出。

## 5. 兼容与迁移原则

- 旧公开类和函数至少保留一个迁移周期。
- 旧 JSON 字段不删除，只追加溯源字段。
- legacy 策略可显式选择，但默认只使用 `tl_regime_breakout_v1`。
- ReleaseManager 可为已有 pickle 生成只读盘点 signature，但生产运行时不信任它；只能显式离线迁移/测试或重新训练后发布。
- 旧历史记忆没有 fingerprint 时标记 `legacy-unknown`，不与新策略统计混合。
- 旧 `AdapterStack` pickle 路径默认且不可配置地停用；状态页只报告遗留文件存在，不读取、移动、删除或参与预测。
- 所有持久化写入使用临时文件替换或 SQLite 事务。
- 发布指针替换前失败时旧 release 保持权威；替换后兼容投影失败时新 release 保持权威且旧视图允许滞后。事件原文/游标使用 SQLite 事务，失败不推进游标。

## 6. 完成标准

满足以下条件才算本轮完成：

1. 五个 TL 消费者对同一输入得到相同基础 Decision。
2. 回测不存在同 bar 收益归属，决策与生效时间可追溯。
3. 生产模型只有一个发布入口，训练失败或发布中断保持旧版本。
4. 舆情 Agent 可以重复运行、断点恢复、区分无事件与数据源失效。
5. 每个告警可以定位到原文、发布时间、首次接收时间和分析版本。
6. 舆情正文、Jev 或 LLM 无论输出什么，都不能直接修改交易方向、仓位或产生订单。
7. ResearchAgent 报告同时包含量化 Decision 和舆情证据，二者保留独立 provenance。
8. 全量测试通过，且本轮不触碰工作区原有未提交产物。

## 7. 提交与发布闸门

实施按以上 Slice 顺序推进，每个 Slice 先 RED、再 GREEN、最后重构。全部实现和审查完成后先展示 diff、测试结果和建议的 conventional commits；未经确认不提交、不推送，也不启动真实训练或外部交易。
