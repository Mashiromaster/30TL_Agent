# A 股实时交易系统：数据源评估与第一阶段方案

更新日期：2026-09-22

## 结论

可行，但需要先区分三件事：网页展示的“当前价格”、可持续接收的实时行情、可以驱动自动下单的合规行情。它们在事件时间、延迟保证、可用性、授权和成本上完全不同。

当前项目先实现只读行情数据面，不连接真实账户、不发送订单。默认适配器直接查询腾讯网页报价（也是开源 `Ashare.py` 的底层数据源之一），返回最新价、源时间和五档盘口；AKShare/东方财富作为显式可选源。两者都适合研究、演示和监控，但都没有延迟 SLA，系统强制标记 `trade_eligible=false`，不能直接驱动实盘下单。

## 数据源全面比较

| 数据源 | 形态 | 实时粒度 | 优点 | 主要限制 | 建议用途 |
|---|---|---|---|---|---|
| Ashare（新浪/腾讯） | HTTP 拉取 | 日/分钟 K 线，最后一根含当前价 | 极简、免费、双网页源 | 不是推送；无 SLA；源码请求无超时；最后一根分钟线不等于逐笔行情 | 研究原型，不直接下单 |
| 腾讯网页报价 | HTTP 拉取 | 最新快照、五档、源时间 | 免费、单股请求轻量；本阶段默认实现有超时和重试 | 未文档化网页接口；时间字段语义和延迟无 SLA | 第一阶段原型与看板 |
| AKShare / 东方财富 | HTTP 拉取 | 最新快照、五档 | 免费、零账号、现有依赖可直接用 | 网页接口可能变更或限流；当前函数没有源事件时间 | 显式备用原型源 |
| Tushare Pro `rt_min` | HTTP 拉取 | 1/5/15/30/60 分钟 | 文档化、多代码批量 | 需要权限；是分钟线，不是逐笔推送 | 分钟策略与备份 |
| QMT / XtQuant | 本地客户端 + 回调推送 | tick、分钟、L1/L2（取决于权限） | 国内券商生态；行情与交易接口相邻；可全推 | 需要支持 QMT 的券商账户和本地 MiniQMT；权限依账户 | **优先的第二阶段实盘通道** |
| RQData | Python SDK / WebSocket | L1 tick 五档、实时分钟 | 文档化推送、带事件时间、适合实盘/备份 | 商业授权与费用 | 生产数据源或 QMT 备源 |
| Futu OpenAPI | OpenD + 推送 | 快照、逐笔、盘口、K 线 | API 完整、推送模型成熟 | A 股权限、额度和可交易范围依账户；国内券商账户不通用 | 已有富途生态时采用 |
| Wind Client API | 终端/API | 多资产实时行情 | 机构级数据覆盖 | 商业终端和授权成本高 | 机构研究与数据治理 |
| 交易所直连 Level-1/2 | 专线/网关/流 | 官方行情流 | 最权威、字段和链路可控 | 行情许可、技术接入和成本高，禁止无授权再分发 | 机构级终局方案 |

主要资料：

- [AKShare 文档](https://akshare.akfamily.xyz/)
- [mpquant/Ashare 源码](https://github.com/mpquant/Ashare/blob/main/Ashare.py)
- [Tushare A 股实时分钟接口](https://tushare.pro/document/2?doc_id=374)
- [XtQuant 行情模块](https://dict.thinktrader.net/nativeApi/xtdata.html)
- [RQData 实时行情推送](https://www.ricequant.com/doc/rqdata/python/generic-api)
- [Futu 行情接口概览](https://openapi.futunn.com/futu-api-doc/en/quote/overview.html)
- [上证所 Level-1 行情](https://www.sseinfo.com/services/assortment/level1/)
- [上证所行情产品价格](https://www.sseinfo.com/services/cpfwjg/)
- [JQData 独立客户端文档](https://www.joinquant.com/help/api/doc?id=9875&name=JQDatadoc)

独立 JQData 更适合历史研究：其独立客户端文档说明当日分钟数据在收盘后才可用，不应把回测平台中的 `get_current_data` 能力误认为外部实时行情接口。

## A 股实盘还需要单独建模的约束

行情接通不等于交易系统完成。后续订单/风控层必须从券商或交易所元数据读取并验证交易日历、可用股份（含 T+1 约束）、申报数量单位、不同板块和证券状态的价格笼子/涨跌幅、集合竞价与连续竞价阶段、停牌以及撤单状态。以上规则会随板块、证券状态和规则修订变化，不能散落成策略代码里的常量。第一阶段只识别通用时段，并明确将节假日状态标为未验证。

## “实时”的工程定义

系统不能把 HTTP 返回时间当成成交所发生的时间。标准行情事件至少需要：

- `provider_timestamp`：行情源给出的事件时间；
- `received_at`：本机收到响应的时间；
- `source`：数据供应商和接口；
- `market_phase`：集合竞价、连续竞价、午休、收盘；
- `is_stale`：基于事件时间计算的陈旧状态；
- `trade_eligible`：是否满足策略下单数据门槛。

统一字段中，`volume_lots` 的单位是手，`bid_volume_shares` / `ask_volume_shares` 的单位是股；适配器必须在边界处完成换算，策略层不接触供应商私有单位。

腾讯适配器解析了行情源返回的时间，并用它进行 15 秒陈旧性检查；但网页接口没有承诺该字段一定是交易所事件时间，因此仍标记 `provider_timestamp_semantics_unverified`。AKShare 当前适配器完全没有 `provider_timestamp`，所以 `is_stale=null`（未知），而不是错误地写成 `false`。两个原型源的 `trade_eligible` 永远为 `false`。生产适配器必须有可验证事件时间、交易日历、重连/心跳、序列去重、丢包检测与授权记录。

### Ashare 的“实时”到底意味着什么

截至本次检查，`Ashare.py` 的分钟实现每次发起 HTTP 请求获取分钟 K 线，并用腾讯报价中的当前价覆盖最后一行 `close`。因此，开盘期间**重新调用** `get_price(..., frequency='1m')` 时，最后一行 `close` 通常会贴近当前价；它不是 WebSocket/tick 推送，也不代表最后一行的 `open/high/low/volume` 都与这个当前价同时更新。日线接口同样是在查询时刷新当日 K 线。可以用于低频展示或分钟研究，不能据此推断“每笔成交实时到达”或“延迟有保证”。

## 推荐架构

```text
腾讯/AKShare(原型) / QMT(主源) / RQData(备源)
                    │
                    ▼
       QuoteProvider 统一适配层
                    │
                    ▼
      时间戳校验 → 去重 → 陈旧性门禁
                    │
                    ▼
       内存最新快照 + 追加式原始事件日志
                    │
             ┌──────┴──────┐
             ▼             ▼
         策略/因子       实时看板
             │
             ▼
       风控 → 模拟撮合 → 券商下单
```

推进顺序：

1. 已实现：统一只读行情接口、A 股代码规范化、腾讯/AKShare 两种适配器、五档快照、源时间、请求超时与重试、市场阶段、质量门禁、一次查询和持续轮询。
2. 下一步：根据实际券商选择 QMT；若没有 QMT 权限，则先接 RQData WebSocket。用同一份接口替换 AKShare，不动策略层。
3. 建立追加式事件日志、断线重连、心跳、双源差异告警、交易日历和盘中回放。
4. 接模拟撮合与组合级风险控制，至少经过一个完整市场周期的影子运行。
5. 最后才接真实委托，并设置人工总开关、单笔/单日限额、重复订单幂等和一键撤单。

## 已实现的第一步

文件：`src/realtime_quotes.py`

单次查询：

```bash
PYTHONPATH=src .venv/bin/python src/realtime_quotes.py quote 600000 000001.SZ
```

每 3 秒查询一次，共 5 次；输出为 JSON Lines：

```bash
PYTHONPATH=src .venv/bin/python src/realtime_quotes.py watch 600000 --interval 3 --count 5
```

如需显式切换到 AKShare/东方财富：

```bash
PYTHONPATH=src .venv/bin/python src/realtime_quotes.py quote 600000 --provider akshare
```

返回值中的关键安全字段：

```json
{
  "symbol": "600000.SH",
  "last": 10.25,
  "provider_timestamp": "2026-09-22T10:00:05+08:00",
  "received_at": "2026-09-22T10:00:10+08:00",
  "is_stale": false,
  "trade_eligible": false,
  "quality_reasons": [
    "unofficial_web_source_no_sla",
    "provider_timestamp_semantics_unverified",
    "trading_calendar_not_verified"
  ]
}
```

交易时段判断目前只用北京时间和工作日作为提示，不会假装识别法定休市日；因此 `trading_day_verified=false`。下一生产适配器必须接入交易所/券商交易日历。

## 验收标准

- 同一股票代码的常见写法统一成 `600000.SH` / `000001.SZ`；
- 网络失败直接报错，不回退到旧行情冒充实时值；
- 源事件时间缺失、陈旧或语义未验证时不能进入下单链路；
- 多次轮询能连续输出结构一致的快照；
- 生产接入前，必须用 QMT/RQData 的盘中事件验证延迟分布、断线恢复、重复/乱序与收盘切换。
