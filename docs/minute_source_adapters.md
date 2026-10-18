# 分钟数据来源适配与近期获取

2026-10-04。在 [分钟接入与覆盖审计](minute_data_pipeline.md) 基础上增加两个入口，继续复用独立批次、分钟快照和因子候选，不另建行情数据库，不接交易账户。

## 1. 本轮交付边界

| 入口 | 可以做什么 | 不代表什么 |
| --- | --- | --- |
| 新浪近期窗口 | 显式请求一个具体 TL 合约，检查响应范围、类型和来源记录 | 不支持任意历史区间；不是交易级行情认证 |
| 天勤单合约 CSV | 离线核验官方导出布局，纳秒精确对齐，转换成规范分钟批次 | 不下载历史、不读取账号；周期与授权仍需要本人确认 |

核心流程：获取/上传 → 不落盘预览 → 确认本地使用及字段语义 → 独立批次 → 显式组装快照 → 原有因子候选链路。任何一步都不会自动训练、发布模型、下单或启动后台轮询。

API 接入技能指导本轮直接扩展已有 `market_data.fetch_bars` 严格模式，并复用 `minute_store.import_minute_csv`，没有引入另一套行情 SDK、缓存或自动覆盖策略。

## 2. 公开近期行情：先看清返回的是什么

官方 AKShare 源码使用固定新浪 `InnerFuturesNewService.getFewMinLine` 端点，以 `symbol`、`type` 请求；没有任意历史起止日期参数。字段说明未证明结束/开始时间标签或持仓量取值时点，示例条数也不是服务保证。[官方源码](https://github.com/akfamily/akshare/blob/main/akshare/futures/futures_zh_sina.py)、[字段文档](https://github.com/akfamily/akshare/blob/main/docs/data/futures/futures.md)

本轮真实只读 CLI 探测（2026-10-04 01:58 上海时间）：

- 合约：TL2612；返回 1,023 条。
- 返回范围：2026-09-23 13:54 ～ 2026-09-30 15:15（按提供方时间解析为上海时区）。
- 响应 SHA256：`6f4b3304d320d00ebad5bc582dba40e2c6e216ca36465604beab1480450a22cd`。
- 结果：`review_required`，`bar_label=unverified`，`accepted_rows=0`，没有生成批次。
- 本次可访问不保证下次可用、许可有效、时效达标或可用于交易；不能因此宣称原始历史缺口已经补齐。

CLI 默认无网络许可；请求必须显式开启：

```bash
# 仅一次请求、只预览，不创建任何批次或覆盖旧缓存
.venv/bin/python src/prepare_market_data.py fetch-recent \
  --symbol TL2612 --allow-network

# 只有已经自行核实 end 标签、单bar成交量、结束OI和本地许可后才使用：
.venv/bin/python src/prepare_market_data.py fetch-recent \
  --symbol TL2612 --allow-network --bar-label end \
  --confirm-semantics --license-note "已确认本地使用许可和字段语义" --commit
```

`--bar-label start` 明确将原始时间加 60 秒；不猜测标签。不知道标签就保留 `unverified`，只能查看返回区间，不能入库。选择标签和勾选确认仍只是调用方声明，`semantics_verified=false` 不会变成认证成功。

获取与保存分开：页面保存的是刚才预览的同一份字节，不再次请求行情。完整 capture 哈希绑定项目、合约、转换参数、规范 CSV 和响应摘要；修改元数据或 CSV、切换请求或项目会阻断旧预览入库。新获取会清除上次语义确认，避免沿用“本次”同意。

尚未结束的尾部 bar 会隔离并计数，截止时间取“检查时点”与“实际响应取得时刻”的较早者。明确超出可能形成中分钟的未来行直接拒绝。标签未知时，形成中条数为 `null`，不伪称已经核验。已有完整分钟不等于当天交易日已经完成，后续因子仍隔离当前日。

严格传输限制：一个具体季度合约、一次 GET、固定主机、请求超时 15 秒、无自动重试/跳转/缓存回退，响应最多 2 MiB、最多 5,000 行。超时是 HTTP 连接/读取超时，并非调度器级的绝对任务时限。拒绝重复 JSON 键、重复时间、异常身份/未知字段、缺 OI、非有限数值、布尔、复数及未声明单位的数字时间戳。限制是本地保护，不是供应商保留量承诺。

## 3. 天勤 DataDownloader CSV：离线、可重放的转换

天勤官方下载器可导出 60 秒 K 线，但下载功能需相应授权。多合约输出会按第一个合约对齐，因此这里仅支持逐合约导出。[官方下载器接口与源码](https://github.com/shinnytech/tqsdk-python/blob/master/tqsdk/tools/downloader.py)

接受以下官方固定顺序的 9 列；其中合约可以是其它明确的 TL 季度合约：

```text
datetime,datetime_nano,CFFEX.TL2606.open,CFFEX.TL2606.high,CFFEX.TL2606.low,CFFEX.TL2606.close,CFFEX.TL2606.volume,CFFEX.TL2606.open_oi,CFFEX.TL2606.close_oi
```

`datetime` 是上海本地无 offset、9 位小数的时间；`datetime_nano` 用十进制整数精确读取，两者必须表示完全相同的时刻。K 线为开始标签，`volume` 为区间成交量，`close_oi` 为结束持仓量。[官方行情语义](https://doc.shinnytech.com/tqsdk/latest/usage/mddatas.html)

适配器不会通过浮点解析纳秒，也不会凭采样间距猜周期。CSV 不含周期列，必须显式声明 `bar_seconds=60`；`interval_verified=false` 保留。缺口允许进入覆盖审计，不插值、不补零、不前向填充。未知列（包括非该官方布局的 `id`）、多合约、主连、非整分钟、时间不一致、重复/逆序、空值及错误金融数值均拒绝。

```bash
# 第一步只校验；不安装天勤SDK、不联网、不使用账号
.venv/bin/python src/prepare_market_data.py import data/imports/TL2606.csv \
  --input-format tqsdk --bar-seconds 60 \
  --provider my-authorized-tqsdk-export --license-note "自有授权，仅本地研究"

# 审查校验输出后，用相同参数加 --commit 保存独立批次
```

每批最多 25 MiB；大历史请按具体合约和时间区间分文件，不要将文件截断成不完整 CSV。

保存结构：

```text
data/minute_batches/minute-<uuid>/
    source.csv        原始天勤导出，保留open_oi、精确纳秒和原字节
    bars.csv          结束标签规范分钟；oi来自close_oi
    manifest.json     两份文件SHA、转换版本、周期声明、来源和授权说明
```

建快照时重新解析 `source.csv` 并比较完整转换记录及规范 CSV 字节。原始文件、转换元数据或规范行情不对应时拒绝组装，不能只靠各自独立哈希蒙混通过。转换血缘随批次进入快照及因子清单。已生成的快照独立验证自身数据与清单，不依赖之后源 CSV 仍在线。

## 4. 页面入口与验收

[本地工作台](http://127.0.0.1:8503/) → 混合决策工作台 → 数据与模型准备 → 分钟行情接入与覆盖审计。

- “公开近期行情”默认关闭联网且标签未核验；获取和保存是两个不同按钮。
- “本地 CSV 格式”可选择标准或天勤格式；后者必须确认导出周期 60 秒。
- 批次不会自动成为研究输入；仍需选择批次、组装快照、选择分钟来源并显式构建因子。
- CLI 失败返回非零退出码及脱敏 JSON；部分数据和未知语义不伪装为已补齐历史。

自动化验证包含合成天勤 CSV → 独立批次 → 分钟快照 → 真正因子计算，以及合成 HTTP 捕获 → 入库 → 快照来源传递。所有写入测试均在临时目录；真实原始行情哈希保持不变，未将合成数据写入项目行情目录。

2026-10-04 验收记录：

- 全量 `pytest`：1,202 项通过，5 项既有警告；本轮涉及的 7 个模块综合覆盖率 98%，`market_data`、`recent_minutes`、`recent_minutes_dashboard` 为 100%。
- 代码编译及 `.gitignore README.md docs src tests` 范围的差异空白检查通过；没有清理既有输出文件变更。
- 重启本地前端后确认默认合约为空、时间标签未核验、联网与保存确认均未勾选。02:12 上海时间通过页面再执行一次 TL2612 只读请求，返回同样的 1,023 条、同样的起止区间；页面显示“规范校验通过 0”“未保存”及不可用于实时交易的提示。本轮共两次真实单次请求（CLI 与页面各一次），均未保存。
- 页面切换天勤格式后确认显示官方 9 列示例、60 秒周期确认框且默认未确认；不再显示容易混淆的标准 CSV 下载模板。此次没有上传真实历史文件。
- 原始 `data/TL分钟级量价数据.pkl` SHA256 仍为 `f4624b962282e3d30b4b523b99ac0eefc3b73b12ef1416ae34dd6ec8a2cf1314`；实际批次与分钟快照列表仍为空。

本轮没有取得完整历史导出、独立确认新浪时间/数量口径、购买授权、启动轮询、训练实际模型候选或推送远程。完整历史修复依然需要合法数据来源、时间标签/单位和合约交易日历核对；已有封存集不再当成全新盲测复用。
