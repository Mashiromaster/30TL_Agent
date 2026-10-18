# TL 分钟数据接入、覆盖审计与补数指南

2026-10-04。继 [混合研究准备链路](hybrid_preparation.md) 后，本轮实现的是**本地数据接入闭环**，不是宣称历史数据已经补齐或策略可以上线。

后续已增加 [天勤离线导出适配与公开近期行情入口](minute_source_adapters.md)。下文“未发起行情请求”等描述是第一阶段验收记录；第二阶段的真实探测与操作边界见新文档。

## 1. 首先纠正缺口的解释

对现有 `data/TL分钟级量价数据.pkl` 做真实只读审计后：

| 指标 | 实际值 |
| --- | ---: |
| 原始记录 | 477,873 行 |
| 明确合约 / 观测日期 | 14 / 752 |
| 已观测合约日 | 2,180 |
| 符合本地常规 255-bar 模板 | 1,845 合约日 |
| 不完整 | 335 合约日，涉及 138 日 |
| 数据截至 | 2026-06-02 15:15，上海时间 |
| 原始数据最大未观测自然日间隔 | 10 天 |

**上一轮因子候选里的 207 天间隔，是稀疏数据被隔离后的候选间隔，不是原始文件连续缺失 207 天。** 例如 2025-11-03，TL2512 只有 17 个 15 分钟时间点，TL2603 只有 9 个约 30 分钟时间点。文件名含“分钟”不能证明其中所有行都是完整的 1 分钟 K 线。

以下是按已观测不完整日期聚合的**待核对范围**，不是交易所认证的应补交易日清单。范围之间可能包含假期、未观测日期或合约上市/到期例外：

| 合约 | 待核对范围 | 不完整合约日 | 本地模板缺少分钟数 |
| --- | --- | ---: | ---: |
| TL2512 | 2025-11-03 ～ 2025-12-11 | 29 | 5,962 |
| TL2603 | 2025-11-03 ～ 2026-03-12 | 86 | 19,743 |
| TL2606 | 2025-12-08 ～ 2026-05-22 | 108 | 26,124 |
| TL2609 | 2025-12-09 ～ 2026-05-22 | 107 | 25,884 |
| TL2606 | 2026-05-27 ～ 2026-06-02 | 5 | 314 |

合计按本地模板缺 78,027 个分钟位置；不能直接将此数字作为数据商应补条数。2025-11 至 2026-04 的已观测合约日均不满足该完整模板。2026-06-02 之后尚未取得可信新分钟历史，本轮没有把日线展开、插值或填零成分钟。

原始文件 SHA256：`f4624b962282e3d30b4b523b99ac0eefc3b73b12ef1416ae34dd6ec8a2cf1314`；本轮审计读取后未改变。

## 2. 数据源选择：先确认覆盖和授权，再接账号

本轮核对了官方文档，未使用数据商凭据、未开通服务、未发起行情 API 请求。

| 方式 | 可核验能力 | 采用边界 |
| --- | --- | --- |
| 已有合法 CSV 导出 | 可以保留具体合约、时间、OHLC、成交量、持仓量 | 本轮已实现统一入口；来源和授权说明是用户声明，不等于独立认证 |
| TqSdk | `DataDownloader` 可按时间区间导出历史；历史下载属于专业版；普通序列有长度限制 | 有授权后逐合约导出，避免多合约横向对齐造成误填或缺行。见 [历史下载](https://doc.shinnytech.com/tqsdk/latest/reference/tqsdk.tools.download.html)、[专业版说明](https://doc.shinnytech.com/tqsdk/latest/profession.html) |
| RQData | `get_price` 支持期货具体合约、分钟、起止日期及 OHLC/volume/open_interest | 核对账号覆盖与许可后接入；不以主力连续价替代交割合约。见 [通用 API](https://www.ricequant.com/doc/rqdata/python/generic-api)、[账号和配额](https://www.ricequant.com/doc/rqdata/python/manual) |
| Tushare | 官方当前接口 `ft_mins`，单次最多 8,000 行，可按时段循环，分钟权限独立申请 | 必须验证实际 TL 历史覆盖与账号权限；积分本身不证明有分钟权限。见 [历史分钟](https://tushare.pro/document/2?doc_id=313)、[权限说明](https://tushare.pro/document/1?doc_id=290) |
| 新浪 / AKShare | 分钟接口以 symbol、period 查询，不提供任意历史起止日期参数 | 只可作为近期窗口补充，不能承诺修复完整长历史；现有仓库已实现新浪适配，不再引入重复库。见 [AKShare 官方期货文档](https://github.com/akfamily/akshare/blob/main/docs/data/futures/futures.md?plain=1) |

选择建议：已有账号就优先验证已有授权能否导出上表的具体合约；无账号则先通过合法本地导出验收。**本轮没有默认购买某家数据商，也没有用公开近期行情冒充历史补数。** 新入口不调用旧 `update_market_data.update_raw_data`，避免直接替换生产原始文件。

### 时间戳与交易日例外

TqSdk 官方明确 `datetime` 是 K 线起点的 epoch 纳秒，`close_oi` 为结束时持仓量。对于确定为 60 秒的 bar，起点转终点可在导出适配时显式加一分钟，但不能对未知数据自动猜测或移位。[TqSdk 行情语义](https://doc.shinnytech.com/tqsdk/latest/usage/mddatas.html)

RQData/Tushare/新浪所核对页面没有统一证明其所有导出格式的起止标注语义。本轮导入器仅接受已确认的结束时间 CSV，不自动做数据商别名映射。

本轮核对了交易所 2023 年发布的 [TL 交易细则原文副本](https://np-newsimg.dfcfw.com/download/A2_cms_f_20230414214217120692?abc9304.pdf=&direct=1)：常规连续竞价 09:30–11:30、13:00–15:15，最后交易日仅上午；最后交易日原则为到期月第二个星期五，异常情形顺延。[官网产品入口](https://www.cffex.com.cn/30tl/) 和 [原通知](https://www.cffex.com.cn/jystz/20230414/32081.html) 本次读取超时，**未取得 2026 完整日历与每个合约到期数据**。

因此 255 根只是常规 end-labelled 本地模板；120 根早盘可能是特殊短日，也可能是缺数，当前不能自动认证。审计 `calendar.verified=false`；缺整日合约也需要上市/到期日历才能判定。

## 3. 本轮实现的最小闭环

```text
CSV文件 / 页面上传
    → 默认只读校验
    → 显式保存独立分钟批次
    → 显式选择批次，必要时纳入旧行情
    → 重叠冲突检查 + 不可变分钟快照
    → 显式构建因子候选
    → 原有就绪检查 / 候选训练 / 混合研究门禁
```

- `minute_quality.py`：纯内存覆盖审计；记录缺分钟、重复、越界时间、OHLC、未来行、当日未完成状态及分合约待核对区间。不修改输入、联网或写文件。
- `minute_store.py`：只接受外部 CSV/bytes；默认 dry-run，显式入库后保留独立哈希及来源声明。快照有完整输入批次清单，不能无选择地把所有文件混到一起。
- `prepare_market_data.py`：本地 CLI，单 JSON stdout，诊断走 stderr；错误退出非零，不静默成功。
- `minute_dashboard.py`：薄前端，先校验再确认导入；批次/快照读取、审计、组装均由按钮触发。页面重渲染不会执行这些操作。
- `factor_candidates.py`：支持显式 `minute_snapshot_id`；绑定快照数据及清单哈希，记录输入来源。选定快照失效时不偷偷回退旧数据。

数据隔离目录：

```text
data/minute_batches/minute-<uuid>/
    bars.csv          # 验证后的规范数据
    manifest.json     # 原文件摘要、规范数据摘要、取得时间和来源声明
artifacts/minute_snapshots/minset-<uuid>/
    bars.pkl          # 仅内部生成的受信任格式，不是上传入口
    manifest.json     # 显式输入、去重计数、覆盖、限制、哈希
artifacts/factor_candidates/local-<uuid>/
    factors.pkl + manifest.json + audit.json
```

旧 `data/TL分钟级量价数据.pkl`、生产因子、预测、参数、模型、交易记忆均不是新入口的写入目标。新分钟目录已加入 `.gitignore`，避免将授权数据和大文件误推送。

## 4. CSV 契约与操作

必需列（UTF-8，可带 BOM；25 MiB 上限）：

```csv
date,ticker,open,high,low,close,volume,oi
```

可额外提供 `trade_dt`，必须与上海本地日期一致。时间为已确认的上海时间 60 秒 bar **结束时刻**；volume 为该 bar 的成交量，oi 为该 bar 结束的持仓量。成交量/持仓量单位和单/双边统计口径仍需供应商独立核对，尤其不能未经核验混用不同来源的 OI 做主力选约。

拒绝未知字段、重复列/重复键、非实数/布尔/非有限数值、非法 OHLC、负数量、TL0 主连或非季度合约、非整分钟、未来行、错误交易日期、模板外时间。不接受外部 pickle。当前日允许存档但不视为完整研究日；部分历史日可以存档，覆盖报告必须保留“不完整”状态。

没有可靠标签或单位说明时先向提供方确认，不要为了通过校验更改时间、价格或合约代码。provider、license_note 只描述数据来源与本地使用授权，不能填 API key、密码或 token。

在仓库根目录：

```bash
# 只读审计现有原始数据；status=incomplete不是“无数据”，而是覆盖不合格
.venv/bin/python src/prepare_market_data.py audit

# 校验文件，不创建批次
.venv/bin/python src/prepare_market_data.py import data/imports/tl_minutes.csv \
  --provider my-authorized-export --license-note "自有授权，仅本地研究"

# 确认校验结果后才入库
.venv/bin/python src/prepare_market_data.py import data/imports/tl_minutes.csv \
  --provider my-authorized-export --license-note "自有授权，仅本地研究" --commit

# 查看已保存批次/快照（有界列表，不反序列化行情）
.venv/bin/python src/prepare_market_data.py list

# 把占位 ID 替换为实际返回值；每个输入都需显式选择
.venv/bin/python src/prepare_market_data.py snapshot \
  --batch-id 'minute-<uuid>' --batch-id 'minute-<another-uuid>'

# 有必要时才显式附加 --include-legacy；旧来源仍标未认证
.venv/bin/python src/prepare_market_data.py audit --snapshot-id 'minset-<uuid>'
.venv/bin/python src/main.py --mode prepare-factors --minute-snapshot 'minset-<uuid>'
```

自定义根目录的 `--base-dir` 放在子命令前。`snapshot` 会写新的独立快照；`audit` 与 `list` 不写文件；`import` 只有 `--commit` 才写入。`incomplete` 可用于定位问题，不代表可交易或可训练；非法输入返回非零退出码。列表每类最多 30 个清单、目录扫描最多 1,000 项，并非无限历史注册表。

前端：[本地工作台](http://127.0.0.1:8503/) →「混合决策工作台 → 数据与模型准备 → 分钟行情接入与覆盖审计」。可下载 CSV 表头和审计 JSON。导入、组装后不会自动切换来源；选择的分钟快照只影响下一次因子构建，不改变已保存因子或研究结果。

### 冲突、修订与可用时间

- 同一时间+合约，所有规范 OHLCV/OI 相同才去重；否则整次快照组装失败，不使用 keep-last，也不拿高优先级来源悄悄覆盖。
- 尤其旧文件可能混有 15/30 分钟 bar，其重叠记录往往与真正 1 分钟数据不等价。修复时应取得可信的完整新导出，单独构建替代快照；不要无条件 `--include-legacy`。
- 当确需人工解决修订冲突，须另行确定来源优先级及受影响范围，本轮未引入隐式修订策略。
- `acquired_at` 记录系统本次取得数据的真实时刻，不以历史 bar 日期伪造历史可用时间。as_of 是校验截止时点，不会回写采集时间。
- 哈希只证明内部字节一致，不能认证数据商、授权、市场真实性或历史修订。持有本地写权限的人仍能改文件及清单，不把内部 pickle 当敌对输入沙箱。

## 5. 验收与尚未执行

合成数据端到端测试已覆盖：CSV 只读检查 → 两个部分重叠批次 → 相同记录去重 → 快照加载 → 真正因子计算 → 来源核验。没有模型 release 或样本不足时，最终就绪状态仍 blocked。全部在测试临时目录内运行，不生成用于展示的假行情，不调用模型/行情服务或下单。

2026-10-04 验收记录：

- 全量自动化测试：**1,003 passed，5 warnings**；既有警告来自日期格式推断、LightGBM 特征名及 py_mini_racer 的结构布局弃用提示。
- 覆盖率：minute_quality 98%、minute_store 98%、minute_dashboard 98%、prepare_market_data 88%、factor_candidates 97%、hybrid_dashboard 100%；六模块合计 98%。
- 默认旧 raw 构建入口也补齐类型门禁：拒绝布尔、复数、重复列，以及单位未声明的数字型时间戳；不改变原有未来/当日数据隔离语义。
- 真实浏览器完成只读审计，显示 477,873 条记录、335 个历史不完整合约日、5 个待核对范围，与 CLI 一致。页面数据保留未认证提示；未点击导入、组装或训练。
- 真实批次/快照列表仍为空，原始文件 SHA256 与审计前一致。编译检查和本轮代码/文档范围的 `git diff --check` 通过；未改动既存 outputs/params.json 的 CRLF 空白提示。

现有真实行情仅做覆盖审计；没有导入新供应商数据，没有覆盖生产原始文件，没有再次训练已看过封存集的模型。后续要真正补齐数据，需要用户选择来源并提供合法导出，或在本地配置其授权账号（无需把密钥发到聊天）。完成补数、时间标签/单位与合约日历核验后，再使用新的验证设计和未使用封存区间评估，不能复用已看过的封存集包装为独立盲测。
