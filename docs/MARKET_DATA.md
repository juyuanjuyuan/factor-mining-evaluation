# 行情数据约定

以下文件统一存放在项目的 `data/` 目录。

所有行情宽表都使用相同的二维结构：

- 行：交易日，升序 `DatetimeIndex`
- 列：六位证券代码，与 `close_df.pq` 顺序完全一致
- 缺失：使用 `NaN`，不以前值、零或代理值填充

| 文件 | 评价符号 | 含义 |
|---|---|---|
| `open_df.pq` | `o` | 复权开盘价 |
| `high_df.pq` | `h` | 复权最高价 |
| `low_df.pq` | `l` | 复权最低价 |
| `close_df.pq` | `c` | 复权收盘价 |
| `volume_df.pq` | `vol` | DataYes `TURNOVER_VOL` 原生数值 |
| `amount_df.pq` | `amt` | 成交额；原文件名为 `vol_df.pq` |
| `vwap_proxy_df.pq` | `vwap` | VWAP 代理：`(high + low) / 2` |
| `market_cap_df.pq` | `cap` | 通联总市值，原始长表对齐后宽表 |
| `float_market_cap_df.pq` | — | 公司行情源提供的流通市值；当前保留为扩展数据，尚未映射为表达式符号 |
| `limit_up_price_df.pq` | — | 公司行情源逐日精确涨停价，与复权 OHLC 使用同一尺度 |
| `limit_down_price_df.pq` | — | 公司行情源逐日精确跌停价，与复权 OHLC 使用同一尺度 |
| `limit_ratio_df.pq` | `limit` | 按代码板块推导的常规涨跌幅限制比例宽表 |
| `st_status_df.pq` | `st` | ST/*ST point-in-time 状态；标准存储为 true-only `day/code/是否st` 长表，加载时规范为布尔宽表 |
| `delisting_period_status_df.pq` | `delisting`（仅 evaluator） | 正式退市整理期 point-in-time 状态；标准存储为 true-only `day/code/is_delisting_period` 长表 |
| `行业数据.parquet` | `industry`（仅 evaluator） | 动态一级行业长表：`trade_date/security_code/industry_l1_code` |
| `security_name_reference.parquet` | — | 持仓展示用证券代码名称表；覆盖历史上市/退市证券，名称为当前或退市前最后简称 |
| `industry_l1_name_reference.parquet` | — | 持仓展示用中信一级行业代码名称表 |

两张展示对照表由
`scripts/data/build_holding_reference_tables.py` 生成，构建来源和覆盖统计记录在
`data/manifests/holding_reference_manifest.json`。证券主表不包含完整的历次更名有效期，因此
`security_name_reference.parquet` 不能被解释为信号日的 point-in-time 简称；网页明确显示为
“当前或退市前最后简称”。行业名称只翻译持仓中已经按信号日确定的行业代码，不参与因子计算、
中性化或收益标签。

## 特殊状态来源与区间口径

`st_status_df.pq` 和 `delisting_period_status_df.pq` 由
`scripts/data/import_company_special_status.py` 从公司数据表
`datayes.equ_inst_sstate` 的下载产物导入。导入器先核对源 `manifest.json` 中的
SHA256，再将状态轴限制在 `close_df.pq` 的交易日与六位证券代码上。项目内的
`data/manifests/company_special_status_import_manifest.json` 记录源文件校验和覆盖统计；
覆盖原 ST 文件前，导入器会备份到
`data/manifests/st_status_df_before_20260803_company_status_import.pq`。

两张状态表都是 **true-only 长表**：存在一行表示该证券当日状态为真，
未出现的 `(day, code)` 组合在对齐后解释为假，不进行前向或后向填充。ST 表的字段为
`day/code/是否st`，表示正式 ST/*ST 状态；撤销事件生效日当天起不再为真。
退市整理期表的字段为 `day/code/is_delisting_period`，由正式状态码 6（整理期首日）
至状态码 7（整理期末日）含首尾构建。若数据截止时某个明确的状态 6 尚未出现
对应状态 7，导入器只将该未闭合区间延长到 `close_df.pq` 当前截止日，不外推到未来。

该来源不覆盖交易所作出终止上市决定之日至退市整理期首日之前的
“已决定退市、尚未进入整理期”阶段。如需从决定日起禁交，必须另外接入可追溯的
交易所决定日数据，不得用当前简称或最后交易日反推。
可交易性过滤另外要求买入日 `amt[t+1]` 为有限数且严格大于零；这会屏蔽停牌、
零成交和仅保留伪持平价格的日期，并覆盖退市整理期状态 7 结束后至正式摘牌前
仍有行情行、但实际无成交的空档。该成交额检查是入场成交约束，不代替缺失的交易所决定日状态。
同时，`amt[t+1]` 是买入日全日汇总，只作为事后执行可行性代理：全日为零可确定不可入场，
但正成交额不代表在开盘时点一定能按代理开盘价成交，也不应被解释为因子在信号日可见的输入。

在两次完整特殊状态下载之间，增量行情更新仍不能直接把
`equ_inst_sstate` 的最后事件当作当前状态。
`scripts/data/merge_company_market_incremental.py` 从更新边界前一交易日的已验证状态开始，
2026-07-06 前结合交易所精确 5%/10% 涨跌停价，仅应用增量区间内实际生效的状态事件；
`scripts/data/repair_st_status_tail.py` 可对已写入的增量尾部执行同口径修复。
获得新的全量事件产物后，仍应以本节 importer 的校验与完整区间重建为准。

## Price-limit ratio 来源

`limit_ratio_df.pq` 由 `scripts/data/build_limit_ratio.py` 从
`close_df.pq` 的证券代码和有效价格覆盖期推导。常规比例为：沪深主板 10%，
创业板/科创板 20%，北交所常见代码段 30%。如果某股票在数据起始日之后才首次
出现有效价格，则前 5 个有效交易日记为 `NaN`，表示新股无涨跌幅限制窗口；数据
起始日已存在的老股票不会被误判为新股。

该矩阵是代理口径，不等同于交易所逐日精确涨跌停价。若后续有
`limit_up_price`/`limit_down_price` 数据，应优先接入精确价格。

## Volume 来源

`volume_df.pq` 由
`scripts/data/import_datayes_volume.py` 从以下已验证的数据适配器导入：

- Python 源码：
  `/Users/huangjuyuan/Documents/Codex/2026-06-29/f-o/src`
- DataYes 月度文件：
  `/Users/huangjuyuan/Desktop/database_summerintern/outputs/clickhouse_market_data/datayes/mkt_equd_adj_af`
- 源字段：`TURNOVER_VOL`

导入前以少量股票验证了 DataYes 复权收盘价与
`data/close_df.pq` 逐值一致。完整导入统计见
`data/manifests/market_data_manifest.json`。

当前宽表日期为 2013-01-14 至 2026-07-10。2026-06-29 至 2026-07-10 的
行情增量来自公司数据库，已与既有矩阵统一到同一组行列轴；该区间的成交量和
成交额也已同步更新。历史上原有的缺失值仍保持 `NaN`，不会以新数据回填。

## Market cap 来源

`market_cap_df.pq` 由 `scripts/data/import_market_cap.py` 从
`通联-股票市值.pq` 的 `day/code/values` 长表转换。源数据从
2014-01-02 开始，规范宽表覆盖率为 66.34%；缺失保持 `NaN`，不前向填充。
导入统计见 `data/manifests/market_cap_manifest.json`。

## 重建命令

```bash
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python scripts/data/import_company_special_status.py \
  --source-dir /Users/huangjuyuan/Desktop/database_summerintern/outputs/a_share_special_status

# 确认 dry-run 校验和区间统计后，再逐文件原子覆盖项目数据
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python scripts/data/import_company_special_status.py \
  --source-dir /Users/huangjuyuan/Desktop/database_summerintern/outputs/a_share_special_status \
  --apply

/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python scripts/data/import_datayes_volume.py \
  --source-root /Users/huangjuyuan/Desktop/database_summerintern/outputs/clickhouse_market_data/datayes/mkt_equd_adj_af \
  --source-python-root /Users/huangjuyuan/Documents/Codex/2026-06-29/f-o/src \
  --reference-close data/close_df.pq \
  --output data/volume_df.pq \
  --manifest data/manifests/market_data_manifest.json

/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python scripts/data/import_market_cap.py \
  --source-file 通联-股票市值.pq \
  --output-file market_cap_df.pq

/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python scripts/data/normalize_market_matrices.py

/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python tests/test_market_data_contract.py
```

不要把 `vol` 和 `amt` 互换。Alpha101 中的 `volume` 应映射到 `vol`；
论文定义的平均成交额 `advN` 应映射为 `adv(amt, N)`。

`vwap_proxy_df.pq` 不是交易所真实 VWAP。使用它的因子必须单独标记为
VWAP proxy，不能与真实输入因子混合比较口径。

## 公司行情增量（2026-07-13 合并）

`company_market_daily.parquet` 保存公司数据库提供的完整原始日频长表，包含
原始/前后复权 OHLC、成交量额、总/流通市值、真实涨跌停价、停牌和 ST 字段等。
宽表价格采用该增量的后复权尺度；为使历史序列与最新复权口径连续，历史 OHLC
按证券代码重标定到增量首日的后复权前收盘价。发生冲突时以该最新增量为准。

增量中 `is_suspended=1` 的记录不代表实际成交：其开高低为零、收盘为前收盘且量额为零。
导入宽表时 OHLC 与成交量额规范为 `NaN`，原始状态仍完整保留在
`company_market_daily.parquet`。ST 长表从增量首日开始也由最新来源整体覆盖。

`cap` 是总市值，不是行业分类。它不能替代 Alpha101 中
`IndClass/IndNeutralize` 所需的 sector、industry 和 subindustry 标签。

## 行业分类

`行业数据.parquet` 是行业或行业-市值联合中性化的可选输入，不属于因子表达式命名空间。每行必须唯一标识
`(trade_date, security_code)`，股票代码为六位字符串，`industry_l1_code` 为当日有效的一级
行业代码。加载器在评价时将其 pivot 为日期×股票矩阵，并仅用因子暴露日 `t` 的同日分类；
缺失分类不会以旧值或未来值填充。

导出数据时必须记录并固定分类体系（中信或申万）和层级；同一研究、训练/测试切分和横向
因子比较不可混用体系。行业数据应覆盖所评价行情区间；覆盖不足时，未分类股票或行业样本少于
3 只的股票会被行业中性化模块剔除，而不是悄然回填。
