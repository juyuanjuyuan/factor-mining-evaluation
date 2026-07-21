# 国泰君安 GTJA191 因子

`factor_registry/gtja191_runnable_factors.json` 是国泰君安 GTJA191 的本地可运行子集。
它从用户提供的 DolphinDB `gtja191Alpha` 日频实现转换而来，保留原始编号和函数名
对应关系：`gtja191_001` 对应 `gtjaAlpha1`，依此类推。

本地 OHLCV 与 VWAP 代理数据可以直接运行其中 186 条。以下 5 条没有被伪造为可运行
公式，因为当前数据契约不含它们需要的外部序列：

- `gtja191_030`：Fama-French `MKT`、`SMB`、`HML`；
- `gtja191_075`、`gtja191_182`：基准指数开盘价和收盘价；
- `gtja191_149`、`gtja191_181`：基准指数收盘价。

40 条依赖 `vwap` 的因子会在注册表中标为 `uses_proxy: true`。本项目的 `vwap` 是
`(high + low) / 2` 代理，并非交易所真实 VWAP；比较结果时不要与真实 VWAP 口径混用。

转换以所给 DolphinDB 函数体为主。其 `gtjaAlpha165`、`gtjaAlpha183` 的 `rowMax/rowMin`
实现会把截面压缩成单行，不适合作为本项目的宽表因子；这里按同一条公式注释中的
`SUMAC` 采用逐证券滚动累计区间。`gtjaAlpha190` 函数体把分母误写为布尔掩码，已按
公式注释采用下行样本计数。其余公式保留源实现中的具体口径。

运行前建议先做干跑检查：

```bash
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python scripts/run_gtja191_evaluations.py --dry-run
```

正式运行全部 186 条：

```bash
MPLCONFIGDIR=/private/tmp/matplotlib /Users/huangjuyuan/miniforge3/envs/rdagent/bin/python scripts/run_gtja191_evaluations.py
```

也可只跑编号范围，例如：

```bash
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python scripts/run_gtja191_evaluations.py --select 1-20 --methods rank_icir
```

输出默认写到 `outputs/factor_evaluation/gtja191_runnable/`。该批次复用了项目统一的
收盘后计算、下一交易日开盘入场、开盘到开盘收益标签。
