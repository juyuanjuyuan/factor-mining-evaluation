# Alpha101 评价脚本

## 可重复使用的因子注册表

`factor_registry/alpha101_runnable_factors.json` 统一记录当前83条可运行因子，
包含稳定名称、可执行表达式、论文原式、所需数据字段和代理标记。该文件不包含
任何一次评价结果，是所有批量评价和漏斗运行的唯一因子定义来源。CSV 快照位于同目录，
仅供人工查看。

公式变化后直接更新 JSON，并运行：

```bash
python scripts/validate_factor_registry.py \
  factor_registry/alpha101_runnable_factors.json
```

对全部可运行因子采用新的评价方法：

```bash
python scripts/run_alpha101_evaluations.py \
  --set runnable \
  --methods your_new_method \
  --select all
```

注册表通过每条记录的 `implementation_set` 区分 52 条 `exact`、30 条
`vwap_proxy` 和 1 条 `market_cap` 因子；`src/` 中不再重复保存公式。
`exact` 因子只使用 OHLC、真实成交量和成交额。评价收益统一为：
因子在 `t` 日收盘后计算，`t+1` 日开盘调仓，默认持有一期至 `t+2` 日开盘，即
`open[t+2] / open[t+1] - 1`。

30 条 `vwap_proxy` 因子中的 `vwap` 明确定义为
`vwap_proxy_df.pq = (high_df + low_df) / 2`，因子名带
`_vwap_proxy` 后缀，结果不能解释为使用真实 VWAP 的回测。

`market_cap` 集合包含新增总市值后解锁的 Alpha056。当前仍有 18 条因子依赖行业分类，市值数据不能替代
`IndClass/IndNeutralize`。

## 预检

```bash
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python scripts/run_alpha101_evaluations.py \
  --select all \
  --dry-run
```

## 运行全部 52 条

```bash
MPLCONFIGDIR=/private/tmp/matplotlib \
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python scripts/run_alpha101_evaluations.py \
  --select all
```

## 运行 30 条 VWAP 代理因子

```bash
MPLCONFIGDIR=/private/tmp/matplotlib \
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python scripts/run_alpha101_evaluations.py \
  --set vwap_proxy \
  --select all
```

## 运行市值因子 Alpha056

```bash
MPLCONFIGDIR=/private/tmp/matplotlib \
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python scripts/run_alpha101_evaluations.py \
  --set market_cap \
  --select all
```

默认一次加载所需行情矩阵以减少重复 I/O。若内存不足，增加
`--no-cache-data`，代价是每条因子重新读取所需文件。

脚本默认跳过 `metrics.csv` 中表达式、收益定义、预测周期和分组数完全相同的成功记录，
因此中断后可直接重复执行。使用 `--rerun` 可强制重跑。

## 选择评价方法

默认评价函数列表包括 `rank_ic`、`rank_icir`、`quantile_returns`、
`quantile_cumulative` 和 `quantile_plot`。也可以只运行指定模块：

```bash
# 只计算 Rank IC 及 IC/IR 汇总
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python scripts/run_alpha101_evaluations.py \
  --select 1-4 \
  --methods rank_icir

# 只计算并绘制分位数组合；依赖函数会自动加入
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python scripts/run_alpha101_evaluations.py \
  --select 1-4 \
  --methods quantile_plot
```

`metrics.csv` 会记录实际使用的有序方法列表和 `return_definition`。恢复运行时，只有
表达式、收益定义、预测周期、分组数和评价方法都一致的记录才会跳过。旧版未记录
`return_definition` 的 close-to-close 结果会自动视为未完成并重跑。

## 分批运行

```bash
# 范围、单个编号和逗号列表可以组合
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python scripts/run_alpha101_evaluations.py \
  --select 1-4,6-10,28,101
```

默认 `--set exact` 只允许 52 条无代理因子；`--set vwap_proxy`
允许 30 条代理因子；`--set market_cap` 运行 Alpha056。
缺少行业分类的其余 18 条仍不执行。

## 输出

正式结果位于 `outputs/factor_evaluation/alpha101_exact/`：

- `metrics.csv`：每个因子的最新指标
- `metrics_history.csv`：所有成功运行历史
- `plots/`：分组累计收益图
- `details/`：日度 IC、分组收益和累计收益
- `code/`：当前及历史可复跑代码
- `batch_status.csv`：批次成功、失败和耗时

每条因子同时计算日度截面 Spearman IC/IR 和十组等权收益。
代理因子的同结构结果位于
`outputs/factor_evaluation/alpha101_vwap_proxy/`。
Alpha056 结果位于
`outputs/factor_evaluation/alpha101_market_cap/`。

## Dashboard

评价完成后可直接打开：

`outputs/factor_evaluation/alpha101_exact/index.html`

重跑因子后，使用以下命令刷新 Dashboard 内嵌指标：

```bash
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python scripts/build_factor_dashboard.py
```

代理因子 Dashboard：

```bash
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python scripts/build_factor_dashboard.py \
  --results-dir outputs/factor_evaluation/alpha101_vwap_proxy \
  --title "Alpha101 VWAP Proxy Performance" \
  --subtitle "30 条因子使用 VWAP=(high+low)/2；以下结果均为代理口径。"
```
