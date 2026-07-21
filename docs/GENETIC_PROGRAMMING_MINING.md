# 遗传规划因子挖掘与自动准入

本模块复现华泰证券 2019 年报告《基于遗传规划的选股因子挖掘》中的核心搜索机制，
并把本项目 Webapp 中的“IC检测”和“盈利能力测试”两套方法组合固化成可审计 CLI。
搜索只使用训练集；表达式冻结后才进入测试集；测试集不参与进化、超参数选择、候选去重
或因子方向调整。

## 论文机制与本项目口径

保留的论文机制：

- 公式由变量、常数和函数组成的表达式树表示；
- ramped half-and-half 随机初始化，默认深度 1-4；
- 锦标赛选择，默认每次抽取 20 个程序；
- 交叉、子树变异、点变异、Hoist 变异与复制；
- 默认每代 1000 个公式、进化 3 代；
- 训练适应度是逐日横截面 Rank IC 均值，并施加可配置的树长度惩罚；
- 从 hall of fame 中按训练集因子暴露相关性去重，再冻结最终表达式。

本地适配边界：

- 统一标签仍是 `open[t+1+horizon]/open[t+1]-1`，默认 `horizon=1`。论文使用
  20 日收盘收益；如需做预测目标对照，可显式传 `--horizon 20`，但分组收益复利解释应
  另行审慎处理，日频盈利准入默认保持 H=1。
- `paper_local` 训练预处理采用论文的“截面中位数 +/- 5 倍未缩放 MAD”去极值，随后
  联合剔除对数总市值、20 日收盘收益、20 日平均成交额和 20 日收益波动率，再做截面
  Z-score。仓库没有自由流通股本，所以这里只能使用“平均成交额流动性代理”，绝不把
  `vol` 冒充论文中的换手率。
- 仓库当前没有已接入引擎、可读取的逐日行业矩阵，因此 `paper_local` 没有做行业中性。
  这部分是与论文的明确差异，不应把当前结果描述为完整的“五风格+行业”复现。
- `vwap` 来自 `(high+low)/2` 代理。遗传程序使用该终端时，入库定义会自动标为代理口径。
- 训练适应度剔除测试项目里同口径的 ST、涨跌停开盘不可交易观测；停牌收益本身为 NaN。

## 两套代码级评价标准

查看固定方法顺序：

```bash
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python \
  scripts/run_factor_standard_evaluation.py list
```

`ic_test`：

```text
prefix_truncation_consistency
-> market_cap_neutralize
-> rank_ic
-> rank_icir
-> newey_west_ic_significance
-> ic_trend_filter
-> ic_peak_decay
```

通过条件为：前缀截断一致性通过、测试集市值中性 Rank IC 均值大于 0、Newey-West
双侧 p 值小于 0.05。阈值均会写入 `standard_evaluation.json`。

`profitability_test`：

```text
market_cap_neutralize
-> tradability_filter
-> quantile_net_returns
-> quantile_cumulative
-> quantile_plot
-> rolling_sharpe
-> rolling_drawdown
-> top_quantile_performance
-> cycle_context
```

通过条件严格为：

```text
最高组全面占优为是
AND (
  最高组分段夏普(60日)中位数 >= 1
  OR 最高组年化收益 > 30%
)
```

这里使用净分组收益：买入和卖出各按 7 bps 计费。分段夏普使用完整、非重叠的 60
交易日区块。最高组年化收益按测试样本内的最高组净日收益几何年化。

单独评价一个已经冻结的表达式：

```bash
MPLCONFIGDIR=/private/tmp/matplotlib \
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python \
  scripts/run_factor_standard_evaluation.py run \
  --factor-name candidate_001 \
  --expression 'ts_corr(vwap / h, h, 10)' \
  --test-start 2019-10-10 \
  --test-end 2026-07-10 \
  --standards all
```

增加 `--admit-if-passed` 后，只有两套标准都通过才运行正式因子库相关性检验。正式因子库
沿用现有契约：所有对齐、有限的日期-证券暴露上的 pooled Pearson，任一非对角元素绝对值
不得超过 0.75。通过后只向 `factor_registry/webapp_factor_library.json` 写因子定义；评价指标
继续保存在输出目录，避免污染注册表。

CLI 在评价未通过或相关性未通过时返回退出码 2，适合 shell、launchd 或监控系统判断。

## 单轮与 7x24 挖掘

下例沿用已经在项目中使用过的训练/测试时间切分，并保持测试集完全冻结：

```bash
MPLCONFIGDIR=/private/tmp/matplotlib \
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python \
  scripts/run_gp_factor_mining.py \
  --campaign paper2019_daily \
  --train-start 2013-01-14 \
  --train-end 2019-09-30 \
  --test-start 2019-10-10 \
  --test-end 2026-07-10 \
  --population-size 1000 \
  --generations 3 \
  --hall-of-fame 100 \
  --components 10 \
  --n-jobs 2
```

启动连续模式只需增加：

```text
--forever --pause-seconds 60
```

每个新 cycle 使用 `seed + cycle - 1`，训练/测试边界和所有门槛保持不变。每个 campaign
持有进程锁，避免两个实例同时写状态或因子库。

长期运行建议使用仓库内的 launchd 示例。先检查参数，再复制并加载：

```bash
plutil -lint deploy/com.factor-mining.gp.plist.example
cp deploy/com.factor-mining.gp.plist.example \
  "$HOME/Library/LaunchAgents/com.factor-mining.gp.plist"
launchctl bootstrap "gui/$(id -u)" \
  "$HOME/Library/LaunchAgents/com.factor-mining.gp.plist"
```

launchd 的 `KeepAlive` 负责进程级重启；程序自身的 `--forever` 负责 cycle 循环。系统重启或
进程崩溃后会从 `active_cycle.json` 和 `checkpoint.json` 恢复，当前 cycle 已算表达式从
根目录临时 `fitness_cache.jsonl` 复用，已完成的测试候选从各自的
`candidate_result.json` 复用。cycle 完成后缓存移入该 cycle 目录并清空内存，避免 7x24
运行时缓存无界增长。

## 输出与可追溯性

```text
outputs/gp_factor_mining/<campaign>/
├── .campaign.lock
├── campaign.json                 固定配置、哈希、下一个 cycle
├── active_cycle.json             运行/测试/完成状态
├── fitness_cache.jsonl           当前未完成 cycle 的恢复缓存（完成后移走）
├── last_error.json               最近一次未处理异常
└── cycles/cycle_000001/
    ├── checkpoint.json           种群、hall of fame、随机数状态
    ├── fitness_cache.jsonl       该 cycle 的完整训练适应度档案
    ├── generations/              每代统计与最优程序
    ├── candidates/               每个冻结候选的两套测试结果
    └── cycle_summary.json         训练、测试、入库汇总
```

同名 campaign 的配置哈希必须完全一致。改变训练/测试日期、数据路径、函数集、阈值或遗传
参数时必须使用新 campaign 名，防止旧适应度缓存混入新实验。

## 测试

```bash
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python \
  tests/test_evaluation_standards.py
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python \
  tests/test_genetic_mining.py
```

`test_training_fitness_cannot_see_test_returns` 会把训练截止日之后的开盘价整体改写，再证明
训练适应度完全不变。合成 smoke test 会跑通“一代进化 -> 训练内去相关 -> 冻结表达式 ->
测试集双标准”，并检查 checkpoint、缓存和候选结果均可恢复。
