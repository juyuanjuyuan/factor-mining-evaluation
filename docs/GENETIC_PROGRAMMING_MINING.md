# 遗传规划因子挖掘与测试筛选

本模块复现华泰证券 2019 年报告《基于遗传规划的选股因子挖掘》中的核心搜索机制，
并把训练期 Rank IC fitness、测试集盈利能力筛选及存活者 IC 检测固化成可审计 CLI。搜索只使用训练集；
表达式冻结后才进入测试集；测试集不参与进化、超参数选择、候选去重或因子方向调整。

## 论文机制与本项目口径

保留的论文机制：

- 公式由变量、常数和函数组成的表达式树表示；
- ramped half-and-half 随机初始化，默认深度 1-4；
- 锦标赛选择，默认每次抽取 20 个程序；
- 交叉、子树变异、点变异、Hoist 变异与复制；
- 默认每代 1000 个公式、进化 3 代；
- 训练适应度是逐日横截面 Rank IC 均值，并施加可配置的树长度惩罚；
- 三代训练结果合并为 Hall of Fame（默认保留前 100 个）；再按训练期 fitness 直接冻结前
  100 个进入独立测试集。HOF 内不做两两相关性筛除：先以测试集盈利能力筛掉大部分候选，
  仅让存活者做测试集 IC 检测；两关均通过后自动交接给因子库服务。因子库负责
  相关性检验及正式入库裁决；GP 不执行相关性检验，也不直接写入正式因子库。

本地适配边界：

- 统一标签仍是 `open[t+1+horizon]/open[t+1]-1`，默认 `horizon=1`。论文使用
  20 日收盘收益；如需做预测目标对照，可显式传 `--horizon 20`，但分组收益复利解释应
  另行审慎处理，日频盈利准入默认保持 H=1。
- `paper_local` 训练预处理采用论文的“截面中位数 +/- 5 倍未缩放 MAD”去极值，随后
  联合剔除对数总市值、20 日收盘收益、20 日平均成交额和 20 日收益波动率，再做截面
  Z-score。仓库没有自由流通股本，所以这里只能使用“平均成交额流动性代理”，绝不把
  `vol` 冒充论文中的换手率。
- `market_cap_industry` 是当前的“市值 + 行业”训练预处理：先做同样的 MAD 去极值，
  再每日一次性拟合 `factor ~ intercept + log(total_market_cap) + industry_l1_fixed_effects`
  并取残差，最后做截面 Z-score。行业少于 3 只、缺失行业或无效市值的观测会置为 NaN。
- 评价模块和 GP 训练共用上述联合 OLS 定义；不使用“先市值、后行业”的顺序残差化。
  `market_cap` 旧模式仅为历史 campaign/checkpoint 兼容保留，Webapp 新建任务不再展示它。
- `paper_local` 仍是四风格本地适配，未自动包含行业；所以它仍不能描述为完整的
  “五风格+行业”复现。
- `vwap` 来自 `(high+low)/2` 代理。后续提交正式因子库时，使用该终端的定义会标为代理口径。
- 训练适应度剔除测试项目里同口径的 ST、涨跌停开盘不可交易观测；停牌收益本身为 NaN。

## 两阶段测试集筛选

查看固定方法顺序：

```bash
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python \
  scripts/run_factor_standard_evaluation.py list
```

GP 不会一开始对全部 100 个候选跑 IC：先运行下面的 `profitability_test`，只有通过者才
运行 `ic_test`。训练期 Rank IC 只用于搜索排序，不能代替独立测试期的 IC 检测。

单独诊断时，`ic_test` 使用“市值+行业”联合中性化后的因子值：

```text
prefix_truncation_consistency
-> industry_market_cap_neutralize
-> rank_ic
-> rank_icir
-> newey_west_ic_significance
-> ic_trend_filter
-> ic_peak_decay
```

通过条件为：前缀截断一致性通过、测试集市值+行业联合中性 Rank IC 均值大于 0、Newey-West
双侧 p 值小于 0.05。阈值均会写入 `standard_evaluation.json`。

`profitability_test`：

```text
industry_market_cap_neutralize
-> tradability_filter
-> quantile_net_returns
-> quantile_cumulative
-> quantile_plot
-> rolling_sharpe
-> rolling_drawdown
-> top_quantile_performance
-> cycle_context
```

不论训练阶段选择 `paper_local`、`market_cap_industry` 或其他兼容模式，这两个冻结测试阶段
都固定从“市值+行业”联合中性化后的因子值开始；若希望训练排序也使用同一口径，应显式选择
`--preprocess-mode market_cap_industry`。

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

GP 在两项测试均通过后会生成持久化交接请求，由因子库服务消费并提交正式因子库；届时才执行
相关性检验。独立标准 CLI 的 `--admit-if-passed` 要求 `ic_test` 与 `profitability_test` 都通过。正式因子库
沿用现有契约：所有对齐、有限的日期-证券暴露上的 pooled Pearson，任一非对角元素绝对值
不得超过 0.75。通过后只向 `factor_registry/webapp_factor_library.json` 写因子定义；评价指标
继续保存在输出目录，避免污染注册表。

该两阶段筛选且不自动入库的协议会写入 campaign 配置指纹。已有的旧版自动入库 campaign
不会被静默改写；要采用本协议，应新建一个 campaign 名称，以保留既有训练/测试结果的审计含义。

独立标准 CLI 在评价未通过或相关性未通过时返回退出码 2，适合 shell、launchd 或监控系统判断。

## 单轮与 7x24 挖掘

### Webapp 操作入口

启动 Webapp 后进入 `/factors`，点击右上角“遗传算法添加因子”。页面使用与本 CLI 相同的
`GeneticMiningRunner` 配置和输出目录，并提供：

- 训练/测试交易日边界、论文默认种群参数和并行线程配置；
- CPU 或 Apple GPU (MPS) 训练适应度后端；MPS 不可用时界面会显示原因并禁用选项；
- 单 cycle、有限多 cycle 或无限连续运行；
- 当前 cycle、训练代数、本代已计算公式进度、测试阶段、测试通过候选数量、候选门槛和日志；
- 通过测试的表达式自动交接至因子库；因子库独立完成相关性检验并返回入库或拒绝结果；
- 停止与 checkpoint 续跑。

遗传挖掘使用独立系统进程，不占用普通因子评价 worker。Webapp 重启只停止状态监控，已经
启动的挖掘进程继续运行；页面重新打开后会依据 PID 和 campaign 文件恢复监控。为避免宽矩阵
重复常驻导致内存压力，Webapp 同时只允许一个 GP campaign 运行。Web API 只接受经过验证的
结构化参数，不接受任意命令、脚本路径或输出路径。

对应 API：

```text
GET  /api/genetic-campaigns
GET  /api/genetic-campaigns/backends
POST /api/genetic-campaigns
GET  /api/genetic-campaigns/{campaign}
POST /api/genetic-campaigns/{campaign}/start
POST /api/genetic-campaigns/{campaign}/stop
```

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
  --components 100 \
  --preprocess-mode market_cap_industry \
  --compute-backend mps \
  --n-jobs 1
```

### Apple GPU (MPS) 训练后端

MPS 后端把对齐后的 OHLCV、成交额、VWAP 代理、市值、行业编码、训练期标签和风格暴露一次转为
`float32` MPS 张量。表达式树的时序窗口、截面排序、MAD 去极值、风格残差、标准化和
逐日 Rank IC 均在 GPU 上计算。`market_cap_industry` 通过行业内去均值的
Frisch-Waugh-Lovell 等价式在 GPU 上完成联合回归，无需构造巨大的行业 dummy 张量，也不将整个因子
重复交给 CPU。`paper_local` 的四风格回归仍只把每日小型广义逆矩阵返回 CPU。
Webapp 启动 MPS 进程时强制 `PYTORCH_ENABLE_MPS_FALLBACK=0`，一旦某个算子不受
MPS 支持就显式失败，不会静默退回 CPU。

MPS 自己管理 GPU 并行，所以必须使用 `--n-jobs 1`。`n_jobs` 只是 CPU 后端的公式线程数，
不是 GPU 核心数。可用性可通过 Web 的 `/api/genetic-campaigns/backends` 或 Python 接口
`fitness_backend_statuses()` 检查。

CPU 和 MPS 是两种不同计算口径（CPU `float64`，MPS `float32`），`compute_backend` 被写入
campaign 配置哈希和 cycle 汇总。因此已存在的 CPU campaign 会继续按 CPU 恢复；切换到
MPS 必须创建新 campaign，避免将 CPU 适应度缓存混入 MPS 试验。

启动连续模式只需增加：

```text
--forever --pause-seconds 60
```

每个新 cycle 使用 `seed + cycle - 1`，训练/测试边界和所有门槛保持不变。每个 campaign
持有进程锁，避免两个实例同时写状态。

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
    ├── candidates/               每个冻结候选的分阶段测试结果
    └── cycle_summary.json         训练与测试候选汇总
```

同名 campaign 的配置哈希必须完全一致。改变训练/测试日期、数据路径、函数集、阈值或遗传
参数时必须使用新 campaign 名，防止旧适应度缓存混入新实验。

## 测试

```bash
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python \
  tests/test_evaluation_standards.py
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python \
  tests/test_genetic_mining.py
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python \
  tests/test_genetic_mining_mps.py
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python \
  webapp/server/tests/test_genetic_mining_webapp.py
```

`test_training_fitness_cannot_see_test_returns` 会把训练截止日之后的开盘价整体改写，再证明
训练适应度完全不变。合成 smoke test 会跑通“一代进化 -> 按训练 fitness 冻结候选 ->
测试集盈利筛选 -> 存活者 IC 检测”，并检查 checkpoint、缓存和候选结果均可恢复。
MPS 合约测试还会遍历全部 GP 算子，对比 CPU/MPS 的有限值、NaN 位置、IC 天数、
有效样本数和适应度，并单独验证 MPS 训练不能读取测试期收益。
