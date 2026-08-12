# 因子评价流程 v1

本文档描述当前采用的单因子评价漏斗。原则：先验证正确性，再做廉价统计筛选，
再去混杂，最后才做最贵的组合层面确认；任何一步淘汰即终止，不再计算后续步骤。
所有步骤共用主流程收益标签 `open[t+1+horizon]/open[t+1]-1`（默认 `horizon=1`，
即 `open[t+2]/open[t+1]-1`），因子在 `t` 日收盘后计算、`t+1` 开盘入场。

各阶段方法列表不同，metrics.csv 按因子名 upsert 会互相覆盖，因此不同阶段
应使用不同的 `--output-dir`（例如 `stage1_validity/`、`stage2_ic/`、
`stage2b_neutral/`、`stage3_portfolio/`）。

## 第 1 步：未来函数检查（正确性门槛）

方法：`future_data_perturbation`（当前配置 4 个 checkpoint）。

对每个 checkpoint `t0`：保留 `t <= t0` 的输入不变，对 `t > t0` 的有限值做
确定性随机扰动，重算因子并比较 `factor[t0]`。任何 checkpoint 的因子值发生
变化即判定含前视依赖，直接淘汰。

放在第一步的理由与成本无关：含未来函数的因子会以极好的成绩通过后面所有
统计检验，必须先堵住。已知盲区：扰动为对数正态乘性噪声（恒正），只依赖
未来数据符号的泄漏路径可能假阴性；该测试也不覆盖 label 构造环节
（那部分由 `returns.py` 的 `RETURN_DEFINITION` 单点约束）。

## 第 2 步：Rank IC 显著性筛选（原始因子）

方法：`rank_ic` → `rank_icir` → `newey_west_ic_significance`。

计算日度 Rank IC 序列、IC 均值、ICIR，并用 Newey-West HAC 标准误检验
IC 均值是否显著异于 0。淘汰规则：不显著即淘汰。

待定/学习中（v1 暂按 5% 双侧显著执行）：

- 显著性阈值与多重检验校正（因子池筛选下 5% 会放进约 5% 的噪声因子；
  候选方案：t > 3 或 BH 校正）；
- 幅度门槛（显著不等于可交易，候选：|IC 均值| 与 ICIR 下限）；
- 当前 NW lag 选择用绝对自协方差阈值 0.05，对日度 IC 序列几乎恒选 lag=0，
  实际退化为普通 t 检验，待改为自相关系数显著带或 plug-in 带宽。

## 第 3 步：市值中性化后复测

方法：`market_cap_neutralize` → `rank_ic` → `rank_icir` →
`newey_west_ic_significance`（独立一次运行，与第 2 步分开出目录）。

每日截面 OLS：`factor = a + b·log(cap) + residual`，以残差为工作因子重复
第 2 步检验，衡量剔除市值暴露后因子是否仍有信息。中性化后不显著即淘汰。
对比原始与中性化 IC 之差可诊断因子的市值暴露程度（`log_cap_beta`、R²
在 details 里逐日可查）。

行业中性化可按需独立运行：`industry_neutralize` → `rank_ic` → `rank_icir` →
`newey_west_ic_significance`。它目前不改变标准四阶段漏斗，避免混淆既有市值中性结果；
只有完整的逐日行业分类覆盖评价区间时才应纳入筛选。建议与原始/市值中性版本并行比较，
而不是用任一单一口径替代全部候选判断。

若研究问题是“同时剔除行业和市值暴露”，使用独立的一次性联合管线：
`industry_market_cap_neutralize` → `rank_ic` → `rank_icir` →
`newey_west_ic_significance`。它每日拟合“截距 + 行业固定效应 + log(total_market_cap)”并取残差；
不要把两个单独中性化模块串联后当作联合回归，因为两种顺序都会产生不同结果。

## 第 4 步：分组回测与组合层面指标

方法（一次运行）：`tradability_filter` → `quantile_returns` →
`quantile_cumulative` → `quantile_plot` → `top_quantile_performance` →
`rolling_sharpe` → `rolling_drawdown`。

- 可交易性过滤：信号日 `t` 或入场日 `t+1` 处于 ST/*ST/正式退市整理期，
  或入场日成交额不满足 `isfinite(amt[t+1]) and amt[t+1] > 0`，
  或入场日开盘触及涨跌停代理边界
  `abs(open[t+1] / close[t] - 1) >= limit[t+1] - 0.002`，都在因子第 `t` 行置 NaN。
  状态恢复后，只有 `t` 和 `t+1` 两个观察日的特殊状态都为 false 才重新纳入候选。
  这样同时覆盖信号日禁交策略和“开盘封板后盘中打开”的入场不可成交情形；
  仍不查出场日。`delisting` 只表示 `datayes.equ_inst_sstate` 状态 6–7 的正式
  退市整理期，不覆盖交易所终止上市决定日到整理期首日前的阶段；
  但买入日成交额检查会屏蔽整理期结束后至摘牌前的零成交/伪持平行情空档。
  `amt[t+1]` 是全日汇总的事后执行代理，不是开盘时点可观测的因子输入；
  它能排除全日无成交，但不能证明正成交额的股票一定能在开盘成交。
- 分组：Q1–Q10 等权，考察最高组 Q10 是否高于其余各组
  （`top_group_above_every_lower_group`），同时看整组排序而非只看端点。
- 最高组表现：统计 Q10 单日跑赢各较低分组的比例，以及 60 日非重叠窗口中
  Q10 累计收益排名第一的比例。
- 分段风险：窗口 20/60/252，使用完整非重叠区块。Rolling Sharpe 输出分布摘要
  `pos_share / min / median`，Rolling Drawdown 输出 `worst / median`
  （末值无筛选信息量，已弃用）；每个完整区块在 details 中仅保留结束日一行。
  另输出 60 日非重叠窗口的年化 Sharpe 折线图，包含全部分位组。
  图上标为结束日 `t` 的 60 日点，使用的是包含 `t` 的 `[t-59,\ldots,t]` 共 60 个
  有效交易日；后一个点从下一个不重叠区块开始。

淘汰标准待定（学习中）：分组单调性度量、换手对应的成本扣减、滚动指标
的量化门槛。当前该步以人工审查图表和指标为主。

## 已知缺口（记录在案，尚未纳入流程）

1. 全流程在全样本上执行，存在样本内选择偏差；候选方案：预留 holdout
   区间或要求 IC 前后半段同号。
2. 未检验与已入库因子的相关性/增量价值。
3. 负 IC 因子在第 4 步顶部分组检验中会被误判，分组前未按 IC 符号翻转。
4. 已提供可选的 `ic_horizon_decay`（H=20–252、固定共同日期样本），但尚未纳入
   默认流水线；半衰期规则和基于衰减曲线的筛选阈值仍未定义。
5. 各步淘汰阈值应在批量运行前固定，避免事后调整。

## 参考

- 方法实现与输出字段：`docs/EXTENDED_EVALUATORS.md`
- 指标契约与审查清单：`docs/FACTOR_EVALUATION_CONTRACT.md`
- 收益标签定义：`src/returns.py`

## 批量执行

全量漏斗由 `scripts/run_alpha101_funnel.py` 编排。它按上述门槛逐阶段执行，
淘汰或失败的因子不进入下一阶段；每一阶段写入独立目录。默认复用表达式、
收益定义、horizon、分组数和方法列表完全匹配，且 code、details、plot（若该
阶段要求）均非空的已有结果。

```bash
# 预检并冻结本次配置
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python \
  scripts/run_alpha101_funnel.py --select all --dry-run

# 执行 83 条当前可运行 Alpha101
MPLCONFIGDIR=/private/tmp/matplotlib \
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python \
  scripts/run_alpha101_funnel.py --select all
```

默认根目录为 `outputs/factor_evaluation/alpha101_funnel_v1/`，包含：

- `stage1_validity/`
- `stage2_ic/`
- `stage2b_neutral/`
- `stage3_portfolio/`
- `funnel_status.csv`：每次运行的逐因子、逐阶段执行/复用及门槛结果
- `funnel_summary.csv`：按因子 upsert 的最终状态和关键跨阶段指标
- `funnel_runs.csv`：每次运行开始前冻结的收益定义、显著性阈值和方法配置
