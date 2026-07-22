# 因子挖掘

模块化的股票因子表达式计算、Rank IC/IR 与分位数组合评价项目。

统一收益口径：因子在 `t` 日收盘后计算，`t+1` 日开盘调仓；默认一期收益为
`open[t+2] / open[t+1] - 1`。

## 目录

```text
data/                         行情宽表和数据构建清单
factor_registry/              统一因子定义来源
src/                          评价引擎、方法、收益标签和报告模块
scripts/                      评价、批处理和数据维护命令
tests/                        合成数据与真实数据契约测试
docs/                         数据与评价规范
notebooks/                    教学和探索性 Notebook
outputs/factor_evaluation/    已归档的评价结果
webapp/                       FastAPI + React 可视化评价平台
skills/                       Codex 技能说明与代理配置
```

## 常用命令

```bash
# 单因子完整评价；默认读取 data/，输出到 outputs/factor_evaluation/custom/
python scripts/run_factor_evaluation.py \
  --factor-name factor_test1 \
  --expression 'ts_mean(abs((h-l)/(h+l+1e-6)), 20)'

# Agent 因子研究 CLI：先回看研究历史，再同步运行一个训练期候选
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python \
  scripts/run_factor_research.py catalog all
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python \
  scripts/run_factor_research.py journal summary \
  --research-id short-term-reversal-v1
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python \
  scripts/run_factor_research.py run \
  --factor-name factor_test1 \
  --expression 'ts_mean(abs((h-l)/(h+l+1e-6)), 20)' \
  --methods rank_ic,rank_icir,quantile_returns,quantile_cumulative,quantile_plot \
  --signal-start 2019-10-10 --signal-end 2021-12-31 \
  --research-id short-term-reversal-v1 --research-phase training \
  --research-direction '短期价格反转' \
  --hypothesis '近期收益冲击会在下一持有期部分反转' \
  --iteration-note '训练期首个候选'

# Alpha101 预检和断点续跑
python scripts/run_alpha101_evaluations.py --select all --dry-run
python scripts/run_alpha101_evaluations.py --select all
# 按正确性 → 原始 IC → 市值中性 IC → 组合确认的漏斗运行全部 83 条
python scripts/run_alpha101_funnel.py --select all --dry-run
python scripts/run_alpha101_funnel.py --select all

# 对当前全部83条可运行因子采用指定评价方法
python scripts/run_alpha101_evaluations.py \
  --set runnable --methods default --select all

# 校验作为唯一运行来源的因子定义 JSON
python scripts/validate_factor_registry.py factor_registry/alpha101_runnable_factors.json

# 刷新 Dashboard
python scripts/build_factor_dashboard.py

# 查看并运行 Webapp 同口径的 IC/盈利能力评价标准
python scripts/run_factor_standard_evaluation.py list
python scripts/run_factor_standard_evaluation.py run \
  --factor-name candidate_001 \
  --expression 'ts_corr(vwap / h, h, 10)' \
  --test-start 2019-10-10 --test-end 2026-07-10

# 论文式遗传规划挖掘；增加 --forever 可连续运行并断点恢复
python scripts/run_gp_factor_mining.py \
  --campaign paper2019_daily \
  --train-start 2013-01-14 --train-end 2019-09-30 \
  --test-start 2019-10-10 --test-end 2026-07-10 \
  --population-size 1000 --generations 3 --n-jobs 2

# 核心回归
python tests/test_evaluation_methods.py
python tests/test_factor_evaluation.py
python tests/test_factor_research_cli.py
python tests/test_market_data_contract.py

# Web 平台（安装、开发双端口、构建后单端口）
cat webapp/README.md
```

跑web平台最简单的是单端口方式（前端已经构建好了，一条命令跑起来）：

cd "/Users/huangjuyuan/Desktop/因子挖掘:评价"
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python -m uvicorn \
  --app-dir webapp/server app.main:app --host 127.0.0.1 --port 8000
然后打开 http://127.0.0.1:8000 即可，API 和网页都在这一个端口上。

如果要改前端代码、需要热更新，就用双终端开发模式：

# 终端 1：后端
cd "/Users/huangjuyuan/Desktop/因子挖掘:评价/webapp/server"
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python run.py

# 终端 2：前端
cd "/Users/huangjuyuan/Desktop/因子挖掘:评价/webapp/frontend"
npm run dev


详细口径见 [因子评价契约](docs/FACTOR_EVALUATION_CONTRACT.md)、
[Alpha101 运行说明](docs/ALPHA101_EVALUATION.md)和
[行情数据约定](docs/MARKET_DATA.md)。可选评价模块见
[扩展评价模块](docs/EXTENDED_EVALUATORS.md)。
[遗传规划挖掘与自动准入](docs/GENETIC_PROGRAMMING_MINING.md)说明训练/测试隔离、
两套 CLI 评价标准、相关性准入、断点恢复与 7x24 运行。Webapp 的正式因子库 `/factors`
也提供“遗传算法添加因子”入口，可配置、启动、停止并查看同一套 campaign 的实时进度。
新因子批次的保存规范见 [因子注册表规范](docs/FACTOR_REGISTRY.md)。
可视化平台的启动与验证见 [Web 平台说明](webapp/README.md)。
LLM 驱动的候选生成、Webapp 控件到 CLI 的对应关系与研究闭环见
[`mine-stock-factors` skill](skills/mine-stock-factors/SKILL.md)。
