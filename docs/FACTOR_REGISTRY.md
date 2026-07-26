# 因子注册表规范

因子定义与评价结果必须分离。`factor_registry/` 只保存公式及其来源；IC、收益、
图表和其他 performance 继续写入 `outputs/factor_evaluation/`。
评价脚本只从批次 JSON 读取因子定义，`src/` 中不得再维护公式副本；CSV 仅用于
人工查看，不作为运行输入。

## 一个批次一个 JSON

同一次论文录入、模型生成或人工提交的因子属于同一批次，统一保存在：

```text
factor_registry/<batch_id>.json
```

文件名必须等于 JSON 中的 `batch_id`。不要把多个来源批次追加进同一个文件，也不要
为同一批次的每条因子分别创建 JSON。修改已存在批次时增加 `batch_version`。

每个因子至少记录：

- `factor_name`：跨评价方法不变的唯一名称
- `entered_at`：首次进入因子注册表的带时区 ISO 时间；重跑、重新导出时不得修改
- `expression`：可执行公式
- `required_symbols`：公式需要的数据字段

批次必须记录来源、公式语言、生成时间和因子数量。代理字段必须显式写入
`uses_proxy` 与 `proxy_description`。禁止写入 `ic_mean`、`ir`、收益等评价结果。

`entered_at` 与评价结果中的 `evaluated_at` 含义不同：前者记录因子入库时间，
后者记录某次评价发生时间。

新批次从
[`factor_registry/templates/factor_batch.template.json`](../factor_registry/templates/factor_batch.template.json)
复制。完成后运行：

```bash
python scripts/validate_factor_registry.py factor_registry/<batch_id>.json
```

字段规范见 `factor_registry/schema/factor_batch.schema.json`。当前正式因子库是
`factor_registry/webapp_factor_library.json`；测试库已清空，新建测试因子时才会创建
相应批次文件。

## Web 因子库相关性冲突裁决

从测试库提交时，正式因子库先按所有对齐、有限的日期-证券暴露执行 pooled Pearson
相关性检验，任何非对角绝对相关系数超过 `0.75` 都视为冲突。冲突不再一律直接退回：

1. 对候选与每一个冲突的正式因子，在**完全一致**的盈利能力评价口径下，先比较
   `gn_rolling_sharpe_60_median`（60 日完整非重叠窗口 Sharpe 中位数）；
2. 只有 Sharpe 中位数并列时才比较 `fitness`；两个指标都是“数值越高越好”，候选必须
   **严格**优于每一个冲突因子，不能以并列替换；
3. 若没有同口径、可比较的结果，系统会为双方分别运行内置的“盈利能力测试”流水线
   （`industry_market_cap_neutralize → tradability_filter → quantile_net_returns → fitness`
   及其后续累计收益、滚动风险和最高组诊断），并把运行保存到普通 Webapp 历史；
4. 候选全部胜出才会移除这些旧的正式库副本并正式入库。旧因子的源测试定义、标签和
   历史评价不删除，因而自动回到/留在测试库；只要有一个冲突因子不败，正式库完全不变，
   候选仍留在测试库。

评价指标和替换裁决均不写入因子注册表 JSON：注册表仍只保存定义，运行指标保留在 Webapp
状态库及评价输出中。
