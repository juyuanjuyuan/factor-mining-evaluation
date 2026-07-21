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

字段规范见 `factor_registry/schema/factor_batch.schema.json`。当前 Alpha101 批次是
`factor_registry/alpha101_runnable_factors.json`。
