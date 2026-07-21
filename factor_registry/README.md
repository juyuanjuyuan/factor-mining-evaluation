# 可运行因子注册表

这里保存的是因子定义，不是任何一次评价的结果。统一采用“一批因子一个 JSON”：
同一次论文录入、模型生成或人工提交的因子放在同一个 `<batch_id>.json` 中。

- `webapp_factor_library.json`：用户从测试库提交后的正式因子库，只有这里的因子在“因子库”页面展示

当前测试库已清空；Web 平台后续新建的测试因子会自动创建
`webapp_test_factors.json`。因子库当前仅保留已正式提交的因子定义。

每条记录包含稳定因子名、可执行表达式、论文原式、所需数据字段、实现集合以及
因子首次入库时间 `entered_at` 和是否使用代理。

Webapp 中的生命周期分两层：

- 测试库：用于持续回测、漏斗筛选和对比，包含当前项目内所有候选因子。
- 因子库：只包含用户显式提交后的因子定义；提交会复制测试库定义并记录
  `submitted_at`、`source_batch_id`、`source_factor_name`，不会写入任何评价指标。

每个因子可带 `project` 分类；批次也可设置默认 `project`。测试库和因子库页面按项目分组并可展开或收缩，因子详情页可单独修改项目归属。提交到因子库时会保留原项目，例如现有 Alpha101 归入 `Alpha101`；国泰君安 GTJA191 批次归入 `国泰君安191`。

研究标签是平台的用户元数据，按 `batch_id + factor_name` 存在 Webapp 状态库中，
不会改写 JSON 注册表的公式、来源或入库时间。因此 Alpha101 等只读定义也能被标记。
在测试库或因子库中可添加、筛选标签，并按一个或多个标签提交评价；任务会在服务端
重新解析标签成员并保存标签选择，避免分页、搜索或页面刷新遗漏固定组合。

Python 中统一通过 JSON 注册表加载：

```python
from factor_registry import load_registered_factors

for factor in load_registered_factors():
    print(factor.factor_name, factor.expression)
```

公式修改后直接更新 JSON，并执行校验：

```bash
python scripts/validate_factor_registry.py \
  factor_registry/webapp_factor_library.json
```

因子定义不会在 `src/` 中重复维护。后续新批次从
`templates/factor_batch.template.json` 开始，并使用
`scripts/validate_factor_registry.py` 校验。完整规范见
`docs/FACTOR_REGISTRY.md`。
