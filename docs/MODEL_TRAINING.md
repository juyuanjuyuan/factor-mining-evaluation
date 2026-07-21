# 多因子模型训练方法

`/models` 将“如何从训练集得到模型参数”和“如何评价训练后的模型”分成两个独立层次：

1. `src/model_training/` 中的注册训练器只读取训练窗口，产出权重、可执行表达式和诊断信息；
2. 现有 `src/evaluators/` 管线使用该表达式分别评价训练集和测试集；
3. 测试任务只读取已经冻结的表达式，不会再次调用训练器。

`/models` 的“评价方式”可直接选择“流水线模板”页中已保存的**方法组合**模板。选择时会复制模板中经过校验的有序方法列表；随后若拖拽、增删方法，界面会显示为“自定义流水线”。模型会话、训练任务和样本外测试保存并执行的都是当时展开后的方法顺序，而非会随模板编辑或删除而变化的引用。漏斗模板不适用于模型训练/测试的单条评价流水线，因此不会显示在该选择器中。

## 注册表

公共接口位于 `src/model_training/base.py` 和 `registry.py`：

- `ModelTrainingContext`：冻结因子定义、预加载行情、训练信号日期、持有期和方法参数；
- `ModelTrainingResult`：与因子数量一一对应的权重、引擎可执行表达式和 JSON-safe 诊断；
- `@model_training_method(...)`：注册方法名称、中文标签、说明、是否拟合、权重是否可编辑和参数定义；
- `run_model_training(...)`：worker 唯一调用入口。

每个具体训练器单独放在一个 Python 文件中。目前包括：

- `manual_weights.py`：保留原有手动权重；
- `winsorized_zscore_ridge.py`：默认训练方式，保留原始幅度的每日去极值＋横截面 z-score Ridge 回归；
- `rank_ridge.py`：横截面秩 Ridge 回归，保留为可选的纯排序基线。

新增训练方式时，不要在 API router 或 worker 中加入方法名分支。新增一个训练器文件、使用装饰器注册，并在 `registry.py` 导入即可；`GET /api/models/training-methods` 会把注册元数据提供给 `/models` 的选择器。

## 默认：原始幅度 Z-score Ridge 定义

对训练信号日 `t` 和当日共同有效股票 `i`，先计算：

```text
x_raw[k,i,t] = factor[k,i,t]
x_clip[k,i,t] = clip(x_raw[k,i,t], Q_t(0.01), Q_t(0.99))
x[k,i,t] = (x_clip[k,i,t] - mean_i(x_clip[k,i,t])) / std_i(x_clip[k,i,t])
x_tilde[k,i,t] = x[k,i,t] - mean_i(x[k,i,t])
y[i,t] = open[t+1+horizon] / open[t+1] - 1
y_tilde[i,t] = y[i,t] - mean_i(y[i,t])
```

随后求解：

```text
argmin_beta mean_t(mean_i((y_tilde[i,t] - x_tilde[i,t] @ beta)^2))
             + ridge_alpha * sum_k(beta[k]^2)
```

每个交易日在目标函数中等权，因此上市股票更多的日期不会获得更高权重。因子和收益都已按日去均值，截距固定为 0；部署表达式仍是：

```text
sum_k(beta[k] * zscore_cs(winsorize_cs(factor[k], 0.01, 0.99)))
```

去极值和 z-score 都是同日横截面操作，不使用未来数据；训练与部署复用完全相同的 `winsorize_cs → zscore_cs` 表达式。训练标签复用 `src/returns.py` 的统一 next-open 定义。训练日期裁切复用引擎的 `restrict_evaluation_window`：因子表达式先在完整历史上计算以保留合法滚动前缀，末尾再裁去不能在训练窗口内完成开平仓的信号日。

## 持久化和冻结

`model_tests` 保存：

- `training_method` 和 `training_params_json`；
- 训练后更新的 `terms_json` 和最终 `expression`；
- `fit_result_json`，包括因子定义、系数、样本量、有效日期、训练 MSE/R²、设计矩阵条件数和收益标签。

worker 还会在训练 run 目录写入 `model_training.json`。训练成功时，supervisor 在同一数据库事务中将拟合表达式写回当前模型和 run；若模型训练引用已变化，旧任务不能覆盖新配置。样本外测试 run 不携带 `model_training` 请求，只执行冻结表达式。

## 验证

训练器数值与注册表：

```bash
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python tests/test_model_training.py
```

API、worker 和前端仍需执行 `AGENTS.md` 中的完整 webapp 验证清单。
