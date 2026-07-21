# 因子评价 Web 平台

FastAPI + SQLite + 单常驻评价 worker，前端为 Vite + React + TypeScript +
Ant Design + ECharts。每个 run 写入独立的
`outputs/webapp/runs/<run_id>/`，不会改动原有 `outputs/factor_evaluation/`。

## 开发启动

项目路径包含冒号，请不要设置 `PYTHONPATH`。使用 editable install：

```bash
cd "/Users/huangjuyuan/Desktop/因子挖掘:评价"
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python -m pip install -e '.[webapp]'

cd webapp/server
MPLCONFIGDIR=/private/tmp/matplotlib \
  /Users/huangjuyuan/miniforge3/envs/rdagent/bin/python run.py
```

另一个终端：

```bash
cd "/Users/huangjuyuan/Desktop/因子挖掘:评价/webapp/frontend"
npm install
npm run dev
```

打开 <http://127.0.0.1:5173>。Vite 将 `/api` 代理到 8000 端口。

## 单端口运行

```bash
cd webapp/frontend
npm run build

cd ../server
MPLCONFIGDIR=/private/tmp/matplotlib \
  /Users/huangjuyuan/miniforge3/envs/rdagent/bin/python \
  -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

打开 <http://127.0.0.1:8000>。API 文档位于 `/docs`。

## 因子生命周期

- 测试库：默认入口，展示 Alpha101、历史 Web 自定义批次和新建测试因子；用于反复回测、漏斗筛选和对比。
- 因子库：只展示从测试库显式提交后的因子。提交会复制定义到
  `factor_registry/webapp_factor_library.json`，不会移动测试库记录，也不会写入评价指标。
- 新增因子默认保存到 `factor_registry/webapp_test_factors.json`，历史
  `webapp_custom_factors.json` 会继续作为测试库读取。
- `/factors` 的“遗传算法添加因子”使用独立 GP 进程，不阻塞评价 worker。页面可以配置
  训练/测试边界、论文默认进化参数、自动入库和连续 cycle，并查看 checkpoint、测试门槛、
  候选与入库结果。GP 输出写入 `outputs/gp_factor_mining/<campaign>/`。

## 验证

```bash
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python \
  webapp/server/tests/test_api.py
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python \
  webapp/server/tests/test_worker_lifecycle.py
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python \
  webapp/server/tests/test_genetic_mining_webapp.py
```

第二项使用合成 parquet 验证任务成功、worker 异常退出隔离和自动重启。
