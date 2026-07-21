# 因子评价平台 · 服务器部署文档

FastAPI + React 单端口 Web 服务，用 Docker 部署。数据（`data/`，A 股行情矩阵）随包提供，
因子库自带 alpha101 83 个示例因子。

---

## 一、环境要求

- Linux 服务器，已安装 **Docker Engine 20+** 与 **Docker Compose v2**（`docker compose` 命令）
- 内存 **≥ 6 GB**（评价 worker 会把全部行情矩阵加载进内存；`docker-compose.yml` 里 `mem_limit: 6g`，可按数据规模调整）
- 磁盘 ≥ 3 GB（镜像约 1.5 GB + 行情数据约 560 MB + 运行产物）
- 无需联网访问外部服务；但**构建镜像时**需要能拉取 Docker 基础镜像和 npm/pip 依赖（见离线方案）

检查环境：

```bash
docker --version
docker compose version
```

---

## 二、快速部署（三步）

假设收到的压缩包为 `factor-platform-bundle.tar.gz`。

```bash
# 1) 解压
tar -xzf factor-platform-bundle.tar.gz
cd 因子挖掘:评价          # 目录名含冒号，后续命令注意加引号

# 2) 构建并后台启动
docker compose up -d --build

# 3) 查看日志确认启动完成
docker compose logs -f
```

看到 `Uvicorn running on http://0.0.0.0:8000` 即启动成功。
浏览器访问 **http://<服务器IP>:8000**。

> 目录名含冒号只影响宿主机上的 shell 命令（记得加引号）；容器内部代码安装在 `/app`，
> 不含冒号，无任何影响。

---

## 三、目录挂载说明

`docker-compose.yml` 把三部分挂到宿主机，便于更新与持久化：

| 宿主机路径 | 容器内 | 模式 | 用途 |
|---|---|---|---|
| `./data` | `/app/data` | 只读 | 行情数据矩阵（parquet），不进镜像 |
| `./factor_registry` | `/app/factor_registry` | 可写 | 因子库定义；网页「新增因子」会写回这里并持久化 |
| 命名卷 `platform-state` | `/app/outputs/webapp` | 可写 | 任务数据库 `platform.db` 与每次回测的结果/图表 |

- **更新行情数据**：替换宿主机 `data/` 下的 `.pq` 文件后 `docker compose restart`。
- **回测结果持久化**：存于 Docker 命名卷 `platform-state`，`docker compose down` 不会丢；
  `docker compose down -v` 会一并删除（谨慎）。

---

## 四、常用运维命令

```bash
docker compose ps                 # 查看运行状态
docker compose logs -f            # 实时日志
docker compose restart            # 重启（改数据/配置后）
docker compose up -d --build      # 改代码后重建并重启
docker compose down               # 停止并移除容器（保留数据卷）
docker compose down -v            # 停止并删除数据卷（清空回测历史，慎用）
```

### 修改对外端口

编辑 `docker-compose.yml` 的 `ports`，例如对外用 80 端口：

```yaml
    ports:
      - "80:8000"
```

改完 `docker compose up -d`。

---

## 五、可选：Nginx 反向代理 + HTTPS

如需域名或 HTTPS，在宿主机 Nginx 加一段反代（容器仍监听 8000）：

```nginx
server {
    listen 80;
    server_name factor.example.com;

    client_max_body_size 20m;
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

HTTPS 用 certbot 申请证书即可，应用本身无需改动（前端用相对路径请求 `/api`）。

---

## 六、离线服务器（无法拉取镜像/依赖）

若目标服务器不能联网，在一台**能联网**、架构相同（通常 linux/amd64）的机器上构建后导出镜像：

```bash
# 联网机器
docker compose build
docker save factor-platform:latest | gzip > factor-platform-image.tar.gz
```

把 `factor-platform-image.tar.gz` 连同项目目录一起拷到服务器：

```bash
# 离线服务器
docker load < factor-platform-image.tar.gz
# 用已加载的镜像启动（不再触发 build）
docker compose up -d --no-build
```

---

## 七、健康检查与排障

- **健康检查接口**：`curl http://127.0.0.1:8000/api/health` 返回 `{"status":"ok"}`。
- **因子库为空**：确认 `factor_registry/alpha101_runnable_factors.json` 存在且已随包解压；
  该目录是可写挂载，容器需有读权限。
- **回测任务一直排队不执行 / worker 反复重启**：多半是内存不足被 OOM。调大
  `docker-compose.yml` 的 `mem_limit` 或服务器内存后 `docker compose up -d`。
- **提交回测报「数据文件缺失」**：检查 `./data` 是否挂载成功、9 个 `.pq` 文件是否齐全
  （`close/open/high/low/volume/amount/vwap_proxy/market_cap/st_status`）。
- **构建时前端依赖或 pip 包下载失败**：服务器构建需联网；否则改用上面的「离线服务器」方案。

---

## 八、架构与技术栈（供接手同事参考）

- **后端**：FastAPI，单进程用 uvicorn 同时托管 REST API（`/api/*`）与前端 SPA。
  评价任务由应用内派生的**常驻 worker 子进程**串行执行（隔离崩溃、复用已加载的行情数据）。
  任务队列与结果记录在 SQLite（`outputs/webapp/platform.db`）。
- **前端**：Vite + React + TypeScript + Ant Design + ECharts，构建为静态资源由后端托管。
- **因子评价库**：`src/` 下的模块化评价引擎；webapp 直接 import，不改动库逻辑。
  修改 `src/` 中评价代码后的兼容性要求见 `AGENTS.md` 的「Webapp Compatibility」一节。
- 更多背景见项目根 `README.md` 与 `webapp/README.md`。
