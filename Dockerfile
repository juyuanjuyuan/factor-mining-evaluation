# syntax=docker/dockerfile:1

# ---------- 阶段 1：构建前端静态资源 ----------
FROM node:20-slim AS frontend
WORKDIR /build
# 先只拷依赖清单，命中构建缓存
COPY webapp/frontend/package.json webapp/frontend/package-lock.json ./
RUN npm ci
COPY webapp/frontend/ ./
RUN npm run build
# 产物在 /build/dist

# ---------- 阶段 2：Python 运行时 ----------
FROM python:3.11-slim AS runtime

# 说明：项目根在本地叫「因子挖掘:评价」，路径含冒号会破坏 PYTHONPATH 等以冒号
# 分隔的配置。容器内一律安装到无冒号的 /app，规避该问题。
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # 覆盖 executor 里 macOS 风格的默认值 /private/tmp/matplotlib
    MPLCONFIGDIR=/tmp/matplotlib

# 拷入库代码与打包清单（data/ 不进镜像，运行时用挂载卷提供，见 docker-compose.yml）
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY webapp/server/ ./webapp/server/
# factor_registry 里的因子定义会被挂载卷覆盖，这里拷一份作为镜像自带的默认因子库
COPY factor_registry/ ./factor_registry/

# editable 安装库本身 + webapp 额外依赖(fastapi/uvicorn)。
# 3.11-slim 下 numpy/pandas/pyarrow/matplotlib 均有预编译 wheel，无需编译工具链。
RUN pip install -e ".[webapp]"

# 放入已构建好的前端静态资源，main.py 会从此目录挂载 SPA
COPY --from=frontend /build/dist ./webapp/frontend/dist

# 生成运行时状态目录（platform.db 与各 run 输出），下方声明为卷以便持久化
RUN mkdir -p /app/outputs/webapp /tmp/matplotlib
VOLUME ["/app/outputs/webapp"]

EXPOSE 8000

# 单进程即托管 API + 前端 SPA（同端口）。评价 worker 由应用内部派生子进程管理。
CMD ["uvicorn", "--app-dir", "webapp/server", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
