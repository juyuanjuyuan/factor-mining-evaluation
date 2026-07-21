#!/usr/bin/env bash
# 打包部署压缩包，发给公司服务器。
# 在项目根目录执行： bash make_bundle.sh
# 产物： ../factor-platform-bundle.tar.gz
#
# 包含：库代码(src/)、后端(webapp/server)、前端源码、行情数据(data/)、
#       alpha101 因子库、部署文件(Dockerfile / docker-compose.yml / DEPLOYMENT.md)。
# 排除：本地历史评价结果(outputs/)、node_modules、构建产物、Python 缓存、
#       网页自定义因子(本地测试遗留)、笔记本与 PDF。

set -euo pipefail

ROOT_NAME="$(basename "$PWD")"
OUT="../factor-platform-bundle.tar.gz"

# 临时把本地测试遗留的自定义因子移出，避免打进包（部署后由使用者重新新增）
CUSTOM="factor_registry/webapp_custom_factors.json"
STASH=""
if [ -f "$CUSTOM" ]; then
  STASH="$(mktemp)"
  mv "$CUSTOM" "$STASH"
  echo "已临时移出 $CUSTOM（不打包）"
fi
restore() { [ -n "$STASH" ] && mv "$STASH" "$CUSTOM" && echo "已恢复 $CUSTOM"; }
trap restore EXIT

# 从上级目录打包整个项目目录，套用排除清单
tar -czf "$OUT" \
  --exclude="$ROOT_NAME/outputs" \
  --exclude="$ROOT_NAME/webapp/frontend/node_modules" \
  --exclude="$ROOT_NAME/webapp/frontend/dist" \
  --exclude="$ROOT_NAME/.git" \
  --exclude="*/__pycache__" \
  --exclude="*.py[cod]" \
  --exclude="*.tsbuildinfo" \
  --exclude="*/.venv" \
  --exclude="*/venv" \
  --exclude="*/.pytest_cache" \
  --exclude="*/.mypy_cache" \
  --exclude="*/.ruff_cache" \
  --exclude=".DS_Store" \
  --exclude="$ROOT_NAME/tmp" \
  --exclude="$ROOT_NAME/notebooks" \
  --exclude="*.pdf" \
  -C .. "$ROOT_NAME"

echo ""
echo "打包完成： $(cd .. && pwd)/$(basename "$OUT")"
du -h "$OUT" | cut -f1 | sed 's/^/压缩包大小： /'
