#!/usr/bin/env bash
# 在仓库根目录执行；先在另一个终端构建并启动上游 server：
#   bash script/serve_openjev_sglang.sh --build       # 首次构建镜像
#   bash script/serve_openjev_sglang.sh              # 前台启动，等待 API 就绪
#   bash script/serve_openjev_sglang.sh --gpus device=0  # 可选：指定 GPU
# 默认接口：http://127.0.0.1:8013/v1/systemone
# 自定义端口时，通过 OPENJEV_SGLANG_ENDPOINT 设置评测接口；
# 自定义权重时，在服务端和评测端设置相同的 OPENJEV_SGLANG_WEIGHTS。
# server 就绪后运行评测：
#   bash script/openjev_sglang.sh --limit-bases 10
# 启动命令加 --dry-run 可只查看命令，不实际启动。
set -euo pipefail
cd "$(dirname "$0")/.."
# 权重默认使用 configs/models.json；可通过对应环境变量覆盖。
source .venv/bin/activate


exec "$VIRTUAL_ENV/bin/python" -m src.eval --model openjev_sglang --input data/final/versions.jsonl --output "results/openjev-sglang-$(date -u +%Y%m%dT%H%M%SZ)" "$@"
