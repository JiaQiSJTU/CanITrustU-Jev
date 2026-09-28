#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

# 在外部先 export JEV_API_KEY，再运行 bash script/jev.sh；脚本不覆盖密钥。
: "${JEV_API_KEY:?请先在外部设置 JEV_API_KEY}"
export JEV_ENDPOINT="${JEV_ENDPOINT:-https://api.typesafe.ai/v1/systemone}"

exec "$VIRTUAL_ENV/bin/python" -m src.eval --model jev --input data/final/versions.jsonl --output "results/jev-$(date -u +%Y%m%dT%H%M%SZ)" "$@"
