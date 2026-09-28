#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# 权重默认使用 configs/models.json；可通过对应环境变量覆盖。
source .venv/bin/activate

export LAYA_SUBFOLDER=typed-decisions

exec "$VIRTUAL_ENV/bin/python" -m src.eval --model laya --input data/final/versions.jsonl --output "results/laya-$(date -u +%Y%m%dT%H%M%SZ)" "$@"
