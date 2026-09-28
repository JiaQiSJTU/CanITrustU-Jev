#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# 权重默认使用 configs/models.json；可通过对应环境变量覆盖。
source .venv-semif/bin/activate

# 使用官方固定版本；可通过 SEMIF_MODEL 指定权重。
export SEMIF_REVISION="${SEMIF_REVISION:-851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}"

exec "$VIRTUAL_ENV/bin/python" -m src.eval --model semif --input data/final/versions.jsonl --output "results/semif-$(date -u +%Y%m%dT%H%M%SZ)" "$@"
