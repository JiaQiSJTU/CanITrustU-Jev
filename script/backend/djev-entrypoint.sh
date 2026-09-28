#!/usr/bin/env bash
set -euo pipefail
vllm serve /weights --served-model-name dgemma --host 127.0.0.1 --port 8000 \
  --diffusion-config '{"canvas_length":64}' --max-logprobs 32 \
  --enable-prefix-caching --async-scheduling --max-model-len "$MODEL_MAX_TOKENS" &
engine_pid=$!
api_pid=''
cleanup() {
  kill -TERM "$engine_pid" ${api_pid:+"$api_pid"} 2>/dev/null || true
  wait || true
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
until curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; do
  if ! kill -0 "$engine_pid" 2>/dev/null; then
    echo 'djev: vLLM exited during startup' >&2
    exit 1
  fi
  sleep 5
done
python /djev/structured_server.py --upstream http://127.0.0.1:8000 \
  --model dgemma --tokenizer /weights --canvas 64 --canvas-step 16 \
  --host 0.0.0.0 --port 8080 &
api_pid=$!
wait -n "$engine_pid" "$api_pid"
exit 1
