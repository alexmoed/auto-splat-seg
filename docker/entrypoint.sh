#!/usr/bin/env bash
# Container entrypoint: start vLLM serving Qwen 3.6 AWQ in the
# background, wait for :8000/v1/models to respond, then exec the
# command passed to `docker run` (defaults to bash via the Dockerfile
# CMD).
#
# Configurable via env:
#   QWEN_MODEL_PATH               (default /models/qwen36-awq)
#   QWEN_MODEL                    served model name (default qwen36-awq)
#   QWEN_GPU_MEMORY_UTILIZATION   (default 0.75  — 48GB GPU)
#   QWEN_MAX_MODEL_LEN            (default 8192  — floor_drop needs >4096)
#   QWEN_PORT                     (default 8000)
#   SKIP_VLLM=1                   skip vLLM launch (use external Qwen)
#
# Logs:
#   /tmp/vllm.log
set -euo pipefail

QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/models/qwen36-awq}"
QWEN_MODEL="${QWEN_MODEL:-qwen36-awq}"
QWEN_PORT="${QWEN_PORT:-8000}"
QWEN_GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-0.75}"
QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-16384}"
SKIP_VLLM="${SKIP_VLLM:-0}"

SAM_SERVER_PORT="${SAM_SERVER_PORT:-8001}"
SKIP_SAM_SERVER="${SKIP_SAM_SERVER:-0}"

if [[ "${SKIP_VLLM}" != "1" ]]; then
  if [[ ! -d "${QWEN_MODEL_PATH}" ]]; then
    echo "[entrypoint] FATAL: ${QWEN_MODEL_PATH} not found." >&2
    echo "  Mount the Qwen 3.6 35B-A3B-AWQ weights with:" >&2
    echo "    -v /host/path/to/Qwen3.6-35B-A3B-AWQ:/models/qwen36-awq:ro" >&2
    echo "  Or set SKIP_VLLM=1 if Qwen is served from elsewhere." >&2
    exit 1
  fi

  echo "[entrypoint] launching vLLM in background:" >&2
  echo "  model_path=${QWEN_MODEL_PATH}" >&2
  echo "  served_name=${QWEN_MODEL}" >&2
  echo "  port=${QWEN_PORT}" >&2
  echo "  gpu_memory_utilization=${QWEN_GPU_MEMORY_UTILIZATION}" >&2
  echo "  max_model_len=${QWEN_MAX_MODEL_LEN}" >&2

  nohup vllm serve "${QWEN_MODEL_PATH}" \
      --served-model-name "${QWEN_MODEL}" \
      --host 0.0.0.0 \
      --port "${QWEN_PORT}" \
      --gpu-memory-utilization "${QWEN_GPU_MEMORY_UTILIZATION}" \
      --max-model-len "${QWEN_MAX_MODEL_LEN}" \
      --quantization awq_marlin \
      > /tmp/vllm.log 2>&1 &

  VLLM_PID=$!
  echo "[entrypoint] vLLM pid=${VLLM_PID}, waiting for readiness..." >&2

  # Wait for vLLM to load the model + warm up torch.compile + open the
  # port. First cold start of Qwen3.6 35B-A3B-AWQ with vLLM's
  # torch.compile takes 200-400s on a 48GB card. Configurable via
  # QWEN_READY_TIMEOUT env (default 900s = 15 min).
  TIMEOUT="${QWEN_READY_TIMEOUT:-900}"
  ELAPSED=0
  until curl -sSf "http://127.0.0.1:${QWEN_PORT}/v1/models" >/dev/null 2>&1; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    if [[ ${ELAPSED} -ge ${TIMEOUT} ]]; then
      echo "[entrypoint] FATAL: vLLM didn't become ready in ${TIMEOUT}s" >&2
      echo "[entrypoint] last 30 log lines:" >&2
      tail -30 /tmp/vllm.log >&2 || true
      exit 1
    fi
    # NOTE: we don't check kill -0 on VLLM_PID anymore — vllm's
    # `vllm serve` command exits early after spawning the EngineCore
    # subprocess, so the parent PID disappears even when vLLM is
    # healthy. Rely on the timeout + curl probe instead.
    echo "[entrypoint] still waiting (${ELAPSED}s)..." >&2
  done

  echo "[entrypoint] vLLM ready at http://127.0.0.1:${QWEN_PORT}" >&2
fi

# ─── SAM3 persistent server ─────────────────────────────────────
# Loads facebook/sam3 ONCE so each subprocess of the pipeline doesn't
# reload the weights. Scripts find it via SAM_URL env var (set below).
if [[ "${SKIP_SAM_SERVER}" != "1" ]]; then
  echo "[entrypoint] launching sam_server on port ${SAM_SERVER_PORT}..." >&2
  SAM_SERVER_PORT="${SAM_SERVER_PORT}" \
    nohup python3 /workspace/pipeline/sam_server.py \
      > /tmp/sam_server.log 2>&1 &
  SAM_PID=$!
  echo "[entrypoint] sam_server pid=${SAM_PID}, waiting for readiness..." >&2

  # SAM3 cold load is ~30-60s (much faster than vLLM). 5 min timeout.
  SAM_TIMEOUT="${SAM_READY_TIMEOUT:-300}"
  ELAPSED=0
  until curl -sSf "http://127.0.0.1:${SAM_SERVER_PORT}/health" >/dev/null 2>&1; do
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    if [[ ${ELAPSED} -ge ${SAM_TIMEOUT} ]]; then
      echo "[entrypoint] WARN: sam_server didn't come up in ${SAM_TIMEOUT}s — pipeline will fall back to in-process load" >&2
      tail -20 /tmp/sam_server.log >&2 || true
      break
    fi
  done

  if curl -sSf "http://127.0.0.1:${SAM_SERVER_PORT}/health" >/dev/null 2>&1; then
    export SAM_URL="http://127.0.0.1:${SAM_SERVER_PORT}"
    echo "[entrypoint] sam_server ready at ${SAM_URL}" >&2
  fi
fi

# Exec whatever the user requested. Default Dockerfile CMD is `bash`.
exec "$@"
