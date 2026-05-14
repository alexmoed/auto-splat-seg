#!/usr/bin/env bash
# run_parallel.sh — flip-the-switch parallel runner for N scenes on N GPUs.
#
# Usage:
#   ./run_parallel.sh <scene_dir_1> [<scene_dir_2> ...]
#
# Examples:
#   ./run_parallel.sh livingroom_a livingroom_b livingroom_c
#   ./run_parallel.sh scene_*
#
# What it does:
#   1. Detects available NVIDIA GPUs (via nvidia-smi).
#   2. For each scene argument, picks the next GPU round-robin and
#      assigns a unique container name + non-colliding ports
#      (vLLM 8000+i, sam_server 9000+i).
#   3. Launches one detached container per scene; runs run_pipeline.sh
#      against each via env-var-routed Qwen + SAM URLs.
#   4. Waits for all pipelines to complete, prints summary.
#
# Single-GPU note: if scenes > GPUs, scenes are run sequentially on the
# same GPU (no concurrency, since vLLM holds the full model in VRAM).
# This script doesn't enforce that — it'll happily start N containers
# on 1 GPU and you'll OOM. Pass at most NGPU scenes at a time.
#
# Env overrides:
#   QWEN_WEIGHTS_HOST_PATH  (default /home/ubuntu/models/Qwen3.6-35B-A3B-AWQ)
#   PIPELINE_IMAGE          (default splat-pipeline:latest)
#   SCENE_ROOT              (default /home/ubuntu/room_pipeline_v002/pipeline)
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <scene_dir_1> [<scene_dir_2> ...]" >&2
  echo "  scene names are dirs under SCENE_ROOT (default: ./pipeline/)" >&2
  exit 1
fi

SCENE_ROOT="${SCENE_ROOT:-/home/ubuntu/room_pipeline_v002/pipeline}"
QWEN_WEIGHTS="${QWEN_WEIGHTS_HOST_PATH:-/home/ubuntu/models/Qwen3.6-35B-A3B-AWQ}"
IMAGE="${PIPELINE_IMAGE:-splat-pipeline:latest}"

# Discover GPUs
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
if [[ "${NGPU}" -lt 1 ]]; then
  echo "[fatal] no NVIDIA GPUs detected (nvidia-smi -L returned 0)" >&2
  exit 1
fi
NSCENES=$#

echo "============================================================"
echo "[$(date '+%F %T')] parallel pipeline start"
echo "  scenes        : ${NSCENES} → $*"
echo "  GPUs detected : ${NGPU}"
echo "  scene root    : ${SCENE_ROOT}"
echo "  image         : ${IMAGE}"
echo "============================================================"

if [[ "${NSCENES}" -gt "${NGPU}" ]]; then
  echo "[warn] ${NSCENES} scenes > ${NGPU} GPUs — multiple containers will fight for the same GPU's VRAM." >&2
  echo "[warn] Run at most ${NGPU} scenes at a time to avoid OOM." >&2
fi

PIDS=()
SCENE_NAMES=()
CONTAINER_NAMES=()
LOG_FILES=()

i=0
for SCENE_NAME in "$@"; do
  SCENE_DIR="${SCENE_ROOT}/${SCENE_NAME}"
  if [[ ! -d "${SCENE_DIR}" ]]; then
    echo "[fatal] scene dir does not exist: ${SCENE_DIR}" >&2
    exit 1
  fi

  GPU_ID=$(( i % NGPU ))
  QWEN_PORT=$(( 8000 + i ))
  SAM_PORT=$(( 9000 + i ))
  CONTAINER="splat-pipe-${i}"
  LOG="${SCENE_DIR}/run_parallel.log"

  echo "  → slot ${i}: scene=${SCENE_NAME}  gpu=${GPU_ID}  vllm=${QWEN_PORT}  sam=${SAM_PORT}  container=${CONTAINER}"

  # Tear down any stale container with this name
  sudo docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

  # Start the container (detached) on the assigned GPU + ports
  sudo docker run -d --name "${CONTAINER}" \
      --gpus "device=${GPU_ID}" --ipc=host --shm-size=16gb \
      -e QWEN_PORT="${QWEN_PORT}" \
      -e SAM_SERVER_PORT="${SAM_PORT}" \
      -v "${QWEN_WEIGHTS}:/models/qwen36-awq:ro" \
      -v "/home/ubuntu/.cache/huggingface:/root/.cache/huggingface" \
      -v "/home/ubuntu/.cache/torch_extensions:/root/.cache/torch_extensions" \
      -v "${SCENE_DIR}:/workspace/scene" \
      -p "127.0.0.1:${QWEN_PORT}:${QWEN_PORT}" \
      -p "127.0.0.1:${SAM_PORT}:${SAM_PORT}" \
      "${IMAGE}" \
      sleep infinity >/dev/null

  SCENE_NAMES+=("${SCENE_NAME}")
  CONTAINER_NAMES+=("${CONTAINER}")
  LOG_FILES+=("${LOG}")

  # Spawn the pipeline driver in background, pointed at this container's URLs
  (
    QWEN_URL="http://127.0.0.1:${QWEN_PORT}/v1" \
    SAM_URL="http://127.0.0.1:${SAM_PORT}" \
    CONTAINER="${CONTAINER}" \
    bash "$(dirname "$0")/run_pipeline.sh" "${SCENE_NAME}" \
        > "${LOG}" 2>&1
  ) &
  PIDS+=($!)

  i=$(( i + 1 ))
done

echo ""
echo "[$(date '+%F %T')] all ${NSCENES} pipelines launched in background"
echo "  tail any log: tail -f <scene_dir>/run_parallel.log"
echo ""

# Wait for all
FAIL=0
for idx in "${!PIDS[@]}"; do
  PID="${PIDS[$idx]}"
  SCENE="${SCENE_NAMES[$idx]}"
  if wait "${PID}"; then
    echo "[$(date '+%T')] ✓ ${SCENE} (slot ${idx}) DONE"
  else
    RC=$?
    echo "[$(date '+%T')] ✗ ${SCENE} (slot ${idx}) FAILED (exit ${RC})" >&2
    FAIL=$(( FAIL + 1 ))
  fi
done

echo ""
echo "============================================================"
if [[ "${FAIL}" -eq 0 ]]; then
  echo "[$(date '+%F %T')] all ${NSCENES} pipelines complete ✓"
else
  echo "[$(date '+%F %T')] ${FAIL} of ${NSCENES} pipelines failed ✗" >&2
fi
echo "============================================================"

# Cleanup containers (leave logs in place)
for CONTAINER in "${CONTAINER_NAMES[@]}"; do
  sudo docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
done

exit "${FAIL}"
