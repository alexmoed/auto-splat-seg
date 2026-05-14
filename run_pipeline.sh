#!/usr/bin/env bash
# run_pipeline.sh — end-to-end Kitchen_living_dining pipeline.
#
# Usage:
#   ./run_pipeline.sh <scene_dir_name>
# Example:
#   ./run_pipeline.sh Kitchen_living_dining_v9
#
# Assumes the scene_dir already contains the foundation PLYs
# (raw_ydown, step1..step8). If you're starting fresh, copy or
# hardlink them from a prior scene first.
#
# Logs to ./<scene_dir>/run_pipeline.log.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <scene_dir_name>"
  echo "  e.g. $0 Kitchen_living_dining_v9"
  exit 1
fi

SCENE_NAME="$1"
SCENE_DIR="/home/ubuntu/room_pipeline_v002/pipeline/${SCENE_NAME}"

# Container + endpoint config — overridable so run_parallel.sh can point
# this driver at a per-GPU instance.
CONTAINER="${CONTAINER:-splat-pipe}"
QWEN_URL="${QWEN_URL:-http://127.0.0.1:8000/v1}"
SAM_URL="${SAM_URL:-http://127.0.0.1:8001}"
QWEN_PROBE="${QWEN_URL%/v1}/v1/models"

LOG="${SCENE_DIR}/run_pipeline.log"

if [[ ! -d "${SCENE_DIR}" ]]; then
  echo "[fatal] scene dir does not exist: ${SCENE_DIR}"
  exit 1
fi

# Critical foundation PLYs the pipeline needs
for f in step7_cardinal_aligned.ply step7_sliced.ply step8_density_filtered.ply; do
  if [[ ! -f "${SCENE_DIR}/${f}" ]]; then
    echo "[fatal] missing ${SCENE_DIR}/${f}"
    exit 1
  fi
done

mkdir -p "${SCENE_DIR}"
exec > >(tee -a "${LOG}") 2>&1

echo "============================================================"
echo "[$(date '+%F %T')] pipeline start — scene: ${SCENE_NAME}"
echo "============================================================"

# ─── 1. start container (if not running) ────────────────────────
if sudo docker ps --format "{{.Names}}" | grep -qx "${CONTAINER}"; then
  CURRENT_MOUNT=$(sudo docker inspect "${CONTAINER}" \
    --format '{{range .Mounts}}{{if eq .Destination "/workspace/scene"}}{{.Source}}{{end}}{{end}}')
  if [[ "${CURRENT_MOUNT}" != "${SCENE_DIR}" ]]; then
    echo "[$(date '+%T')] container mounted at ${CURRENT_MOUNT}, restarting at ${SCENE_DIR}"
    sudo docker rm -f "${CONTAINER}"
  else
    echo "[$(date '+%T')] container already running with correct mount"
  fi
fi

if ! sudo docker ps --format "{{.Names}}" | grep -qx "${CONTAINER}"; then
  echo "[$(date '+%T')] starting container"
  sudo docker run -d --name "${CONTAINER}" --gpus all --ipc=host --shm-size=16gb \
    -v /home/ubuntu/models/Qwen3.6-35B-A3B-AWQ:/models/qwen36-awq:ro \
    -v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface \
    -v /home/ubuntu/.cache/torch_extensions:/root/.cache/torch_extensions \
    -v "${SCENE_DIR}:/workspace/scene" \
    splat-pipeline:latest \
    sleep infinity >/dev/null
fi

# ─── 2. wait for vLLM ───────────────────────────────────────────
echo "[$(date '+%T')] waiting for vLLM ..."
echo "[$(date '+%T')] (cold start typically 5-7 min — model is loading 9 checkpoint shards)"
START=$SECONDS
LAST_HEARTBEAT=0
while true; do
  if sudo docker exec "${CONTAINER}" curl -sf -m 2 \
       "${QWEN_PROBE}" >/dev/null 2>&1; then
    echo "[$(date '+%T')] vLLM ready ($((SECONDS - START))s)"
    break
  fi
  # Bail if container died
  if ! sudo docker ps --format "{{.Names}}" | grep -qx "${CONTAINER}"; then
    echo "[fatal] container died while waiting for vLLM"
    sudo docker logs "${CONTAINER}" 2>&1 | tail -30 || true
    exit 1
  fi
  if (( SECONDS - START > 900 )); then
    echo "[fatal] vLLM didn't come up in 15min"
    exit 1
  fi
  # Heartbeat every 30s with shard-load progress from the vLLM log
  if (( SECONDS - LAST_HEARTBEAT >= 30 )); then
    LAST_HEARTBEAT=$SECONDS
    elapsed=$((SECONDS - START))
    progress=$(sudo docker exec "${CONTAINER}" \
        grep -oE 'Loading safetensors checkpoint shards: +[0-9]+%' \
        /tmp/vllm.log 2>/dev/null | tail -1 || echo "starting...")
    echo "[$(date '+%T')] waited ${elapsed}s — ${progress}"
  fi
  sleep 15
done

# ─── 3. run the pipeline stages ─────────────────────────────────
run_stage() {
  local label="$1" ; shift
  echo ""
  echo "============================================================"
  echo "[$(date '+%T')] ${label}"
  echo "============================================================"
  if sudo docker exec \
       -e QWEN_URL="${QWEN_URL}" \
       -e SAM_URL="${SAM_URL}" \
       "${CONTAINER}" "$@"; then
    echo "[$(date '+%T')] ${label} — OK"
  else
    rc=$?
    echo "[fatal] ${label} exited ${rc}"
    exit ${rc}
  fi
}

run_stage "step 1 (inventory + phase 1/2/3 extract)" \
    python3 /workspace/pipeline/run_all.py /workspace/scene --step 1

run_stage "step 2 (per-object dispatch — sam_carve, floor_drop, sam_tight)" \
    python3 /workspace/pipeline/run_all.py /workspace/scene --step 2

run_stage "rename_to_qwen (folders → Qwen-refined labels)" \
    python3 /workspace/pipeline/rename_to_qwen.py /workspace/scene

run_stage "step 3 (parent/child grouping + subtract)" \
    python3 /workspace/pipeline/run_all.py /workspace/scene --step 3

run_stage "extract_background" \
    python3 /workspace/pipeline/extract_background.py /workspace/scene

run_stage "extract_final_outputs (PLY → .splat)" \
    python3 /workspace/pipeline/extract_final_outputs.py /workspace/scene

run_stage "merge_scene (reassembled PLY + verification renders)" \
    python3 /workspace/pipeline/merge_scene.py /workspace/scene

echo ""
echo "============================================================"
echo "[$(date '+%F %T')] PIPELINE COMPLETE"
echo "============================================================"
echo "Final outputs: ${SCENE_DIR}/final_outputs/"
echo "Reassembled  : ${SCENE_DIR}/scene_reassembled.ply"
echo "Renders      : ${SCENE_DIR}/renders/reassembled/"
ls "${SCENE_DIR}/final_outputs/" 2>/dev/null | head -5
echo "..."
echo "Full log     : ${LOG}"
