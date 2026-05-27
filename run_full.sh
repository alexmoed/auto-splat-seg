#!/usr/bin/env bash
# run_full.sh — end-to-end driver: raw_ydown.ply → final .splat outputs.
#
# Usage:
#   ./run_full.sh <scene_dir_name>
# Example:
#   ./run_full.sh Kitchen_living_dining_v32
#
# scene_dir_name is the directory name under pipeline/. It must contain
# raw_ydown.ply. Everything else (step PLYs, 02_<slug>/ dirs, final_outputs/)
# is produced by this script.
#
# Chain:
#   1. orient.py            — y-down rotation + tilt + Hough + Qwen yaw sweep
#   2. cardinal_pick.py 1-3 — Qwen-driven cardinal rotation (the big yaw)
#   3. slice.py 1-7         — geometric pre-cleanup slicing
#   4. cleanup.py 1         — density filter
#   5. run_pipeline.sh      — docker: inventory + phase 1/2/3 + dispatch +
#                             rename + grouping + background + finals + merge
#
# Halts on any non-zero exit. Logs to <scene>/full_run.log.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <scene_dir_name>"
  echo "  e.g. $0 Kitchen_living_dining_v32"
  exit 1
fi

SCENE_NAME="$1"
ROOT="/home/ubuntu/room_pipeline_v002"
SCENE_DIR="${ROOT}/pipeline/${SCENE_NAME}"
LOG="${SCENE_DIR}/full_run.log"

if [[ ! -d "${SCENE_DIR}" ]]; then
  echo "[fatal] scene dir does not exist: ${SCENE_DIR}"
  exit 1
fi
if [[ ! -f "${SCENE_DIR}/raw_ydown.ply" ]]; then
  echo "[fatal] missing ${SCENE_DIR}/raw_ydown.ply"
  echo "  Put the raw y-down PLY there first."
  exit 1
fi

exec > >(tee -a "${LOG}") 2>&1

echo "============================================================"
echo "[$(date '+%F %T')] run_full start — scene: ${SCENE_NAME}"
echo "============================================================"

# ─── host-side foundation stages (need conda env) ────────────────
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate claude_seg

stage() {
  echo ""
  echo "============================================================"
  echo "[$(date '+%T')] $1"
  echo "============================================================"
}

cd "${ROOT}"

stage "1. orient.py — rotate to y-down + tilt + PCA + Hough + Qwen yaw"
python3 pipeline/orient.py "${SCENE_DIR}"

stage "2. cardinal_pick.py 1-3 — Qwen cardinal rotation"
for s in 1 2 3; do
  echo "--- cardinal_pick step ${s} ---"
  python3 pipeline/cardinal_pick.py "${SCENE_DIR}" --step "${s}"
done

stage "3. slice.py 1-7 — geometric pre-cleanup"
for s in 1 2 3 4 5 6 7; do
  echo "--- slice step ${s} ---"
  python3 pipeline/slice.py "${SCENE_DIR}" --step "${s}"
done

stage "4. cleanup.py 1 — density filter"
python3 pipeline/cleanup.py "${SCENE_DIR}" --step 1

# ─── docker-side per-object dispatch + finalize ──────────────────
stage "5. run_pipeline.sh — inventory + phase 1/2/3 + dispatch + finalize"
./run_pipeline.sh "${SCENE_NAME}"

echo ""
echo "============================================================"
echo "[$(date '+%F %T')] FULL PIPELINE COMPLETE — ${SCENE_NAME}"
echo "============================================================"
echo "  Foundation:    ${SCENE_DIR}/step{1,3,4,6,7,8}*.ply"
echo "  Aligned:       ${SCENE_DIR}/step7_cardinal_aligned.ply"
echo "  Sliced:        ${SCENE_DIR}/step7_sliced.ply"
echo "  Density-filt:  ${SCENE_DIR}/step8_density_filtered.ply"
echo "  Per-object:    ${SCENE_DIR}/02_<slug>/"
echo "  Final splats:  ${SCENE_DIR}/final_outputs/"
echo "  Reassembled:   ${SCENE_DIR}/scene_reassembled.ply"
echo "  Log:           ${LOG}"
