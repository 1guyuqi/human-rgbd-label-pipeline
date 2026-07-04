#!/usr/bin/env bash
# Re-run 3D export for all RVideo tasks under GT_ROOT (kpst/pcd alignment fix).
#
# Required:
#   GT_ROOT          folder containing task subdirs with step1_kpst_2d_info.json
#   RECORD_ROOT      local Record dataset root (remaps paths in clip JSON)
#
# Optional:
#   RECORDED_RGBD_ROOT   alias / depth fallback root
#   PIPELINE_ROOT        repo root (auto-detected)
#
# Example:
#   export GT_ROOT=/path/to/process_data
#   export RECORD_ROOT=/path/to/Record
#   bash scripts/batch_recompute_3d.sh
set -euo pipefail

PIPELINE_ROOT="${PIPELINE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
GT_ROOT="${GT_ROOT:-./process_data}"
PYTHON="${PYTHON:-python}"

if [[ ! -d "$GT_ROOT" ]]; then
  echo "ERROR: GT_ROOT is not a directory: $GT_ROOT" >&2
  echo "Set GT_ROOT to the folder that contains your RVideo task outputs." >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "${RECORD_ROOT:-}" ]]; then
  EXTRA_ARGS+=(--record_root "$RECORD_ROOT")
fi
if [[ -n "${RECORDED_RGBD_ROOT:-}" ]]; then
  EXTRA_ARGS+=(--recorded_rgbd_root "$RECORDED_RGBD_ROOT")
fi
if [[ "${USE_REFINED_DEPTH:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--use_refined_depth)
fi
if [[ "${NO_AUG:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--aug_rot_deg 0 --aug_jitter_std 0)
fi

cd "$PIPELINE_ROOT/rvideo"
exec "$PYTHON" label_gen.py \
  --gt_root "$GT_ROOT" \
  --only_3d \
  --recompute_3d \
  "${EXTRA_ARGS[@]}" \
  "$@"
