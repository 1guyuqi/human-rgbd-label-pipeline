#!/usr/bin/env bash
# Re-run 3D export for all RVideo tasks under process_data (kpst/pcd alignment fix).
# Requires raw RGB-D on disk (mount UBU or set RECORD_ROOT).
set -euo pipefail

PIPELINE_ROOT="${PIPELINE_ROOT:-/home/ljx/code/human-rgbd-label-pipeline}"
GT_ROOT="${GT_ROOT:-/home/ljx/文档/work/general_flow/process_data}"
PYTHON="${PYTHON:-python}"

EXTRA_ARGS=()
if [[ -n "${RECORD_ROOT:-}" ]]; then
  EXTRA_ARGS+=(--record_root "$RECORD_ROOT")
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
