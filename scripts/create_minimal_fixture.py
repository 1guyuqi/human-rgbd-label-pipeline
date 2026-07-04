#!/usr/bin/env python3
"""Create a tiny synthetic RGB-D clip + step1_2d cache for smoke testing."""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "examples" / "minimal_rvideo"
TRAJ_ROOT = FIXTURE_ROOT / "traj_000"
OUTPUT_ROOT = FIXTURE_ROOT / "output"
STEP1_2D = OUTPUT_ROOT / "step1_2d" / "0"

H, W = 240, 320
N_FRAMES = 4
N_TRACKS = 8
FX = FY = 300.0
CX, CY = W / 2.0, H / 2.0
BG_DEPTH_MM = 1000
OBJ_DEPTH_MM = 600


def _camera_in() -> np.ndarray:
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = FX
    K[1, 1] = FY
    K[0, 2] = CX
    K[1, 2] = CY
    return K


def _object_mask() -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[70:170, 110:210] = 1
    return mask


def _pixel_from_xyz(x: float, y: float, z: float) -> tuple[int, int]:
    u = int(round(x * FX / z + CX))
    v = int(round(y * FY / z + CY))
    return u, v


def main() -> None:
    mask_other = _object_mask()
    mask_tool = np.zeros_like(mask_other)

    TRAJ_ROOT.mkdir(parents=True, exist_ok=True)
    STEP1_2D.mkdir(parents=True, exist_ok=True)

    np.save(TRAJ_ROOT / "camera_in.npy", _camera_in())

    kps_tracks = np.zeros((N_FRAMES, N_TRACKS, 2), dtype=np.float32)
    kps_visibility = np.ones((N_FRAMES, N_TRACKS), dtype=np.float32)

    for t in range(N_FRAMES):
        depth = np.full((H, W), BG_DEPTH_MM, dtype=np.uint16)
        depth[mask_other > 0] = OBJ_DEPTH_MM

        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        rgb[..., 0] = 40 + t * 10
        rgb[..., 1] = 80
        rgb[mask_other > 0, 2] = 200

        cv2.imwrite(str(TRAJ_ROOT / f"rgb_{t}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(TRAJ_ROOT / f"depth_{t}.png"), depth)

        for n in range(N_TRACKS):
            x = 0.05 + 0.03 * n + 0.025 * t
            y = 0.0
            z = OBJ_DEPTH_MM / 1000.0
            u, v = _pixel_from_xyz(x, y, z)
            u = int(np.clip(u, 110, 209))
            v = int(np.clip(v, 70, 169))
            kps_tracks[t, n] = [u, v]

    np.save(STEP1_2D / "mask_tool.npy", mask_tool)
    np.save(STEP1_2D / "mask_other.npy", mask_other)
    np.save(STEP1_2D / "kps_tracks.npy", kps_tracks)
    np.save(STEP1_2D / "kps_visibility.npy", kps_visibility)

    clip_info = [
        {
            "id": 0,
            "index": str(TRAJ_ROOT),
            "st": 0,
            "ed": N_FRAMES - 1,
            "task_description": "minimal_smoke_test",
            "action": "pull",
            "object": "block",
            "tool": "hand",
            "other": "block",
            "other_only": True,
            "seq_index": list(range(N_FRAMES)),
        }
    ]
    with open(OUTPUT_ROOT / "step1_kpst_2d_info.json", "w") as f:
        json.dump(clip_info, f, indent=2)

    print(f"Fixture written under {FIXTURE_ROOT}")


if __name__ == "__main__":
    main()
