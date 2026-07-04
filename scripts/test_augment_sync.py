#!/usr/bin/env python3
"""Unit check: kpst_3d receives the same global SE(3) as point clouds."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rvideo"))
from utils import apply_global_se3


def main() -> None:
    rng = np.random.default_rng(0)
    R_z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    t = np.array([0.01, -0.02, 0.03], dtype=np.float32)

    pcd_pts = rng.normal(size=(100, 3)).astype(np.float32)
    kpst = rng.normal(size=(16, 4, 3)).astype(np.float32)

    pcd_aug = apply_global_se3(pcd_pts, R_z, t)
    kpst_aug = apply_global_se3(kpst, R_z, t)

    expected_pcd = (pcd_pts @ R_z.T) + t
    expected_kpst = (kpst @ R_z.T) + t.reshape(1, 1, 3)

    assert np.allclose(pcd_aug, expected_pcd), "pcd augmentation mismatch"
    assert np.allclose(kpst_aug, expected_kpst), "kpst augmentation mismatch"
    print("augment sync OK")


if __name__ == "__main__":
    main()
