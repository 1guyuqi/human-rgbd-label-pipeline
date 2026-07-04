#!/usr/bin/env python3
"""Run 3D-only labeling on the synthetic fixture and verify core outputs."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "examples" / "minimal_rvideo"
OUTPUT_ROOT = FIXTURE_ROOT / "output"
CLIP_DIR = OUTPUT_ROOT / "data" / "0"


def _check(path: Path, desc: str) -> None:
    if not path.is_file():
        raise SystemExit(f"FAIL: missing {desc}: {path}")
    print(f"OK: {desc} -> {path} ({path.stat().st_size} bytes)")


def main() -> None:
    create_script = REPO_ROOT / "scripts" / "create_minimal_fixture.py"
    subprocess.run([sys.executable, str(create_script)], check=True, cwd=REPO_ROOT)

    label_gen = REPO_ROOT / "rvideo" / "label_gen.py"
    cmd = [
        sys.executable,
        str(label_gen),
        "--save_root",
        str(OUTPUT_ROOT),
        "--only_3d",
        "--recompute_3d",
        "--static_threshold",
        "0.01",
        "--n_pcd_points",
        "512",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT / "rvideo")

    _check(CLIP_DIR / "pcd.npy", "pcd.npy")
    _check(CLIP_DIR / "kpst_traj.npy", "kpst_traj.npy")
    _check(OUTPUT_ROOT / "metadata_egosoft_demo.json", "metadata_egosoft_demo.json")
    meta_json = OUTPUT_ROOT / "metadata.json"
    shutil.copy2(OUTPUT_ROOT / "metadata_egosoft_demo.json", meta_json)
    _check(meta_json, "metadata.json (copy of RVideo clip metadata)")

    pcd = np.load(CLIP_DIR / "pcd.npy")
    kpst = np.load(CLIP_DIR / "kpst_traj.npy")
    if pcd.ndim != 2 or pcd.shape[1] != 7 or pcd.shape[0] == 0:
        raise SystemExit(f"FAIL: bad pcd shape {pcd.shape}, expected (N, 7)")
    if kpst.ndim != 3 or kpst.shape[0] == 0:
        raise SystemExit(f"FAIL: bad kpst_traj shape {kpst.shape}, expected (N, T, 3)")

    print(f"Shapes: pcd={pcd.shape}, kpst_traj={kpst.shape}")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
