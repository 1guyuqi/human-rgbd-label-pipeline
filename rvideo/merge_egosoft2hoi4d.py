#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge ego-centric (RVideo) labeled clips into an HOI4D-style dataset folder and
assign train/val/test splits with a stable hash-based rule.

Idempotent merge behavior:
1) Append to the current hoi4d_dir/metadata.json (initialize if missing)
2) Allocate new clip IDs from max(existing data/) + 1
3) On first run, backup metadata.json / metadata_stat.json as *_hoi4d_org.json
4) Optional deduplication by split_key when re-importing the same batch

Usage:
  python merge_egosoft2hoi4d.py \\
    --hoi4d_dir ./output/hoi4d \\
    --egosoft_dir ./output/rvideo/my_task \\
    --train_ratio 0.8 \\
    --val_ratio 0.1
"""

import os
import json
import copy
import hashlib
import shutil
import numpy as np
from pathlib import Path


def stable_split(key: str, train_ratio=0.8, val_ratio=0.1):
    """Stable, reproducible split by hashing key -> [0,1). Returns: 'train'/'val'/'test'."""
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    r = (int(h[:8], 16) % 1_000_000) / 1_000_000.0
    if r < train_ratio:
        return "train"
    elif r < train_ratio + val_ratio:
        return "val"
    else:
        return "test"


def load_json(p: Path):
    with p.open("rb") as f:
        return json.load(f)


def save_json(obj, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def copy_dir_contents(src_dir: Path, dst_dir: Path):
    """
    Copy all files under src_dir into dst_dir (flat copy: src/* -> dst/).
    Keeps behavior similar to: cp {src}/* {dst}/
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    if not src_dir.exists():
        raise FileNotFoundError(f"Source dir not found: {src_dir}")

    for item in src_dir.iterdir():
        if item.is_file():
            shutil.copy2(str(item), str(dst_dir / item.name))
        elif item.is_dir():
            shutil.copytree(str(item), str(dst_dir / item.name), dirs_exist_ok=True)


def ensure_split_keys(metadata: dict):
    for k in ["train", "val", "test"]:
        if k not in metadata or metadata[k] is None:
            metadata[k] = []


def get_current_max_fold_idx(data_dir: Path) -> int:
    """Scan data_dir for numeric subfolders and return max index; 0 if none."""
    data_dir.mkdir(parents=True, exist_ok=True)
    fold_list = []
    for x in data_dir.iterdir():
        if x.is_dir():
            try:
                fold_list.append(int(x.name))
            except ValueError:
                pass
    return max(fold_list) if fold_list else 0


def build_existing_keys(metadata: dict):
    """
    Build a set of split_keys already in metadata to avoid duplicates.
    We store split_key into each clip (new field) so it’s robust across runs.
    If old entries don't have split_key, we fallback to index|action|object.
    """
    keys = set()
    for sp in ["train", "val", "test"]:
        for c in metadata.get(sp, []):
            if isinstance(c, dict):
                if "split_key" in c:
                    keys.add(c["split_key"])
                else:
                    # best-effort fallback
                    idx = c.get("index", "")
                    act = c.get("action", "")
                    obj = c.get("object", "")
                    if idx != "" and act != "" and obj != "":
                        keys.add(f"{idx}|{act}|{obj}")
    return keys


def main():
    import argparse

    parser = argparse.ArgumentParser("TAP-KPST Label Extraction (merge soft into HOI4D)")
    parser.add_argument("--hoi4d_dir", type=str, default="/home/ycb/HOI4D_KPST")
    parser.add_argument("--egosoft_dir", type=str, default="./Soft_KPST")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--allow_duplicate", action="store_true",
                        help="Allow importing the same soft clip multiple times (default: False)")
    args = parser.parse_args()

    hoi4d_dir = Path(args.hoi4d_dir)
    egosoft_dir = Path(args.egosoft_dir)

    # --------- Paths ---------
    meta_fp = hoi4d_dir / "metadata.json"
    stat_fp = hoi4d_dir / "metadata_stat.json"
    org_meta_fp = hoi4d_dir / "metadata_hoi4d_org.json"
    org_stat_fp = hoi4d_dir / "metadata_stat_hoi4d_org.json"

    # --------- Load CURRENT HOI4D metadata (append-based) ---------
    if meta_fp.exists():
        hoi4d_metadata = load_json(meta_fp)
    else:
        hoi4d_metadata = {"train": [], "val": [], "test": []}

    if stat_fp.exists():
        hoi4d_metadata_stat = load_json(stat_fp)
    else:
        hoi4d_metadata_stat = {}

    ensure_split_keys(hoi4d_metadata)

    # --------- Backup original (copy, only once) ---------
    # If you want a "true original HOI4D" backup, run once on pristine folder.
    if meta_fp.exists() and (not org_meta_fp.exists()):
        shutil.copy2(str(meta_fp), str(org_meta_fp))
    if stat_fp.exists() and (not org_stat_fp.exists()):
        shutil.copy2(str(stat_fp), str(org_stat_fp))

    # --------- Determine new id start from CURRENT data/ max + 1 ---------
    data_dir = hoi4d_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    hoi4d_fold_idx_max = get_current_max_fold_idx(data_dir)
    st_soft_idx = hoi4d_fold_idx_max + 1

    # --------- Load Soft metadata ---------
    soft_meta_fp = egosoft_dir / "metadata_egosoft_demo.json"
    if not soft_meta_fp.exists():
        raise FileNotFoundError(f"Soft metadata not found: {soft_meta_fp}")
    soft_metadata = load_json(soft_meta_fp)

    # --------- Merge ---------
    metadata = copy.deepcopy(hoi4d_metadata)
    metadata_stat = copy.deepcopy(hoi4d_metadata_stat)

    # Existing split_keys to avoid duplicates
    existing_keys = build_existing_keys(metadata)

    stat_soft_data = {"train": 0, "val": 0, "test": 0}
    stat_soft_skipped_dup = 0

    # To avoid collisions even if something weird happens, keep incrementing st_soft_idx
    next_new_id_base = st_soft_idx

    for sclip in soft_metadata:
        # sclip fields assumed: id, index, action, object, st
        sclip_id = int(sclip["id"])
        split_key = f'{sclip["index"]}|{sclip["action"]}|{sclip["object"]}'

        if (not args.allow_duplicate) and (split_key in existing_keys):
            stat_soft_skipped_dup += 1
            continue

        # Pick split deterministically
        split = stable_split(split_key, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

        # new_id: use monotonic base + sclip_id, but ensure final folder doesn't exist
        # This guarantees: never overwrite.
        while True:
            new_id = sclip_id + next_new_id_base
            target_fp = data_dir / str(new_id)
            if not target_fp.exists():
                break
            # If exists, push base forward to avoid clash
            next_new_id_base += 1

        clip = {
            "id": new_id,
            "index": sclip["index"],
            "action": sclip["action"],
            "object": sclip["object"],
            "img": sclip["st"],
            "kpst_part": ["body"],
            # "split_key": split_key,      # ✅ persist key for future dedup
            # "source": str(egosoft_dir),  # (optional) provenance
        }

        metadata[split].append(clip)
        existing_keys.add(split_key)

        source_fp = egosoft_dir / "data" / str(sclip_id)
        copy_dir_contents(source_fp, target_fp)

        stat_soft_data[split] += 1

    # Update stat
    if "soft_data_stat" not in metadata_stat or not isinstance(metadata_stat.get("soft_data_stat"), dict):
        metadata_stat["soft_data_stat"] = {"train": 0, "val": 0, "test": 0}

    # Accumulate counts (so multiple runs add up)
    for k in ["train", "val", "test"]:
        metadata_stat["soft_data_stat"][k] = int(metadata_stat["soft_data_stat"].get(k, 0)) + int(stat_soft_data[k])

    metadata_stat["soft_data_skipped_dup"] = int(metadata_stat.get("soft_data_skipped_dup", 0)) + int(stat_soft_skipped_dup)

    # Write merged outputs
    save_json(metadata, meta_fp)
    save_json(metadata_stat, stat_fp)

    print("[Done] Soft merged into HOI4D (append mode).")
    print("  HOI4D dir:", hoi4d_dir)
    print("  Soft dir :", egosoft_dir)
    print("  This run added:", stat_soft_data, f"(skipped_dup={stat_soft_skipped_dup})")
    print("  Current data max idx:", get_current_max_fold_idx(data_dir))
    print("  Metadata saved to:")
    print("   -", meta_fp)
    print("   -", stat_fp)
    if org_meta_fp.exists():
        print("  Backup exists:")
        print("   -", org_meta_fp)
        print("   -", org_stat_fp)


if __name__ == "__main__":
    main()
