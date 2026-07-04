#!/usr/bin/env python3
"""Re-run step1_2d for one clip: manual SAM2 masks + CoTracker (for GT refresh)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_RVIDEO = Path(__file__).resolve().parent
_ROOT = _RVIDEO.parent
for p in (_ROOT, _RVIDEO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from label_gen import (  # noqa: E402
    EgoHOIAnalysts,
    get_mark_image,
    get_video,
    parse_task_description,
    resolve_dino_other_prompt,
    sample_kps_two_masks,
    segment_other_mask,
)


def load_clip_meta(save_root: str, clip_id: int) -> dict:
    for name in ("step1_kpst_2d_info.json", "metadata_egosoft_demo.json"):
        fp = os.path.join(save_root, name)
        if not os.path.isfile(fp):
            continue
        with open(fp) as f:
            clips = json.load(f)
        for c in clips:
            if int(c["id"]) == int(clip_id):
                return c
    raise FileNotFoundError(f"clip id={clip_id} not found under {save_root}")


def main():
    ap = argparse.ArgumentParser("Manual mask + CoTracker for one GT clip (step1_2d only)")
    ap.add_argument("--save_root", required=True, help="task root, e.g. process_data/PRESS/press_the-bottle")
    ap.add_argument("--clip_id", type=int, required=True, help="clip id, e.g. 23")
    ap.add_argument(
        "--mode",
        choices=["both", "tool", "other", "review"],
        default="both",
        help="both=manual hand+object; tool/other=manual one side + DINO the other; review=DINO then y/m",
    )
    ap.add_argument("--n_sample_max", type=int, default=1024)
    ap.add_argument("--tool_point_ratio", type=float, default=0.5)
    ap.add_argument("--tool_prompt", type=str, default=None, help="override tool entity name (default from task)")
    ap.add_argument("--dino_other_prompt", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    clip = load_clip_meta(args.save_root, args.clip_id)
    base_fp = clip["index"]
    st, ed = int(clip["st"]), int(clip["ed"])
    task = clip.get("task_description", "")
    action, tool_name, other_name = parse_task_description(task)
    if clip.get("tool") is not None:
        tool_name = clip["tool"]
    if clip.get("other") is not None:
        other_name = clip["other"]
    tool_prompt = (args.tool_prompt or tool_name or "hand").replace("_", " ")
    other_prompt = resolve_dino_other_prompt(args, task, action, other_name or "object")

    save_fp = os.path.join(args.save_root, "step1_2d", str(args.clip_id))
    os.makedirs(save_fp, exist_ok=True)

    print(f"[Clip {args.clip_id}] {task} | {base_fp} | frames [{st}, {ed}]")
    print(f"[Seg] tool={tool_prompt!r} other={other_prompt!r} mode={args.mode}")

    np.random.seed(int(args.seed))
    video_rgb = get_video(base_fp, "rgb")
    rgb_video_clip = video_rgb[st : ed + 1]
    rgb_image = rgb_video_clip[0]
    rgb0_path = os.path.join(base_fp, f"rgb_{st}.png")

    analysts = EgoHOIAnalysts(device=args.device)

    class SegArgs:
        dino_other_prompt = args.dino_other_prompt
        manual_seg_fallback = args.mode == "review"
        manual_other_only = args.mode == "other"
        lid_mask_top_ratio = 0.0

    seg_args = SegArgs()

    if args.mode == "both":
        print("[Manual] TOOL then OTHER (close each window when done; q or X)")
        mask_tool, mask_other = analysts.segment_hoi_pair_manual(
            rgb_image, tool_prompt, other_prompt, vis_dir=save_fp,
        )
        if mask_tool is None or mask_other is None:
            raise RuntimeError("manual segmentation failed")
        m_tool = mask_tool[0].astype(np.uint8)
        m_other = mask_other[0].astype(np.uint8)
    elif args.mode == "tool":
        print("[Manual] TOOL only; OTHER via DINO")
        mask_tool = analysts.segment_entity_manual(
            rgb_image, tool_prompt, role="tool", vis_dir=save_fp,
        )
        mask_other = segment_other_mask(
            analysts, rgb_image, other_prompt, rgb0_path, save_fp, seg_args,
            action=action, other_name=other_name,
        )
        if mask_tool is None or mask_other is None:
            raise RuntimeError("segmentation failed")
        m_tool = mask_tool[0].astype(np.uint8)
        m_other = mask_other[0].astype(np.uint8)
    elif args.mode == "other":
        print("[Manual] OTHER only; TOOL via DINO")
        mask_other = analysts.segment_entity_manual(
            rgb_image, other_prompt, role="other", vis_dir=save_fp,
        )
        mask_tool, _ = analysts.segment_hoi_pair(
            rgb_image, tool_name=tool_prompt, object_name=other_prompt,
            image_path=rgb0_path, vis_dir=save_fp,
        )
        if mask_tool is None or mask_other is None:
            raise RuntimeError("segmentation failed")
        m_tool = mask_tool[0].astype(np.uint8)
        m_other = mask_other[0].astype(np.uint8)
    else:
        print("[Review] DINO + y=accept / m=manual")
        mask_tool, mask_other = analysts.segment_hoi_pair(
            rgb_image, tool_name=tool_prompt, object_name=other_prompt,
            image_path=rgb0_path, vis_dir=save_fp,
        )
        if mask_tool is None and seg_args.manual_seg_fallback:
            mask_tool = analysts.segment_entity_manual(
                rgb_image, tool_prompt, role="tool", vis_dir=save_fp,
            )
        mask_other = segment_other_mask(
            analysts, rgb_image, other_prompt, rgb0_path, save_fp, seg_args,
            action=action, other_name=other_name,
        )
        if mask_tool is None or mask_other is None:
            raise RuntimeError("segmentation failed")
        m_tool = mask_tool[0].astype(np.uint8)
        m_other = mask_other[0].astype(np.uint8)

    if tool_name == "hand":
        m_hand = m_tool.copy()
    else:
        m_hand = np.zeros_like(m_tool, dtype=np.uint8)

    kps = sample_kps_two_masks(
        m_tool, m_other,
        n_total=int(args.n_sample_max),
        tool_ratio=float(args.tool_point_ratio),
    )
    if kps.shape[0] == 0:
        raise RuntimeError("no query points sampled from masks")

    print(f"[CoTracker] tracking {kps.shape[0]} queries over {rgb_video_clip.shape[0]} frames ...")
    pred_tracks, pred_visibility = analysts.get_kpst_track(
        rgb_video_clip, kps, vis_dir=save_fp,
    )
    mark_image = get_mark_image(rgb_image, kps)

    cv2.imwrite(os.path.join(save_fp, "mark.jpg"), cv2.cvtColor(mark_image, cv2.COLOR_RGB2BGR))
    np.save(os.path.join(save_fp, "mask_tool.npy"), m_tool)
    np.save(os.path.join(save_fp, "mask_other.npy"), m_other)
    np.save(os.path.join(save_fp, "mask_hand.npy"), m_hand)
    np.save(os.path.join(save_fp, "kps_tracks.npy"), pred_tracks)
    np.save(os.path.join(save_fp, "kps_visibility.npy"), pred_visibility)

    print(f"[OK] saved step1_2d -> {save_fp}")
    print("       mark.jpg, mask_*.npy, kps_tracks.npy, queries_pred_track.mp4")
    print("\nNext (3D GT refresh for this clip):")
    print(f"  cd {_RVIDEO}")
    print("  python label_gen.py \\")
    print("    --only_3d --recompute_3d \\")
    print(f"    --save_root {args.save_root} \\")
    print('    --record_root "$RECORD_ROOT" \\')
    print('    --recorded_rgbd_root "$RECORDED_RGBD_ROOT"')


if __name__ == "__main__":
    main()
