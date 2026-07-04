from __future__ import annotations

from __future__ import annotations

import os
import torch
import pdb
import numpy as np 
import pandas as pd
from tqdm import tqdm

from PIL import Image
import cv2
import json
import time
import open3d as o3d
import argparse
import scipy.ndimage
from base64 import b64encode

from plyfile import PlyData
import sys
from pathlib import Path

_RVIDEO = Path(__file__).resolve().parent
_ROOT = _RVIDEO.parent
for p in (_ROOT, _RVIDEO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from repo_paths import get_path, load_paths

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import traceback
import matplotlib.pyplot as plt

# FPS = 15

import numpy as np
import cv2

from utils import (
    remove_fly_points_xyzrgb,
    filter_bg_remove_islands,
    filter_kpst_long_trajectories,
    apply_global_se3,
)


def get_camera(camera_in, W, H):
    """Build Open3D intrinsic that matches the actual image resolution (W,H)."""
    camera = o3d.camera.PinholeCameraIntrinsic()
    camera.set_intrinsics(
        int(W), int(H),
        float(camera_in[0, 0]), float(camera_in[1, 1]),
        float(camera_in[0, 2]), float(camera_in[1, 2]),
    )
    return camera



def save_video(video_save_fp, video):
    video_np = video.squeeze(0).permute(0, 2, 3, 1).numpy()
    # pdb.set_trace()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_save_fp, fourcc, 30, (1920, 1080))
    for frame in video_np:
        frame_bgr = cv2.cvtColor(np.uint8(frame), cv2.COLOR_RGB2BGR) 
        video_writer.write(frame_bgr)
    video_writer.release()


class EgoHOIAnalysts(object):

    def __init__(self,
                 fastsam_fp=None,
                 tracker_pth=None,
                 downsample_ratio=1,
                 device='cuda',
                 use_dino=True,
                 dino_cfg=None,
                 dino_ckpt=None,
                 dino_box_thresh=0.30,
                 dino_text_thresh=0.30,
                 ego_hoi_det_cfg=None,
                 ego_hoi_det_pth=None,
                 sam2_checkpoint=None,
                 sam2_config=None,
                 ):

        from cotracker.predictor import CoTrackerPredictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.build_sam import build_sam2
        from dino_hoi_detector import GroundingDINODetector
        from ego_hoi_detector import EgoHOIDetector

        cfg = load_paths()
        self.device = device
        self.fastsam_fp = fastsam_fp or get_path("fastsam_fp", "", cfg)
        self.downsample_ratio = downsample_ratio

        sam2_checkpoint = sam2_checkpoint or get_path("sam2_checkpoint", "", cfg)
        model_cfg = sam2_config or get_path("sam2_config", "configs/sam2.1/sam2.1_hiera_l.yaml", cfg)
        tracker_pth = tracker_pth or get_path("cotracker_ckpt", "", cfg)
        dino_cfg = dino_cfg or get_path("groundingdino_config", "", cfg)
        dino_ckpt = dino_ckpt or get_path("groundingdino_ckpt", "", cfg)
        ego_hoi_det_cfg = ego_hoi_det_cfg or get_path("hand_detector_cfg", "", cfg)
        ego_hoi_det_pth = ego_hoi_det_pth or get_path("hand_detector_ckpt", "", cfg)

        sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
        self.sam2_image_predictor = SAM2ImagePredictor(sam2_image_model)


        # --- segmentation (FastSAM) ---
        # self.sam_model = FastSAM(self.fastsam_fp)
        # self.sam_prompt_model = FastSAMPrompt(device=device)

        # --- HOI detection (bbox) ---
        self.use_dino = use_dino
        self.dino_box_thresh = float(dino_box_thresh)
        self.dino_text_thresh = float(dino_text_thresh)

        if self.use_dino:
            self.dino_det = GroundingDINODetector(
                config_path=dino_cfg,
                checkpoint_path=dino_ckpt,
                device=device,
                box_threshold=self.dino_box_thresh,
                text_threshold=self.dino_text_thresh,
            )
            self.hoi_det_model = None
        else:
            # legacy FasterRCNN detector
            self.ego_hoi_det_cfg = ego_hoi_det_cfg
            self.ego_hoi_det_pth = ego_hoi_det_pth
            self.hoi_det_model = EgoHOIDetector(cfg_file=self.ego_hoi_det_cfg,
                                                pretrained_path=self.ego_hoi_det_pth)
            self.dino_det = None

        kps_tracker = CoTrackerPredictor(checkpoint=tracker_pth)
        self.kps_tracker = kps_tracker.to(device)
        

    def _sam2_mask_from_xyxy(self, image_rgb, box_xyxy):
        """SAM2 box-prompt segmentation. box_xyxy: [x1,y1,x2,y2] in pixels."""
        if image_rgb.dtype != np.uint8:
            img_u8 = image_rgb.astype(np.uint8)
        else:
            img_u8 = image_rgb
        img_u8 = np.ascontiguousarray(img_u8)

        self.sam2_image_predictor.set_image(img_u8)

        box = np.array(box_xyxy, dtype=np.float32)
        box_in = box[None, :] if box.ndim == 1 else box

        masks, scores, logits, *_ = self.sam2_image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_in,
            multimask_output=False,
        )

        masks = np.array(masks)
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        if masks.ndim == 3:
            m = masks[0]
        elif masks.ndim == 2:
            m = masks
        else:
            return None

        m = (m > 0).astype(np.uint8)
        if m.sum() == 0:
            return None
        return m[None]

    def _sam2_mask_small_part_from_box(self, image_rgb, box_xyxy):
        """For handle/lid-sized parts: prefer a tight SAM mask via center point."""
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        pts = np.array([[cx, cy]], dtype=np.float32)
        lbl = np.ones((1,), dtype=np.int32)

        img = np.ascontiguousarray(image_rgb.astype(np.uint8))
        self.sam2_image_predictor.set_image(img)
        masks, scores, logits, *_ = self.sam2_image_predictor.predict(
            point_coords=pts,
            point_labels=lbl,
            box=np.array(box_xyxy, dtype=np.float32)[None, :],
            multimask_output=True,
        )
        masks = np.array(masks)
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        if masks.ndim != 3 or masks.shape[0] == 0:
            return None

        H, W = img.shape[:2]
        best = None
        best_area = None
        for mi in range(masks.shape[0]):
            m = (masks[mi] > 0).astype(np.uint8)
            area = int(m.sum())
            if area == 0:
                continue
            if not m[int(np.clip(cy, 0, H - 1)), int(np.clip(cx, 0, W - 1))]:
                continue
            if best is None or area < best_area:
                best, best_area = m, area
        if best is None:
            return None
        return best[None]

    def _keep_mask_cc_at_center(self, mask, cx: float, cy: float):
        """Keep the connected component that covers (cx, cy), or the nearest one."""
        m = mask[0] if mask.ndim == 3 else mask
        m = (m > 0).astype(np.uint8)
        if m.sum() == 0:
            return mask

        n_cc, labels = cv2.connectedComponents(m)
        if n_cc <= 2:
            out = m
        else:
            ix = int(np.clip(round(cx), 0, m.shape[1] - 1))
            iy = int(np.clip(round(cy), 0, m.shape[0] - 1))
            lbl = labels[iy, ix]
            if lbl == 0:
                best_lbl, best_dist = 0, None
                for lab in range(1, n_cc):
                    ys, xs = np.where(labels == lab)
                    if ys.size == 0:
                        continue
                    d = (ys.mean() - iy) ** 2 + (xs.mean() - ix) ** 2
                    if best_dist is None or d < best_dist:
                        best_dist, best_lbl = d, lab
                lbl = best_lbl
            out = (labels == lbl).astype(np.uint8) if lbl > 0 else m

        if mask.ndim == 3:
            return out[None]
        return out[None]

    def _is_small_part_entity(self, entity_name: str) -> bool:
        name = entity_name.lower()
        return any(k in name for k in ("handle", "knob", "lid", "cover"))

    def _mask_covers_box_center(self, mask, box_xyxy) -> bool:
        m = mask[0] if mask.ndim == 3 else mask
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        cx = int(np.clip(round(0.5 * (x1 + x2)), 0, m.shape[1] - 1))
        cy = int(np.clip(round(0.5 * (y1 + y2)), 0, m.shape[0] - 1))
        return bool(m[cy, cx] > 0)

    def _segment_small_part_mask_from_box(self, image, box_xyxy, entity_name: str):
        """Point-first SAM for handle/lid; keep CC at box center."""
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

        mask = self._sam2_mask_small_part_from_box(image, box_xyxy)
        if mask is None:
            mask = self._sam2_mask_from_xyxy(image, box_xyxy)
        if mask is None:
            return None

        mask = self._keep_mask_cc_at_center(mask, cx, cy)
        H, W = image.shape[:2]
        if mask[0].sum() / float(H * W) > 0.05:
            tight = self._sam2_mask_small_part_from_box(image, box_xyxy)
            if tight is not None and tight[0].sum() < mask[0].sum():
                mask = self._keep_mask_cc_at_center(tight, cx, cy)
        return mask

    def _segment_entity_mask_from_box(self, image, box_xyxy, entity_name: str):
        """SAM2 mask from DINO box."""
        if self._is_small_part_entity(entity_name):
            return self._segment_small_part_mask_from_box(image, box_xyxy, entity_name)
        return self._sam2_mask_from_xyxy(image, box_xyxy)

    def _pick_best_small_part_mask(self, image, boxes, phrases, entity_name: str):
        """Try several DINO boxes; keep smallest valid mask covering box center."""
        H, W = image.shape[:2]
        best_mask, best_area, best_phrase = None, None, None

        for box, phrase in zip(boxes, phrases):
            try:
                mask = self._segment_small_part_mask_from_box(image, box, entity_name)
            except Exception:
                continue
            if mask is None or mask[0].sum() == 0:
                continue
            if not self._mask_covers_box_center(mask, box):
                continue
            area_frac = mask[0].sum() / float(H * W)
            if area_frac > 0.05:
                continue
            area = int(mask[0].sum())
            if best_area is None or area < best_area:
                best_mask, best_area, best_phrase = mask, area, phrase

        if best_mask is not None:
            print(f"[SAM2] picked mask area={best_area}px phrase={best_phrase!r}")
        return best_mask

    def segment_hoi_pair(self, image, tool_name: str, object_name: str, image_path=None, vis_dir=None):
        """Segment a *pair* of interacting entities (tool, object) using:
        - GroundingDINO for bbox (tool + object)
        - SAM2 (SAM2ImagePredictor) for mask from bbox prompt

        Returns:
            mask_tool: (1,H,W) uint8 {0,1} or None
            mask_obj:  (1,H,W) uint8 {0,1} or None
        """
        start_time = time.time()
        vis = vis_dir is not None 

        H0, W0 = image.shape[:2]

        # --- downsample image only for legacy FasterRCNN branch / (optional) speed ---
        # (SAM2 uses ORIGINAL image; DINO boxes are in ORIGINAL coords; do NOT rescale for SAM2)
        img_pil = Image.fromarray(image)
        new_height = img_pil.height // self.downsample_ratio
        new_width = img_pil.width // self.downsample_ratio
        resized_img_pil = img_pil.resize((new_width, new_height))
        resized_img = np.asarray(resized_img_pil)
        Hr, Wr = resized_img.shape[:2]

        bboxes_tool_xyxy = []
        bboxes_obj_xyxy = []

        # --- 1) bbox detection ---
        if self.use_dino:
            if tool_name is None or object_name is None:
                raise ValueError("segment_hoi_pair: tool_name and object_name are required when use_dino=True")

            det = self.dino_det.detect(
                tool_name=tool_name,
                object_name=object_name,
                image_rgb=image,          # fallback if image_path is None
                image_path=image_path,
                max_tools=1,
            )

            tool_boxes = det.get("tool_boxes_xyxy", [])
            obj_boxes = det.get("obj_boxes_xyxy", [])

            # boxes already in ORIGINAL image XYXY
            if len(tool_boxes) > 0:
                bboxes_tool_xyxy = [tool_boxes[0]]
            if len(obj_boxes) > 0:
                bboxes_obj_xyxy = [obj_boxes[0]]

        else:
            # legacy FasterRCNN (expects resized image). It returns boxes in resized coords.
            obj_det, hand_det = self.hoi_det_model.detect(resized_img, vis=vis)  # <x1, y1, x2, y2>
            if hand_det is not None and hand_det.shape[0] >= 1 and hand_det[0, 4] > 0.5:
                # scale boxes from resized -> ORIGINAL
                sx = float(W0) / float(Wr)
                sy = float(H0) / float(Hr)

                def _scale_up_xyxy(bb):
                    x1, y1, x2, y2 = [float(v) for v in bb]
                    x1 *= sx; x2 *= sx
                    y1 *= sy; y2 *= sy
                    x1 = max(0.0, min(x1, W0 - 1.0))
                    x2 = max(0.0, min(x2, W0 - 1.0))
                    y1 = max(0.0, min(y1, H0 - 1.0))
                    y2 = max(0.0, min(y2, H0 - 1.0))
                    if x2 <= x1 + 1 or y2 <= y1 + 1:
                        return None
                    return [x1, y1, x2, y2]

                ob = _scale_up_xyxy(obj_det[0, :4].tolist())
                tb = _scale_up_xyxy(hand_det[0, :4].tolist())
                if ob is not None and tb is not None:
                    bboxes_obj_xyxy = [ob]
                    bboxes_tool_xyxy = [tb]

        mask_tool = None
        mask_obj = None

        if bboxes_tool_xyxy and len(bboxes_tool_xyxy) > 0:
            try:
                mask_tool = self._sam2_mask_from_xyxy(image, bboxes_tool_xyxy[0])
            except Exception as e:
                print(f"[SAM2] tool mask failed ({tool_name}):", repr(e))
                traceback.print_exc()
        else:
            print(f"[DINO] tool bbox missing: {tool_name}")

        if bboxes_obj_xyxy and len(bboxes_obj_xyxy) > 0:
            try:
                mask_obj = self._sam2_mask_from_xyxy(image, bboxes_obj_xyxy[0])
            except Exception as e:
                print(f"[SAM2] object mask failed ({object_name}):", repr(e))
                traceback.print_exc()
        else:
            print(f"[DINO] object bbox missing: {object_name}")

        if mask_tool is None and mask_obj is None:
            return None, None

        if vis:
            os.makedirs(vis_dir, exist_ok=True)
            if mask_tool is not None:
                cv2.imwrite(os.path.join(vis_dir, "mask_tool.png"), (mask_tool[0] * 255).astype(np.uint8))
            if mask_obj is not None:
                cv2.imwrite(os.path.join(vis_dir, "mask_obj.png"), (mask_obj[0] * 255).astype(np.uint8))

        return mask_tool, mask_obj



    @staticmethod
    def display_and_capture_points(image, title="pick_points"):
        """OpenCV point picker: left-click to add FG points, close window when done."""
        points = []
        image_u8 = np.ascontiguousarray(np.uint8(image))
        canvas = image_u8.copy()
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        elif canvas.shape[2] == 4:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_RGBA2BGR)
        else:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

        win_name = title if isinstance(title, str) and len(title) > 0 else "pick_points"
        hint = f"{win_name} | LMB=FG, close window or q=done"

        def _on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((float(x), float(y)))
                cv2.circle(canvas, (x, y), 4, (0, 0, 255), -1)
                cv2.imshow(win_name, canvas)
                print(f"Point: ({x}, {y})")

        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win_name, _on_mouse)
        cv2.imshow(win_name, canvas)
        print(hint)

        while True:
            try:
                key = cv2.waitKey(20) & 0xFF
            except cv2.error:
                break
            if key in (27, ord("q")):
                break
            try:
                visible = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE)
            except cv2.error:
                break
            if visible < 1:
                break

        try:
            cv2.destroyWindow(win_name)
        except cv2.error:
            pass
        try:
            cv2.waitKey(1)
        except cv2.error:
            pass
        if len(points) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return np.asarray(points, dtype=np.float32)

    def _sam2_predict_from_points(self, image_rgb, points_xy, labels=None):
        """SAM2 point-prompt segmentation. points_xy: (N,2) in pixel (x,y)."""
        if points_xy is None or len(points_xy) == 0:
            return None

        pts = np.asarray(points_xy, dtype=np.float32)
        if labels is None:
            lbl = np.ones((pts.shape[0],), dtype=np.int32)
        else:
            lbl = np.asarray(labels, dtype=np.int32)

        img = image_rgb
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        img = np.ascontiguousarray(img)

        self.sam2_image_predictor.set_image(img)
        masks, scores, logits, *_ = self.sam2_image_predictor.predict(
            point_coords=pts,
            point_labels=lbl,
            multimask_output=False,
        )

        masks = np.array(masks)
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        if masks.ndim == 3:
            m = masks[0]
        elif masks.ndim == 2:
            m = masks
        else:
            return None

        m = (m > 0).astype(np.uint8)
        if m.sum() == 0:
            return None
        return m[None]

    def segment_entity_manual(self, image, entity_name: str, role: str = "tool", vis_dir=None):
        """Manual SAM2 segmentation for a single entity (tool or other)."""
        role_key = "tool" if role == "tool" else "other"
        role_label = "TOOL" if role_key == "tool" else "OTHER"
        print(f"[Manual Seg] click points for {role_label} ({entity_name}).")
        pts = self.display_and_capture_points(
            image,
            title=f"{role_label}: {entity_name} (click FG, close window)",
        )
        if pts.shape[0] == 0:
            print(f"[Manual Seg] No {role_key} points selected.")
            return None

        mask = self._sam2_predict_from_points(image, pts, labels=None)
        if mask is None:
            print(f"[Manual Seg] SAM2 failed for {role_key}.")
            return None

        if vis_dir is not None:
            os.makedirs(vis_dir, exist_ok=True)
            cv2.imwrite(
                os.path.join(vis_dir, f"mask_{role_key}_manual.png"),
                (mask[0] * 255).astype(np.uint8),
            )
            np.save(os.path.join(vis_dir, f"manual_{role_key}_points.npy"), pts)
        return mask

    def segment_entity_dino(self, image, entity_name: str, role: str = "other", image_path=None, vis_dir=None):
        """DINO bbox + SAM2 mask for a single entity."""
        role_key = "tool" if role == "tool" else "other"
        if not self.use_dino or self.dino_det is None:
            print(f"[DINO] disabled; skip auto {role_key} seg for {entity_name}.")
            return None

        det = self.dino_det.detect_entity_candidates(
            entity_name=entity_name,
            image_rgb=image,
            image_path=image_path,
            max_candidates=5,
        )
        boxes = det.get("boxes_xyxy", [])
        phrases = det.get("phrases", [])
        if not boxes:
            print(f"[DINO] {role_key} bbox missing: {entity_name}")
            return None

        try:
            if self._is_small_part_entity(entity_name) and len(boxes) > 1:
                mask = self._pick_best_small_part_mask(image, boxes, phrases, entity_name)
            else:
                mask = None
            if mask is None:
                mask = self._segment_entity_mask_from_box(image, boxes[0], entity_name)
        except Exception as e:
            print(f"[SAM2] {role_key} mask failed ({entity_name}):", repr(e))
            traceback.print_exc()
            return None

        if mask is None:
            print(f"[SAM2] empty {role_key} mask: {entity_name}")
            return None

        if vis_dir is not None:
            os.makedirs(vis_dir, exist_ok=True)
            cv2.imwrite(
                os.path.join(vis_dir, f"mask_{role_key}_dino.png"),
                (mask[0] * 255).astype(np.uint8),
            )
        print(f"[DINO] auto {role_key} seg ok: {entity_name}")
        return mask

    def review_dino_mask_or_manual(self, image, mask, entity_name: str, role: str = "other", vis_dir=None):
        """Show DINO mask; accept with y/Enter, or press m to re-annotate manually."""
        role_key = "tool" if role == "tool" else "other"
        role_label = "TOOL" if role_key == "tool" else "OTHER"
        m = mask[0] if mask.ndim == 3 else mask

        overlay = np.ascontiguousarray(np.uint8(image)).copy()
        if overlay.ndim == 3 and overlay.shape[2] == 3:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        colored = np.zeros_like(overlay)
        colored[:, :, 1] = (m > 0).astype(np.uint8) * 180
        overlay = cv2.addWeighted(overlay, 0.65, colored, 0.35, 0)

        win_name = f"Review {role_label}: {entity_name}"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, overlay)
        print(f"[Review] {entity_name} | y/Enter=accept DINO, m=manual re-pick, Esc=reject")

        accepted = False
        while True:
            try:
                key = cv2.waitKey(0) & 0xFF
            except cv2.error:
                break
            if key in (27,):
                break
            if key in (13, ord("y"), ord("Y")):
                accepted = True
                break
            if key in (ord("m"), ord("M")):
                break

        try:
            cv2.destroyWindow(win_name)
        except cv2.error:
            pass
        try:
            cv2.waitKey(1)
        except cv2.error:
            pass

        if accepted:
            return mask
        print(f"[Review] DINO {role_key} rejected -> manual for {entity_name}.")
        return self.segment_entity_manual(image, entity_name, role=role, vis_dir=vis_dir)

    def segment_entity_with_fallback(
        self,
        image,
        entity_name: str,
        role: str = "other",
        image_path=None,
        vis_dir=None,
        manual_on_fail: bool = True,
        review_on_success: bool = False,
        manual_only: bool = False,
    ):
        """Try DINO auto seg; on fail or bad result -> manual point picking."""
        if manual_only:
            return self.segment_entity_manual(
                image, entity_name, role=role, vis_dir=vis_dir,
            )

        mask = self.segment_entity_dino(
            image, entity_name, role=role, image_path=image_path, vis_dir=vis_dir,
        )
        if mask is not None:
            if review_on_success:
                return self.review_dino_mask_or_manual(
                    image, mask, entity_name, role=role, vis_dir=vis_dir,
                )
            return mask

        if manual_on_fail:
            return self.segment_entity_manual(image, entity_name, role=role, vis_dir=vis_dir)
        return None

    def segment_hoi_pair_manual(self, image, tool_name: str, object_name: str, vis_dir=None):
        """Interactive SAM2 segmentation when DINO fails on both tool and other."""
        mask_tool = self.segment_entity_manual(image, tool_name, role="tool", vis_dir=vis_dir)
        mask_obj = self.segment_entity_manual(image, object_name, role="other", vis_dir=vis_dir)
        if mask_tool is None or mask_obj is None:
            return None, None

        if vis_dir is not None:
            tool_pts = np.load(os.path.join(vis_dir, "manual_tool_points.npy"))
            obj_pts = np.load(os.path.join(vis_dir, "manual_other_points.npy"))
            overlay = np.ascontiguousarray(np.uint8(image)).copy()
            if overlay.ndim == 3 and overlay.shape[2] == 3:
                overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            for x, y in tool_pts:
                cv2.circle(overlay, (int(x), int(y)), 4, (0, 0, 255), -1)
            for x, y in obj_pts:
                cv2.circle(overlay, (int(x), int(y)), 4, (0, 255, 0), -1)
            cv2.imwrite(os.path.join(vis_dir, "manual_points.jpg"), overlay)

        return mask_tool, mask_obj


    # --- backward-compatible wrapper ---
    def segment_hoi_object(self, image, object_name=None, image_path=None, vis_dir=None):
        """Deprecated: kept for older callers.

        Old behavior was prompting "hand . {object}" and returning (mask_obj, mask_hand).
        We now call segment_hoi_pair(tool_name='hand', object_name=object_name) and
        return (mask_obj, mask_hand==mask_tool).
        """
        if object_name is None:
            raise ValueError("segment_hoi_object: object_name is required")
        mask_tool, mask_obj = self.segment_hoi_pair(
            image=image,
            tool_name="hand",
            object_name=object_name,
            image_path=image_path,
            vis_dir=vis_dir,
        )
        return mask_obj, mask_tool

    def get_kps(self, mask, n_sample_max=1024):
        if mask.ndim == 2:
            mask = mask[None]
        H, W = mask.shape[1], mask.shape[2]  
        mask_flattened = mask.reshape(-1) 
        ones_indices = np.where(mask_flattened == 1)[0]
        if len(ones_indices) > n_sample_max:
            selected_indices = np.random.choice(ones_indices, size=n_sample_max, replace=False)
        else:
            selected_indices = ones_indices
        selected_2d_indices = np.array([np.unravel_index(idx, (H, W)) for idx in selected_indices])
        return selected_2d_indices
    
    def get_kpst_track(self, video, kps_2d, vis_dir=None):
        # kps_2d: (h, w) --> (w, h) format
        query = np.concatenate([np.zeros((kps_2d.shape[0], 1)), kps_2d[:, 1:2], kps_2d[:, 0:1]], axis=1)
        query = torch.tensor(query).cuda()

        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()  # torch.Size([1, 113, 3, 1080, 1920])
        video = video.to(self.device)
        pred_tracks, pred_visibility = self.kps_tracker(video.float(), queries=query[None].float())

        if vis_dir is not None:
            from cotracker.utils.visualizer import Visualizer
            vis = Visualizer(save_dir=vis_dir, linewidth=3, mode='cool', tracks_leave_trace=-1)
            vis.visualize(video=video, tracks=pred_tracks, visibility=pred_visibility, filename='queries')
        
        pred_tracks = pred_tracks[0].cpu().numpy()
        pred_visibility = pred_visibility[0].cpu().numpy()   # (w, h) - format
        return pred_tracks, pred_visibility

# HEIGHT = 720
# WIDTH = 1280


def get_video(base_fp, prefix='none'):
    n_file = count_rgb_frames(base_fp, prefix=prefix)
    if n_file <= 0:
        raise RuntimeError(f"No {prefix}_*.png frames under: {base_fp}")
    video = []
    for fl_id in range(n_file):
        image = Image.open(os.path.join(base_fp, f'{prefix}_{fl_id}.png'))
        image = np.asarray(image)
        video.append(image)
    video = np.stack(video, axis=0)          # (T, H, W, C)
    return video


def get_mark_image(image, mark_coordinates):
    im = image.copy()
    for x, y in mark_coordinates:
        cv2.circle(im, (y, x), radius=5, color=(0, 255, 0), thickness=-1)
    return im


# =========================
# NEW helpers: task parsing + balanced sampling + simple augmentations
# =========================
def parse_task_description(task_desc: str):
    """
    Examples:
      pickup_cup -> action=pickup, tool=hand, other=cup
      hangon_cup_to_branch -> action=hangon, tool=cup, other=branch
      pull_out_the_drawer -> action=pull, tool=hand, other=drawer
      lift_lid -> action=lift, tool=hand, other=lid
    """
    if task_desc in TASK_SEMANTIC_OVERRIDES:
        return TASK_SEMANTIC_OVERRIDES[task_desc]

    parts = task_desc.split('_')
    action = parts[0]
    tool = None
    other = None
    if "to" in parts:
        to_idx = parts.index("to")
        if to_idx >= 2:
            tool = parts[1]
            other = parts[to_idx + 1] if (to_idx + 1) < len(parts) else None
    elif "_the_" in task_desc:
        action, rest = task_desc.split("_the_", 1)
        action = action.split("_")[0]
        tool = "hand"
        other = rest.replace("-", "_")
    else:
        tool = "hand"
        other = "_".join(parts[1:]) if len(parts) > 1 else None
    return action, tool, other


TASK_SEMANTIC_OVERRIDES = {
    "lift_lid": ("lift", "hand", "lid"),
    "pickup_cover": ("pickup", "hand", "cover"),
    "pull_drawer": ("pull", "hand", "drawer"),
    "pull_out_the_drawer": ("pull", "hand", "drawer"),
    "pull_black_drawer": ("pull", "hand", "drawer"),
    "push_drawer": ("push", "hand", "drawer"),
    "open_drawer": ("open", "hand", "drawer"),
    "drag_drawer": ("drag", "hand", "drawer"),
}

# Task-specific DINO text prompts (avoid ambiguous names like "lid" -> whole cup)
DINO_OTHER_PROMPT_OVERRIDES = {
    "lift_lid": "white cup lid",
    "pickup_cover": "blue cover",
    "pull_drawer": "metal pull handle on drawer",
    "pull_out_the_drawer": "metal pull handle on drawer",
    "pull_black_drawer": "black drawer pull handle",
    "push_drawer": "drawer front panel handle",
    "open_drawer": "metal pull handle on drawer",
    "drag_drawer": "metal pull handle on drawer",
}
DINO_OTHER_PROMPT_BY_ACTION_OTHER = {
    ("lift", "lid"): "white cup lid",
    ("pickup", "cover"): "blue cover",
    ("pull", "drawer"): "metal pull handle on drawer",
    ("push", "drawer"): "drawer front panel handle",
    ("open", "drawer"): "metal pull handle on drawer",
    ("drag", "drawer"): "metal pull handle on drawer",
}


def list_trajectory_dirs(task_dir: str) -> list[str]:
    """Prefer traj_XXX folders; otherwise any subdir that already has rgb_0.png."""
    entries = [e for e in os.listdir(task_dir) if not e.startswith(".")]
    traj_dirs = [
        e for e in entries
        if e.startswith("traj_") and os.path.isdir(os.path.join(task_dir, e))
    ]
    if traj_dirs:
        def _traj_key(name: str):
            suf = name.split("_")[-1]
            return int(suf) if suf.isdigit() else 10 ** 9
        return sorted(traj_dirs, key=_traj_key)

    rgb_dirs = []
    for e in entries:
        p = os.path.join(task_dir, e)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "rgb_0.png")):
            rgb_dirs.append(e)
    return sorted(
        rgb_dirs,
        key=lambda x: (0, int(x)) if x.isdigit() else (1, x),
    )


def should_process_task(task_desc: str, args) -> bool:
    task_filter = getattr(args, "task_name", None)
    if task_filter and task_desc != task_filter:
        return False
    action, tool, other = parse_task_description(task_desc)
    if not action or not tool or not other:
        print(f"[Skip] cannot parse task: {task_desc!r}")
        return False
    return True


def load_existing_2d_clips(save_root_task: str, append_new: bool) -> tuple[list, set]:
    """Load prior step1_kpst_2d_info.json when appending new trajectories."""
    fp = os.path.join(save_root_task, "step1_kpst_2d_info.json")
    if not append_new or not os.path.isfile(fp):
        return [], set()
    with open(fp) as f:
        clips = json.load(f)
    done = {os.path.normpath(c["index"]) for c in clips}
    print(f"[Append] keep {len(clips)} existing 2D clips, skip {len(done)} traj paths")
    return clips, done


def load_existing_3d_clips(save_root_task: str, append_new: bool) -> dict:
    fp = os.path.join(save_root_task, "metadata_egosoft_demo.json")
    if not append_new or not os.path.isfile(fp):
        return {}
    with open(fp) as f:
        clips = json.load(f)
    return {c["id"]: c for c in clips}


STEP1_2D_REQUIRED = (
    "kps_tracks.npy",
    "kps_visibility.npy",
    "mask_tool.npy",
    "mask_other.npy",
)


def step1_2d_ready(step_fp: str) -> bool:
    return all(os.path.isfile(os.path.join(step_fp, fn)) for fn in STEP1_2D_REQUIRED)


def load_metadata_3d_clips(save_root_task: str) -> dict:
    fp = os.path.join(save_root_task, "metadata_egosoft_demo.json")
    if not os.path.isfile(fp):
        return {}
    with open(fp) as f:
        clips = json.load(f)
    return {c["id"]: c for c in clips}


def discover_gt_task_roots(root: str) -> list[str]:
    """Find GT task folders that already have 2D outputs (step1_kpst_2d_info.json)."""
    roots: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "step1_kpst_2d_info.json" not in filenames:
            continue
        if not os.path.isdir(os.path.join(dirpath, "step1_2d")):
            continue
        roots.append(dirpath)
    return sorted(roots)


DEPTH_BASE_FP_ALIASES = (
    ("/media/ljx/UBU/data/recorded_rgbd/", "/media/ljx/UBU/data/Record/"),
)

RECORD_ROOT_PREFIXES = (
    "/media/ljx/UBU/data/Record",
)


def _symlink_record_candidates(traj_dir: str, record_root: str | None) -> list[str]:
    """Remap broken staging symlinks via RECORD_ROOT (match path after .../Record/)."""
    if not record_root:
        return []
    cam = os.path.join(traj_dir, "camera_in.npy")
    if not os.path.islink(cam):
        return []
    target = os.readlink(cam)
    if not os.path.isabs(target):
        target = os.path.normpath(os.path.join(os.path.dirname(cam), target))
    target = os.path.normpath(target)
    record_root = os.path.normpath(record_root)
    marker = os.sep + "Record" + os.sep
    idx = target.find(marker)
    if idx >= 0:
        rel = target[idx + len(marker):]
        return [os.path.dirname(os.path.join(record_root, rel))]
    out: list[str] = []
    for prefix in ("/media/ljx/UBU/data/Record",):
        pnorm = os.path.normpath(prefix)
        pmarker = pnorm + os.sep
        if target.startswith(pmarker):
            rel = target[len(pmarker):]
            out.append(os.path.dirname(os.path.join(record_root, rel)))
            break
    return out


def _traj_root_ready(traj_dir: str) -> bool:
    cam = os.path.join(traj_dir, "camera_in.npy")
    return os.path.isfile(cam)


def _missing_raw_hint(base_fp: str) -> str:
    cam = os.path.join(base_fp, "camera_in.npy")
    if os.path.islink(cam) and not os.path.exists(cam):
        return f"{base_fp} (broken symlink -> {os.readlink(cam)}; mount disk or set RECORD_ROOT)"
    return base_fp


def resolve_traj_root(base_fp: str, record_root: str | None = None) -> str | None:
    """Return traj folder with camera_in.npy; prefer the path stored in clip JSON."""
    candidates: list[str] = []
    norm = os.path.normpath(base_fp)

    candidates.append(norm)
    for src, dst in DEPTH_BASE_FP_ALIASES:
        src_n = os.path.normpath(src)
        dst_n = os.path.normpath(dst)
        if norm.startswith(src_n):
            candidates.append(os.path.normpath(dst_n + norm[len(src_n):]))
        elif norm.startswith(dst_n):
            candidates.append(os.path.normpath(src_n + norm[len(dst_n):]))

    if record_root:
        record_root = os.path.normpath(record_root)
        for prefix in RECORD_ROOT_PREFIXES:
            pnorm = os.path.normpath(prefix)
            if norm == pnorm or norm.startswith(pnorm + os.sep):
                rel = norm[len(pnorm):].lstrip(os.sep)
                candidates.append(os.path.join(record_root, rel))
                break

    candidates.extend(_symlink_record_candidates(norm, record_root))

    seen: set[str] = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        if _traj_root_ready(cand):
            return cand
    return None


def depth_base_fp_candidates(base_fp: str) -> list[str]:
    out = [base_fp]
    norm = os.path.normpath(base_fp)
    for src, dst in DEPTH_BASE_FP_ALIASES:
        src_n = os.path.normpath(src)
        dst_n = os.path.normpath(dst)
        if norm.startswith(src_n):
            alt = os.path.normpath(dst_n + norm[len(src_n):])
        elif norm.startswith(dst_n):
            alt = os.path.normpath(src_n + norm[len(dst_n):])
        else:
            continue
        if alt not in out and os.path.isdir(alt):
            out.append(alt)
        break
    return out


def _load_depth_uint16_from_root(
    base_fp: str, frame_idx: int, use_refined_depth: bool = False,
) -> np.ndarray:
    idx = int(frame_idx)
    raw_png = os.path.join(base_fp, f"depth_{idx}.png")

    if use_refined_depth:
        refined_png = os.path.join(base_fp, f"depth_refined_{idx}.png")
        refined_npy = os.path.join(base_fp, f"depth_refined_{idx}.npy")
        if os.path.isfile(refined_png):
            dep = np.asarray(Image.open(refined_png))
        elif os.path.isfile(refined_npy):
            depth_m = np.load(refined_npy).astype(np.float32)
            dep = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
        elif os.path.isfile(raw_png):
            dep = np.asarray(Image.open(raw_png))
        else:
            raise FileNotFoundError(f"No depth for frame {idx} under {base_fp}")
    elif os.path.isfile(raw_png):
        dep = np.asarray(Image.open(raw_png))
    else:
        raise FileNotFoundError(f"depth_{idx}.png not found under {base_fp}")

    if dep.ndim == 3:
        dep = dep[..., 0]
    return dep.astype(np.uint16)


def load_depth_uint16(base_fp: str, frame_idx: int, use_refined_depth: bool = False) -> np.ndarray:
    """
    Load depth as uint16 millimeters for GT 3D back-projection / RGBD.
    Tries alias roots (recorded_rgbd <-> Record) when a frame is missing.
    """
    last_err: FileNotFoundError | None = None
    for cand in depth_base_fp_candidates(base_fp):
        try:
            return _load_depth_uint16_from_root(cand, frame_idx, use_refined_depth=use_refined_depth)
        except FileNotFoundError as exc:
            last_err = exc
            continue
    if last_err is not None:
        raise last_err
    raise FileNotFoundError(f"depth_{int(frame_idx)}.png not found under {base_fp}")


def refine_lid_mask(mask: np.ndarray, top_ratio: float = 0.28) -> np.ndarray:
    """
    When DINO/SAM segments the whole cup, keep only the top band of the mask bbox.
    top_ratio=0.28 -> keep upper 28%% of bbox height (cup lid region).
    """
    if top_ratio <= 0 or top_ratio >= 1.0:
        return (mask > 0).astype(np.uint8)

    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return m

    ys, _ = np.where(m > 0)
    y0, y1 = int(ys.min()), int(ys.max())
    h = y1 - y0 + 1
    cut_y = y0 + max(2, int(round(h * top_ratio)))

    out = np.zeros_like(m)
    out[y0:cut_y, :] = m[y0:cut_y, :]
    return out


def refine_mask_for_lift_lid(mask, args, action=None, other_name=None, vis_dir=None):
    """
    Optional post-process when DINO/SAM segments the whole cup instead of the lid.
    Off by default (lid_mask_top_ratio=0); mask_other.npy uses DINO result directly.
    """
    if mask is None or action != "lift" or other_name != "lid":
        return mask

    ratio = float(getattr(args, "lid_mask_top_ratio", 0.0))
    if ratio <= 0:
        return mask

    m2d = mask[0] if mask.ndim == 3 else mask
    refined = refine_lid_mask(m2d, top_ratio=ratio)
    if refined.sum() == 0:
        print("[lift_lid] lid refine emptied mask; keep original.")
        return mask

    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(vis_dir, "mask_other_lid_refined.png"),
            (refined * 255).astype(np.uint8),
        )
    print(f"[lift_lid] refined other mask to lid top_ratio={ratio:.2f}, "
          f"pixels {int((m2d > 0).sum())} -> {int(refined.sum())}")
    return refined[None].astype(np.uint8)


def resolve_dino_other_prompt(args, task_desc: str, action: str, other_name: str) -> str:
    """Resolve DINO caption for other/object segmentation."""
    if getattr(args, "dino_other_prompt", None):
        return str(args.dino_other_prompt).strip()
    if task_desc in DINO_OTHER_PROMPT_OVERRIDES:
        return DINO_OTHER_PROMPT_OVERRIDES[task_desc]
    key = (action, other_name)
    if key in DINO_OTHER_PROMPT_BY_ACTION_OTHER:
        return DINO_OTHER_PROMPT_BY_ACTION_OTHER[key]
    return other_name.replace("_", " ").replace("-", " ")


def segment_other_mask(analysts, image, other_prompt, image_path, vis_dir, args, action=None, other_name=None):
    """Segment other: manual-only, or DINO with optional review/fallback."""
    manual_other_only = bool(getattr(args, "manual_other_only", False))
    manual_review = bool(getattr(args, "manual_seg_fallback", False))
    if manual_other_only:
        print(f"[Manual Only] other ({other_prompt})")
        mask = analysts.segment_entity_with_fallback(
            image,
            other_prompt,
            role="other",
            image_path=image_path,
            vis_dir=vis_dir,
            manual_only=True,
        )
    else:
        print(f"[Seg] DINO auto other ({other_prompt}); manual if fail/bad.")
        mask = analysts.segment_entity_with_fallback(
            image,
            other_prompt,
            role="other",
            image_path=image_path,
            vis_dir=vis_dir,
            manual_on_fail=True,
            review_on_success=manual_review,
        )

    return refine_mask_for_lift_lid(
        mask, args, action=action, other_name=other_name, vis_dir=vis_dir,
    )


def sample_indices_from_mask(mask_2d: np.ndarray, n: int):
    """
    mask_2d: (H,W) bool/0-1
    return: (n,2) indices in (row, col) = (h, w)
    """
    mask = (mask_2d > 0)
    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.zeros((0,2), dtype=np.int64)
    if ys.size <= n:
        sel = np.arange(ys.size)
    else:
        sel = np.random.choice(ys.size, size=n, replace=False)
    return np.stack([ys[sel], xs[sel]], axis=1)


def count_rgb_frames(base_fp: str, prefix: str = "rgb") -> int:
    n = 0
    while os.path.isfile(os.path.join(base_fp, f"{prefix}_{n}.png")):
        n += 1
    return n


def sample_kps_two_masks(mask_tool: np.ndarray, mask_other: np.ndarray, n_total: int, tool_ratio: float = 0.7):
    """
    Return 2D keypoints in (h,w) format.
    """
    n_tool = int(round(n_total * tool_ratio))
    n_other = n_total - n_tool
    k_tool = sample_indices_from_mask(mask_tool, n_tool)
    k_other = sample_indices_from_mask(mask_other, n_other)
    if k_tool.shape[0] == 0 and k_other.shape[0] == 0:
        return np.zeros((0,2), dtype=np.int64)
    return np.concatenate([k_tool, k_other], axis=0)

def _rand_choice_rows(arr: np.ndarray, n: int, replace: bool):
    if arr.shape[0] == 0:
        return arr
    if arr.shape[0] <= n and not replace:
        return arr
    idx = np.random.choice(arr.shape[0], size=n, replace=replace)
    return arr[idx]

def balanced_sample_points(xyz, rgb, label, total_points: int, bg_ratio: float = 0.015, tool_ratio: float = 0.7):
    """
    label: 0=bg, 1=tool, 2=other
    Keep a tiny bit background, then split remaining between tool/other by tool_ratio.
    """
    # total_points = int(total_points)
    # n_bg = max(1, int(round(total_points * bg_ratio)))
    # n_rem = max(1, total_points - n_bg)
    # n_tool = int(round(n_rem * tool_ratio))
    # n_other = max(1, n_rem - n_tool)

    # out_xyz, out_rgb, out_label = [], [], []

    # for lab, n in [(0, n_bg), (1, n_tool), (2, n_other)]:
    #     m = (label.reshape(-1) == lab)
    #     xyz_l = xyz[m]
    #     rgb_l = rgb[m]
    #     replace = xyz_l.shape[0] < n
    #     xyz_s = _rand_choice_rows(xyz_l, n, replace)
    #     rgb_s = _rand_choice_rows(rgb_l, n, replace)
    #     out_xyz.append(xyz_s)
    #     out_rgb.append(rgb_s)
    #     out_label.append(np.full((xyz_s.shape[0], 1), lab, dtype=np.float32))

    # xyz_o = np.concatenate(out_xyz, axis=0)
    # rgb_o = np.concatenate(out_rgb, axis=0)
    # label_o = np.concatenate(out_label, axis=0)
    # return xyz_o, rgb_o, label_o
    # ==========================================

    if label.ndim == 1:
        label_o = label.astype(np.float32).reshape(-1, 1)
    else:
        label_o = label.astype(np.float32)
    return xyz.astype(np.float32), rgb.astype(np.float32), label_o    

def augment_points_xyz(xyz: np.ndarray, rot_deg: float = 10.0, jitter_std: float = 0.002, dropout: float = 0.0):
    """
    Simple augmentation: small random Z-rotation + jitter + random dropout (handled by caller via sampling).
    Units in meters for jitter_std.
    """
    if xyz.shape[0] == 0:
        return xyz
    # rotation about Z axis
    theta = (np.random.rand() * 2 - 1) * np.deg2rad(rot_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    xyz2 = (xyz @ R.T)
    # jitter
    xyz2 = xyz2 + np.random.randn(*xyz2.shape).astype(np.float32) * float(jitter_std)
    return xyz2


def gripperify_hand_contact_shell(
    tool_xyz: np.ndarray,
    tool_rgb: np.ndarray,
    obj_xyz: np.ndarray,
    q_contact: float = 0.04,
    min_pts: int = 80,
    relax: float = 0.1, 
    q_max: float = 0.3,
    return_idx: bool = False
):
    """
    Keep near-contact shell points on hand: for each hand point, compute NN distance to object,
    keep those with d <= quantile(d, q). If too few, relax q -> min(q_max, q+relax) iteratively.
    """
    if tool_xyz is None or obj_xyz is None or tool_xyz.shape[0] == 0 or obj_xyz.shape[0] == 0:
        return tool_xyz, tool_rgb

    # sanitize
    tool_xyz = tool_xyz.astype(np.float32, copy=False)
    obj_xyz  = obj_xyz.astype(np.float32, copy=False)
    tool_rgb = tool_rgb.astype(np.float32, copy=False) if tool_rgb is not None else tool_rgb

    # remove non-finite rows just in case
    m_tool = np.isfinite(tool_xyz).all(axis=1)
    m_obj  = np.isfinite(obj_xyz).all(axis=1)
    tool_xyz2 = tool_xyz[m_tool]
    tool_rgb2 = tool_rgb[m_tool] if tool_rgb is not None else None
    obj_xyz2  = obj_xyz[m_obj]

    if tool_xyz2.shape[0] == 0 or obj_xyz2.shape[0] == 0:
        return tool_xyz, tool_rgb

    try:
        from scipy.spatial import cKDTree

        d, _ = cKDTree(obj_xyz2).query(tool_xyz2, k=1, workers=-1)

        # drop any inf (can happen if obj empty, but we already checked)
        d = d[np.isfinite(d)]
        if d.size == 0:
            return tool_xyz, tool_rgb

        q = float(np.clip(q_contact, 0.0, 1.0))
        q_cur = q

        idx = np.array([], dtype=np.int64)
        while True:
            thr = float(np.quantile(d, q_cur))
            # Threshold distance on full tool_xyz2, not on the filtered subset d.
            d_full, _ = cKDTree(obj_xyz2).query(tool_xyz2, k=1, workers=-1)
            idx = np.where(np.isfinite(d_full) & (d_full <= thr))[0]

            if idx.shape[0] >= int(min_pts) or q_cur >= float(q_max):
                break
            q_cur = min(float(q_max), q_cur + float(relax))

        if idx.shape[0] == 0:
            return tool_xyz, tool_rgb

        # return tool_xyz2[idx], (tool_rgb2[idx] if tool_rgb2 is not None else tool_rgb2)
        out_xyz = tool_xyz2[idx]
        out_rgb = tool_rgb2[idx] if tool_rgb2 is not None else tool_rgb2
        if return_idx:
            return out_xyz, out_rgb, idx, tool_xyz2, tool_rgb2  # return full tool cloud for set-difference
        return out_xyz, out_rgb


    except Exception:
        # scipy missing or any error -> don't destroy the hand
        return tool_xyz, tool_rgb



def kpst_label_gen_demo_2d(analysts, args):
    """
    Modified:
      - Parse task_description to decide (tool, other)
      - Generate TWO object masks per clip: mask_tool, mask_other (+ optional mask_hand debug)
      - Sample 2D keypoints only on the TWO objects with ratio:
            tool: args.tool_point_ratio (default 0.7)
            other: 1 - tool_ratio
      - Save masks into step1_2d/<clip_id>/ :
            mask_tool.npy, mask_other.npy, mask_hand.npy (optional)
    """
    task_list = os.listdir(args.raw_data_root)
    clip_list = []
    fps = 15
    base_root = args.save_root

    for task_desc in task_list:
        if task_desc.startswith('.'):
            continue

        action, tool_name, other_name = parse_task_description(task_desc)

        if not should_process_task(task_desc, args):
            continue

        print("action:", action)
        print("tool_name:", tool_name)
        print("other_name:", other_name)



        # keep original folder structure under save_root
        args.save_root = os.path.join(base_root, task_desc)
        os.makedirs(os.path.join(args.save_root), exist_ok=True)
        os.makedirs(os.path.join(args.save_root, 'step1_2d'), exist_ok=True)

        append_new = bool(getattr(args, "append_new", False))
        clip_list, done_traj_paths = load_existing_2d_clips(args.save_root, append_new)

        base_task_read_fp = os.path.join(args.raw_data_root, task_desc)
        exec_fp_list = list_trajectory_dirs(base_task_read_fp)
        if not exec_fp_list:
            print(f"[Skip] no traj_* / rgb_0.png dirs under {base_task_read_fp}")
            continue

        for exec_fp in tqdm(exec_fp_list):
            base_fp = os.path.join(base_task_read_fp, exec_fp)
            if not os.path.isdir(base_fp):
                continue
            if os.path.normpath(base_fp) in done_traj_paths:
                print(f"[Skip] already processed: {exec_fp}")
                continue
            video_rgb = get_video(base_fp, 'rgb')  # (T,H,W,3)
            n_frame = video_rgb.shape[0]
            traj_st, traj_ed = 0, n_frame - 1

            # one clip per trajectory (no sliding window)
            traj_clip_idx = [(traj_st, traj_ed)]

            for traj_idx in traj_clip_idx:
                traj_st_real, traj_ed_real = traj_idx
                if traj_ed_real - traj_st_real < args.traj_len:
                    continue

                # sample traj_len+1 keyframes (inclusive end frame)
                traj_idx_list = np.linspace(traj_st_real, traj_ed_real, args.traj_len + 1).tolist()
                traj_idx_list = [int(idx) for idx in traj_idx_list]
                if tool_name == "hand":
                    object_name = other_name
                else:
                    object_name = f'{tool_name}+{other_name}'

                if tool_name is None or other_name is None:
                    continue

                tool_prompt = tool_name.replace('_', ' ')
                other_prompt = resolve_dino_other_prompt(args, task_desc, action, other_name)
                print(f"[DINO Prompt] other: {other_prompt!r} (semantic other={other_name!r})")

                clip = {
                    'id': len(clip_list),
                    'index': base_fp,
                    'st': traj_st_real,
                    'ed': traj_ed_real,
                    'task_description': task_desc,
                    'action': action,
                    'object': object_name,
                    'tool': tool_name,
                    'other': other_name,
                    'other_dino_prompt': other_prompt,
                    'other_only': bool(getattr(args, 'other_only', False)),
                    'manual_other_only': bool(getattr(args, 'manual_other_only', False)),
                    'seq_index': traj_idx_list,
                }

                save_fp = os.path.join(args.save_root, 'step1_2d', str(len(clip_list)))
                os.makedirs(save_fp, exist_ok=True)

                rgb_video_clip = video_rgb[traj_st_real: traj_ed_real + 1]
                rgb_image = rgb_video_clip[0]
                rgb0_path = os.path.join(base_fp, f"rgb_{traj_st_real}.png")

                from pathlib import Path

                last_dir = Path(base_fp).name

                # --- get masks (tool + other) ---
                # GroundingDINO prompt now uses TWO entities: "{tool} . {other} ."
                #     tool_prompt = f'dark green {tool_prompt}'
                # else:
                #     tool_prompt = f'red {tool_prompt}'

                other_only = bool(getattr(args, "other_only", False))
                manual_review = bool(getattr(args, "manual_seg_fallback", False))
                if other_only:
                    # other_only: only segment cup lid (other), no tool/hand
                    mask_other = segment_other_mask(
                        analysts, rgb_image, other_prompt, rgb0_path, save_fp, args,
                        action=action, other_name=other_name,
                    )
                    if mask_other is None:
                        print("Segment Fail, Continue.")
                        continue
                    m_other = mask_other[0].astype(np.uint8)
                    m_tool = np.zeros_like(m_other, dtype=np.uint8)
                    m_hand = np.zeros_like(m_other, dtype=np.uint8)
                    kps = sample_indices_from_mask(m_other, args.n_sample_max)
                else:
                    mask_tool, _ = analysts.segment_hoi_pair(
                        rgb_image,
                        tool_name=tool_prompt,
                        object_name=other_prompt,
                        image_path=rgb0_path,
                        vis_dir=save_fp,
                    )
                    if mask_tool is None and manual_review:
                        print("Segment fail: tool missing -> manual tool.")
                        mask_tool = analysts.segment_entity_manual(
                            rgb_image, tool_prompt, role="tool", vis_dir=save_fp,
                        )

                    mask_other = segment_other_mask(
                        analysts, rgb_image, other_prompt, rgb0_path, save_fp, args,
                        action=action, other_name=other_name,
                    )
                    if mask_tool is None or mask_other is None:
                        print("Segment Fail, Continue.")
                        continue

                    m_tool = mask_tool[0].astype(np.uint8)
                    m_other = mask_other[0].astype(np.uint8)

                    if tool_name == "hand":
                        m_hand = m_tool.copy()
                    else:
                        mask_hand, _ = analysts.segment_hoi_pair(
                            rgb_image,
                            tool_name="hand",
                            object_name=tool_prompt,
                            image_path=rgb0_path,
                            vis_dir=None,
                        )

                        if mask_hand is None:
                            m_hand = np.zeros_like(m_tool, dtype=np.uint8)
                        else:
                            m_hand = mask_hand[0].astype(np.uint8)
                            m_hand = (m_hand > 0).astype(np.uint8)
                            m_hand = (m_hand & (1 - (m_tool > 0).astype(np.uint8))).astype(np.uint8)

                    kps = sample_kps_two_masks(
                        m_tool, m_other,
                        n_total=args.n_sample_max,
                        tool_ratio=float(args.tool_point_ratio),
                    )
                if kps.shape[0] == 0:
                    print("Empty kps, Continue.")
                    continue

                pred_tracks, pred_visibility = analysts.get_kpst_track(rgb_video_clip, kps, vis_dir=save_fp)
                mark_image = get_mark_image(rgb_image, kps)

                cv2.imwrite(os.path.join(save_fp, 'mark.jpg'), cv2.cvtColor(mark_image, cv2.COLOR_RGB2BGR))
                np.save(os.path.join(save_fp, 'mask_tool.npy'), m_tool)
                np.save(os.path.join(save_fp, 'mask_other.npy'), m_other)
                np.save(os.path.join(save_fp, 'mask_hand.npy'), m_hand)

                np.save(os.path.join(save_fp, 'kps_tracks.npy'), pred_tracks)            # (T, N, 2) (w,h)
                np.save(os.path.join(save_fp, 'kps_visibility.npy'), pred_visibility)    # (T, N)

                clip_list.append(clip)

            # break
    fp = os.path.join(args.save_root, 'step1_kpst_2d_info.json')
    with open(fp, 'w') as f:
        json.dump(clip_list, f, indent=4)


# def get_camera(camera_in):
#     camera = o3d.camera.PinholeCameraIntrinsic()
#     camera.set_intrinsics(WIDTH, HEIGHT, camera_in[0,0], camera_in[1,1], camera_in[0,2], camera_in[1,2])
#     return camera

def get_intrinsic_parameter(camera_in):
    cx = camera_in[0, 2]
    cy = camera_in[1, 2]
    fx = camera_in[0, 0]
    fy = camera_in[1, 1]
    return cx, cy, fx, fy


def remove_mask_border(mask: np.ndarray, border_px: int = 1) -> np.ndarray:
    """
    Remove only the boundary band of a binary mask, keeping the interior.
    border_px controls boundary thickness in pixels.
    """
    if border_px <= 0:
        return (mask > 0).astype(np.uint8)

    m = (mask > 0).astype(np.uint8)
    k = 2 * int(border_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    inner = cv2.erode(m, kernel, iterations=1)  # interior after removing border band
    return inner
# [0, 1, 2,3,4,5,6,7,8,9,10,27,28,29,30,31,32,33,34,35,50,51,52,53,54,59,60,61]
# [11-26]

def kpst_label_gen_demo_3d(args):
    """
    Modified 3D generation:

    - Use masks from step1_2d/<clip_id>/ :
        mask_tool.npy, mask_other.npy (+ mask_hand.npy for debug)
    - Build point clouds ONLY from masked regions:
        tool (label=1), other (label=2)
      plus a tiny fraction of background (label=0) for augmentation
    - Final sampling ratios (per-clip):
        bg: args.bg_point_ratio (default 0.015)
        remaining: tool/other = args.tool_point_ratio (default 0.7) / (1-tool_ratio)
    - Tool == hand: convert hand points to gripper-like (near-contact shell) before sampling.
    - Apply light augmentation to tool/other points (global SE(3) + jitter).
    """
    with open(os.path.join(args.save_root, 'step1_kpst_2d_info.json')) as fp:
        org_clip_list = json.load(fp)

    print("Total number of step1_kpst_label_gen_2d clips: ", len(org_clip_list))
    os.makedirs(os.path.join(args.save_root, 'data'), exist_ok=True)

    append_new = bool(getattr(args, "append_new", False))
    recompute_3d = bool(getattr(args, "recompute_3d", False))
    use_refined_depth = bool(getattr(args, "use_refined_depth", False))
    existing_3d = load_existing_3d_clips(args.save_root, append_new and not recompute_3d)
    prior_3d = load_metadata_3d_clips(args.save_root)
    clip_list = []
    n_skip_incomplete = 0
    n_skip_missing_raw = 0
    record_root = getattr(args, "record_root", None) or os.environ.get("RECORD_ROOT")
    if use_refined_depth:
        print("[3D] using LingBot refined depth (depth_refined_*.png/npy)")
    for _, clip_org in tqdm(enumerate(org_clip_list)):
        step_fp = os.path.join(args.save_root, 'step1_2d', str(clip_org['id']))
        save_fp = os.path.join(args.save_root, 'data', str(clip_org['id']))
        kpst_fp = os.path.join(save_fp, 'kpst_traj.npy')
        if append_new and not recompute_3d and os.path.isfile(kpst_fp):
            clip_list.append(existing_3d.get(clip_org['id'], clip_org))
            continue

        if not step1_2d_ready(step_fp):
            n_skip_incomplete += 1
            if os.path.isfile(kpst_fp):
                clip_list.append(prior_3d.get(clip_org['id'], clip_org))
            else:
                print(f"[Skip] incomplete step1_2d clip {clip_org['id']}: {step_fp}")
            continue

        base_fp_orig = clip_org['index']
        base_fp = resolve_traj_root(base_fp_orig, record_root)
        if base_fp is None:
            n_skip_missing_raw += 1
            print(f"[Skip] raw RGB-D not found: {_missing_raw_hint(base_fp_orig)}")
            if os.path.isfile(kpst_fp):
                clip_list.append(prior_3d.get(clip_org['id'], clip_org))
            continue
        if base_fp != os.path.normpath(base_fp_orig):
            print(f"[Remap] {base_fp_orig} -> {base_fp}")

        os.makedirs(save_fp, exist_ok=True)

        camera_in = np.load(os.path.join(base_fp, "camera_in.npy"))

        # use REAL image size from depth (or rgb) to build intrinsic
        st = int(clip_org['st'])
        dep0 = load_depth_uint16(base_fp, st, use_refined_depth=use_refined_depth)
        H0, W0 = dep0.shape[:2]

        print("dep dtype:", dep0.dtype, "min:", dep0.min(), "max:", dep0.max(),
            "nonzero:", np.count_nonzero(dep0), "refined:", use_refined_depth)
        
        camera = get_camera(camera_in, W0, H0)


        # ---- 2D keypoint tracks -> 3D keypoint tracks ----
        seq_idx_list = clip_org['seq_index']
        kps_track = np.load(os.path.join(step_fp, 'kps_tracks.npy'))                # (T, N, 2)
        kps_visibility = np.load(os.path.join(step_fp, 'kps_visibility.npy'))       # (T, N)
        kps_pos, kps_vis = [], []
        st = int(clip_org['st'])

        for sidx in seq_idx_list:
            kps_pos.append(kps_track[sidx-st])
            kps_vis.append(kps_visibility[sidx-st])
        kps_pos, kps_vis = np.stack(kps_pos, axis=1), np.stack(kps_vis, axis=1)  # (N, T, 2), (N, T)

        kps_vis_sum = np.sum(kps_vis, axis=1)
        other_only = bool(clip_org.get("other_only", getattr(args, "other_only", False)))
        is_available = (kps_vis_sum == kps_vis.shape[1])
        if np.sum(is_available) == 0:
            continue
        kps_pos = kps_pos[is_available]
        kps_vis = kps_vis[is_available]

        # ---- only keep queries on (tool OR object) at t=0 ----
        pos0 = kps_pos[:, 0]  # (N,2) in (x,y) = (w,h)
        pos0_ij = np.round(pos0).astype(np.int32)

        mask_tool = np.load(os.path.join(step_fp, "mask_tool.npy")).astype(np.uint8)
        mask_other = np.load(os.path.join(step_fp, "mask_other.npy")).astype(np.uint8)
        if mask_tool.ndim == 3:  mask_tool = mask_tool[0]
        if mask_other.ndim == 3: mask_other = mask_other[0]

        # --- erode mask edges a bit (remove boundary pixels) ---
        # erode_ksize = int(getattr(args, "mask_erode_ksize", 3))   # 3 or 5 typical
        # erode_iter  = int(getattr(args, "mask_erode_iter", 1))    # 1~2 typical
        # if erode_ksize > 1 and erode_iter > 0:
        #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_ksize, erode_ksize))
        #     mask_tool  = cv2.erode((mask_tool  > 0).astype(np.uint8), kernel, iterations=erode_iter)
        #     mask_other = cv2.erode((mask_other > 0).astype(np.uint8), kernel, iterations=erode_iter)

        border_px = int(getattr(args, "mask_border_px", 5))
        mask_tool  = remove_mask_border(mask_tool, border_px=border_px)
        mask_other = remove_mask_border(mask_other, border_px=border_px)


        pos0_ij[:, 0] = np.clip(pos0_ij[:, 0], 0, mask_tool.shape[1]-1)
        pos0_ij[:, 1] = np.clip(pos0_ij[:, 1], 0, mask_tool.shape[0]-1)

        on_tool0 = mask_tool[pos0_ij[:, 1], pos0_ij[:, 0]] > 0
        on_obj0  = mask_other[pos0_ij[:, 1], pos0_ij[:, 0]] > 0
        keep0 = on_obj0 if other_only else (on_tool0 | on_obj0)
        if np.sum(keep0) == 0:
            continue

        kps_pos = kps_pos[keep0]
        kps_vis = kps_vis[keep0]
        on_tool0 = on_tool0[keep0]
        on_obj0  = on_obj0[keep0]
        pos0_ij  = pos0_ij[keep0]



        point_3d_list = []
        cx, cy, fx, fy = get_intrinsic_parameter(camera_in)

        for img_idx, t in zip(seq_idx_list, range(kps_vis.shape[1])):
            pos = kps_pos[:, t]  # (N,2) in (w,h)
            dep = load_depth_uint16(base_fp, img_idx, use_refined_depth=use_refined_depth)
            Hd, Wd = dep.shape[:2]

            # IMPORTANT: int32 (avoid uint16 wrap if negative)
            pos_ij = np.round(pos).astype(np.int32)
            pos_ij[:, 0] = np.clip(pos_ij[:, 0], 0, Wd - 1)  # x in [0, Wd)
            pos_ij[:, 1] = np.clip(pos_ij[:, 1], 0, Hd - 1)  # y in [0, Hd)

            pos_dep = dep[pos_ij[:, 1], pos_ij[:, 0]].astype(np.float32)

            # print("pos_dep:", pos_dep)
            z = pos_dep / 1000.0  # meters

            x = (pos_ij[:, 0].astype(np.float32) - float(cx)) * z / float(fx)
            y = (pos_ij[:, 1].astype(np.float32) - float(cy)) * z / float(fy)

            point_3d = np.stack([x, y, z], axis=-1)
            point_3d_list.append(point_3d)


        # kpst_3d = np.stack(point_3d_list, axis=1)  # (N, T, 3)
        # is_available = (kpst_3d[:, :, -1] > 0).sum(axis=1) == kpst_3d.shape[1]
        # if np.sum(is_available) == 0:
        #     continue
        # kpst_3d = kpst_3d[is_available]
        # kpst_dist = np.sum(np.linalg.norm(kpst_3d[:, 1:] - kpst_3d[:, :-1], axis=-1), axis=-1)
        # is_available = kpst_dist > args.static_threshold
        # if np.sum(is_available) == 0:
        #     continue
        kpst_3d = np.stack(point_3d_list, axis=1)  # (N,T,3)

        # ---- filter 2: valid depth + motion + outlier rejection ----
        keep_z = (kpst_3d[:, :, 2] > 0).all(axis=1)
        if np.sum(keep_z) == 0:
            continue
        kpst_3d  = kpst_3d[keep_z]
        kps_pos  = kps_pos[keep_z]
        kps_vis  = kps_vis[keep_z]
        pos0_ij  = pos0_ij[keep_z]
        on_tool0 = on_tool0[keep_z]
        on_obj0  = on_obj0[keep_z]

        if float(args.static_threshold) > 0:
            kpst_dist = np.sum(np.linalg.norm(kpst_3d[:, 1:] - kpst_3d[:, :-1], axis=-1), axis=-1)
            keep_move = kpst_dist > float(args.static_threshold)
            if np.sum(keep_move) == 0:
                continue
            kpst_3d  = kpst_3d[keep_move]
            kps_pos  = kps_pos[keep_move]
            kps_vis  = kps_vis[keep_move]
            pos0_ij  = pos0_ij[keep_move]
            on_tool0 = on_tool0[keep_move]
            on_obj0  = on_obj0[keep_move]

        if other_only:
            # lift_lid: allow larger motion than hand-tool tasks
            max_step = 0.50
            max_disp = 1.0
            max_total_len = 1.5
        elif str(clip_org.get("action", "")).lower() == "place" or (
            clip_org.get("tool") not in (None, "hand")
            and clip_org.get("other") is not None
            and str(clip_org.get("action", "")).lower() != "pickup"
        ):
            # place_apple_to_plate etc.: tool moves ~20-40cm between viz keyframes
            max_step = 0.50
            max_disp = 0.45
            max_total_len = 1.20
        else:
            max_step = float(args.kpst_max_step)
            max_disp = float(args.kpst_max_disp)
            max_total_len = float(args.kpst_max_total_len)

        kpst_3d, keep_long = filter_kpst_long_trajectories(
            kpst_3d,
            max_total_len=max_total_len,
            max_step=max_step,
            max_disp=max_disp,
            max_outlier_step_ratio=float(getattr(args, "kpst_max_outlier_step_ratio", 0.35)),
            outlier_step_thr=float(getattr(args, "kpst_outlier_step_thr", 0.08)),
        )
        if kpst_3d.shape[0] == 0:
            continue

        kps_pos  = kps_pos[keep_long]
        kps_vis  = kps_vis[keep_long]
        pos0_ij  = pos0_ij[keep_long]
        on_tool0 = on_tool0[keep_long]
        on_obj0  = on_obj0[keep_long]




        # ---- build 3D scene pcd for first frame only ----
        dep = load_depth_uint16(base_fp, st, use_refined_depth=use_refined_depth)
        color_raw = o3d.io.read_image(os.path.join(base_fp, f"rgb_{st}.png"))


        # mask_tool = np.load(os.path.join(step_fp, "mask_tool.npy")).astype(np.uint8)
        # mask_other = np.load(os.path.join(step_fp, "mask_other.npy")).astype(np.uint8)

        other_only = bool(clip_org.get("other_only", getattr(args, "other_only", False)))
        mask_bg = (mask_other == 0) if other_only else ((mask_tool == 0) & (mask_other == 0))

        def _pcd_from_mask(mask_2d_uint8):
            depth_t = dep.copy()
            mk = (mask_2d_uint8 > 0)
            depth_t[~mk] = 0
            # Open3D wants an image; write temp in-memory via numpy->Image is annoying, easiest is cv2 tmp
            tmp_fp = os.path.join(step_fp, "_tmp_depth_mask.png")
            cv2.imwrite(tmp_fp, depth_t.astype(np.uint16))
            depth_img = o3d.io.read_image(tmp_fp)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_raw, depth_img, convert_rgb_to_intensity=False, 
                depth_trunc=1.6,
            )
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera, extrinsic=np.eye(4))
            if args.voxel_size is not None and float(args.voxel_size) > 0:
                pcd = pcd.voxel_down_sample(voxel_size=float(args.voxel_size))
            xyz = np.asarray(pcd.points).astype(np.float32)
            rgb = np.asarray(pcd.colors).astype(np.float32)
            return xyz, rgb

        tool_xyz, tool_rgb = _pcd_from_mask(mask_tool)
        other_xyz, other_rgb = _pcd_from_mask(mask_other)
        bg_xyz, bg_rgb = _pcd_from_mask(mask_bg.astype(np.uint8))

        z_min = float(getattr(args, "pcd_z_min", 0.05))
        z_max = float(getattr(args, "pcd_z_max", 1.6))

        bg_xyz, bg_rgb = remove_fly_points_xyzrgb(
            bg_xyz, bg_rgb,
            z_min=z_min, z_max=z_max,
            use_statistical=True,
            nb_neighbors=int(getattr(args, "bg_sor_k", 20)),
            std_ratio=float(getattr(args, "bg_sor_std", 2.5)),
            use_radius=True,
            radius=float(getattr(args, "bg_ror_radius", 0.04)),
            min_neighbors=int(getattr(args, "bg_ror_min_nb", 4)),
        )

        # foreground = tool + other
        fg_xyz = np.concatenate([tool_xyz, other_xyz], axis=0)

        bg_xyz, bg_rgb = filter_bg_remove_islands(
            bg_xyz=bg_xyz,
            bg_rgb=bg_rgb,
            fg_xyz=fg_xyz,
            max_fg_dist=float(getattr(args, "bg_max_fg_dist", 0.6)),
            dbscan_eps=float(getattr(args, "bg_dbscan_eps", 0.03)),
            dbscan_min_points=int(getattr(args, "bg_dbscan_min_points", 30)),
            keep_largest_cluster=True,
        )



        if other_xyz.shape[0] == 0:
            continue
        if not other_only and tool_xyz.shape[0] == 0:
            continue

        # ---- tool == hand -> gripper-like ----
        tool_name = clip_org.get("tool", None)

        # print("tool_name:", tool_name)
        if tool_name is None:
            # backward compatibility (older json)
            task_desc = clip_org.get("task_description", f"{clip_org.get('action','')}_{clip_org.get('object','')}")
            _, tool_name, _ = parse_task_description(task_desc)

        tool_xyz_filt = tool_xyz
        tool_rgb_filt = tool_rgb

        if not other_only and tool_name == "hand":
            tool_xyz_filt, tool_rgb_filt = gripperify_hand_contact_shell(
                tool_xyz=tool_xyz,
                tool_rgb=tool_rgb,
                obj_xyz=other_xyz,
                q_contact=0.10,
                min_pts=400,
                relax=0.4,
                q_max=0.80,
            )
            # use gripperified hand for downstream pcd as well
            tool_xyz, tool_rgb = tool_xyz_filt, tool_rgb_filt

        elif not other_only:
            hand_mask_fp = os.path.join(step_fp, "mask_hand.npy")
            if os.path.exists(hand_mask_fp):
                mask_hand = np.load(hand_mask_fp).astype(np.uint8)
                if mask_hand.ndim == 3:
                    mask_hand = mask_hand[0]

                # hand point cloud from hand mask (not tool mask)
                hand_xyz, hand_rgb = _pcd_from_mask(mask_hand)

                if hand_xyz.shape[0] > 0 and tool_xyz.shape[0] > 0:
                    # gripperify in 3D: keep hand shell near tool
                    hand_grip_xyz, hand_grip_rgb, idx_keep, hand_xyz2, hand_rgb2 = gripperify_hand_contact_shell(
                        tool_xyz=hand_xyz,
                        tool_rgb=hand_rgb,
                        obj_xyz=tool_xyz,   # reference object = tool (nearest-neighbor shell)
                        q_contact=float(getattr(args, "hand_q_contact", 0.10)),
                        min_pts=int(getattr(args, "hand_min_pts", 80)),
                        relax=float(getattr(args, "hand_relax", 0.15)),
                        q_max=float(getattr(args, "hand_q_max", 0.90)),
                        return_idx=True,
                    )

                    # rest = hand - grip (strict complement)
                    keep_mask = np.zeros((hand_xyz2.shape[0],), dtype=bool)
                    if idx_keep is not None and idx_keep.size > 0:
                        keep_mask[idx_keep] = True
                    hand_rest_xyz = hand_xyz2[~keep_mask]

                    # project hand_rest back to 2D for bg cleanup
                    if hand_rest_xyz.shape[0] > 0:
                        cx, cy, fx, fy = get_intrinsic_parameter(camera_in)

                        X = hand_rest_xyz[:, 0]
                        Y = hand_rest_xyz[:, 1]
                        Z = hand_rest_xyz[:, 2]
                        valid = Z > 1e-6
                        X, Y, Z = X[valid], Y[valid], Z[valid]

                        u = np.round(fx * (X / Z) + cx).astype(np.int32)
                        v = np.round(fy * (Y / Z) + cy).astype(np.int32)

                        Hm, Wm = mask_tool.shape[:2]
                        u = np.clip(u, 0, Wm - 1)
                        v = np.clip(v, 0, Hm - 1)

                        mask_hand_rest2d = np.zeros((Hm, Wm), dtype=np.uint8)
                        mask_hand_rest2d[v, u] = 1

                        # bg = bg minus hand_rest (strict 2D set difference)
                        mask_bg_clean = (mask_bg & (mask_hand_rest2d == 0))
                        bg_xyz, bg_rgb = _pcd_from_mask(mask_bg_clean.astype(np.uint8))

            # optional: re-detect hand with GroundingDINO then gripperify



        # ---- remove trajectories whose initial point is NOT in gripperified-hand region ----
        if (not other_only) and tool_name == "hand" and tool_xyz_filt is not None and tool_xyz_filt.shape[0] > 0:
            cx, cy, fx, fy = get_intrinsic_parameter(camera_in)

            X = tool_xyz_filt[:, 0]; Y = tool_xyz_filt[:, 1]; Z = tool_xyz_filt[:, 2]
            valid = Z > 1e-6
            X, Y, Z = X[valid], Y[valid], Z[valid]

            u = np.round(fx * (X / Z) + cx).astype(np.int32)
            v = np.round(fy * (Y / Z) + cy).astype(np.int32)
            u = np.clip(u, 0, mask_tool.shape[1]-1)
            v = np.clip(v, 0, mask_tool.shape[0]-1)

            mask_tool_grip2d = np.zeros_like(mask_tool, dtype=np.uint8)
            mask_tool_grip2d[v, u] = 1

            keep_tool_traj = mask_tool_grip2d[pos0_ij[:, 1], pos0_ij[:, 0]] > 0
            keep = on_obj0 | (on_tool0 & keep_tool_traj)

            if np.sum(keep) == 0:
                continue

            kpst_3d  = kpst_3d[keep]
            kps_pos  = kps_pos[keep]
            kps_vis  = kps_vis[keep]
            pos0_ij  = pos0_ij[keep]
            on_tool0 = on_tool0[keep]
            on_obj0  = on_obj0[keep]

        fg_xyz = other_xyz if other_only else np.concatenate([tool_xyz, other_xyz], axis=0)

        bg_xyz, bg_rgb = filter_bg_remove_islands(
            bg_xyz=bg_xyz,
            bg_rgb=bg_rgb,
            fg_xyz=fg_xyz,
            max_fg_dist=float(getattr(args, "bg_max_fg_dist", 0.6)),
            dbscan_eps=float(getattr(args, "bg_dbscan_eps", 0.03)),
            dbscan_min_points=int(getattr(args, "bg_dbscan_min_points", 30)),
            keep_largest_cluster=True,
        )

        # ---- simple augmentation (preserve relative pose) ----
        # global small Z-rot + jitter
        theta = (np.random.rand() * 2 - 1) * np.deg2rad(float(args.aug_rot_deg))
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0.0],
                      [s,  c, 0.0],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
        t = (np.random.randn(3).astype(np.float32) * float(args.aug_trans_std))

        tool_xyz = apply_global_se3(tool_xyz, R, t)
        other_xyz = apply_global_se3(other_xyz, R, t)
        bg_xyz = apply_global_se3(bg_xyz, R, t)
        kpst_3d = apply_global_se3(kpst_3d, R, t)

        tool_xyz = tool_xyz + np.random.randn(*tool_xyz.shape).astype(np.float32) * float(args.aug_jitter_std)
        other_xyz = other_xyz + np.random.randn(*other_xyz.shape).astype(np.float32) * float(args.aug_jitter_std)

        # ---- compose labels ----
        xyz = np.concatenate([bg_xyz, tool_xyz, other_xyz], axis=0)
        rgb = np.concatenate([bg_rgb, tool_rgb, other_rgb], axis=0)
        label = np.concatenate([
            np.zeros((bg_xyz.shape[0], 1), dtype=np.float32),
            np.ones((tool_xyz.shape[0], 1), dtype=np.float32),
            np.ones((other_xyz.shape[0], 1), dtype=np.float32) * 2.0
        ], axis=0)

        # ---- final balanced sampling ----
        xyz_s, rgb_s, label_s = balanced_sample_points(
            xyz, rgb, label,
            total_points=int(args.n_pcd_points),
            bg_ratio=float(args.bg_point_ratio),
            tool_ratio=float(args.tool_point_ratio)
        )

        spcd = np.concatenate([xyz_s, rgb_s, label_s], axis=1)  # (N, 7)
        np.save(os.path.join(save_fp, 'pcd.npy'), spcd)

        np.save(os.path.join(save_fp, 'kpst_traj.npy'), kpst_3d)   # (N, T, 3)
        if other_only:
            part_id = np.full((kpst_3d.shape[0], 1), 2.0, dtype=np.float32)
        else:
            part_id = np.where(on_tool0.reshape(-1, 1), 1.0, 2.0).astype(np.float32)
        np.save(os.path.join(save_fp, 'kpst_part_id'), part_id)


        clip_list.append(clip_org)



    if n_skip_incomplete:
        print(f"[3D] skipped {n_skip_incomplete} clip(s) with incomplete step1_2d")
    if n_skip_missing_raw:
        print(f"[3D] skipped {n_skip_missing_raw} clip(s) with missing raw RGB-D (mount disk or set --record_root)")

    fp = os.path.join(args.save_root, 'metadata_egosoft_demo.json')
    with open(fp, 'w') as f:
        json.dump(clip_list, f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser('TAP-KPST Label Extraction')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--downsample_ratio', type=int, default=1)
    parser.add_argument('--n_sample_max', type=int, default=1024)

    parser.add_argument('--aft_clip_interval', type=float, default=0.15)    
    parser.add_argument('--clip_duration', type=float, default=1.5)
    parser.add_argument('--traj_len', type=int, default=3)
    parser.add_argument('--voxel_size', default=0.01, type=float)
    parser.add_argument('--static_threshold', default=0.02, type=float,
                        help='Min total 3D path length (m) to keep a track (0=disable)')
    parser.add_argument('--kpst_max_step', type=float, default=0.12,
                        help='Max single-step 3D jump (m); other_only uses 0.50 internally')
    parser.add_argument('--kpst_max_disp', type=float, default=0.35,
                        help='Max 3D endpoint displacement (m); other_only uses 1.0 internally')
    parser.add_argument('--kpst_max_total_len', type=float, default=0.60,
                        help='Max total 3D path length (m); other_only uses 1.5 internally')

    parser.add_argument('--n_pcd_points', type=int, default=4096)
    parser.add_argument('--bg_point_ratio', type=float, default=0.015)   # 1~2%
    parser.add_argument('--tool_point_ratio', type=float, default=0.7)   # tool vs other
    parser.add_argument('--aug_rot_deg', type=float, default=10.0)
    parser.add_argument('--aug_trans_std', type=float, default=0.0)
    parser.add_argument('--aug_jitter_std', type=float, default=0.002)

    parser.add_argument('--raw_data_root', type=str, default=None, help='Root of human RGB-D trajectories (required unless --only_3d)')
    parser.add_argument('--save_root', type=str, default='./output/rvideo')
    parser.add_argument(
        '--task_name',
        type=str,
        default=None,
        help='Process only this task folder under raw_data_root (e.g. pull_out_the_drawer)',
    )
    parser.add_argument(
        '--dino_other_prompt',
        type=str,
        default=None,
        help='Override DINO text prompt for other/object',
    )
    parser.add_argument(
        '--lid_mask_top_ratio',
        type=float,
        default=0.0,
        help='Optional: crop other mask to top fraction of bbox when DINO segments whole cup (0=off, use DINO mask as-is)',
    )
    parser.add_argument(
        '--manual_seg_fallback',
        action='store_true',
        help='Review DINO masks (y=accept, m=manual); also manual fallback for tool when DINO fails',
    )
    parser.add_argument(
        '--other_only',
        action='store_true',
        help='Only segment other on frame 0; skip tool, sample kps on other only',
    )
    parser.add_argument(
        '--append_new',
        action='store_true',
        help='Only process new traj not in existing step1_kpst_2d_info.json; merge outputs',
    )
    parser.add_argument(
        '--manual_other_only',
        action='store_true',
        help='Skip DINO for other; always open manual point picker on frame 0',
    )
    parser.add_argument(
        '--only_3d',
        action='store_true',
        help='Skip 2D; re-run 3D from existing step1_2d + step1_kpst_2d_info.json',
    )
    parser.add_argument(
        '--recompute_3d',
        action='store_true',
        help='Overwrite existing data/*/kpst_traj.npy (use with --only_3d)',
    )
    parser.add_argument(
        '--use_refined_depth',
        action='store_true',
        help='Use depth_refined_*.png/npy from traj folder for 3D back-projection',
    )
    parser.add_argument(
        '--gt_root',
        type=str,
        default=None,
        help='With --only_3d: batch all GT tasks under this root (e.g. process_data)',
    )
    parser.add_argument(
        '--record_root',
        type=str,
        default=None,
        help='Remap /media/ljx/UBU/data/Record to this path when the UBU disk is mounted elsewhere '
             '(or set env RECORD_ROOT)',
    )
    args = parser.parse_args()

    os.makedirs(args.save_root, exist_ok=True)

    if args.only_3d:
        if os.path.isfile(os.path.join(args.save_root, "step1_kpst_2d_info.json")):
            gt_task_roots = [args.save_root]
        elif args.gt_root:
            gt_task_roots = discover_gt_task_roots(args.gt_root)
        else:
            gt_task_roots = discover_gt_task_roots(args.save_root)
        if not gt_task_roots:
            raise RuntimeError(
                f"No GT task with step1_kpst_2d_info.json under {args.gt_root or args.save_root}"
            )
        print(f"[Only3D] {len(gt_task_roots)} task(s)")
        for gt_root in gt_task_roots:
            args.save_root = gt_root
            print(f"\n======== 3D: {gt_root} ========")
            kpst_label_gen_demo_3d(args)
    else:
        if not args.raw_data_root:
            parser.error('--raw_data_root is required unless --only_3d is set')
        analysts = EgoHOIAnalysts(downsample_ratio=args.downsample_ratio, device=args.device)
        kpst_label_gen_demo_2d(analysts, args)
        kpst_label_gen_demo_3d(args)