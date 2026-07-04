#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GroundingDINO-based HOI detector (bbox only).

This is a drop-in replacement for the FasterRCNN-based EgoHOIDetector usage in label_gen_demo.py:
- Returns "hand" boxes (top-K) and "object" boxes (top-1 by default) in XYXY pixel coordinates.
- Boxes are in ORIGINAL image pixel coordinates (before any downsampling). Caller can scale as needed.

Assumptions (matches GroundingDINO util.inference defaults):
- predict() returns:
    boxes:  (N,4) in normalized CXCYWH (0..1)
    logits: (N,)
    phrases: list[str] length N
"""

from __future__ import annotations
import os
import time
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except Exception as e:
    raise ImportError("GroundingDINODetector requires torch.") from e

# GroundingDINO util.inference (from the official repo / pip package)
try:
    from groundingdino.util.inference import load_model, load_image, predict
except Exception as e:
    raise ImportError(
        "Cannot import groundingdino.util.inference. "
        "Make sure GroundingDINO is installed or its repo is in PYTHONPATH."
    ) from e


def _normalize_phrase(s: str) -> str:
    s = s.lower().strip()
    # remove common punctuation from GroundingDINO phrases
    for ch in [".", ",", ";", ":", "!", "?", "\"", "'"]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s


def filter_bbox(image_source, boxes, logits, phrases, m):
    unique_labels = set(phrases)
    filtered_boxes = []
    filtered_logits = []
    filtered_phrases = []

    for label in unique_labels:
        label_indices = [i for i, phrase in enumerate(phrases) if phrase == label]
        label_logits = logits[label_indices]
        max_logit_index = torch.argmax(label_logits)
        original_index = label_indices[max_logit_index]

        filtered_boxes.append(boxes[original_index])
        filtered_logits.append(logits[original_index])
        filtered_phrases.append(phrases[original_index])

    filtered_boxes = torch.stack(filtered_boxes)
    filtered_logits = torch.stack(filtered_logits)

    top7_indices = torch.topk(filtered_logits, k=min(m, len(filtered_logits)))[1]
    filtered_boxes = filtered_boxes[top7_indices]
    filtered_logits = filtered_logits[top7_indices]
    filtered_phrases = [filtered_phrases[i] for i in top7_indices]

    print(filtered_phrases)
    print(filtered_logits)

    return filtered_boxes, filtered_logits, filtered_phrases


def _cxcywh_to_xyxy_pixel(boxes_cxcywh_norm: torch.Tensor, w: int, h: int) -> np.ndarray:
    """
    boxes_cxcywh_norm: (N,4) tensor in normalized cx,cy,w,h (0..1)
    returns: (N,4) float32 array in pixel xyxy
    """
    if boxes_cxcywh_norm.numel() == 0:
        return np.zeros((0, 4), dtype=np.float32)

    b = boxes_cxcywh_norm.detach().cpu().float().numpy()
    cx = b[:, 0] * w
    cy = b[:, 1] * h
    bw = b[:, 2] * w
    bh = b[:, 3] * h
    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0
    out = np.stack([x1, y1, x2, y2], axis=-1).astype(np.float32)
    # clip
    out[:, 0] = np.clip(out[:, 0], 0, w - 1)
    out[:, 2] = np.clip(out[:, 2], 0, w - 1)
    out[:, 1] = np.clip(out[:, 1], 0, h - 1)
    out[:, 3] = np.clip(out[:, 3], 0, h - 1)
    return out


def _build_entity_candidates(
    boxes: torch.Tensor,
    logits: torch.Tensor,
    phrases: List[str],
    entity_label: str,
    w: int,
    h: int,
) -> List[dict]:
    """Build DINO detection candidates with metadata."""
    import re

    def _norm(p: str) -> str:
        p = p.lower()
        p = re.sub(r"\(.*?\)", "", p)
        p = re.sub(r"[^a-z0-9 ]", " ", p)
        return " ".join(p.split())

    if boxes is None or boxes.numel() == 0:
        return []

    entity_norm = _norm(entity_label)
    entity_words = set(entity_norm.split()) or {entity_norm}
    part_kw = None
    for kw in ("handle", "lid", "cover", "knob", "pull"):
        if kw in entity_words:
            part_kw = kw
            break

    parent_words = {
        "handle": {"drawer", "cabinet", "door"},
        "lid": {"cup", "mug", "pot", "container"},
        "cover": {"box", "container", "bottle", "jar", "pot", "cup"},
        "knob": {"drawer", "door", "cabinet"},
        "pull": {"drawer", "cabinet"},
    }

    img_area = float(max(1, w * h))
    candidates = []
    for i in range(len(phrases)):
        phrase_norm = _norm(phrases[i])
        phrase_words = set(phrase_norm.split())
        box = _cxcywh_to_xyxy_pixel(boxes[i].unsqueeze(0), w=w, h=h)[0]
        area = float(max(1.0, (box[2] - box[0]) * (box[3] - box[1])))
        has_part = part_kw is not None and part_kw in phrase_norm
        parent_set = parent_words.get(part_kw or "", set())
        has_parent_only = (
            part_kw is not None
            and bool(phrase_words & parent_set)
            and not has_part
            and not any(k in phrase_norm for k in ("handle", "knob", "pull", "lid", "cover"))
        )
        candidates.append(
            {
                "box": box,
                "area": area,
                "area_frac": area / img_area,
                "conf": float(logits[i]),
                "phrase": phrases[i],
                "has_part": has_part,
                "parent_only": has_parent_only,
            }
        )
    return candidates


def _rank_part_candidates(candidates: List[dict]) -> List[dict]:
    """Rank small-part candidates: prefer handle/lid phrase, skip parent-only, smallest box first."""
    if not candidates:
        return []
    part_cands = [c for c in candidates if c["has_part"]]
    pool = part_cands if part_cands else candidates
    non_parent = [c for c in pool if not c["parent_only"]]
    pool = non_parent if non_parent else pool
    small = [c for c in pool if c["area_frac"] <= 0.06]
    pool = small if small else pool
    return sorted(pool, key=lambda c: (c["area"], -c["conf"]))


def _select_entity_box(
    boxes: torch.Tensor,
    logits: torch.Tensor,
    phrases: List[str],
    entity_label: str,
    w: int,
    h: int,
) -> Tuple[Optional[np.ndarray], List[str], List[float]]:
    """
    Pick one DINO box for entity-only segmentation.
    For small parts (handle / lid / knob): prefer matching phrase, then smallest area.
    """
    entity_norm = entity_label.lower()
    part_kw = next((kw for kw in ("handle", "lid", "cover", "knob", "pull") if kw in entity_norm), None)
    candidates = _build_entity_candidates(boxes, logits, phrases, entity_label, w, h)
    if not candidates:
        return None, [], []

    if part_kw is not None:
        ranked = _rank_part_candidates(candidates)
        best = ranked[0]
        print(
            f"[DINO part pick:{part_kw}] phrase={best['phrase']!r} "
            f"area={best['area']:.0f} ({best['area_frac']*100:.1f}% frame) conf={best['conf']:.3f}"
        )
    else:
        import re
        def _norm(p: str) -> str:
            p = re.sub(r"[^a-z0-9 ]", " ", p.lower())
            return " ".join(p.split())
        entity_words = set(_norm(entity_label).split())
        matched = [c for c in candidates if len(set(_norm(c["phrase"]).split()) & entity_words) > 0]
        pool = matched if matched else candidates
        best = max(pool, key=lambda c: c["conf"])

    return best["box"].copy(), [best["phrase"]], [best["conf"]]


def _list_entity_box_candidates(
    boxes: torch.Tensor,
    logits: torch.Tensor,
    phrases: List[str],
    entity_label: str,
    w: int,
    h: int,
    max_candidates: int = 5,
) -> List[Tuple[np.ndarray, str, float]]:
    """Return up to N DINO boxes for small-part queries (smallest first)."""
    entity_norm = entity_label.lower()
    part_kw = next((kw for kw in ("handle", "lid", "cover", "knob", "pull") if kw in entity_norm), None)
    candidates = _build_entity_candidates(boxes, logits, phrases, entity_label, w, h)
    if not candidates:
        return []

    if part_kw is not None:
        ranked = _rank_part_candidates(candidates)[:max_candidates]
    else:
        ranked = sorted(candidates, key=lambda c: -c["conf"])[:max_candidates]

    out = []
    for c in ranked:
        out.append((c["box"].copy(), c["phrase"], c["conf"]))
    return out


class GroundingDINODetector:
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda",
        box_threshold: float = 0.30,
        text_threshold: float = 0.30,
    ):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"GroundingDINO config not found: {config_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"GroundingDINO checkpoint not found: {checkpoint_path}")

        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)

        # Load model ONCE
        self.model = load_model(self.config_path, self.checkpoint_path)
        if self.device.startswith("cuda") and torch.cuda.is_available():
            self.model = self.model.to(self.device)
        self.model.eval()


    def detect(
        self,
        tool_name: str,
        object_name: str,
        image_path: Optional[str] = None,
        image_rgb: Optional[np.ndarray] = None,
        max_tools: int = 1,
    ) -> Dict[str, List[np.ndarray]]:
        """
        Returns:
        {
            "ok": bool,
            "tool_boxes_xyxy": [np.ndarray(4,), ...],   # up to max_tools (default 1)
            "obj_boxes_xyxy":  [np.ndarray(4,), ...],   # 1
            "phrases": [...],   # filtered phrases
            "logits":  [...],

            # backward-compatible aliases:
            "hand_boxes_xyxy":  [...],  # only present when tool_name == 'hand'
        }
        """
        if image_path is None and image_rgb is None:
            raise ValueError("Provide either image_path or image_rgb.")

        import re
        tool_label = tool_name.replace("_", " ").strip().lower()
        obj_label = object_name.replace("_", " ").strip().lower()
        # prompt BOTH tool and object (instead of hard-coding hand)
        text_prompt = f"{tool_label} . {obj_label} ."

        tmp_path = None
        try:
            if image_path is None:
                fd, tmp_path = tempfile.mkstemp(suffix=".jpg", prefix="dino_tmp_")
                os.close(fd)
                import cv2
                bgr = image_rgb[:, :, ::-1].copy()
                cv2.imwrite(tmp_path, bgr)
                img_path = tmp_path
            else:
                img_path = image_path

            image_source, image = load_image(img_path)
            h, w = image_source.shape[:2]

            image = image.to(torch.float32)
            boxes, logits, phrases = predict(
                model=self.model,
                image=image,
                caption=text_prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device,
            )
            # boxes: (N,4) cxcywh normalized (tensor)
            # logits: (N,) (tensor)
            # phrases: list[str]

            def _norm(p: str) -> str:
                p = p.lower()
                p = re.sub(r"\(.*?\)", "", p)
                p = re.sub(r"[^a-z0-9 ]", " ", p)
                p = " ".join(p.split())
                return p

            tool_words = set(_norm(tool_label).split())
            obj_words = set(_norm(obj_label).split())
            if len(tool_words) == 0:
                tool_words = set([_norm(tool_label)])
            if len(obj_words) == 0:
                obj_words = set([_norm(obj_label)])

            phrases_simple = []
            for p in phrases:
                pn = _norm(p)
                pw = set(pn.split())
                # Prefer tool match first (so "hand" won't swallow the object if tool isn't hand)
                if len(pw & tool_words) > 0:
                    phrases_simple.append(tool_label)
                elif len(pw & obj_words) > 0:
                    phrases_simple.append(obj_label)
                else:
                    phrases_simple.append("other")

            filtered_boxes, filtered_logits, filtered_phrases = filter_bbox(
                image_source=image_source,
                boxes=boxes,
                logits=logits,
                phrases=phrases_simple,
                m=7
            )

            # Return whichever side(s) DINO found; missing side stays empty.
            label2idx = {p: i for i, p in enumerate(filtered_phrases)}
            tool_boxes_xyxy: List[np.ndarray] = []
            obj_boxes_xyxy: List[np.ndarray] = []

            if tool_label in label2idx:
                tool_i = label2idx[tool_label]
                tool_box = _cxcywh_to_xyxy_pixel(filtered_boxes[tool_i].unsqueeze(0), w=w, h=h)[0]
                tool_boxes_xyxy = [tool_box.copy()]

            if obj_label in label2idx:
                obj_i = label2idx[obj_label]
                obj_box = _cxcywh_to_xyxy_pixel(filtered_boxes[obj_i].unsqueeze(0), w=w, h=h)[0]
                obj_boxes_xyxy = [obj_box.copy()]

            if not tool_boxes_xyxy and not obj_boxes_xyxy:
                return {
                    "ok": False,
                    "tool_boxes_xyxy": [],
                    "obj_boxes_xyxy": [],
                    "hand_boxes_xyxy": [],
                    "phrases": filtered_phrases,
                    "logits": filtered_logits.detach().cpu().tolist() if hasattr(filtered_logits, "detach") else [],
                }

            out = {
                "ok": True,
                "tool_boxes_xyxy": tool_boxes_xyxy,
                "obj_boxes_xyxy": obj_boxes_xyxy,
                "phrases": filtered_phrases,
                "logits": filtered_logits.detach().cpu().tolist() if hasattr(filtered_logits, "detach") else [],
            }
            out["hand_boxes_xyxy"] = tool_boxes_xyxy if tool_label == "hand" else []
            return out

        finally:
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _predict_entity_boxes(
        self,
        entity_name: str,
        image_path: Optional[str] = None,
        image_rgb: Optional[np.ndarray] = None,
    ):
        """Run GroundingDINO once; return (boxes, logits, phrases, w, h, entity_label)."""
        if image_path is None and image_rgb is None:
            raise ValueError("Provide either image_path or image_rgb.")

        entity_label = entity_name.replace("_", " ").strip().lower()
        if "handle" in entity_label:
            text_prompt = f"{entity_label} . metal handle . pull handle ."
        elif "lid" in entity_label or "cover" in entity_label:
            text_prompt = f"{entity_label} ."
        else:
            text_prompt = f"{entity_label} ."

        tmp_path = None
        try:
            if image_path is None:
                fd, tmp_path = tempfile.mkstemp(suffix=".jpg", prefix="dino_tmp_")
                os.close(fd)
                import cv2
                bgr = image_rgb[:, :, ::-1].copy()
                cv2.imwrite(tmp_path, bgr)
                img_path = tmp_path
            else:
                img_path = image_path

            image_source, image = load_image(img_path)
            h, w = image_source.shape[:2]
            image = image.to(torch.float32)
            boxes, logits, phrases = predict(
                model=self.model,
                image=image,
                caption=text_prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device,
            )
            return boxes, logits, phrases, w, h, entity_label
        finally:
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def detect_entity_candidates(
        self,
        entity_name: str,
        image_path: Optional[str] = None,
        image_rgb: Optional[np.ndarray] = None,
        max_candidates: int = 5,
    ) -> Dict[str, object]:
        """Return multiple DINO box candidates (smallest-first for handle/lid)."""
        boxes, logits, phrases, w, h, entity_label = self._predict_entity_boxes(
            entity_name, image_path=image_path, image_rgb=image_rgb,
        )
        cands = _list_entity_box_candidates(
            boxes, logits, phrases, entity_label, w=w, h=h, max_candidates=max_candidates,
        )
        return {
            "ok": len(cands) > 0,
            "boxes_xyxy": [c[0] for c in cands],
            "phrases": [c[1] for c in cands],
            "logits": [c[2] for c in cands],
        }

    def detect_entity_only(
        self,
        entity_name: str,
        image_path: Optional[str] = None,
        image_rgb: Optional[np.ndarray] = None,
    ) -> Dict[str, List[np.ndarray]]:
        """Detect a single entity (e.g. lid) with GroundingDINO; bbox only."""
        boxes, logits, phrases, w, h, entity_label = self._predict_entity_boxes(
            entity_name, image_path=image_path, image_rgb=image_rgb,
        )

        picked_box, picked_phrases, picked_logits = _select_entity_box(
            boxes, logits, phrases, entity_label, w=w, h=h,
        )

        if picked_box is None:
            return {
                "ok": False,
                "boxes_xyxy": [],
                "phrases": list(phrases),
                "logits": logits.detach().cpu().tolist() if hasattr(logits, "detach") else [],
            }

        return {
            "ok": True,
            "boxes_xyxy": [picked_box],
            "phrases": picked_phrases,
            "logits": picked_logits,
        }
