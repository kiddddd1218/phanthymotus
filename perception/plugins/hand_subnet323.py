#!/usr/bin/env python3
"""
plugins/hand_subnet323.py — Portable <30 MB ONNX Runtime pipeline for Darvin Subnet 323.

Python 3.8+ (`from __future__ import annotations`), pure NumPy + OpenCV + ONNX Runtime.
Works identically on x86, JetPack 5.1.1 (cp38, ORT 1.15.1), and JetPack 6.1 (cp310, ORT 1.18.0)
with 0% GPU utilization on CPUExecutionProvider and bounded memory (`enable_cpu_mem_arena=False`).
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from plugins.vision_runtime import LetterboxMeta, letterbox, to_blob

COCO_NOSE = 0
COCO_LEFT_EAR = 3
COCO_RIGHT_EAR = 4
COCO_LEFT_SHOULDER = 5
COCO_RIGHT_SHOULDER = 6
COCO_LEFT_ELBOW = 7
COCO_RIGHT_ELBOW = 8
COCO_LEFT_WRIST = 9
COCO_RIGHT_WRIST = 10

HAND_WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_CHAINS = (
    (HAND_WRIST, THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
    (HAND_WRIST, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    (HAND_WRIST, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    (HAND_WRIST, RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    (HAND_WRIST, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
)
FINGERTIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
FINGER_PIPS = (THUMB_MCP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP)

OFFICIAL_30_CLASSES: Tuple[str, ...] = (
    "stop", "fist", "pointing", "victory", "ok", "thumbs_up", "one", "thumbs_down",
    "palm", "two_up", "three", "three2", "three3", "four", "call", "mute",
    "rock", "little_finger", "middle_finger", "grabbing", "grip", "thumb_index", "three_gun",
    "holy", "timeout", "xsign", "hand_heart", "hand_heart2", "take_picture", "thumb_index2",
)
TWO_HAND_CLASSES: Tuple[str, ...] = OFFICIAL_30_CLASSES[23:]
NO_GESTURE_IDX = 30

IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMAGENET_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

DEFAULT_CALIBRATION = {
    "pose_conf_thresh": 0.22,
    "hand_det_conf_thresh": 0.24,
    "hand_det_low_anchor_thresh": 0.14,
    "temperature_hand": 1.0,
    "temperature_pair": 1.0,
    "pair_alpha": 0.60,
    "pair_threshold": 0.48,
    "null_logit_bias": 0.0,
    "min_single_prob": 0.32,
    "second_hand_min_prob": 0.78,
}


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-9)


def bbox_iou_np(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 1e-6 else 0.0


def nms_boxes_np(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.45) -> np.ndarray:
    if boxes.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return np.asarray(keep, dtype=np.int64)


def expand_square_crop(
    box_xyxy: np.ndarray, img_w: int, img_h: int, expand_ratio: float = 1.35
) -> Tuple[float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    side = max(x2 - x1, y2 - y1, 16.0) * expand_ratio
    side = min(side, float(max(img_w, img_h)) * 1.25)
    return cx, cy, side


def crop_imagenet_blob(bgr: np.ndarray, cx: float, cy: float, side: float, out_size: int = 224) -> np.ndarray:
    import cv2

    half = 0.5 * side
    src = np.array(
        [[cx - half, cy - half], [cx + half, cy - half], [cx - half, cy + half]],
        dtype=np.float32,
    )
    dst = np.array([[0.0, 0.0], [float(out_size), 0.0], [0.0, float(out_size)]], dtype=np.float32)
    mat = cv2.getAffineTransform(src, dst)
    patch_bgr = cv2.warpAffine(
        bgr, mat, (out_size, out_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    rgb = patch_bgr[:, :, ::-1].astype(np.float32)
    norm = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(norm.transpose(2, 0, 1), dtype=np.float32)


def estimate_palm_anchors_from_body(
    body_kpts: Optional[np.ndarray],
    person_box: Optional[np.ndarray],
    min_kpt_conf: float = 0.20,
) -> Dict[str, Optional[Tuple[np.ndarray, float]]]:
    out: Dict[str, Optional[Tuple[np.ndarray, float]]] = {"left": None, "right": None}
    if body_kpts is None:
        return out
    k = np.asarray(body_kpts, dtype=np.float32)
    if k.shape != (17, 3):
        return out

    sh_l, sh_r = k[COCO_LEFT_SHOULDER], k[COCO_RIGHT_SHOULDER]
    shoulder_w = (
        float(np.linalg.norm(sh_l[:2] - sh_r[:2]))
        if (sh_l[2] >= min_kpt_conf and sh_r[2] >= min_kpt_conf)
        else 0.0
    )
    fallback_scale = shoulder_w * 0.85 if shoulder_w > 8.0 else 0.0
    if fallback_scale <= 8.0 and person_box is not None:
        pw = max(1.0, float(person_box[2] - person_box[0]))
        ph = max(1.0, float(person_box[3] - person_box[1]))
        fallback_scale = 0.20 * max(pw, ph)

    for side, idx_sh, idx_el, idx_wr in (
        ("left", COCO_LEFT_SHOULDER, COCO_LEFT_ELBOW, COCO_LEFT_WRIST),
        ("right", COCO_RIGHT_SHOULDER, COCO_RIGHT_ELBOW, COCO_RIGHT_WRIST),
    ):
        wr = k[idx_wr]
        if wr[2] < min_kpt_conf:
            continue
        el = k[idx_el]
        sh = k[idx_sh]
        if el[2] >= min_kpt_conf:
            vec = wr[:2] - el[:2]
            forearm = float(np.linalg.norm(vec))
            if forearm > 6.0:
                out[side] = (wr[:2] + 0.35 * vec, max(forearm * 0.90, 24.0))
                continue
        if sh[2] >= min_kpt_conf:
            vec = wr[:2] - sh[:2]
            arm_len = float(np.linalg.norm(vec))
            if arm_len > 10.0:
                out[side] = (wr[:2] + 0.18 * vec, max(arm_len * 0.45, 24.0))
                continue
        if fallback_scale > 8.0:
            out[side] = (wr[:2].copy(), max(fallback_scale, 24.0))
    return out


def assign_hands_to_sides(
    hand_boxes: np.ndarray,
    hand_scores: np.ndarray,
    body_kpts: Optional[np.ndarray],
    person_box: Optional[np.ndarray],
    img_w: int,
    img_h: int,
    side_probs_right: Optional[np.ndarray] = None,
) -> Dict[str, Optional[int]]:
    assigned: Dict[str, Optional[int]] = {"left": None, "right": None}
    m = int(hand_boxes.shape[0])
    if m == 0:
        return assigned

    if person_box is not None:
        px1, py1, px2, py2 = [float(v) for v in person_box]
        pw, ph = max(px2 - px1, 1.0), max(py2 - py1, 1.0)
        ex1, ey1, ex2, ey2 = px1 - 0.35 * pw, py1 - 0.25 * ph, px2 + 0.35 * pw, py2 + 0.25 * ph
        centers = 0.5 * (hand_boxes[:, :2] + hand_boxes[:, 2:4])
        in_person = (
            (centers[:, 0] >= ex1) & (centers[:, 0] <= ex2)
            & (centers[:, 1] >= ey1) & (centers[:, 1] <= ey2)
        )
        valid_idx = np.where(in_person)[0]
        if valid_idx.size == 0:
            valid_idx = np.arange(m)
    else:
        valid_idx = np.arange(m)

    valid_idx = valid_idx[np.argsort(hand_scores[valid_idx])[::-1][:4]]
    k_cand = int(valid_idx.size)
    anchors = estimate_palm_anchors_from_body(body_kpts, person_box)

    if person_box is not None:
        mid_x = 0.5 * float(person_box[0] + person_box[2])
        ref_scale = max(float(person_box[2] - person_box[0]), float(person_box[3] - person_box[1]), 100.0)
    else:
        mid_x = 0.5 * float(img_w)
        ref_scale = float(max(img_w, img_h))

    cost = np.full((2, k_cand), 1e6, dtype=np.float32)
    for col, det_i in enumerate(valid_idx):
        bx = hand_boxes[det_i]
        cx = 0.5 * float(bx[0] + bx[2])
        cy = 0.5 * float(bx[1] + bx[3])
        c_xy = np.array([cx, cy], dtype=np.float32)
        det_bonus = -0.15 * float(hand_scores[det_i])
        for row, side in enumerate(("left", "right")):
            anc = anchors[side]
            if anc is not None:
                palm_xy, scale_px = anc
                c_val = float(np.linalg.norm(c_xy - palm_xy)) / max(scale_px, 40.0)
            else:
                rel_x = (cx - mid_x) / max(0.5 * ref_scale, 50.0)
                c_val = (-0.25 * rel_x if side == "left" else 0.25 * rel_x) + 1.2
            if side_probs_right is not None:
                p_right = float(side_probs_right[det_i])
                p_side = p_right if side == "right" else (1.0 - p_right)
                c_val += 0.45 * (1.0 - p_side)
            cost[row, col] = c_val + det_bonus

    max_accept = 3.2
    if k_cand == 1:
        best_row = int(np.argmin(cost[:, 0]))
        if cost[best_row, 0] <= max_accept:
            assigned["left" if best_row == 0 else "right"] = int(valid_idx[0])
        return assigned

    pen_l = 2.35 if anchors["left"] is not None else 1.65
    pen_r = 2.35 if anchors["right"] is not None else 1.65
    best_total = pen_l + pen_r
    best_pair: Tuple[Optional[int], Optional[int]] = (None, None)
    options: List[Optional[int]] = [None] + list(range(k_cand))
    for c_l in options:
        for c_r in options:
            if c_l is not None and c_l == c_r:
                continue
            val_l = float(cost[0, c_l]) if c_l is not None else pen_l
            val_r = float(cost[1, c_r]) if c_r is not None else pen_r
            if (c_l is not None and val_l > max_accept) or (c_r is not None and val_r > max_accept):
                continue
            if val_l + val_r < best_total:
                best_total = val_l + val_r
                best_pair = (c_l, c_r)
    if best_pair[0] is not None:
        assigned["left"] = int(valid_idx[best_pair[0]])
    if best_pair[1] is not None:
        assigned["right"] = int(valid_idx[best_pair[1]])
    return assigned


def decode_simcc_np(simcc_x: np.ndarray, simcc_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    px = _softmax(simcc_x, axis=-1)
    py = _softmax(simcc_y, axis=-1)
    wx = simcc_x.shape[-1]
    wy = simcc_y.shape[-1]
    ix = np.argmax(px, axis=-1)
    iy = np.argmax(py, axis=-1)
    conf_x = np.take_along_axis(px, ix[..., None], axis=-1)[..., 0]
    conf_y = np.take_along_axis(py, iy[..., None], axis=-1)[..., 0]
    conf = np.clip(np.sqrt(conf_x * conf_y) * 3.0, 0.0, 1.0)
    exp_x = np.sum(px * np.arange(wx, dtype=np.float32)[None, None, :], axis=-1)
    exp_y = np.sum(py * np.arange(wy, dtype=np.float32)[None, None, :], axis=-1)
    sub_x = (0.7 * ix.astype(np.float32) + 0.3 * exp_x) / float(wx)
    sub_y = (0.7 * iy.astype(np.float32) + 0.3 * exp_y) / float(wy)
    return np.stack([sub_x, sub_y], axis=-1).astype(np.float32), conf.astype(np.float32)


def hand_geometric_features(
    kpts_xyc: np.ndarray, side: str, hand_box: np.ndarray, body_kpts: Optional[np.ndarray] = None
) -> np.ndarray:
    k = np.asarray(kpts_xyc, dtype=np.float32)
    if k.shape != (21, 3):
        return np.zeros((118,), dtype=np.float32)
    pts = k[:, :2]
    wrist = pts[HAND_WRIST]
    mid_mcp = pts[MIDDLE_MCP]
    palm_center = 0.5 * (pts[INDEX_MCP] + pts[PINKY_MCP])
    bw = max(float(hand_box[2] - hand_box[0]), 1.0)
    bh = max(float(hand_box[3] - hand_box[1]), 1.0)
    box_scale = max(bw, bh)
    scale = max(float(np.linalg.norm(mid_mcp - wrist)), 0.22 * box_scale, 1e-3)

    cam_flat = np.clip(((pts - palm_center[None, :]) / scale).reshape(-1), -5.0, 5.0)
    axis_y = mid_mcp - wrist
    ny = float(np.linalg.norm(axis_y))
    uy = (axis_y / ny) if ny > 1e-4 else np.array([0.0, -1.0], dtype=np.float32)
    ux = np.array([-uy[1], uy[0]], dtype=np.float32)
    rel_w = (pts - wrist[None, :]) / scale
    ax = rel_w @ ux
    ay = rel_w @ uy
    if side == "left":
        ax = -ax
    aligned_flat = np.clip(np.stack([ax, ay], axis=-1).reshape(-1), -5.0, 5.0)

    reaches = np.array(
        [
            (float(np.linalg.norm(pts[tip] - wrist)) - float(np.linalg.norm(pts[pip] - wrist))) / scale
            for tip, pip in zip(FINGERTIPS, FINGER_PIPS)
        ],
        dtype=np.float32,
    )
    cosines = np.zeros((10,), dtype=np.float32)
    ci = 0
    for chain in FINGER_CHAINS:
        for j in (1, 2):
            v1 = pts[chain[j]] - pts[chain[j - 1]]
            v2 = pts[chain[j + 1]] - pts[chain[j]]
            cosines[ci] = float(np.dot(v1, v2) / (float(np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6))
            ci += 1

    tip_dists = np.zeros((10,), dtype=np.float32)
    ti = 0
    for a in range(5):
        for b in range(a + 1, 5):
            tip_dists[ti] = min(float(np.linalg.norm(pts[FINGERTIPS[a]] - pts[FINGERTIPS[b]])) / scale, 6.0)
            ti += 1

    body_feat = np.zeros((9,), dtype=np.float32)
    body_feat[0] = 1.0 if side == "right" else -1.0
    body_feat[1] = bw / max(bh, 1.0)
    if body_kpts is not None and np.asarray(body_kpts).shape == (17, 3):
        bk = np.asarray(body_kpts, dtype=np.float32)
        sh_l, sh_r = bk[COCO_LEFT_SHOULDER], bk[COCO_RIGHT_SHOULDER]
        b_scale = (
            float(np.linalg.norm(sh_l[:2] - sh_r[:2]))
            if (sh_l[2] >= 0.2 and sh_r[2] >= 0.2)
            else box_scale * 1.5
        )
        b_scale = max(b_scale, 20.0)
        nose = bk[COCO_NOSE]
        if nose[2] >= 0.2:
            body_feat[2] = 1.0
            mouth_xy = np.array([nose[0], nose[1] + 0.18 * b_scale], dtype=np.float32)
            body_feat[3:5] = np.clip((pts[INDEX_TIP] - mouth_xy) / b_scale, -4.0, 4.0)
            body_feat[5] = min(float(np.linalg.norm(pts[INDEX_TIP] - nose[:2])) / b_scale, 5.0)
        ear_dists = [
            float(np.linalg.norm(pts[THUMB_TIP] - bk[ei, :2])) / b_scale
            for ei in (COCO_LEFT_EAR, COCO_RIGHT_EAR)
            if bk[ei, 2] >= 0.2
        ]
        if ear_dists:
            body_feat[6] = min(min(ear_dists), 5.0)
        el_i = COCO_RIGHT_ELBOW if side == "right" else COCO_LEFT_ELBOW
        wr_i = COCO_RIGHT_WRIST if side == "right" else COCO_LEFT_WRIST
        if bk[el_i, 2] >= 0.2 and bk[wr_i, 2] >= 0.2:
            fv = bk[wr_i, :2] - bk[el_i, :2]
            fn = float(np.linalg.norm(fv))
            if fn > 4.0:
                body_feat[7:9] = fv / fn
    return np.concatenate([cam_flat, aligned_flat, reaches, cosines, tip_dists, body_feat]).astype(np.float32)


def pair_relational_features(
    kpts_left: Optional[np.ndarray],
    box_left: Optional[np.ndarray],
    kpts_right: Optional[np.ndarray],
    box_right: Optional[np.ndarray],
) -> np.ndarray:
    feat = np.zeros((30,), dtype=np.float32)
    if kpts_left is None or kpts_right is None or box_left is None or box_right is None:
        return feat
    bl = np.asarray(box_left, dtype=np.float32)
    br = np.asarray(box_right, dtype=np.float32)
    kl = np.asarray(kpts_left, dtype=np.float32)[:, :2]
    kr = np.asarray(kpts_right, dtype=np.float32)[:, :2]
    cl, cr = 0.5 * (bl[:2] + bl[2:4]), 0.5 * (br[:2] + br[2:4])
    sl = max(float(bl[2] - bl[0]), float(bl[3] - bl[1]), 1.0)
    sr = max(float(br[2] - br[0]), float(br[3] - br[1]), 1.0)
    pair_scale = max(0.5 * (sl + sr), 16.0)

    feat[0] = 1.0
    feat[1:3] = np.clip((cl - cr) / pair_scale, -6.0, 6.0)
    feat[3] = 1.0 if cl[0] < cr[0] else -1.0
    feat[4] = bbox_iou_np(bl, br)
    feat[5] = min(float(np.linalg.norm(cl - cr)) / pair_scale, 8.0)

    al = kl[MIDDLE_MCP] - kl[HAND_WRIST]
    ar = kr[MIDDLE_MCP] - kr[HAND_WRIST]
    ul = al / (float(np.linalg.norm(al)) + 1e-6)
    ur = ar / (float(np.linalg.norm(ar)) + 1e-6)
    feat[6] = float(np.dot(ul, ur))
    feat[7] = float(ul[0] * ur[1] - ul[1] * ur[0])
    feat[8:10] = ul
    feat[10:12] = ur
    for i, (ia, ib) in enumerate((
        (HAND_WRIST, HAND_WRIST), (THUMB_TIP, THUMB_TIP), (INDEX_TIP, INDEX_TIP),
        (MIDDLE_TIP, MIDDLE_TIP), (RING_TIP, RING_TIP), (PINKY_TIP, PINKY_TIP),
        (THUMB_TIP, INDEX_TIP), (INDEX_TIP, THUMB_TIP), (INDEX_TIP, MIDDLE_MCP),
        (MIDDLE_MCP, INDEX_TIP), (MIDDLE_TIP, HAND_WRIST), (HAND_WRIST, MIDDLE_TIP),
        (THUMB_IP, THUMB_IP), (INDEX_PIP, INDEX_PIP),
    )):
        feat[12 + i] = min(float(np.linalg.norm(kl[ia] - kr[ib])) / pair_scale, 8.0)
    feat[26] = np.clip(((kl[INDEX_TIP, 1] + kr[INDEX_TIP, 1]) * 0.5 - (kl[THUMB_TIP, 1] + kr[THUMB_TIP, 1]) * 0.5) / pair_scale, -4.0, 4.0)
    feat[27] = np.clip((kl[INDEX_TIP, 1] - kr[INDEX_TIP, 1]) / pair_scale, -4.0, 4.0)
    feat[28] = np.clip((kl[THUMB_TIP, 1] - kr[THUMB_TIP, 1]) / pair_scale, -4.0, 4.0)
    feat[29] = sl / max(sr, 1.0)
    return feat


def _new_ort_session(onnx_path: Path, num_threads: int = 1):
    """Build a strictly CPU-only ONNX Runtime session.

    Never loads a CUDA provider (`providers=["CPUExecutionProvider"]` only), so it
    never touches `libonnxruntime_providers_shared.so` (`Provider_SetHost`) on JP5.11
    and consumes 0% GPU (<10% Subnet 323 budget) with zero CUDA context overhead.
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, int(num_threads))
    opts.inter_op_num_threads = 1
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    return ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])


class OnnxPoseEngineAdapter:
    """Drop-in `VisionEngineSession` replacement backed by `person_pose.onnx` (`YOLO11n-pose` 640x640).

    Emits `[rows_1xNx57]` in 640x640 letterbox coordinates + `LetterboxMeta`, matching
    `plugins.vision_runtime.decode_poses` byte-for-byte without TensorRT engine deserialization.
    """

    def __init__(self, onnx_path: Union[str, Path], num_threads: int = 1) -> None:
        self.onnx_path = Path(onnx_path)
        self._lock = threading.Lock()
        self._sess = _new_ort_session(self.onnx_path, num_threads=num_threads)
        self._in_name = self._sess.get_inputs()[0].name
        self._last_blob: Optional[np.ndarray] = None
        self._last_meta: Optional[LetterboxMeta] = None
        self._last_rows_frame: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    @property
    def input_size(self) -> Tuple[int, int]:
        return (640, 640)

    def infer(self, frame: np.ndarray) -> Tuple[List[np.ndarray], LetterboxMeta]:
        canvas, meta = letterbox(frame, 640, 640)
        blob = to_blob(canvas, np.float32)
        with self._lock:
            raw = self._sess.run(None, {self._in_name: blob})[0]
            self._last_blob = blob
            self._last_meta = meta

        arr = raw[0]
        if arr.shape[0] > arr.shape[1]:
            arr = arr.T
        scores = arr[4]
        keep = scores >= 0.10
        if not np.any(keep):
            self._last_rows_frame = None
            return [np.zeros((1, 0, 57), dtype=np.float32)], meta

        sub = arr[:, keep]
        cx, cy, bw, bh, sc = sub[0], sub[1], sub[2], sub[3], sub[4]
        boxes = np.stack([cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh], axis=-1).astype(np.float32)
        idx = nms_boxes_np(boxes, sc.astype(np.float32), iou_thresh=0.45)[:15]
        n = len(idx)
        rows = np.zeros((1, n, 57), dtype=np.float32)
        rows[0, :, 0:4] = boxes[idx]
        rows[0, :, 4] = np.clip(sc[idx], 0.0, 1.0)
        rows[0, :, 5] = 0.0
        kpts_flat = sub[5:, idx].T.astype(np.float32)
        kpts_flat[:, 2::3] = np.clip(kpts_flat[:, 2::3], 0.0, 1.0)
        rows[0, :, 6:57] = kpts_flat

        # Also cache native-frame fallback person boxes (>=0.10) for tight close-ups
        fx1 = np.clip((boxes[idx, 0] - meta.pad_x) / meta.scale, 0.0, float(meta.orig_w))
        fy1 = np.clip((boxes[idx, 1] - meta.pad_y) / meta.scale, 0.0, float(meta.orig_h))
        fx2 = np.clip((boxes[idx, 2] - meta.pad_x) / meta.scale, 0.0, float(meta.orig_w))
        fy2 = np.clip((boxes[idx, 3] - meta.pad_y) / meta.scale, 0.0, float(meta.orig_h))
        fboxes = np.stack([fx1, fy1, fx2, fy2], axis=-1).astype(np.float32)
        fkpts = kpts_flat.reshape(n, 17, 3).copy()
        fkpts[:, :, 0] = np.clip((fkpts[:, :, 0] - meta.pad_x) / meta.scale, 0.0, float(meta.orig_w))
        fkpts[:, :, 1] = np.clip((fkpts[:, :, 1] - meta.pad_y) / meta.scale, 0.0, float(meta.orig_h))
        self._last_rows_frame = (fboxes, np.clip(sc[idx], 0.0, 1.0).astype(np.float32), fkpts)
        return [rows], meta


class Subnet323HandRuntime:
    """Singleton thread-safe runtime running `hand_det.onnx` + `hand_crop_net.onnx` + `gesture_fusion.onnx`."""

    def __init__(
        self,
        bundle_dir: Union[str, Path],
        calibration_path: Optional[Union[str, Path]] = None,
        num_threads: int = 1,
    ) -> None:
        self.bundle_dir = Path(bundle_dir)
        self._lock = threading.Lock()
        self.calib = dict(DEFAULT_CALIBRATION)
        for cand in (
            Path(calibration_path) if calibration_path else None,
            self.bundle_dir / "calibration.json",
            Path(__file__).resolve().parent.parent / "data" / "hand_subnet323_calibration.json",
        ):
            if cand is not None and cand.is_file():
                self.calib.update(json.loads(cand.read_text(encoding="utf-8")))
                break

        self.hand_sess = _new_ort_session(self.bundle_dir / "hand_det.onnx", num_threads=num_threads)
        self.crop_sess = _new_ort_session(self.bundle_dir / "hand_crop_net.onnx", num_threads=num_threads)
        self.fuse_sess = _new_ort_session(self.bundle_dir / "gesture_fusion.onnx", num_threads=num_threads)
        self._hand_in = self.hand_sess.get_inputs()[0].name
        self._crop_in = self.crop_sess.get_inputs()[0].name
        self.ran = 0
        self.skipped: Dict[str, int] = {
            "wrist_occluded": 0,
            "no_scale": 0,
            "too_far": 0,
            "throttled": 0,
            "low_confidence": 0,
        }
        self.last_error: Optional[str] = None

    @property
    def input_size(self) -> int:
        return 224

    def reset(self) -> None:
        """Interface hook called by `_PoseNode.stop()`; sessions stay alive across restarts."""
        return None

    def _detect_hands(self, frame: np.ndarray, cached_blob: Optional[np.ndarray], cached_meta: Optional[LetterboxMeta]):
        h, w = frame.shape[:2]
        if cached_blob is not None and cached_meta is not None and cached_meta.orig_w == w and cached_meta.orig_h == h:
            blob, meta = cached_blob, cached_meta
        else:
            canvas, meta = letterbox(frame, 640, 640)
            blob = to_blob(canvas, np.float32)

        with self._lock:
            raw = self.hand_sess.run(None, {self._hand_in: blob})[0]
            self.ran += 1

        arr = raw[0]
        if arr.shape[0] > arr.shape[1]:
            arr = arr.T
        scores = arr[4]
        det_th = float(self.calib["hand_det_conf_thresh"])
        keep = scores >= det_th
        if not np.any(keep):
            keep = scores >= float(self.calib["hand_det_low_anchor_thresh"])
        if not np.any(keep):
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        sub = arr[:, keep]
        cx, cy, bw, bh, sc = sub[0], sub[1], sub[2], sub[3], sub[4]
        x1 = np.clip((cx - 0.5 * bw - meta.pad_x) / meta.scale, 0.0, float(w))
        y1 = np.clip((cy - 0.5 * bh - meta.pad_y) / meta.scale, 0.0, float(h))
        x2 = np.clip((cx + 0.5 * bw - meta.pad_x) / meta.scale, 0.0, float(w))
        y2 = np.clip((cy + 0.5 * bh - meta.pad_y) / meta.scale, 0.0, float(h))
        boxes = np.stack([x1, y1, x2, y2], axis=-1).astype(np.float32)
        idx = nms_boxes_np(boxes, sc.astype(np.float32), iou_thresh=0.45)
        return boxes[idx], sc[idx].astype(np.float32)

    @staticmethod
    def _decode_gestures(
        logits_left: np.ndarray,
        logits_right: np.ndarray,
        logits_pair: np.ndarray,
        present_left: bool,
        present_right: bool,
        calib: dict,
    ) -> Tuple[Dict[str, Optional[str]], Dict[str, Optional[float]]]:
        gestures: Dict[str, Optional[str]] = {"left": None, "right": None}
        scores: Dict[str, Optional[float]] = {"left": None, "right": None}
        if not present_left and not present_right:
            return gestures, scores

        t_h = max(float(calib.get("temperature_hand", 1.0)), 0.1)
        t_p = max(float(calib.get("temperature_pair", 1.0)), 0.1)
        null_bias = float(calib.get("null_logit_bias", 0.0))

        adj_l = logits_left.astype(np.float32, copy=True) / t_h
        adj_r = logits_right.astype(np.float32, copy=True) / t_h
        adj_l[NO_GESTURE_IDX] += null_bias
        adj_r[NO_GESTURE_IDX] += null_bias

        if present_left and present_right:
            p_l_all = _softmax(adj_l)
            p_r_all = _softmax(adj_r)
            p_pair = _softmax(logits_pair.astype(np.float32) / t_p)
            alpha = float(calib.get("pair_alpha", 0.60))
            two_scores = np.zeros((7,), dtype=np.float32)
            for idx_7 in range(7):
                c30 = 23 + idx_7
                geom_hand = float(np.sqrt(max(p_l_all[c30] * p_r_all[c30], 0.0)))
                two_scores[idx_7] = alpha * float(p_pair[1 + idx_7]) + (1.0 - alpha) * geom_hand
            best_7 = int(np.argmax(two_scores))
            best_two_prob = float(two_scores[best_7])
            if best_two_prob >= float(calib.get("pair_threshold", 0.48)):
                lbl = TWO_HAND_CLASSES[best_7]
                sc = round(min(max(best_two_prob, 0.0), 1.0), 4)
                return {"left": lbl, "right": lbl}, {"left": sc, "right": sc}

        single_candidates: Dict[str, Tuple[Optional[str], float]] = {}
        min_prob = float(calib.get("min_single_prob", 0.32))
        for side, present, adj in (("left", present_left, adj_l), ("right", present_right, adj_r)):
            if not present:
                single_candidates[side] = (None, 0.0)
                continue
            masked = adj.copy()
            masked[23:30] = -1e9
            probs = _softmax(masked)
            best_c = int(np.argmax(probs[:23]))
            best_p = float(probs[best_c])
            if best_p > float(probs[NO_GESTURE_IDX]) and best_p >= min_prob:
                single_candidates[side] = (OFFICIAL_30_CLASSES[best_c], best_p)
            else:
                single_candidates[side] = (None, 0.0)

        if single_candidates["left"][0] is not None and single_candidates["right"][0] is not None:
            sec_min = float(calib.get("second_hand_min_prob", 0.78))
            p_l = single_candidates["left"][1]
            p_r = single_candidates["right"][1]
            if p_l >= p_r and p_r < sec_min:
                single_candidates["right"] = (None, 0.0)
            elif p_r > p_l and p_l < sec_min:
                single_candidates["left"] = (None, 0.0)

        for side in ("left", "right"):
            lbl, pr = single_candidates[side]
            if lbl is not None:
                gestures[side] = lbl
                scores[side] = round(min(max(pr, 0.0), 1.0), 4)
        return gestures, scores

    def populate_persons(
        self,
        persons: List[dict],
        frame: np.ndarray,
        pose_adapter: Optional[OnnxPoseEngineAdapter] = None,
    ) -> int:
        """Populate `hands`, `hand_bboxes`, `hand_gestures`, and `hand_gesture_scores` on `persons` in-place."""
        h, w = frame.shape[:2]
        cached_blob = pose_adapter._last_blob if isinstance(pose_adapter, OnnxPoseEngineAdapter) else None
        cached_meta = pose_adapter._last_meta if isinstance(pose_adapter, OnnxPoseEngineAdapter) else None
        h_boxes, h_scores = self._detect_hands(frame, cached_blob, cached_meta)

        # If pose detector missed a close-up person/arm shot at the main threshold, check 0.10 pose fallback or synthesize 1 person!
        if not persons and len(h_boxes) > 0:
            fb_box = [0.0, 0.0, float(w), float(h)]
            fb_kpts = np.zeros((17, 3), dtype=np.float32)
            fb_score = round(float(h_scores[0]), 2)
            if isinstance(pose_adapter, OnnxPoseEngineAdapter) and pose_adapter._last_rows_frame is not None:
                fboxes, fscores, fkpts = pose_adapter._last_rows_frame
                if len(fboxes) > 0:
                    fb_box = [float(v) for v in fboxes[0]]
                    fb_kpts = fkpts[0]
                    fb_score = round(float(fscores[0]), 2)
            persons.append({
                "id": 1,
                "score": fb_score,
                "box": fb_box,
                "keypoints": fb_kpts,
                "position": [0.0, 0.0],
                "verdict": {"posture": "unknown", "posture_confidence": 0.0, "evidence": {}},
            })

        if not persons:
            return 0

        # Sort primary gesturing person to index 0 when multiple people exist so `persons[0]` (`id: 1`) is the foreground subject!
        if len(persons) > 1:
            def _prominence(p: dict) -> float:
                bx = p["box"]
                area = max(float(bx[2] - bx[0]), 1.0) * max(float(bx[3] - bx[1]), 1.0)
                pcx = 0.5 * float(bx[0] + bx[2])
                cw = 1.0 - 0.35 * abs(pcx - 0.5 * w) / max(0.5 * w, 1.0)
                return area * float(p.get("score", 0.5)) * cw

            persons.sort(key=_prominence, reverse=True)
            for idx_p, p in enumerate(persons, start=1):
                p["id"] = idx_p

        cand_n = min(len(h_boxes), 6)
        h_boxes_c = h_boxes[:cand_n]
        h_scores_c = h_scores[:cand_n]

        side_probs_r: Optional[np.ndarray] = None
        cand_meta: List[Tuple[float, float, float]] = []
        cand_outs = None
        if cand_n > 0:
            blobs = []
            for i in range(cand_n):
                cx, cy, s_px = expand_square_crop(h_boxes_c[i], w, h, expand_ratio=1.35)
                cand_meta.append((cx, cy, s_px))
                blobs.append(crop_imagenet_blob(frame, cx, cy, s_px, 224))
            batch_crops = np.stack(blobs, axis=0)
            with self._lock:
                sx, sy, emb, c_logits, r_logit = self.crop_sess.run(None, {self._crop_in: batch_crops})
            side_probs_r = 1.0 / (1.0 + np.exp(-np.clip(r_logit[:, 0], -20.0, 20.0)))
            cand_outs = (sx, sy, emb, c_logits)

        used_mask = np.zeros((cand_n,), dtype=bool)
        total_found = 0

        for person in persons:
            avail_idx = np.where(~used_mask)[0]
            p_box = np.asarray(person["box"], dtype=np.float32)
            body_k = np.asarray(person["keypoints"], dtype=np.float32)
            if avail_idx.size > 0:
                sub_assign = assign_hands_to_sides(
                    h_boxes_c[avail_idx],
                    h_scores_c[avail_idx],
                    body_k,
                    p_box,
                    w,
                    h,
                    side_probs_right=(side_probs_r[avail_idx] if side_probs_r is not None else None),
                )
                assigned = {
                    s: (int(avail_idx[v]) if v is not None else None)
                    for s, v in sub_assign.items()
                }
            else:
                assigned = {"left": None, "right": None}

            side_data: Dict[str, dict] = {}
            for side in ("left", "right"):
                di = assigned[side]
                if di is None or cand_outs is None:
                    side_data[side] = {
                        "present": False,
                        "bbox": [],
                        "kpts": None,
                        "vis": np.zeros((256,), dtype=np.float32),
                        "geo": np.zeros((118,), dtype=np.float32),
                        "crop_logits": np.zeros((31,), dtype=np.float32),
                    }
                else:
                    used_mask[di] = True
                    total_found += 1
                    coords_01, k_conf = decode_simcc_np(cand_outs[0][di : di + 1], cand_outs[1][di : di + 1])
                    cx, cy, s_px = cand_meta[di]
                    x0, y0 = cx - 0.5 * s_px, cy - 0.5 * s_px
                    pts_px = np.zeros((21, 3), dtype=np.float32)
                    pts_px[:, 0] = np.clip(x0 + coords_01[0, :, 0] * s_px, 0.0, float(w))
                    pts_px[:, 1] = np.clip(y0 + coords_01[0, :, 1] * s_px, 0.0, float(h))
                    pts_px[:, 2] = k_conf[0]
                    bx = h_boxes_c[di]
                    side_data[side] = {
                        "present": True,
                        "bbox": [round(float(v), 2) for v in bx],
                        "bbox_np": bx,
                        "kpts": pts_px,
                        "vis": cand_outs[2][di].astype(np.float32),
                        "geo": hand_geometric_features(pts_px, side, bx, body_k),
                        "crop_logits": cand_outs[3][di].astype(np.float32),
                    }

            both_p = side_data["left"]["present"] and side_data["right"]["present"]
            if both_p:
                bl, br = side_data["left"]["bbox_np"], side_data["right"]["bbox_np"]
                ux1, uy1 = min(float(bl[0]), float(br[0])), min(float(bl[1]), float(br[1]))
                ux2, uy2 = max(float(bl[2]), float(br[2])), max(float(bl[3]), float(br[3]))
                uw = max(ux2 - ux1, 16.0)
                ubox = np.array([ux1, uy1, ux2, min(float(h), uy2 + 0.30 * uw)], dtype=np.float32)
                ucx, ucy, uside = expand_square_crop(ubox, w, h, expand_ratio=1.20)
                ublob = crop_imagenet_blob(frame, ucx, ucy, uside, 224)[None]
                with self._lock:
                    _, _, u_emb, u_clog, _ = self.crop_sess.run(None, {self._crop_in: ublob})
                vis_union = u_emb[0].astype(np.float32)
                clog_union = u_clog[0].astype(np.float32)
                p_geo = pair_relational_features(side_data["left"]["kpts"], bl, side_data["right"]["kpts"], br)
            else:
                vis_union = np.zeros((256,), dtype=np.float32)
                clog_union = np.zeros((31,), dtype=np.float32)
                p_geo = np.zeros((30,), dtype=np.float32)

            if side_data["left"]["present"] or side_data["right"]["present"]:
                feed = {
                    "vis_left": side_data["left"]["vis"][None],
                    "geo_left": side_data["left"]["geo"][None],
                    "crop_logits_left": side_data["left"]["crop_logits"][None],
                    "mask_left": np.array([[1.0 if side_data["left"]["present"] else 0.0]], dtype=np.float32),
                    "vis_right": side_data["right"]["vis"][None],
                    "geo_right": side_data["right"]["geo"][None],
                    "crop_logits_right": side_data["right"]["crop_logits"][None],
                    "mask_right": np.array([[1.0 if side_data["right"]["present"] else 0.0]], dtype=np.float32),
                    "vis_union": vis_union[None],
                    "pair_geo": p_geo[None],
                    "crop_logits_union": clog_union[None],
                    "mask_pair": np.array([[1.0 if both_p else 0.0]], dtype=np.float32),
                }
                with self._lock:
                    ll, lr, lp = self.fuse_sess.run(None, feed)
                gestures, scores = self._decode_gestures(
                    ll[0], lr[0], lp[0],
                    side_data["left"]["present"], side_data["right"]["present"],
                    self.calib,
                )
            else:
                gestures = {"left": None, "right": None}
                scores = {"left": None, "right": None}

            person["hands"] = {
                "left": side_data["left"]["kpts"],
                "right": side_data["right"]["kpts"],
            }
            person["hand_bboxes"] = {
                "left": side_data["left"]["bbox"],
                "right": side_data["right"]["bbox"],
            }
            person["hand_gestures"] = gestures
            person["hand_gesture_scores"] = scores

        return total_found
