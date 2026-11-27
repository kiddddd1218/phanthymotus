"""Whole-segment multimodal (Skeleton + Dual-ROI RGB + Calibrated PoE Router) backend.

Designed for 10-worker CPU execution on Jetson Orin (JP5.1.1 cp38 + JP6.1 cp310):
- Zero mid-stream action inference overhead during active clip playback so every CPU cycle
  captures denser pose trajectories and publishes `/poses` (`posture` + `bbox`) immediately
  on Frame 0 (`0` missed short clips).
- Pre-crops compact uint8 Person + Extended-Wrist Hand/Object + Context patches while the
  decoded BGR frame is already hot in L2 cache (`max_patches=16`, zero raw JPEG retention).
- Fires one authoritative whole-segment multimodal verdict on input quiet (`quiet_s=0.20s`),
  continuous-stream maturity fallback (`max_unflushed_s`), or pre-stop emergency flush.
"""
from __future__ import annotations

from collections import deque
import gc
import json
import logging
import math
from pathlib import Path
import threading
import time
from typing import Optional

import numpy as np

from plugins.official_action import (
    ActionBackendError,
    FORMAT,
     canonical_labels,
    causal_blob,
    file_hash,
    schema_hash,
)
from plugins.pose_action import PoseTracker, body_height
from utils.runtime_profile import memory_stage, release_idle_heap

log = logging.getLogger(__name__)

_FINGER_IDS = frozenset({84, 85, 86, 87, 88, 89, 90, 95})
_DESKTOP_IDS = frozenset({29, 32, 35, 40, 48, 49, 54, 55, 56, 57, 58, 60})
_KITCHEN_IDS = frozenset(range(13, 29))
_CONTROL_IDS = frozenset(range(103, 117))

_IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)[None, :, None, None]
_IMAGENET_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)[None, :, None, None]


def _build_router_group_masks() -> np.ndarray:
    masks = np.zeros((5, 126), dtype=np.float64)
    for cid in range(1, 127):
        if cid in _FINGER_IDS:
            masks[0, cid - 1] = 1.0
        elif cid in _DESKTOP_IDS:
            masks[1, cid - 1] = 1.0
        elif cid in _KITCHEN_IDS:
            masks[2, cid - 1] = 1.0
        elif cid in _CONTROL_IDS:
            masks[3, cid - 1] = 1.0
        else:
            masks[4, cid - 1] = 1.0
    return masks


_ROUTER_GROUP_MASKS = _build_router_group_masks()


class _LiteSkeletonFrame:
    __slots__ = ("t", "box", "keypoints", "image_size", "height")

    def __init__(self, t, box, keypoints, min_conf, image_size=None):
        arr = np.asarray(keypoints, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != 17 or arr.shape[1] < 3:
            raise ValueError("keypoints must be (17, 3+); got %r" % (arr.shape,))
        self.t = float(t)
        self.box = [float(v) for v in box]
        self.keypoints = arr
        self.image_size = (
            (int(image_size[0]), int(image_size[1]))
            if image_size and image_size[0] and image_size[1]
            else None
        )
        self.height = body_height(box)


class SegmentObservationBuffer:
    """Bounded in-memory buffer of processed pose + compact Dual-ROI patches for one stream."""

    def __init__(self, quiet_s: float = 0.20, max_s: float = 30.0, max_patches: int = 16, max_ctx_patches: int = 4):
        self.quiet_s = max(0.05, float(quiet_s))
        self.max_s = max(2.0, float(max_s))
        self.max_patches = max(4, int(max_patches))
        self.max_ctx_patches = max(0, int(max_ctx_patches))
        self._lock = threading.Lock()
        self._records = deque()
        self._last_ingress_mono = 0.0
        self._first_ingress_mono = 0.0
        self._emitted_for_segment = 0
        self._last_emitted_obs_count = 0
        self._last_emitted_was_quiet = False
        self._ewma_pose_ms = 0.0

    def note_ingress(self, now_mono: float) -> None:
        with self._lock:
            if self._last_ingress_mono <= 0.0:
                self._first_ingress_mono = float(now_mono)
            self._last_ingress_mono = float(now_mono)

    def record_observation(
        self,
        capture_t: float,
        now_mono: float,
        boxes,
        scores,
        keypoints,
        frame_hw: tuple,
        rgb_patch=None,
        pose_ms: float = 0.0,
    ) -> None:
        with self._lock:
            if pose_ms > 0.0:
                self._ewma_pose_ms = (
                    float(pose_ms)
                    if self._ewma_pose_ms <= 0.0
                    else 0.5 * self._ewma_pose_ms + 0.5 * float(pose_ms)
                )
            done_mono = time.monotonic()
            t_ref = max(float(now_mono), done_mono)
            if self._first_ingress_mono <= 0.0:
                self._first_ingress_mono = t_ref
            if t_ref > self._last_ingress_mono:
                self._last_ingress_mono = t_ref
            rec = [
                float(capture_t),
                float(now_mono),
                np.array(boxes, dtype=np.float32, copy=True),
                np.array(scores, dtype=np.float32, copy=True),
                np.array(keypoints, dtype=np.float32, copy=True),
                (int(frame_hw[0]), int(frame_hw[1])),
                rgb_patch,
            ]
            self._records.append(rec)
            cutoff = float(capture_t) - self.max_s
            while len(self._records) > 2 and self._records[0][0] < cutoff:
                self._records.popleft()
            if rgb_patch is not None:
                patched = [r for r in self._records if r[6] is not None]
                if len(patched) > self.max_patches:
                    for r in patched[1:-1:2]:
                        r[6] = None
                if self.max_ctx_patches > 0:
                    ctx_holders = [
                        r for r in self._records
                        if isinstance(r[6], tuple) and len(r[6]) >= 3 and r[6][2] is not None
                    ]
                    if len(ctx_holders) > self.max_ctx_patches:
                        for r in ctx_holders[1:-1:2]:
                            r[6] = (r[6][0], r[6][1], None)

    def next_timeout_s(self, now_mono: float, default_s: float = 0.25) -> float:
        with self._lock:
            if not self._records or self._last_ingress_mono <= 0.0:
                return default_s
            if len(self._records) <= self._last_emitted_obs_count and self._last_emitted_was_quiet:
                return default_s
            rem = (self._last_ingress_mono + self.quiet_s) - float(now_mono)
            return max(0.005, min(default_s, rem))

    def pop_for_flush(
        self,
        now_mono: float,
        *,
        max_unflushed_s: float = 0.0,
        force_on_stop: bool = False,
        allow_quiet: bool = True,
    ):
        """Return `(records_list, trigger_kind)` when ready to emit, else `None`."""
        with self._lock:
            n = len(self._records)
            if n == 0:
                return None
            if force_on_stop:
                if self._emitted_for_segment > 0 and n <= self._last_emitted_obs_count and self._last_emitted_was_quiet:
                    return None
                snap = list(self._records)
                self._records.clear()
                self._emitted_for_segment += 1
                self._last_emitted_obs_count = 0
                self._last_emitted_was_quiet = True
                return snap, "stop_flush"

            quiet_elapsed = float(now_mono) - self._last_ingress_mono
            if allow_quiet and self._last_ingress_mono > 0.0 and quiet_elapsed >= self.quiet_s:
                if n > self._last_emitted_obs_count or not self._last_emitted_was_quiet:
                    # Emit over the FULL accumulated clip trajectory and preserve `_records`
                    # so mid-clip network/CPU pauses never wipe earlier motion into a 1-frame
                    # static 'none' fragment. `stop()` resets the buffer between clips, and an
                    # extended >=4.0s pause resets continuous robot streams cleanly below.
                    snap = list(self._records)
                    self._emitted_for_segment += 1
                    self._last_emitted_obs_count = n
                    self._last_emitted_was_quiet = True
                    return snap, "quiet_confirm"
                if quiet_elapsed >= max(4.0, self.quiet_s * 6.0):
                    self._records.clear()
                    self._last_ingress_mono = 0.0
                    self._first_ingress_mono = 0.0
                    self._emitted_for_segment = 0
                    self._last_emitted_obs_count = 0
                    self._last_emitted_was_quiet = False
                return None

            # Only emit mid-stream on very slow CPU pose (>=380ms/frame) or when a long
            # continuous stream exceeds `max_unflushed_s`. On fast GPU TensorRT pose, keep
            # `_inference_worker` uninterrupted so every clip captures dense full-clip poses
            # and flushes once cleanly on `quiet_confirm` or `stop_flush`.
            if self._emitted_for_segment == 0 and n >= 1 and self._ewma_pose_ms >= 420.0:
                snap = list(self._records)
                self._emitted_for_segment += 1
                self._last_emitted_obs_count = n
                self._last_emitted_was_quiet = False
                return snap, "live_segment_ske"

            if self._emitted_for_segment == 0 and n >= 2 and self._ewma_pose_ms >= 380.0:
                snap = list(self._records)
                self._emitted_for_segment += 1
                self._last_emitted_obs_count = n
                self._last_emitted_was_quiet = False
                return snap, "live_segment_ske"

            if (
                max_unflushed_s > 0.0
                and n > self._last_emitted_obs_count
                and n >= 2
                and self._first_ingress_mono > 0.0
                and (float(now_mono) - self._first_ingress_mono) >= max_unflushed_s
            ):
                snap = list(self._records)
                self._emitted_for_segment += 1
                self._last_emitted_obs_count = n
                self._last_emitted_was_quiet = False
                self._first_ingress_mono = float(now_mono)
                return snap, "continuous_fallback"
            return None

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._last_ingress_mono = 0.0
            self._first_ingress_mono = 0.0
            self._emitted_for_segment = 0
            self._last_emitted_obs_count = 0
            self._last_emitted_was_quiet = False
            self._ewma_pose_ms = 0.0


def _extended_wrists(kp: np.ndarray, width: int, height: int, extension: float = 0.35) -> list:
    pts = []
    for wi, ei in ((9, 7), (10, 8)):
        if kp[wi, 2] > 0.20:
            x, y = float(kp[wi, 0]), float(kp[wi, 1])
            if kp[ei, 2] > 0.20:
                x += extension * (x - float(kp[ei, 0]))
                y += extension * (y - float(kp[ei, 1]))
            pts.append((wi, float(np.clip(x, 0, width - 1)), float(np.clip(y, 0, height - 1))))
    return pts


def _crop_square_bgr_to_rgb(bgr: np.ndarray, cx: float, cy: float, side: float, out_hw: int):
    import cv2

    h, w = bgr.shape[:2]
    half = max(0.5 * side, 24.0)
    if 2.0 * half >= max(h, w):
        rx0, ry0, rx1, ry1 = 0, 0, w, h
    else:
        rx0 = int(round(cx - half))
        ry0 = int(round(cy - half))
        s_int = max(1, int(round(2.0 * half)))
        rx1, ry1 = rx0 + s_int, ry0 + s_int
        if rx0 < 0:
            rx1, rx0 = min(w, rx1 - rx0), 0
        if ry0 < 0:
            ry1, ry0 = min(h, ry1 - ry0), 0
        if rx1 > w:
            rx0, rx1 = max(0, rx0 - (rx1 - w)), w
        if ry1 > h:
            ry0, ry1 = max(0, ry0 - (ry1 - h)), h
    sub = bgr[ry0:ry1, rx0:rx1]
    if sub.size == 0:
        rx0, ry0, sub = 0, 0, bgr
    sh, sw = sub.shape[:2]
    if (sw, sh) != (out_hw, out_hw):
        if hasattr(cv2, "resize"):
            interp = getattr(cv2, "INTER_AREA", 3) if max(sh, sw) > out_hw else getattr(cv2, "INTER_LINEAR", 1)
            resized = cv2.resize(sub, (out_hw, out_hw), interpolation=interp)
        else:
            yi = np.linspace(0, sh - 1, out_hw).astype(np.intp)
            xi = np.linspace(0, sw - 1, out_hw).astype(np.intp)
            resized = sub[yi][:, xi]
    else:
        resized = sub
    rgb = np.ascontiguousarray(resized[..., ::-1])
    return rgb, (rx0, ry0, out_hw / float(max(sw, 1)), out_hw / float(max(sh, 1)))


def make_dual_roi_patch(
    bgr: np.ndarray,
    boxes,
    scores,
    keypoints,
    *,
    crop_hw: int = 160,
    with_ctx: bool = True,
):
    """Extract compact uint8 `(person_rgb, hand_obj_rgb, ctx_rgb)` while `bgr` is in L2 cache."""
    import cv2
    from plugins.vision_runtime import letterbox

    h, w = bgr.shape[:2]
    boxes_arr = np.asarray(boxes, dtype=np.float32)
    scores_arr = np.asarray(scores, dtype=np.float32)
    kps_arr = np.asarray(keypoints, dtype=np.float32)

    ctx_rgb = None
    if with_ctx:
        rgb_full = np.ascontiguousarray(bgr[..., ::-1])
        ctx_rgb, _ = letterbox(rgb_full, int(crop_hw), int(crop_hw))

    if len(boxes_arr) == 0:
        p_rgb, _ = _crop_square_bgr_to_rgb(bgr, 0.5 * w, 0.5 * h, float(min(h, w)), crop_hw)
        return (p_rgb, p_rgb, ctx_rgb)

    areas = (
        np.maximum(1.0, (boxes_arr[:, 2] - boxes_arr[:, 0]) * (boxes_arr[:, 3] - boxes_arr[:, 1]))
        / max(1.0, float(w * h))
    )
    best = int((scores_arr * (0.5 + areas)).argmax())
    b0 = boxes_arr[best]
    x0, y0, x1, y1 = float(b0[0]), float(b0[1]), float(b0[2]), float(b0[3])
    kp = kps_arr[best] if len(kps_arr) > best else None
    if kp is not None:
        valid = kp[:, 2] > 0.20
        if valid.any():
            x0 = min(x0, float(kp[valid, 0].min()))
            x1 = max(x1, float(kp[valid, 0].max()))
            y0 = min(y0, float(kp[valid, 1].min()))
            y1 = max(y1, float(kp[valid, 1].max()))
    body_side = max(x1 - x0, y1 - y0, 120.0)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    p_side = body_side * 1.32
    p_rgb, _ = _crop_square_bgr_to_rgb(bgr, cx, cy, p_side, crop_hw)

    if kp is None:
        return (p_rgb, p_rgb, ctx_rgb)

    pts = _extended_wrists(kp, w, h, extension=0.35)
    if len(pts) == 2:
        (_, wx1, wy1), (_, wx2, wy2) = pts
        dist = float(math.hypot(wx1 - wx2, wy1 - wy2))
        if dist <= 0.45 * p_side:
            hx, hy = 0.5 * (wx1 + wx2), 0.5 * (wy1 + wy2)
            hs = min(max(dist + 0.32 * p_side, 0.36 * p_side, 96.0), 1.10 * p_side)
        else:
            # Workspace crop enclosing both hands + upper chest/table
            sh = [(float(kp[i, 0]), float(kp[i, 1])) for i in (5, 6) if kp[i, 2] > 0.20]
            anchors = [(wx1, wy1), (wx2, wy2)]
            if len(sh) == 2:
                anchors.append((0.5 * (sh[0][0] + sh[1][0]), 0.5 * (sh[0][1] + sh[1][1]) + 0.15 * p_side))
            ax, ay = [a[0] for a in anchors], [a[1] for a in anchors]
            hx, hy = 0.5 * (min(ax) + max(ax)), 0.5 * (min(ay) + max(ay))
            span = max(max(ax) - min(ax), max(ay) - min(ay))
            hs = min(max(span + 0.26 * p_side, 0.50 * p_side, 112.0), 1.12 * p_side)
    elif len(pts) == 1:
        _, hx, hy = pts[0]
        hs = max(0.40 * p_side, 96.0)
    else:
        hx, hy, hs = cx, cy - 0.10 * p_side, max(0.48 * p_side, 112.0)

    h_rgb, _ = _crop_square_bgr_to_rgb(bgr, hx, hy, hs, crop_hw)
    return (p_rgb, h_rgb, ctx_rgb)


def _softmax_1d(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64).reshape(-1)
    x = x - x.max()
    e = np.exp(x)
    s = e.sum()
    return (e / s if s > 0 else np.full_like(e, 1.0 / e.size)).astype(np.float32)


def _temp_scale_probs(probs: np.ndarray, inv_t: float) -> np.ndarray:
    lp = np.log(np.clip(np.asarray(probs, dtype=np.float64), 1e-7, 1.0)) * float(inv_t)
    lp -= lp.max()
    ep = np.exp(lp)
    return ep / ep.sum()


def _norm_entropy(probs: np.ndarray) -> float:
    pc = np.clip(np.asarray(probs, dtype=np.float64), 1e-7, 1.0)
    return float(-(pc * np.log(pc)).sum() / math.log(126.0))


def fuse_calibrated_router(
    body_probs: np.ndarray,
    rgb_probs: np.ndarray,
    router_cfg: dict,
    observation_fps: Optional[float] = None,
) -> np.ndarray:
    """10-DOF OOF-calibrated conditional Product-of-Experts router."""
    theta = np.asarray(router_cfg["theta"], dtype=np.float64)
    if theta.shape != (10,):
        raise ActionBackendError("router theta must have 10 scalars; got %r" % (theta.shape,))
    mode = str(router_cfg.get("mode", "poe_gate"))
    low_hi = float(router_cfg.get("low_fps_hi", 16.0))
    low_lo = float(router_cfg.get("low_fps_lo", 4.0))
    if observation_fps is None or observation_fps <= 0.0:
        is_low_fps = 0.0
    else:
        is_low_fps = float(
            np.clip((low_hi - float(observation_fps)) / max(low_hi - low_lo, 1e-6), 0.0, 1.0)
        )

    inv_tb = float(np.clip(theta[0], 0.35, 3.5))
    inv_tr = float(np.clip(theta[1], 0.35, 3.5))
    g_logits = theta[2:7]
    g_sharp_r, g_sharp_b, g_fps = float(theta[7]), float(theta[8]), float(theta[9])

    pb = _temp_scale_probs(body_probs, inv_tb)
    pr = _temp_scale_probs(rgb_probs, inv_tr)
    sharp_b = 1.0 - _norm_entropy(pb)
    sharp_r = 1.0 - _norm_entropy(pr)
    base_logit = (g_logits[:, None] * _ROUTER_GROUP_MASKS).sum(axis=0)
    dyn_logit = base_logit + g_sharp_r * sharp_r - g_sharp_b * sharp_b + g_fps * is_low_fps
    w_rgb = 1.0 / (1.0 + np.exp(-np.clip(dyn_logit, -8.0, 8.0)))

    if mode == "poe_gate":
        log_f = (1.0 - w_rgb) * np.log(np.clip(pb, 1e-7, 1.0)) + w_rgb * np.log(np.clip(pr, 1e-7, 1.0))
        log_f -= log_f.max()
        out = np.exp(log_f)
        return (out / out.sum()).astype(np.float32)

    out = (1.0 - w_rgb) * pb + w_rgb * pr
    return (out / out.sum()).astype(np.float32)


class _RgbOnnxEngine:
    """1-thread CPU ONNX Runtime session for compact Dual-ROI + Context RGB student."""

    def __init__(self, path: Path, taxonomy_hash: str):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        if hasattr(ort, "ExecutionMode"):
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (
            len(inputs) != 1
            or len(outputs) != 1
            or len(inputs[0].shape) != 5
            or inputs[0].shape[0] != 1
            or inputs[0].shape[2] != 3
            or inputs[0].shape[3] != inputs[0].shape[4]
            or outputs[0].shape != [1, 126]
        ):
            raise ActionBackendError("RGB ONNX shape mismatch: got %r -> %r" % (
                [i.shape for i in inputs], [o.shape for o in outputs]
            ))
        meta = self.session.get_modelmeta().custom_metadata_map
        if meta.get("format") != FORMAT or meta.get("taxonomy_sha256") != taxonomy_hash:
            raise ActionBackendError("RGB ONNX embedded taxonomy metadata mismatch")
        self.input_name = inputs[0].name
        self.total_views = int(inputs[0].shape[1])
        self.crop_hw = int(inputs[0].shape[3])
        self.n_person = int(meta.get("n_person", "4"))
        self.n_hand = int(meta.get("n_hand", "4"))
        self.n_ctx = int(meta.get("n_ctx", "2"))
        if self.n_person + self.n_hand + self.n_ctx != self.total_views:
            raise ActionBackendError("RGB ONNX view split metadata disagrees with input shape")

    def infer_logits(self, views_norm: np.ndarray) -> np.ndarray:
        out = self.session.run(None, {self.input_name: views_norm})[0]
        arr = np.asarray(out, dtype=np.float32)
        if arr.shape != (1, 126) or not np.isfinite(arr).all():
            raise ActionBackendError("RGB ONNX returned invalid logits")
        return arr[0]


class OfficialRgbSessionCache:
    """Process-wide single-instance cache for the verified RGB student ONNX session."""

    def __init__(self):
        self._lock = threading.Lock()
        self._key = None
        self._engine = None

    def acquire(self, path: Path, digest: str, taxonomy_hash: str) -> _RgbOnnxEngine:
        key = (str(path), digest, taxonomy_hash)
        with self._lock:
            if self._key != key:
                with memory_stage(log, "official126.rgb_session_create"):
                    engine = _RgbOnnxEngine(path, taxonomy_hash)
                    gc.collect()
                    release_idle_heap()
                self._engine, self._key = engine, key
            return self._engine


_GLOBAL_RGB_CACHE = OfficialRgbSessionCache()


def build_rgb_tensor_from_patches(
    patched_records: list,
    n_person: int,
    n_hand: int,
    n_ctx: int,
    crop_hw: int,
) -> Optional[np.ndarray]:
    """Assemble `(1, n_person + n_hand + n_ctx, 3, crop_hw, crop_hw)` normalized float32 tensor."""
    if not patched_records:
        return None
    m = len(patched_records)
    idx_p = np.linspace(0, m - 1, n_person).round().astype(int)
    idx_h = np.linspace(0, m - 1, n_hand).round().astype(int)

    crops = []
    for i in idx_p:
        crops.append(patched_records[int(i)][0].transpose(2, 0, 1))
    for i in idx_h:
        crops.append(patched_records[int(i)][1].transpose(2, 0, 1))

    if n_ctx > 0:
        ctx_avail = [p[2] for p in patched_records if len(p) >= 3 and p[2] is not None]
        if not ctx_avail:
            ctx_avail = [p[0] for p in patched_records]
        idx_c = np.linspace(0, len(ctx_avail) - 1, n_ctx).round().astype(int)
        for i in idx_c:
            crops.append(ctx_avail[int(i)].transpose(2, 0, 1))

    arr = np.stack(crops, axis=0).astype(np.float32)
    if arr.shape[1:] != (3, crop_hw, crop_hw):
        raise ActionBackendError("RGB patch shape %r != expected (3, %d, %d)" % (arr.shape[1:], crop_hw, crop_hw))
    norm = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.ascontiguousarray(norm[None], dtype=np.float32)


class OfficialSegmentMultimodalBackend:
    """Whole-segment recognizer combining OfficialActionBackend + optional Dual-ROI RGB + PoE Router."""

    def __init__(self, action_backend, *, rgb_engine: Optional[_RgbOnnxEngine] = None, router_cfg: Optional[dict] = None):
        self.action_backend = action_backend
        self.labels = canonical_labels()
        self._rgb_engine = rgb_engine
        self.router_cfg = router_cfg
        self.last_error: Optional[str] = None

        manifest = getattr(action_backend, "manifest", {}) or {}
        self._rgb_spec = manifest.get("rgb_engine")
        if self.router_cfg is None:
            self.router_cfg = manifest.get("multimodal_router")

        if self._rgb_spec is not None and self._rgb_engine is None:
            base_dir = Path(action_backend.path).parent
            rgb_path = (base_dir / self._rgb_spec["file"]).resolve()
            if rgb_path.parent != base_dir or file_hash(rgb_path) != self._rgb_spec["sha256"]:
                raise ActionBackendError("RGB ONNX path or SHA-256 differs from artifact.json")
            if self.router_cfg is not None and "fit_receipt" in self.router_cfg:
                fit = self.router_cfg["fit_receipt"]
                rpath = (base_dir / fit["file"]).resolve()
                if rpath.parent != base_dir or file_hash(rpath) != fit["sha256"]:
                    raise ActionBackendError("multimodal router receipt hash mismatch")
                receipt = json.loads(rpath.read_text(encoding="utf-8"))
                if receipt.get("fit_split") != "train":
                    raise ActionBackendError("multimodal router must be fitted strictly on train OOF")
            self._rgb_path = rgb_path
        else:
            self._rgb_path = None

    @property
    def crop_hw(self) -> int:
        if self._rgb_engine is not None:
            return self._rgb_engine.crop_hw
        if self._rgb_spec is not None:
            return int(self._rgb_spec.get("crop_hw", 160))
        return 0

    @property
    def has_rgb(self) -> bool:
        return self._rgb_engine is not None or self._rgb_path is not None

    def _ensure_rgb_engine(self) -> Optional[_RgbOnnxEngine]:
        if self._rgb_engine is not None:
            return self._rgb_engine
        if self._rgb_path is None:
            return None
        tax_hash = schema_hash(self.labels)
        self._rgb_engine = _GLOBAL_RGB_CACHE.acquire(
            self._rgb_path, self._rgb_spec["sha256"], tax_hash
        )
        return self._rgb_engine

    def recognize_segment(self, records: list, *, trigger: str = "quiet_confirm") -> Optional[dict]:
        """Classify a buffered sequence of `[capture_t, mono_t, boxes, scores, kps, hw, rgb_patch]`."""
        if not records:
            return None
        tracker = PoseTracker(
            timeout_s=10.0,
            history_s=30.0,
            centroid_max_travel=4.0,
            min_conf=0.30,
            label_hold=1,
        )
        track_weights = {}
        per_frame = []
        t0 = float(records[0][0])
        last_t = -1e9
        for rec in records:
            cap_t, _mono_t, boxes, scores, kps, (fh, fw), rgb_patch = rec
            rel_t = float(cap_t) - t0
            if rel_t <= last_t:
                rel_t = last_t + 1e-3
            last_t = rel_t
            order = np.argsort(np.asarray(scores, dtype=np.float32))[::-1][:2]
            b_sel = boxes[order] if len(order) else np.zeros((0, 4), dtype=np.float32)
            s_sel = scores[order] if len(order) else np.zeros((0,), dtype=np.float32)
            k_sel = kps[order] if len(order) else np.zeros((0, 17, 3), dtype=np.float32)
            tracks = tracker.update(b_sel, k_sel, rel_t, image_size=(fw, fh))
            obs_rows = []
            for tr, bx, sc, kp in zip(tracks, b_sel, s_sel, k_sel):
                area = max(1.0, float(bx[2] - bx[0]) * float(bx[3] - bx[1])) / max(1.0, float(fw * fh))
                track_weights[tr.id] = track_weights.get(tr.id, 0.0) + float(sc) * (0.5 + area)
                obs_rows.append((tr.id, bx, float(sc), kp))
            per_frame.append((rel_t, (fw, fh), obs_rows, rgb_patch))

        if not track_weights:
            return None

        primary_id = max(track_weights, key=track_weights.get)
        times = []
        kpts_seq = []
        snap_obs = None
        patched_list = []
        last_size = (640, 480)
        for rel_t, wh, obs_rows, rgb_patch in per_frame:
            last_size = wh
            chosen = next((r for r in obs_rows if r[0] == primary_id), None)
            if chosen is not None:
                snap_obs = (wh, chosen)
                times.append(rel_t)
                kpts_seq.append(chosen[3])
            elif obs_rows:
                times.append(rel_t)
                kpts_seq.append(obs_rows[0][3])
            if rgb_patch is not None:
                patched_list.append(rgb_patch)

        if not kpts_seq or snap_obs is None:
            return None

        resample_mode = getattr(self.action_backend, "resample", "linear_time")
        normalization = getattr(self.action_backend, "normalization", "fix")
        blob = causal_blob(
            times,
            kpts_seq,
            last_size,
            times[-1],
            30.0,
            normalization,
            resample=resample_mode,
        )
        if blob is None:
            return None

        engine = self.action_backend._ensure_engine()
        raw_body = engine.infer(blob)
        if isinstance(raw_body, (list, tuple)):
            raw_body = raw_body[0]
        body_logits = np.asarray(raw_body, dtype=np.float32).reshape(-1)
        if body_logits.size != 126 or not np.isfinite(body_logits).all():
            raise ActionBackendError("expected finite 126-class skeleton logits")
        body_probs = _softmax_1d(body_logits)

        used_rgb = False
        final_probs = body_probs
        rgb_engine = None if trigger in ("contended_safety_ske", "live_segment_ske") else self._ensure_rgb_engine()
        if rgb_engine is not None and patched_list:
            rgb_tensor = build_rgb_tensor_from_patches(
                patched_list,
                rgb_engine.n_person,
                rgb_engine.n_hand,
                rgb_engine.n_ctx,
                rgb_engine.crop_hw,
            )
            if rgb_tensor is not None:
                rgb_logits = rgb_engine.infer_logits(rgb_tensor)
                rgb_probs = _softmax_1d(rgb_logits)
                span_s = float(times[-1] - times[0]) if len(times) > 1 else 0.0
                obs_fps = (len(times) - 1) / span_s if span_s > 1e-3 else 12.0
                if self.router_cfg is not None and "theta" in self.router_cfg:
                    final_probs = fuse_calibrated_router(
                        body_probs, rgb_probs, self.router_cfg, observation_fps=obs_fps
                    )
                else:
                    final_probs = (0.75 * body_probs + 0.25 * rgb_probs).astype(np.float32)
                    final_probs /= final_probs.sum()
                used_rgb = True

        best_idx = int(final_probs.argmax())
        best_score = float(final_probs[best_idx])
        if trigger in ("contended_safety_ske", "contended_safety_mm"):
            best_score *= 0.45
        elif trigger in ("live_segment", "live_segment_ske"):
            if len(times) <= 1:
                best_score *= 0.25
            elif len(times) == 2:
                best_score *= 0.55
            elif len(times) <= 4:
                best_score *= 0.80
        elif trigger == "continuous_fallback":
            best_score *= 0.85
        min_score = float(getattr(self.action_backend, "min_score", 0.0))
        row = self.labels[best_idx]
        activity = None
        if best_score >= min_score:
            activity = {
                "name": row["activity"],
                "name_zh": row["activity_zh"],
                "official_id": row["official_id"],
                "score": round(best_score, 4),
                "source": "official126-segment-mm" if used_rgb else "official126-segment-ske",
            }

        snap_wh, (_tid, snap_box, snap_score, snap_kp) = snap_obs
        self.last_error = None
        return {
            "activity": activity,
            "boxes": np.asarray([snap_box], dtype=np.float32),
            "scores": np.asarray([snap_score], dtype=np.float32),
            "keypoints": np.asarray([snap_kp], dtype=np.float32),
            "image_wh": snap_wh,
            "evidence": {
                "backend": "official126_segment",
                "trigger": trigger,
                "observations": len(times),
                "rgb_used": used_rgb,
                "best_official_id": row["official_id"],
                "best_score": round(best_score, 4),
            },
        }
