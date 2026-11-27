"""Tests for v4 whole-segment multimodal recognition, instant Frame-0 /poses, and stop-flush."""
from __future__ import annotations

import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import numpy as np

from vision_stubs import _FakeCompressedImage, _wait_until, frame_bytes
from test_pose_plugin import _plugin
from plugins.official_action import (
    ActionBackendError,
    FORMAT,
    INPUT_SHAPE,
    OfficialActionBackend,
    canonical_labels,
    file_hash,
    schema_hash,
)
from plugins.official_segment import (
    OfficialSegmentMultimodalBackend,
    fuse_calibrated_router,
    make_dual_roi_patch,
)


class _FakeBodyEngine:
    input_shape = INPUT_SHAPE

    def __init__(self, target_idx: int = 0):
        self.calls = 0
        self.logits = np.zeros((1, 126), dtype=np.float32)
        self.logits[0, target_idx] = 6.0

    def infer(self, blob):
        self.calls += 1
        return [self.logits.copy()]


class _FakeRgbEngine:
    crop_hw = 160
    n_person = 4
    n_hand = 4
    n_ctx = 2

    def __init__(self, target_idx: int = 10):
        self.calls = 0
        self.logits = np.zeros((126,), dtype=np.float32)
        self.logits[target_idx] = 8.0

    def infer_logits(self, views_norm: np.ndarray) -> np.ndarray:
        self.calls += 1
        assert views_norm.shape == (1, 10, 3, 160, 160), views_norm.shape
        return self.logits.copy()


class OfficialSegmentTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "model.onnx").write_bytes(b"fake-body-onnx")
        labels = canonical_labels()
        manifest = {
            "format": FORMAT,
            "classes": labels,
            "taxonomy_sha256": schema_hash(labels),
            "input_shape": list(INPUT_SHAPE),
            "output_shape": [1, 126],
            "output_kind": "logits",
            "preprocessing": {
                "normalization": "fix",
                "order": "normalize_then_resample",
                "resample": "linear_time",
                "joint_channels": "COCO17 x/y/confidence",
                "person_slots": "primary_then_zero",
            },
            "engine": {
                "kind": "onnx_cpu",
                "file": "model.onnx",
                "sha256": file_hash(self.root / "model.onnx"),
            },
        }
        (self.root / "artifact.json").write_text(json.dumps(manifest))

    def test_segment_mode_publishes_frame0_pose_without_early_action_then_flushes_on_quiet(self):
        body_engine = _FakeBodyEngine(target_idx=0)  # official_id=1 "drink water"
        rgb_engine = _FakeRgbEngine(target_idx=0)
        plugin, _ = _plugin(
            {
                "action_backend": "official126",
                "official_action_manifest": str(self.root / "artifact.json"),
                "activity_mode": "segment",
                "activity_segment_pause_s": 0.12,
                "activity_max_unflushed_s": 5.0,
                "fps": 24,
                "max_persons": 1,
                "label_hold": 1,
            }
        )
        with patch("plugins.official_action._OnnxEngine", return_value=body_engine):
            plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
            node = plugin._nodes["/cam/rgb"]
            node._segment_backend._rgb_engine = rgb_engine

            # Send 3 frames rapidly
            for _ in range(3):
                msg = _FakeCompressedImage(frame_bytes())
                node._image_cb(msg)
                time.sleep(0.05)

            assert _wait_until(lambda: len(node._pub.messages) >= 1, timeout=2.0)
            first_payload = json.loads(node._pub.messages[0])
            self.assertGreaterEqual(first_payload["count"], 1)
            self.assertNotIn("activity", first_payload["persons"][0])

            # Wait for quiet_s (0.12s) final whole-clip verdict (short <1.2s GPU clip avoids wasted 2-frame flush)
            assert _wait_until(
                lambda: body_engine.calls >= 1
                and any(
                    "activity" in json.loads(m)["persons"][0]
                    for m in node._pub.messages
                    if json.loads(m)["persons"]
                ),
                timeout=2.0,
            )
            action_msgs = [
                json.loads(m)
                for m in node._pub.messages
                if json.loads(m)["persons"] and "activity" in json.loads(m)["persons"][0]
            ]
            self.assertEqual(len(action_msgs), 1)
            self.assertEqual(action_msgs[-1]["persons"][0]["activity"], "drink water")
            self.assertEqual(body_engine.calls, 1)
            self.assertGreaterEqual(rgb_engine.calls, 1)
            plugin.dispatch("pose", {"action": "stop"})

    def test_pre_stop_emergency_flush_emits_verdict_before_worker_exits(self):
        body_engine = _FakeBodyEngine(target_idx=1)  # "eat meal"
        plugin, _ = _plugin(
            {
                "action_backend": "official126",
                "official_action_manifest": str(self.root / "artifact.json"),
                "activity_mode": "segment",
                "activity_segment_pause_s": 5.0,  # Long pause so only stop-flush can fire
                "activity_max_unflushed_s": 30.0,
                "fps": 24,
                "max_persons": 1,
                "label_hold": 1,
            }
        )
        with patch("plugins.official_action._OnnxEngine", return_value=body_engine):
            plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
            node = plugin._nodes["/cam/rgb"]
            pub_ref = node._pub
            node._image_cb(_FakeCompressedImage(frame_bytes()))
            assert _wait_until(lambda: len(pub_ref.messages) >= 1, timeout=2.0)
            self.assertEqual(body_engine.calls, 0)

            # Immediately stop before quiet_s=5.0s elapses -> emergency stop flush must fire!
            plugin.dispatch("pose", {"action": "stop"})
            action_msgs = [
                json.loads(m)
                for m in pub_ref.messages
                if json.loads(m)["persons"] and "activity" in json.loads(m)["persons"][0]
            ]
            self.assertEqual(len(action_msgs), 1)
            self.assertEqual(action_msgs[0]["persons"][0]["activity"], "eat meal")
            self.assertEqual(body_engine.calls, 1)

    def test_continuous_fallback_downweights_provisional_then_quiet_emits_full_verdict(self):
        body_engine = _FakeBodyEngine(target_idx=3)
        rgb_engine = _FakeRgbEngine(target_idx=3)
        rgb_engine.n_person = 3
        rgb_engine.n_hand = 3
        rgb_engine.n_ctx = 2
        rgb_engine.infer_logits = lambda v: (
            setattr(rgb_engine, "calls", rgb_engine.calls + 1)
            or (
                (_ for _ in ()).throw(AssertionError(v.shape))
                if v.shape != (1, 8, 3, 160, 160)
                else rgb_engine.logits.copy()
            )
        )
        plugin, _ = _plugin(
            {
                "action_backend": "official126",
                "official_action_manifest": str(self.root / "artifact.json"),
                "activity_mode": "segment",
                "activity_segment_pause_s": 0.25,
                "activity_max_unflushed_s": 0.08,
                "fps": 24,
                "max_persons": 1,
                "label_hold": 1,
            }
        )
        with patch("plugins.official_action._OnnxEngine", return_value=body_engine):
            plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
            node = plugin._nodes["/cam/rgb"]
            node._segment_backend._rgb_engine = rgb_engine
            node._image_cb(_FakeCompressedImage(frame_bytes()))
            time.sleep(0.10)
            node._image_cb(_FakeCompressedImage(frame_bytes()))
            assert _wait_until(
                lambda: sum(
                    1
                    for m in node._pub.messages
                    if json.loads(m)["persons"] and "activity" in json.loads(m)["persons"][0]
                )
                >= 1,
                timeout=2.0,
            )
            # Now send one more frame and let quiet_s (0.25s) expire
            time.sleep(0.06)
            node._image_cb(_FakeCompressedImage(frame_bytes()))
            assert _wait_until(
                lambda: node._last_prediction.get("trigger") == "quiet_confirm",
                timeout=2.0,
            )
            action_msgs = [
                json.loads(m)["persons"][0]
                for m in node._pub.messages
                if json.loads(m)["persons"] and "activity" in json.loads(m)["persons"][0]
            ]
            self.assertGreaterEqual(len(action_msgs), 2)
            self.assertLess(action_msgs[0]["activity_score"], action_msgs[-1]["activity_score"])
            plugin.dispatch("pose", {"action": "stop"})

    def test_dual_roi_patch_and_calibrated_poe_router(self):
        bgr = np.full((480, 640, 3), 120, dtype=np.uint8)
        boxes = np.array([[200.0, 80.0, 420.0, 440.0]], dtype=np.float32)
        scores = np.array([0.92], dtype=np.float32)
        kps = np.zeros((1, 17, 3), dtype=np.float32)
        kps[0, :, 0] = 310.0
        kps[0, :, 1] = np.linspace(100.0, 420.0, 17)
        kps[0, :, 2] = 0.9
        p_rgb, h_rgb, c_rgb = make_dual_roi_patch(bgr, boxes, scores, kps, crop_hw=160, with_ctx=True)
        self.assertEqual(p_rgb.shape, (160, 160, 3))
        self.assertEqual(h_rgb.shape, (160, 160, 3))
        self.assertEqual(c_rgb.shape, (160, 160, 3))

        pb = np.full(126, 1.0 / 126.0, dtype=np.float32)
        pr = np.full(126, 1e-4, dtype=np.float32)
        pr[17] = 0.98  # kitchen/household class 18
        pr /= pr.sum()
        router_cfg = {
            "mode": "poe_gate",
            "theta": [1.0, 1.2, 0.2, 0.4, 1.5, 0.2, -1.2, 1.0, 1.0, 0.5],
        }
        fused = fuse_calibrated_router(pb, pr, router_cfg, observation_fps=6.0)
        self.assertEqual(int(fused.argmax()), 17)
        self.assertAlmostEqual(float(fused.sum()), 1.0, places=5)

    def test_shared_tensorrt_device_scratch_pool_reuses_and_grows(self):
        from utils.tensorrt_runtime import _SharedDeviceScratchPool

        class _FakeCuda:
            def __init__(self):
                self.allocs = []
                self.frees = []
                self._next = 0x1000

            def malloc(self, size):
                ptr = self._next
                self._next += 0x1000
                self.allocs.append((ptr, size))
                return ptr

            def free(self, ptr):
                self.frees.append(ptr)

        pool = _SharedDeviceScratchPool()
        cuda = _FakeCuda()
        ptr1, cap1 = pool.ensure(0, cuda, 1024)
        ptr2, cap2 = pool.ensure(0, cuda, 512)
        self.assertEqual(ptr1, ptr2)
        self.assertEqual(cap1, 1024)
        self.assertEqual(cap2, 1024)
        self.assertEqual(len(cuda.allocs), 1)
        ptr3, cap3 = pool.ensure(0, cuda, 4096)
        self.assertNotEqual(ptr3, ptr1)
        self.assertEqual(cap3, 4096)
        self.assertEqual(cuda.frees, [ptr1])

    def test_prebaked_pose_model_dir_resolves_without_models_root_check(self):
        import os

        prebaked = self.root / "work_pose"
        prebaked.mkdir()
        eng = prebaked / "yolo26s-pose.engine"
        eng.write_bytes(b"fake-trt-plan")
        plugin, _ = _plugin({"action_backend": "rules"})
        with patch.dict(os.environ, {"POSE_MODEL_DIR": str(prebaked)}, clear=False):
            resolved = plugin._resolve_engine()
        self.assertEqual(resolved, str(eng))

    def test_dockerfile_pose_precache_uses_verified_bundle(self):
        root = Path(__file__).resolve().parents[2]
        recipe = (root / "perception/Dockerfile.jetson").read_text(encoding="utf-8")
        self.assertIn("ENV POSE_MODEL_DIR=/work/models/pose", recipe)
        self.assertIn("POSE_MODEL_BUNDLES", recipe)
        self.assertIn("ensure_verified_bundle", recipe)
        self.assertNotIn("POSE_CPU_ONNX_PATH", recipe)


if __name__ == "__main__":
    unittest.main()
