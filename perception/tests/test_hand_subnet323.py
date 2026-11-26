"""
tests/test_hand_subnet323.py — Darvin Subnet 323 (<30 MB ONNX) unit, compatibility & E2E plugin tests.
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import yaml

from vision_stubs import _FakeCompressedImage, _FakeExecutor, _wait_until, frame_bytes  # noqa: F401
import plugins.pose as pose_plugin
from plugins.hand_subnet323 import (
    DEFAULT_CALIBRATION,
    NO_GESTURE_IDX,
    OFFICIAL_30_CLASSES,
    Subnet323HandRuntime,
)
from plugins.hand_subnet323_assets import (
    FALLBACK_SUBNET323_BASE_URL,
    SUBNET323_MODEL_FILES,
    resolve_subnet323_base_url,
    verify_local_bundle,
)

PERCEPTION_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = PERCEPTION_DIR.parent


def _find_local_bundle_dir() -> Optional[Path]:
    for cand in (
        os.environ.get("HAND_SUBNET323_BUNDLE_DIR"),
        "/work/models/hand_subnet323",
        "/mnt/data/wangxinghong/hand_rec/models/onnx_bundle",
    ):
        if cand:
            p = Path(cand)
            if all((p / fn).is_file() for fn in SUBNET323_MODEL_FILES):
                return p
    return None


def test_python38_ast_and_annotation_compatibility():
    """JP5.1.1 ships Python 3.8 (`cp38`): every touched module must parse cleanly as Python 3.8."""
    targets = [
        PERCEPTION_DIR / "main.py",
        PERCEPTION_DIR / "plugins" / "pose.py",
        PERCEPTION_DIR / "plugins" / "hand_subnet323.py",
        PERCEPTION_DIR / "plugins" / "hand_subnet323_assets.py",
        PERCEPTION_DIR / "tests" / "test_hand_subnet323.py",
    ]
    for path in targets:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path), feature_version=(3, 8))
        assert tree is not None

    # Ensure our new runtime modules avoid PEP 604 `X | Y` even inside stringified annotations
    for path in (
        PERCEPTION_DIR / "plugins" / "hand_subnet323.py",
        PERCEPTION_DIR / "plugins" / "hand_subnet323_assets.py",
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.arg) and node.annotation is not None:
                assert not any(isinstance(sub, ast.BitOr) for sub in ast.walk(node.annotation)), path
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns is not None:
                assert not any(isinstance(sub, ast.BitOr) for sub in ast.walk(node.returns)), path


def test_config_yaml_and_env_ports_hardened_for_8gb_orin():
    cfg = yaml.safe_load((PERCEPTION_DIR / "config.yaml").read_text(encoding="utf-8"))
    plugins = cfg["plugins"]
    for disabled_name in (
        "asr",
        "tts",
        "vop",
        "visual_depth",
        "ocr",
        "barcode",
        "soundevent",
        "echo_cancellation",
        "speaker_recognition",
        "face_recognition",
        "phanthy_music",
        "audio_mixer",
    ):
        assert plugins[disabled_name]["enabled"] is False, disabled_name

    pose_cfg = plugins["pose"]
    assert pose_cfg["enabled"] is True
    assert pose_cfg["hands"] is True
    assert pose_cfg["hand_backend"] == "subnet323_onnx"
    assert pose_cfg["action_backend"] == "rules"
    assert "/uploads" in pose_cfg["image_roots"]
    assert "/models" in pose_cfg["image_roots"]

    main_src = (PERCEPTION_DIR / "main.py").read_text(encoding="utf-8")
    assert 'os.environ.get("MCP_PORT")' in main_src
    assert 'os.environ.get("WS_PORT")' in main_src

    # Verify CPU-only ORT invariant so `hand_subnet323.py` can never trigger JP5.11 Provider_SetHost collision
    rt_src = (PERCEPTION_DIR / "plugins" / "hand_subnet323.py").read_text(encoding="utf-8")
    assert "CUDAExecutionProvider" not in rt_src


def test_empty_arg_env_falls_back_to_github_release_url(monkeypatch):
    """Regression test for JP5 Dockerfile ARG HAND_SUBNET323_BASE_URL="" empty-env bug."""
    assert FALLBACK_SUBNET323_BASE_URL.startswith("https://github.com/kiddddd1218/phanthymotus/releases/download/")

    monkeypatch.delenv("HAND_SUBNET323_BASE_URL", raising=False)
    assert resolve_subnet323_base_url() == FALLBACK_SUBNET323_BASE_URL

    monkeypatch.setenv("HAND_SUBNET323_BASE_URL", "")
    assert resolve_subnet323_base_url() == FALLBACK_SUBNET323_BASE_URL

    monkeypatch.setenv("HAND_SUBNET323_BASE_URL", "   ")
    assert resolve_subnet323_base_url() == FALLBACK_SUBNET323_BASE_URL

    monkeypatch.setenv("HAND_SUBNET323_BASE_URL", "https://example.com/mirror")
    assert resolve_subnet323_base_url() == "https://example.com/mirror"
    assert resolve_subnet323_base_url("  ") == "https://example.com/mirror"
    assert resolve_subnet323_base_url("https://override.example/path") == "https://override.example/path"


def test_decode_gestures_two_hand_and_resting_second_hand_guard():
    assert len(OFFICIAL_30_CLASSES) == 30
    calib = dict(DEFAULT_CALIBRATION)

    # 1) Two-hand symmetric emission (`xsign` = index 25 -> pair index 1 + 2 = 3)
    ll = np.zeros((31,), dtype=np.float32)
    lr = np.zeros((31,), dtype=np.float32)
    lp = np.zeros((8,), dtype=np.float32)
    ll[25] = 6.0
    lr[25] = 6.0
    lp[3] = 6.0
    gestures, scores = Subnet323HandRuntime._decode_gestures(ll, lr, lp, True, True, calib)
    assert gestures == {"left": "xsign", "right": "xsign"}
    assert scores["left"] is not None and scores["left"] == scores["right"]
    assert 0.5 < scores["left"] <= 1.0

    # 2) Single-hand active (`victory` = index 3 on right) + visible resting left hand -> left suppressed to None!
    ll2 = np.zeros((31,), dtype=np.float32)
    lr2 = np.zeros((31,), dtype=np.float32)
    lp2 = np.array([6.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0], dtype=np.float32)
    ll2[1] = 1.1       # mild `fist` temptation on resting hand
    ll2[NO_GESTURE_IDX] = 0.5
    lr2[3] = 7.0       # strong `victory` on gesturing right hand
    g2, s2 = Subnet323HandRuntime._decode_gestures(ll2, lr2, lp2, True, True, calib)
    assert g2 == {"left": None, "right": "victory"}
    assert s2["left"] is None and (s2["right"] is not None and s2["right"] > 0.8)

    # 3) Pure `no_gesture` abstention
    ll3 = np.zeros((31,), dtype=np.float32)
    ll3[NO_GESTURE_IDX] = 5.0
    g3, s3 = Subnet323HandRuntime._decode_gestures(ll3, ll3, lp2, True, False, calib)
    assert g3 == {"left": None, "right": None}
    assert s3 == {"left": None, "right": None}


def test_no_oversized_source_files_under_1mb_redline():
    if shutil.which("git") and (REPO_ROOT / ".git").exists():
        proc = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files"],
            check=True,
            capture_output=True,
            text=True,
        )
        candidates = [REPO_ROOT / r for r in proc.stdout.splitlines() if r.strip()]
    else:
        candidates = [
            p for p in PERCEPTION_DIR.rglob("*")
            if p.is_file() and "models" not in p.parts and "__pycache__" not in p.parts
        ]
    oversized = [(str(fp), fp.stat().st_size) for fp in candidates if fp.is_file() and fp.stat().st_size >= 1_000_000]
    assert not oversized, f"Source files >= 1 MB redline: {oversized}"


def test_pinned_onnx_bundle_budget_and_end_to_end_photo_schema(tmp_path, monkeypatch):
    bundle_dir = _find_local_bundle_dir()
    if bundle_dir is None:
        pytest.skip("ONNX bundle not present in test environment")

    verified = verify_local_bundle(bundle_dir)
    total_bytes = sum(Path(p).stat().st_size for p in verified.values())
    assert total_bytes < 30_000_000, f"Total ONNX bundle {total_bytes} exceeds 30 MB!"

    monkeypatch.setenv("HAND_SUBNET323_BUNDLE_DIR", str(bundle_dir))
    cfg = yaml.safe_load((PERCEPTION_DIR / "config.yaml").read_text(encoding="utf-8"))["plugins"]["pose"]
    cfg = dict(cfg)
    cfg["image_roots"] = list(cfg["image_roots"]) + [str(tmp_path.resolve())]

    # Pick a real official val image if available, otherwise synthetic JPEG
    val_dir = Path("/mnt/data/wangxinghong/hand_rec/data/extracted/训练集")
    sample_jpg = tmp_path / "sample.jpg"
    real_imgs = sorted(val_dir.glob("*.jpg")) if val_dir.is_dir() else []
    if real_imgs:
        sample_jpg.write_bytes(real_imgs[0].read_bytes())
    else:
        sample_jpg.write_bytes(frame_bytes(640, 480))

    executor = _FakeExecutor()
    plugin = pose_plugin.PosePerceptionPlugin(cfg, "subnet323_test", executor)
    res = plugin.dispatch("pose", {"action": "recognize_by_photo", "image_path": str(sample_jpg)})
    assert res["ok"] is True, res
    assert isinstance(res["persons"], list)
    if real_imgs:
        assert res["count"] >= 1
    for person in res["persons"]:
        assert set(person["hand_bboxes"].keys()) == {"left", "right"}
        assert set(person["hand_gestures"].keys()) == {"left", "right"}
        assert set(person["hand_gesture_scores"].keys()) == {"left", "right"}
        for side in ("left", "right"):
            hb = person["hand_bboxes"][side]
            assert isinstance(hb, list) and len(hb) in (0, 4)
            hg = person["hand_gestures"][side]
            hs = person["hand_gesture_scores"][side]
            if hg is None:
                assert hs is None
            else:
                assert hg in OFFICIAL_30_CLASSES
                assert isinstance(hs, float) and 0.0 <= hs <= 1.0

    # Also test start -> info -> stop lifecycle (ensuring `reset()` and `input_size` work cleanly)
    plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
    info = plugin.dispatch("pose", {"action": "info"})
    assert info["state"] == "running"
    assert info["instances"]["/cam/rgb"]["hands"] is True
    assert info["instances"]["/cam/rgb"]["hand_input"] == 224
    stop_res = plugin.dispatch("pose", {"action": "stop"})
    assert stop_res["state"] == "idle"
