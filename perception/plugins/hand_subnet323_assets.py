#!/usr/bin/env python3
"""
plugins/hand_subnet323_assets.py — SHA-256 + byte-size verified bundle loader for Darvin Subnet 323.

Total model files strictly < 30,000,000 bytes (< 30 MB).
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Union

# Populated automatically with exact byte sizes & SHA-256 digests after ONNX export
SUBNET323_MODEL_FILES: Dict[str, Dict[str, object]] = {
    "person_pose.onnx": {
        "size": 11810678,
        "sha256": "62cdc049fbcc60580729a35882872dabcae6c35cb73fc1610bb9032a8c2dd563",
    },
    "hand_det.onnx": {
        "size": 10564810,
        "sha256": "234442c86fd12d601453e8ad03b3d80ad092766b9ccef954d8887ddb70c90ca2",
    },
    "hand_crop_net.onnx": {
        "size": 5803910,
        "sha256": "f6fbb4730fda716e3995a6fc617676e53a91f5054fc445fcce487f1999be14e8",
    },
    "gesture_fusion.onnx": {
        "size": 1484102,
        "sha256": "dd3b40b5ddb0b4695918e7efec3ebd7db816dc6c2d711960d7d49d75c1fde232",
    },
}

FALLBACK_SUBNET323_BASE_URL = (
    "https://github.com/kiddddd1218/phanthymotus/releases/download/darvin323-models-v1"
)
DEFAULT_SUBNET323_BASE_URL = FALLBACK_SUBNET323_BASE_URL


def resolve_subnet323_base_url(override: Optional[str] = None) -> str:
    """Resolve non-empty base URL; ignores empty Docker ARG env strings (`""`)."""
    for candidate in (override, os.environ.get("HAND_SUBNET323_BASE_URL")):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return FALLBACK_SUBNET323_BASE_URL


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(4 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_local_bundle(bundle_dir: Union[str, Path]) -> Dict[str, str]:
    root = Path(bundle_dir)
    total_bytes = 0
    verified: Dict[str, str] = {}
    for fname, meta in SUBNET323_MODEL_FILES.items():
        fp = root / fname
        if not fp.is_file():
            raise FileNotFoundError(f"Missing model file {fp}")
        sz = fp.stat().st_size
        total_bytes += sz
        exp_sz = int(meta.get("size") or 0)
        exp_sha = str(meta.get("sha256") or "")
        if exp_sz > 0 and sz != exp_sz:
            raise RuntimeError(f"Size mismatch for {fp}: {sz} != {exp_sz}")
        if exp_sha:
            actual_sha = _sha256_path(fp)
            if actual_sha != exp_sha:
                raise RuntimeError(f"SHA-256 mismatch for {fp}: {actual_sha} != {exp_sha}")
        verified[fname] = str(fp)
    if total_bytes >= 30_000_000:
        raise RuntimeError(f"Total model bundle {total_bytes} bytes exceeds 30 MB limit!")
    return verified


def ensure_subnet323_bundle(
    bundle_dir: str = "/work/models/hand_subnet323",
    base_url: Optional[str] = None,
) -> Dict[str, str]:
    root = Path(bundle_dir)
    if all((root / fn).is_file() for fn in SUBNET323_MODEL_FILES):
        return verify_local_bundle(root)

    from utils.model_downloader import ensure_verified_bundle

    resolved_url = resolve_subnet323_base_url(base_url)
    last_err: Optional[Exception] = None
    for outer_attempt in range(1, 3):
        try:
            ensure_verified_bundle("hand_subnet323", str(root), resolved_url, SUBNET323_MODEL_FILES)
            return verify_local_bundle(root)
        except Exception as exc:  # noqa: BLE001 — retry once on transient CDN reset
            last_err = exc
            if outer_attempt < 2:
                time.sleep(5)
    assert last_err is not None
    raise last_err


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    target = sys.argv[1] if len(sys.argv) > 1 else "/work/models/hand_subnet323"
    print(f"[hand_subnet323] Ensuring <30 MB ONNX bundle at {target} from {resolve_subnet323_base_url()}")
    paths = ensure_subnet323_bundle(target)
    total = sum(Path(p).stat().st_size for p in paths.values())
    print(f"[hand_subnet323] Verified {len(paths)} ONNX files ({total} bytes < 30,000,000) in {target}")
