"""Explicit 126-label action artifacts and causal skeleton inputs.

This backend never downloads models or falls back to the NTU60/geometry label
space. An artifact declares its backend, exact label order and preprocessing.
Capture/video time must be supplied by the caller; processing time is not used.
"""
from collections import deque
import hashlib
import json
import logging
import math
import threading
from pathlib import Path

import numpy as np

from plugins.pose_stgcn import ActionBackendError, pre_normalize_2d, resample_clip, _nothing
from utils.runtime_profile import memory_stage

INPUT_SHAPE = (1, 2, 100, 17, 3)
FORMAT = "darvin-official126-v1"


def canonical_labels():
    path = Path(__file__).resolve().parents[1] / "data/official_labels.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows = sorted(rows, key=lambda row: row["official_id"])
    if len(rows) != 126 or [r["official_id"] for r in rows] != list(range(1, 127)):
        raise ActionBackendError("bundled official taxonomy must contain IDs 1..126")
    if len({r["activity"] for r in rows}) != 126 or len({r["activity_zh"] for r in rows}) != 126:
        raise ActionBackendError("bundled taxonomy contains duplicate labels")
    return [{**row, "output_index": i} for i, row in enumerate(rows)]


def schema_hash(labels):
    return hashlib.sha256(json.dumps(labels, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_temporal_calibration(value):
    """Optional artifact-bound confidence calibration; labels are unchanged."""
    if not isinstance(value, dict) or value.get("format") != "darvin-temporal-temperature-v1":
        raise ActionBackendError("invalid temporal calibration format")
    for key, lower, upper in (("history_boundary_s", 0., 60.),
                               ("early_temperature", 1., 20.), ("mature_temperature", 1., 20.)):
        number = value.get(key)
        if (type(number) not in (int, float) or not math.isfinite(number)
                or number < lower or number > upper or (key == "history_boundary_s" and number == 0)):
            raise ActionBackendError("invalid temporal calibration parameter: " + key)
    receipt = value.get("fit_receipt", {})
    if not isinstance(receipt, dict):
        raise ActionBackendError("invalid temporal calibration training receipt")
    name, digest = receipt.get("file", ""), receipt.get("sha256", "")
    if (value.get("fit_split") != "train" or not isinstance(name, str) or not isinstance(digest, str)
            or not name or Path(name).name != name
            or name in (".", "..") or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)):
        raise ActionBackendError("invalid temporal calibration training receipt")


def validate_manifest(manifest):
    labels = canonical_labels()
    if (manifest.get("format") != FORMAT or manifest.get("classes") != labels
            or manifest.get("taxonomy_sha256") != schema_hash(labels)):
        raise ActionBackendError("artifact format or exact canonical 126-label order differs")
    for row in manifest["classes"]:
        if type(row.get("official_id")) is not int or type(row.get("output_index")) is not int:
            raise ActionBackendError("class IDs and output indices must be integers")
    if (manifest.get("input_shape") != list(INPUT_SHAPE)
            or manifest.get("output_shape") != [1, 126]
            or manifest.get("output_kind") != "logits"):
        raise ActionBackendError("artifact requires fixed (1,2,100,17,3) input and (1,126) logits")
    preprocessing = manifest.get("preprocessing", {})
    if (preprocessing.get("normalization") not in ("fix", "auto")
            or preprocessing.get("order") != "normalize_then_resample"
            or preprocessing.get("joint_channels") != "COCO17 x/y/confidence"
            or preprocessing.get("person_slots") != "primary_then_zero"):
        raise ActionBackendError("unsupported or absent preprocessing contract")
    engine = manifest.get("engine", {})
    name = engine.get("file", "")
    digest = engine.get("sha256", "")
    if (engine.get("kind") not in ("onnx_cpu", "tensorrt") or not name
            or Path(name).name != name or name in (".", "..")
            or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
        raise ActionBackendError("invalid explicit engine kind/file/hash")
    if "temporal_calibration" in manifest:
        validate_temporal_calibration(manifest["temporal_calibration"])
    return labels


def pose_blob(keypoints, image_size, normalization="fix"):
    """Shared train/inference transform; confidence remains a confidence channel."""
    raw = np.asarray(keypoints, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[1:] != (17, 3) or not len(raw) or not np.isfinite(raw).all():
        raise ActionBackendError("nonempty finite (T,17,3) keypoints required")
    if (raw[..., 2] < 0).any() or (raw[..., 2] > 1).any():
        raise ActionBackendError("joint confidence outside [0,1]")
    normalized = pre_normalize_2d(raw, image_size, normalization)
    result = np.zeros(INPUT_SHAPE, dtype=np.float32)
    result[0, 0] = resample_clip(normalized, 100)
    return result


def causal_blob(times, keypoints, image_size, now, window_s, normalization="fix"):
    """Select only observations in [now-window_s, now], then build the input."""
    times = np.asarray(times, dtype=np.float64)
    raw = np.asarray(keypoints, dtype=np.float32)
    if (times.ndim != 1 or raw.ndim != 3 or len(times) != len(raw)
            or not np.isfinite(times).all() or (np.diff(times) <= 0).any()
            or not math.isfinite(now) or not math.isfinite(window_s) or window_s <= 0):
        raise ActionBackendError("ordered timestamps and a positive finite window are required")
    selected = (times >= now - window_s) & (times <= now)
    if not selected.any():
        return None
    return pose_blob(raw[selected], image_size, normalization)


class CausalWindow:
    """One stream/person. Explicit reset is mandatory at a new clip boundary."""
    def __init__(self, window_s=2.5, normalization="fix"):
        if not math.isfinite(window_s) or window_s <= 0:
            raise ValueError("window_s must be positive and finite")
        self.window_s, self.normalization = float(window_s), normalization
        self.reset()

    def reset(self):
        self.frames = deque()
        self.last_time = None
        self.image_size = None

    def push(self, timestamp, keypoints, image_size):
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or (self.last_time is not None and timestamp <= self.last_time):
            raise ActionBackendError("timestamps must increase; reset before a new clip")
        if self.image_size is not None and tuple(image_size) != self.image_size:
            raise ActionBackendError("image size changed without a stream reset")
        self.image_size, self.last_time = tuple(image_size), timestamp
        if keypoints is not None:
            raw = np.asarray(keypoints, dtype=np.float32)
            pose_blob(raw[None], self.image_size, self.normalization)  # Validate at ingress.
            self.frames.append((timestamp, raw.copy()))
        while self.frames and self.frames[0][0] < timestamp - self.window_s:
            self.frames.popleft()

    def blob(self):
        if not self.frames:
            return None
        return pose_blob(np.stack([x[1] for x in self.frames]), self.image_size, self.normalization)


class _OnnxEngine:
    def __init__(self, path, taxonomy_hash, normalization):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or len(outputs) != 1 or tuple(inputs[0].shape) != INPUT_SHAPE
                or outputs[0].shape != [1, 126] or inputs[0].type != "tensor(float)"
                or outputs[0].type != "tensor(float)"):
            raise ActionBackendError("ONNX input/output declarations disagree with the artifact")
        metadata = self.session.get_modelmeta().custom_metadata_map
        if metadata.get("format") != FORMAT or metadata.get("taxonomy_sha256") != taxonomy_hash:
            raise ActionBackendError("ONNX embedded schema metadata differs from the artifact")
        if (metadata.get("normalization") != normalization
                or metadata.get("preprocessing_order") != "normalize_then_resample"
                or metadata.get("output_kind") != "logits"):
            raise ActionBackendError("ONNX embedded preprocessing/logit contract differs from the artifact")
        self.name = inputs[0].name
        self.input_shape = INPUT_SHAPE

    def infer(self, blob):
        return self.session.run(None, {self.name: blob})


class OfficialOnnxSessionCache:
    """One verified CPU engine per pose plugin, never a tracker or classifier."""
    def __init__(self):
        self._lock = threading.Lock()
        self._key = None
        self._engine = None

    def acquire(self, path, digest, taxonomy_hash, normalization):
        key = (path, digest, taxonomy_hash, normalization)
        with self._lock:
            if self._key != key:
                with memory_stage(logging.getLogger(__name__), "official126.session_create"):
                    engine = _OnnxEngine(path, taxonomy_hash, normalization)
                self._engine, self._key = engine, key
            return self._engine


class OfficialActionBackend:
    uses_official_taxonomy = True

    def __init__(self, *, manifest_path, window_s=2.5, min_score=0., engine=None):
        self.path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        self.labels = validate_manifest(self.manifest)
        self.window_s, self.min_score = float(window_s), float(min_score)
        if not math.isfinite(self.window_s) or self.window_s <= 0 or not 0 <= self.min_score <= 1:
            raise ActionBackendError("invalid official action window/score threshold")
        self.engine_path = (self.path.parent / self.manifest["engine"]["file"]).resolve()
        if self.engine_path.parent != self.path.parent or file_hash(self.engine_path) != self.manifest["engine"]["sha256"]:
            raise ActionBackendError("engine path/hash differs from the artifact")
        self.normalization = self.manifest["preprocessing"]["normalization"]
        self.temporal_calibration = self.manifest.get("temporal_calibration")
        if self.temporal_calibration is not None:
            fit = self.temporal_calibration["fit_receipt"]
            receipt_path = (self.path.parent / fit["file"]).resolve()
            if receipt_path.parent != self.path.parent or file_hash(receipt_path) != fit["sha256"]:
                raise ActionBackendError("temporal calibration receipt path/hash differs")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            expected = {key: self.temporal_calibration[key] for key in
                        ("history_boundary_s", "early_temperature", "mature_temperature")}
            if (receipt.get("fit_split") != "train" or receipt.get("parameters") != expected
                    or receipt.get("engine_sha256") != self.manifest["engine"]["sha256"]):
                raise ActionBackendError("temporal calibration parameters differ from fit receipt")
        self._engine, self._load_error = engine, None

    @property
    def history_s(self):
        return self.window_s

    @property
    def last_error(self):
        return self._load_error

    def _ensure_engine(self, session_cache=None):
        if self._engine is None:
            if self.manifest["engine"]["kind"] == "onnx_cpu":
                if session_cache is None:
                    self._engine = _OnnxEngine(self.engine_path, self.manifest["taxonomy_sha256"], self.normalization)
                else:
                    self._engine = session_cache.acquire(self.engine_path, self.manifest["engine"]["sha256"],
                                                         self.manifest["taxonomy_sha256"], self.normalization)
            else:
                from utils.tensorrt_runtime import TensorRTEngine
                self._engine = TensorRTEngine(str(self.engine_path))
        shape = getattr(self._engine, "input_shape", None)
        if shape is not None and tuple(shape) != INPUT_SHAPE:
            raise ActionBackendError("engine input shape differs from the official artifact")
        return self._engine

    def predict_blob(self, blob, *, temperature=1.):
        blob = np.asarray(blob, dtype=np.float32)
        if blob.shape != INPUT_SHAPE or not np.isfinite(blob).all():
            raise ActionBackendError("invalid official model input tensor")
        values = self._ensure_engine().infer(blob)
        if isinstance(values, (list, tuple)):
            if len(values) != 1:
                raise ActionBackendError("expected exactly one logits output")
            values = values[0]
        logits = np.asarray(values, dtype=np.float32)
        if logits.shape != (1, 126) or not np.isfinite(logits).all():
            raise ActionBackendError("expected finite (1,126) logits; no NTU60 decoding is permitted")
        if not math.isfinite(temperature) or not 1. <= temperature <= 20.:
            raise ActionBackendError("temperature must be finite and between 1 and 20")
        scaled = logits / temperature
        probabilities = np.exp(scaled[0] - scaled.max())
        probabilities /= probabilities.sum()
        index = int(probabilities.argmax())
        row = self.labels[index]
        return {"name": row["activity"], "name_zh": row["activity_zh"],
                "official_id": row["official_id"], "score": float(probabilities[index]),
                "source": "official126"}

    def classify(self, frames, want_activity=True):
        if not frames or not want_activity:
            return _nothing("no frames" if not frames else "activity not requested")
        try:
            sizes = [tuple(f.image_size) for f in frames]
            if any(size != sizes[-1] for size in sizes):
                raise ActionBackendError("mixed image sizes in one action history")
            times = [float(f.t) for f in frames]
            blob = causal_blob(times, [f.keypoints for f in frames], sizes[-1], times[-1],
                               self.window_s, self.normalization)
            temperature = 1.
            if self.temporal_calibration is not None:
                included = [t for t in times if t >= times[-1] - self.window_s]
                span = included[-1] - included[0]
                key = "early_temperature" if span < self.temporal_calibration["history_boundary_s"] else "mature_temperature"
                temperature = self.temporal_calibration[key]
                activity = self.predict_blob(blob, temperature=temperature) if blob is not None else None
            else:
                activity = self.predict_blob(blob) if blob is not None else None
            if activity is not None and activity["score"] < self.min_score:
                activity = None
            self._load_error = None
            return {"posture": None, "posture_confidence": 0., "activity": activity,
                    "evidence": {"backend": "official126", "window_s": self.window_s,
                                 "normalization": self.normalization,
                                 "score_kind": "temporal_temperature_softmax" if self.temporal_calibration is not None else "uncalibrated_softmax",
                                 "temperature": temperature}}
        except Exception as error:
            self._load_error = str(error)
            return _nothing(str(error), backend_error=True)

    def classify_frame(self, frame):
        return self.classify([frame])
