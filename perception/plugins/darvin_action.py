"""Pinned public artifacts for the two Darvin314 streaming candidates.

Downloads reuse the existing verified-bundle downloader. Selection is explicit;
artifact failures must not fall back to the original model or geometry-only mode.
"""
import json
import logging
from pathlib import Path

import numpy as np

from plugins.official_action import OfficialActionBackend, INPUT_SHAPE
from plugins.pose_stgcn import ActionBackendError, NTU60, build_backend
from plugins import darvin_assets
from utils.runtime_profile import log_process_memory, memory_stage

log = logging.getLogger(__name__)


def candidate_spec(name):
    try:
        return darvin_assets.candidate_spec(name)
    except ValueError as error:
        raise ActionBackendError(str(error)) from error


class Native60OnnxEngine:
    """CPU-only ONNX adapter retaining the reference NTU60 classifier semantics."""
    def __init__(self, path, manifest):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or len(outputs) != 1 or tuple(inputs[0].shape) != INPUT_SHAPE
                or outputs[0].shape != [1, 60] or inputs[0].type != "tensor(float)"
                or outputs[0].type != "tensor(float)"):
            raise ActionBackendError("native60 ONNX shape/type mismatch")
        metadata = self.session.get_modelmeta().custom_metadata_map
        if (metadata.get("format") != "darvin-native60-comparison-v1"
                or metadata.get("classes") != "60" or metadata.get("output_kind") != "logits"
                or metadata.get("checkpoint_sha256") != manifest["checkpoint_sha256"]):
            raise ActionBackendError("native60 ONNX identity mismatch")
        self.name = inputs[0].name
        self.input_shape = INPUT_SHAPE

    def infer(self, blob):
        blob = np.asarray(blob, dtype=np.float32)
        if blob.shape != INPUT_SHAPE or not np.isfinite(blob).all():
            raise ActionBackendError("invalid native60 input")
        outputs = self.session.run(None, {self.name: blob})
        if len(outputs) != 1 or outputs[0].shape != (1, 60) or not np.isfinite(outputs[0]).all():
            raise ActionBackendError("invalid native60 logits")
        return outputs


def validate_native60(manifest):
    if (manifest.get("format") != "darvin-native60-comparison-v1"
            or manifest.get("classes") != 60 or manifest.get("native_labels") != [list(row) for row in NTU60]
            or manifest.get("input_shape") != list(INPUT_SHAPE) or manifest.get("output_shape") != [1, 60]
            or manifest.get("training_normalization") != "fix"
            or manifest.get("engine", {}).get("kind") != "onnx_cpu"):
        raise ActionBackendError("native60 artifact contract mismatch")


def build_darvin_backend(name, merged, thresholds, *, session_cache=None):
    spec = candidate_spec(name)
    if merged["action_backend"] != spec["backend"]:
        raise ActionBackendError("Darvin candidate and action_backend disagree")
    if merged.get("official_action_manifest"):
        raise ActionBackendError("choose either a pinned Darvin candidate or a local official manifest")
    paths = darvin_assets.ensure_candidate_bundle(name, spec)
    manifest = json.loads(Path(paths["artifact.json"]).read_text(encoding="utf-8"))
    engine_name = manifest.get("engine", {}).get("file")
    if (engine_name not in spec["files"] or manifest["engine"]["sha256"] != spec["files"][engine_name]["sha256"]):
        raise ActionBackendError("engine identity differs from pinned candidate")
    rgb_spec = manifest.get("rgb_engine")
    if isinstance(rgb_spec, dict):
        rgb_name = rgb_spec.get("file")
        if rgb_name not in spec["files"] or rgb_spec.get("sha256") != spec["files"][rgb_name]["sha256"]:
            raise ActionBackendError("rgb engine identity differs from pinned candidate")
    rtr_spec = manifest.get("multimodal_router")
    if isinstance(rtr_spec, dict) and isinstance(rtr_spec.get("fit_receipt"), dict):
        rtr_file = rtr_spec["fit_receipt"].get("file")
        if rtr_file not in spec["files"] or rtr_spec["fit_receipt"].get("sha256") != spec["files"][rtr_file]["sha256"]:
            raise ActionBackendError("router receipt identity differs from pinned candidate")
    if spec["backend"] == "official126":
        with memory_stage(log, "official126.init"):
            backend = OfficialActionBackend(manifest_path=paths["artifact.json"],
                        window_s=merged["action_window_s"], min_score=merged["official_action_min_score"])
            backend._ensure_engine(session_cache)
    else:
        validate_native60(manifest)
        engine = Native60OnnxEngine(paths[engine_name], manifest)
        backend = build_backend("hybrid", engine=engine, prenorm_mode="fix", thresholds=thresholds,
                    action_window_s=merged["action_window_s"], window_s=merged["action_window_s"],
                    min_score=merged["action_min_score"])
    log.info("[pose/darvin] candidate=%s model_revision=%s engine_sha256=%s provider=CPUExecutionProvider",
             name, spec["revision"], manifest["engine"]["sha256"])
    log_process_memory(log, "action_model_ready")
    return backend
