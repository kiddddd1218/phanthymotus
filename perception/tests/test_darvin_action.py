"""Artifact-selection failures must never quietly choose another model."""
import copy
from types import SimpleNamespace
import sys

import pytest

from plugins.darvin_action import Native60OnnxEngine, build_darvin_backend, validate_native60
from plugins.pose_stgcn import ActionBackendError, NTU60
from plugins.pose import PosePerceptionPlugin


def manifest():
    return dict(format="darvin-native60-comparison-v1", classes=60,
                native_labels=[list(row) for row in NTU60],
                input_shape=[1, 2, 100, 17, 3], output_shape=[1, 60],
                training_normalization="fix", engine={"kind": "onnx_cpu"},
                checkpoint_sha256="a" * 64)


def test_reordered_labels_are_rejected():
    item = manifest()
    validate_native60(item)
    item["native_labels"][0], item["native_labels"][1] = item["native_labels"][1], item["native_labels"][0]
    with pytest.raises(ActionBackendError, match="contract mismatch"):
        validate_native60(item)


@pytest.mark.parametrize("field,value", [("classes", 126), ("training_normalization", "auto"), ("output_shape", [1, 126])])
def test_incompatible_native_contract_is_rejected(field, value):
    item = manifest()
    item[field] = value
    with pytest.raises(ActionBackendError):
        validate_native60(item)


def test_126_outputs_cannot_enter_native60_decoder(monkeypatch):
    tensor = lambda shape: SimpleNamespace(shape=shape, type="tensor(float)", name="x")
    session = SimpleNamespace(get_inputs=lambda: [tensor([1, 2, 100, 17, 3])],
                              get_outputs=lambda: [tensor([1, 126])])
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        SessionOptions=SimpleNamespace, InferenceSession=lambda *a, **kw: session))
    with pytest.raises(ActionBackendError, match="shape/type mismatch"):
        Native60OnnxEngine("unused.onnx", manifest())


def test_candidate_mismatch_fails_before_download():
    with pytest.raises(ActionBackendError, match="disagree"):
        build_darvin_backend("native60_streaming", {"action_backend": "official126"}, {})


def test_explicit_candidate_load_error_never_falls_back(monkeypatch):
    plugin = PosePerceptionPlugin({"action_backend": "hybrid", "darvin_candidate": "native60_streaming"}, "test", None)
    def unavailable(*args, **kwargs):
        raise ActionBackendError("unavailable model")
    monkeypatch.setattr("plugins.darvin_action.build_darvin_backend", unavailable)
    with pytest.raises(ActionBackendError, match="unavailable model"):
        plugin._classifier_for(plugin._merged_config(""))
    assert plugin._backend_fallback is None


def test_explicit_candidate_does_not_silently_become_rules():
    plugin = PosePerceptionPlugin({"action_backend": "rules", "darvin_candidate": "native60_streaming"}, "test", None)
    with pytest.raises(ActionBackendError, match="disagree"):
        plugin._classifier_for(plugin._merged_config(""))
