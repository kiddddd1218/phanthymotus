"""Prevent unrelated eager model startup in action benchmark containers."""
import copy
import logging
from pathlib import Path
import sys
import types

import pytest
import yaml

from utils.runtime_profile import apply_runtime_profile, executor_options, log_process_memory

ROOT = Path(__file__).resolve().parents[1]


def test_pose_profile_blocks_unrelated_imports_even_with_enabled_flags(monkeypatch, tmp_path):
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    assert config["runtime_profile"] == "pose_only"
    for values in config["plugins"].values():
        values["enabled"] = True
    # Missing sections must not activate plugins whose legacy default is True.
    config["plugins"].pop("echo_cancellation")
    config["plugins"].pop("audio_mixer")
    original = copy.deepcopy(config)
    import builtins
    actual_import = builtins.__import__
    attempted = []
    constructed = []

    class Pose:
        PREFIX = "pose"
        def __init__(self, values, namespace, executor):
            constructed.append(values)
        def get_tools(self):
            return [{"name": "info"}]
        def dispatch(self, name, args):
            return {"name": name}

    monkeypatch.setitem(sys.modules, "plugins.pose", types.SimpleNamespace(PosePerceptionPlugin=Pose))

    def guarded_import(name, *args, **kwargs):
        if name.startswith("plugins.") and name != "plugins.pose":
            attempted.append(name)
            raise AssertionError("unrelated plugin imported: " + name)
        return actual_import(name, *args, **kwargs)

    from main import PerceptionBundle, _load_config
    config_path = tmp_path / "edited.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setenv("CONFIG_PATH", str(config_path))
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    bundle = PerceptionBundle(_load_config(), object())
    assert not attempted
    assert constructed == [original["plugins"]["pose"]]
    assert apply_runtime_profile(config)["plugins"]["pose"] == original["plugins"]["pose"]
    assert config == original
    assert bundle.get_all_tools() == [{"name": "pose_info"}]
    assert bundle.dispatch("pose_info", {}) == {"name": "info"}


def test_default_profile_preserves_regular_robot_settings():
    config = {"plugins": {"asr": {"enabled": True}, "pose": {"enabled": False}}}
    assert apply_runtime_profile(config) is config
    assert executor_options(config) == {}


@pytest.mark.parametrize("config", [
    {"runtime_profile": "typo"},
    {"runtime_profile": "pose_only"},
    {"runtime_profile": "pose_only", "plugins": {"pose": {"enabled": False}}},
])
def test_invalid_profile_fails_before_model_initialization(config):
    with pytest.raises(ValueError):
        apply_runtime_profile(config)


def test_pose_settings_and_small_executor_pool_are_preserved():
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    selected = apply_runtime_profile(config)
    assert selected["plugins"]["pose"] == config["plugins"]["pose"]
    assert [k for k, v in selected["plugins"].items() if v.get("enabled")] == ["pose"]
    assert executor_options(selected) == {"num_threads": 2}


def test_memory_snapshot_does_not_log_configuration(caplog):
    with caplog.at_level(logging.INFO):
        log_process_memory(logging.getLogger("snapshot_test"), "after_bundle")
    assert '"stage": "after_bundle"' in caplog.text
    assert "VmRSS" in caplog.text or "unavailable" in caplog.text


def test_required_pose_constructor_failure_stops_benchmark_startup(monkeypatch):
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    def broken(*args):
        raise RuntimeError("pose initialization failed")
    monkeypatch.setitem(sys.modules, "plugins.pose", types.SimpleNamespace(PosePerceptionPlugin=broken))
    from main import PerceptionBundle
    with pytest.raises(RuntimeError, match="pose initialization failed"):
        PerceptionBundle(apply_runtime_profile(config), object())
