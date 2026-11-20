"""Explicit startup profiles; no model or ROS imports belong in this module."""
import json
from pathlib import Path


def apply_runtime_profile(config):
    """Keep the benchmark's pose-only contract even if plugin flags are edited.

    The ordinary robot configuration is unchanged unless it opts into this
    profile. Copy nested plugin dictionaries so callers retain their input.
    """
    profile = config.get("runtime_profile", "default")
    if profile == "default":
        return config
    if profile != "pose_only":
        raise ValueError("unknown perception runtime_profile: %r" % profile)
    plugins = config.get("plugins", {})
    pose = plugins.get("pose")
    if not isinstance(pose, dict) or pose.get("enabled") is not True:
        raise ValueError("pose_only requires plugins.pose.enabled: true")
    selected = {}
    for name, values in plugins.items():
        selected[name] = dict(values) if isinstance(values, dict) else {}
        if name != "pose":
            selected[name]["enabled"] = False
    # These bundle plugins default to enabled even when their keys are absent.
    for name in ("echo_cancellation", "audio_mixer"):
        selected.setdefault(name, {})["enabled"] = False
    return dict(config, plugins=selected)


def executor_options(config):
    # ROS otherwise sizes its pool from host CPU affinity in every instance.
    return {"num_threads": 2} if config.get("runtime_profile") == "pose_only" else {}


def log_process_memory(logger, stage):
    """Small Linux process snapshot, not GPU usage or proof of an OOM kill."""
    result = {"stage": stage}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in ("VmRSS", "VmHWM", "Threads"):
                result[key] = value.strip()
    except OSError:
        result["process_status"] = "unavailable"
    logger.info("[runtime_memory] %s", json.dumps(result, sort_keys=True))
