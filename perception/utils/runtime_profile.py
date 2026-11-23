"""Explicit startup profiles; no model or ROS imports belong in this module."""
import json
import os
import time
from contextlib import contextmanager
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


def _memory_group(proc=Path("/proc/self")):
    groups = {}
    for row in (proc / "cgroup").read_text().splitlines():
        _, controllers, member = row.split(":", 2)
        if member.startswith("/") and ".." not in Path(member).parts:
            for controller in controllers.split(","):
                groups[controller] = Path(member)
    for row in (proc / "mountinfo").read_text().splitlines():
        before, after = row.split(" - ", 1)
        kind, _, options = after.split()[:3]
        if kind == "cgroup" and "memory" in options.split(","):
            controller = "memory"
        elif kind == "cgroup2":
            controller = ""
        else:
            continue
        member = groups.get(controller)
        if member is None:
            continue
        root, mount = (Path(value.replace("\\040", " ")) for value in before.split()[3:5])
        try:
            relative = member.relative_to(root)
        except ValueError:
            relative = member.relative_to("/")
        path = mount / relative
        counter = "memory.current" if kind == "cgroup2" else "memory.usage_in_bytes"
        if (path / counter).is_file():
            return path, kind
    return None


def _group_counters(proc=Path("/proc/self")):
    group = _memory_group(proc)
    if not group:
        return {}
    path, kind = group
    fields = (("memory.current", "memory.peak", "memory.events") if kind == "cgroup2"
              else ("memory.usage_in_bytes", "memory.max_usage_in_bytes", "memory.failcnt", "memory.oom_control"))
    values = {"kind": kind}
    for name in fields:
        try:
            raw = (path / name).read_text()
            if name in ("memory.events", "memory.oom_control"):
                values[name] = {key: int(value) for key, value in (row.split() for row in raw.splitlines())}
            else:
                values[name] = int(raw)
        except (OSError, ValueError):
            continue
    return values


def trace_enabled():
    return os.environ.get("POSE_STARTUP_TRACE") == "1"


@contextmanager
def memory_stage(logger, name):
    enabled = trace_enabled()
    if enabled:
        log_process_memory(logger, name + ".begin")
    yield
    if enabled:
        log_process_memory(logger, name + ".end")


def log_process_memory(logger, stage):
    """Small Linux process snapshot, not GPU usage or proof of an OOM kill."""
    result = {"stage": stage, "pid": os.getpid(), "monotonic_s": time.monotonic()}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in ("VmRSS", "VmHWM", "Threads"):
                result[key] = value.strip()
    except OSError:
        result["process_status"] = "unavailable"
    if trace_enabled():
        try:
            for row in Path("/proc/meminfo").read_text().splitlines():
                if row.startswith("MemAvailable:"):
                    result["host_available_kib"] = int(row.split()[1])
                    break
        except (OSError, ValueError, IndexError):
            pass
        try:
            result["cgroup"] = _group_counters()
        except (OSError, ValueError, IndexError):
            result["cgroup"] = {}
    logger.info("[runtime_memory] %s", json.dumps(result, sort_keys=True))
