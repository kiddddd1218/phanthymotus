"""Explicit startup profiles; no model or ROS imports belong in this module."""
import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path


_POSE_LOAD_LOCK = Path("/dev/shm/phanthymotus-pose-engine-load")


def release_idle_heap():
    """Return free glibc heap pages at worker lifecycle boundaries on Jetson."""
    import ctypes

    trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
    if trim is not None:
        trim(ctypes.c_size_t(0))


_ADMISSION_POLL_S = 0.5
DEFAULT_ADMISSION_TIMEOUT_S = 180.0


class PoseEngineAdmissionError(RuntimeError):
    """A cold pose load was refused before anything was allocated.

    Nothing native was created, so a later start may try again.
    """


def host_available_mib(meminfo=Path("/proc/meminfo")):
    """Whole-host MemAvailable in MiB, or None when it cannot be read.

    Host-wide on purpose: Jetson GPU memory is ordinary system RAM, and the
    judge's containers share one pool, so a per-cgroup figure would miss the
    neighbours whose load high-water this is meant to keep apart.
    """
    try:
        for row in meminfo.read_text().splitlines():
            if row.startswith("MemAvailable:"):
                return int(row.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _admission_budget_s(deadline_s):
    """The configured deadline as finite seconds; a bad value is a config error.

    Raised as ValueError, not as an admission refusal: a refusal is retried on
    the next start, and retrying cannot repair a setting.
    """
    budget = DEFAULT_ADMISSION_TIMEOUT_S if deadline_s is None else float(deadline_s)
    if not math.isfinite(budget) or budget < 0:
        raise ValueError(
            "engine_load_admission_timeout_s must be a finite number of "
            "seconds >= 0; got %r" % (deadline_s,))
    return budget


def _admit(logger, fd, floor, deadline):
    """Take the load lock, then hold it until the host has `floor` MiB available.

    One deadline spans both waits. The first lock attempt and the first
    reading are unconditional; once this call has slept, nothing observed at
    or after the deadline is accepted — a late poll that finds the lock free
    or the memory back is still a refusal — and no sleep outlasts the time
    remaining.
    """
    import fcntl

    waited = False

    def pause(reason):
        nonlocal waited
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise PoseEngineAdmissionError(reason)
        waited = True
        time.sleep(min(_ADMISSION_POLL_S, remaining))
        # The sleep may overrun (scheduler, memory pressure). Do not take
        # another sample for a deadline that has already passed.
        if time.monotonic() >= deadline:
            raise PoseEngineAdmissionError(reason)

    def expired():
        return waited and time.monotonic() >= deadline

    busy = ("pose engine load not admitted: the shared load lock stayed busy "
            "until the admission deadline")
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pause(busy)
            continue
        if expired():
            raise PoseEngineAdmissionError(busy)
        break

    warned = False
    while True:
        available = host_available_mib()
        if available is None:
            # Unknown is not "enough": proceeding blind is the failure this
            # gate exists to prevent.
            raise PoseEngineAdmissionError(
                "pose engine load not admitted: host MemAvailable is unreadable")
        short = ("pose engine load not admitted: host available %d MiB was not "
                 "confirmed at or above the %d MiB floor before the admission "
                 "deadline" % (available, floor))
        if expired():
            raise PoseEngineAdmissionError(short)
        if available >= floor:
            logger.info("[pose] engine load admitted: host available %d MiB "
                        ">= floor %d MiB%s", available, floor,
                        " (after waiting)" if waited else "")
            return
        if not warned:
            warned = True
            logger.warning("[pose] engine load waiting: host available %d MiB "
                           "< floor %d MiB", available, floor)
        pause(short)


@contextmanager
def pose_engine_load_slot(logger, enabled, *, min_available_mib=0, deadline_s=None):
    """Coordinate cold pose constructors across containers sharing /dev/shm.

    With `min_available_mib` > 0 the slot is also an admission gate: the
    constructor only runs once host MemAvailable is at or above the floor,
    checked while holding the lock so two workers cannot both be admitted on
    the same reading. Lock wait and headroom wait share one finite deadline;
    past it PoseEngineAdmissionError is raised and the lock is released. A
    nonfinite or negative `deadline_s` is a ValueError.
    A floor always takes the lock, whatever `enabled` says — an unlocked check
    would be a check two workers can pass together.
    """
    floor = max(0, int(min_available_mib or 0))
    if not enabled and not floor:
        yield
        return

    import fcntl

    deadline = None
    if floor:
        # Validated before the lock is touched.
        deadline = time.monotonic() + _admission_budget_s(deadline_s)

    # Keep the directory in place so all waiters lock the same inode.
    # Linux drops the advisory lock when the holder closes the fd or exits.
    _POSE_LOAD_LOCK.mkdir(mode=0o755, exist_ok=True)
    fd = os.open(_POSE_LOAD_LOCK, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        with memory_stage(logger, "pose.engine_load_wait"):
            if deadline is None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            else:
                _admit(logger, fd, floor, deadline)
        with memory_stage(logger, "pose.engine_load_slot"):
            yield
    finally:
        os.close(fd)


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


def _group_counters(proc=Path("/proc/self"), *, detailed=False):
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
    if detailed:
        # v1 reports cache/rss and v2 reports file/anon. Keep the kernel's
        # names and byte units so the two families can be compared correctly.
        wanted = {"cache", "rss", "mapped_file", "swap", "anon", "file",
                  "shmem", "active_file", "inactive_file",
                  "total_cache", "total_rss", "total_swap"}
        try:
            values["memory.stat"] = {key: int(value) for key, value in
                                     (row.split() for row in (path / "memory.stat").read_text().splitlines())
                                     if key in wanted}
        except (OSError, ValueError):
            pass
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
        detailed = stage in {"trt.deserialize.begin", "trt.deserialize.end",
                             "pose.engine_load_slot.end", "pose.warmup_retired"}
        if detailed:
            try:
                rollup = Path("/proc/self/smaps_rollup").read_text().splitlines()
                result["process_kib"] = {
                    key: int(value.split()[0]) for key, value in
                    (row.split(":", 1) for row in rollup if ":" in row)
                    if key in {"Pss", "Private_Clean", "Private_Dirty", "SwapPss"}}
            except (OSError, ValueError, IndexError):
                pass
        try:
            for row in Path("/proc/meminfo").read_text().splitlines():
                if row.startswith("MemAvailable:"):
                    result["host_available_kib"] = int(row.split()[1])
                    break
        except (OSError, ValueError, IndexError):
            pass
        try:
            result["cgroup"] = _group_counters(detailed=detailed)
        except (OSError, ValueError, IndexError):
            result["cgroup"] = {}
    logger.info("[runtime_memory] %s", json.dumps(result, sort_keys=True))
