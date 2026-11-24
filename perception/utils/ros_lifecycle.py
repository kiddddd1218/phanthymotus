"""
utils/ros_lifecycle.py — Safe teardown of per-instance ROS2 nodes.

Multi-instance plugins create one rclpy Node per input topic and register it
with the bundle's MultiThreadedExecutor. When an instance is retired (topic
change, stop-all, shutdown) the node must be removed from the executor *and*
destroyed, otherwise its subscriptions keep matching camera publishers and
its callbacks keep waking the executor for the rest of the process. The ASR
plugin established this order; OCR and obstacle reuse it here.
"""

from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)


class PoseClipSpin:
    """Own one clip's ROS wait set and callbacks until its node is retired."""

    def __init__(self, node):
        from rclpy.executors import SingleThreadedExecutor

        self._executor = SingleThreadedExecutor()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, name="pose_clip_spin", daemon=True)
        try:
            self._executor.add_node(node)
            self._thread.start()
        except Exception:
            self._executor.remove_node(node)
            self._executor.shutdown()
            raise

    def _spin(self):
        while not self._stop.is_set():
            try:
                # Bound the wait even if stop arrives just before spin_once.
                self._executor.spin_once(timeout_sec=0.25)
            except Exception:
                log.exception("pose clip ROS callback failed")
                self._stop.wait(1.0)

    def request_stop(self):
        self._stop.set()
        self._executor.wake()

    def wait(self, timeout):
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def close(self, node):
        # The caller has already joined the spin thread.
        self._executor.remove_node(node)
        self._executor.shutdown()


def dispose_node(executor, node, *, label: str = "") -> None:
    """Remove `node` from `executor` and destroy it, logging (not raising) errors.

    Callers stop the node's own workers/subscriptions first; this helper only
    handles the executor + rclpy handle lifecycle. Every step is attempted
    even if a previous one failed.
    """
    name = label or getattr(node, "get_name", lambda: type(node).__name__)()
    try:
        executor.remove_node(node)
    except Exception as error:
        log.warning("failed to remove ROS node %r from executor: %s", name, error)
    try:
        node.destroy_node()
    except Exception as error:
        log.warning("failed to destroy ROS node %r: %s", name, error)


__all__ = ["dispose_node"]
