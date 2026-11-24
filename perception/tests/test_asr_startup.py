"""Startup interleavings without ROS, a model, or child processes."""
from types import SimpleNamespace

import pytest

from vision_stubs import _FakeAudioChunk
from plugins import asr


@pytest.fixture
def pipeline(monkeypatch):
    node = asr._ASRNode('/test/audio', adapter=object(), language='en')
    queues, processes = [], []
    controls = SimpleNamespace(fail_start=False, dead_on_start=False,
                               before_start=lambda: None)

    class Queue:
        def __init__(self, **kwargs):
            self.items, self.closed = [], False
            queues.append(self)

        def put_nowait(self, item):
            assert not self.closed
            self.items.append(item)

        def cancel_join_thread(self):
            pass

        def close(self):
            self.closed = True

    class Process:
        def __init__(self, **kwargs):
            self.pid = self.exitcode = None
            self.alive = self.closed = False
            processes.append(self)

        def start(self):
            controls.before_start()
            if controls.fail_start:
                raise RuntimeError('VAD launch failed')
            self.pid = 123
            self.alive = not controls.dead_on_start
            self.exitcode = 1 if controls.dead_on_start else None

        def is_alive(self):
            return self.alive

        def join(self, **kwargs):
            assert node._vad_stop.is_set()
            self.alive, self.exitcode = False, 0

        def close(self):
            self.closed = True

    class Worker:
        def __init__(self, **kwargs):
            pass

        def start(self):
            node._worker_ready.set()

        def is_alive(self):
            return False

    monkeypatch.setattr(asr.multiprocessing, 'Queue', Queue)
    monkeypatch.setattr(asr.multiprocessing, 'Process', Process)
    monkeypatch.setattr(asr.multiprocessing, 'Event', asr.threading.Event)
    monkeypatch.setattr(asr.threading, 'Thread', Worker)

    msg = _FakeAudioChunk()
    msg.format, msg.data = 'audio/pcm-16k', bytes(1024)
    msg.header.stamp.sec = 1700000000
    subscribe = node.create_subscription

    def subscribe_with_immediate_audio(*args):
        sub = subscribe(*args)
        if sub.topic == node._input_topic:
            sub.callback(msg)
        return sub

    monkeypatch.setattr(node, 'create_subscription', subscribe_with_immediate_audio)

    def early_audio():
        for sub in list(node.subscriptions):
            if sub.topic == node._input_topic:
                sub.callback(msg)

    controls.before_start = early_audio
    yield node, controls, queues, processes, msg
    node.stop()


def test_first_available_audio_reaches_ready_vad_and_restart_is_clean(pipeline):
    node, controls, queues, processes, msg = pipeline
    for cycle in range(2):
        assert node.start()['state'] == 'running'
        assert queues[-2].items == [(bytes(1024), 1700000000.0, True)]
        assert not queues[-2].closed
        assert node.stop() == {'state': 'idle'}
        assert not node.subscriptions
        assert processes[-1].closed
        assert all(q.closed for q in queues)
    assert len(processes) == 2


def test_vad_launch_failure_releases_partial_pipeline(pipeline):
    node, controls, queues, processes, msg = pipeline
    controls.fail_start = True
    assert node.start()['state'] == 'error'
    assert node._vad_proc is None
    assert node._pcm_queue is None
    assert not node.subscriptions
    assert all(q.closed for q in queues)
    assert processes[0].closed


def test_vad_death_during_start_is_an_error_not_idle_or_running(pipeline):
    node, controls, queues, processes, msg = pipeline
    controls.dead_on_start = True
    assert node.start()['state'] == 'error'
    assert node.state == 'error'


def test_stop_during_process_launch_is_preserved(pipeline):
    node, controls, queues, processes, msg = pipeline
    controls.before_start = node.request_stop
    assert node.start()['state'] == 'idle'
    assert node._stop_event.is_set()
    assert queues[0].items == []


def test_stop_during_worker_ready_wait_cannot_return_running(pipeline, monkeypatch):
    node, controls, queues, processes, msg = pipeline
    subscribe = node.create_subscription

    def subscribe_then_cancel_during_wait(*args):
        sub = subscribe(*args)
        # Deterministically cancel after the first-chunk wait, while start()
        # is waiting for worker initialization.
        monkeypatch.setattr(node._worker_ready, 'wait',
                            lambda **kw: node.request_stop())
        return sub

    monkeypatch.setattr(node, 'create_subscription', subscribe_then_cancel_during_wait)
    assert node.start()['state'] == 'idle'
    assert node._stop_event.is_set()


def test_running_vad_death_still_stops_audio(pipeline):
    node, controls, queues, processes, msg = pipeline
    assert node.start()['state'] == 'running'
    processes[0].alive, processes[0].exitcode = False, 7
    node._audio_cb(msg)
    assert node.state == 'error'
    assert node._stop_event.is_set()
    assert all(q.closed for q in queues)
