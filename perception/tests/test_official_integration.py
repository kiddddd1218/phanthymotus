"""ROS-stub tests of timestamp propagation, explicit errors and 126 config."""
import queue
from types import SimpleNamespace

import pytest
import vision_stubs
from plugins import pose
from plugins.pose_action import PoseTracker


def bare_node():
    node = pose._PoseNode.__new__(pose._PoseNode)
    node._last_inference_time = 0.
    node._last_capture_time = None
    node._capture_clock = None
    node._processed_capture_time = None
    node._processed_capture_clock = None
    node._frame_interval = .1
    node._frame_queue = queue.Queue(maxsize=1)
    node._tracker = PoseTracker()
    node._gesture_tracker = None
    node._hand_channel = None
    node._last_prediction = {}
    node._last_actions = {}
    return node


def msg(sec, nsec=0):
    return SimpleNamespace(data=b'jpeg', header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nsec)))


def test_capture_cadence_is_independent_of_processing_clock(monkeypatch):
    node = bare_node()
    monkeypatch.setattr(pose.time, 'monotonic', lambda: 1000.)
    node._image_cb(msg(10))
    assert node._frame_queue.get_nowait() == (b'jpeg', 10., 'header')
    node._image_cb(msg(11))
    assert node._frame_queue.get_nowait() == (b'jpeg', 11., 'header')
    node._image_cb(msg(11, 50000000))
    assert node._frame_queue.empty()


def test_unstamped_input_records_ingress_not_processing_time(monkeypatch):
    node = bare_node()
    monkeypatch.setattr(pose.time, 'monotonic', lambda: 22.)
    node._image_cb(msg(0))
    assert node._frame_queue.get_nowait() == (b'jpeg', 22., 'ingress_monotonic')


def test_24fps_capture_yields_12fps_without_roundoff_drops(monkeypatch):
    node = bare_node()
    node._frame_interval = 1. / 12
    monkeypatch.setattr(pose.time, 'monotonic', lambda: 1000.)
    kept = 0
    for frame in range(240):
        nano = round((1 + frame / 24) * 1e9)
        node._image_cb(msg(nano // 1000000000, nano % 1000000000))
        if not node._frame_queue.empty():
            node._frame_queue.get_nowait()
            kept += 1
    assert kept == 120


def test_photo_passes_dimensions_to_official_backend(tmp_path, monkeypatch):
    import numpy as np
    from test_pose_plugin import _photo_plugin, FRAME_W, FRAME_H
    from plugins.official_action import OfficialActionBackend, canonical_labels
    photo = tmp_path / 'photo.jpg'
    photo.write_bytes(vision_stubs.frame_bytes(FRAME_W, FRAME_H))
    plugin, _ = _photo_plugin(str(photo))
    backend = OfficialActionBackend.__new__(OfficialActionBackend)
    backend.window_s, backend.min_score, backend.normalization = 2.5, 0., 'fix'
    backend.temporal_calibration = None
    row = next(r for r in canonical_labels() if r['activity'] == 'none')
    def predict(blob):
        assert blob.shape == (1, 2, 100, 17, 3)
        assert np.isfinite(blob).all()
        return {'name': row['activity'], 'name_zh': row['activity_zh'], 'score': .8, 'source': 'test'}
    backend.predict_blob = predict
    monkeypatch.setattr(plugin, '_classifier_for', lambda merged: backend)
    result = plugin.dispatch('pose', {'action': 'recognize_by_photo', 'image_path': str(photo)})
    assert result['persons'][0]['activity'] == 'none'
    assert result['persons'][0]['activity_zh'] == row['activity_zh']


def test_backward_timestamp_and_clock_switch_reset_all_temporal_consumers():
    node = bare_node()
    resets = []
    node._tracker = SimpleNamespace(reset=lambda: resets.append('tracks'))
    node._gesture_tracker = SimpleNamespace(reset=lambda: resets.append('gestures'))
    node._hand_channel = SimpleNamespace(reset=lambda: resets.append('hands'))
    node._accept_capture_time(10., 'header')
    node._accept_capture_time(11., 'header')
    assert not resets
    node._accept_capture_time(1., 'header')
    assert resets == ['tracks', 'gestures', 'hands']
    node._accept_capture_time(100., 'ingress_monotonic')
    assert len(resets) == 6


def test_broken_official_manifest_never_falls_back_to_geometry(tmp_path):
    plugin = pose.PosePerceptionPlugin({'plugins': {'pose': {'action_backend': 'official126'}}}, 'test', None)
    merged = plugin._merged_config('test')
    merged['action_backend'] = 'official126'
    merged['official_action_manifest'] = str(tmp_path / 'missing.json')
    with pytest.raises(FileNotFoundError):
        plugin._classifier_for(merged)
    assert plugin._backend_fallback is None


def test_list_actions_exposes_exact_126_catalogue():
    plugin = pose.PosePerceptionPlugin({}, 'test', None)
    plugin._action_backend = 'official126'
    result = plugin.dispatch('list_actions', {})
    assert len(result['activities']) == 126
    assert [x['official_id'] for x in result['activities']] == list(range(1, 127))
    assert any(x['name'] == 'none' for x in result['activities'])
