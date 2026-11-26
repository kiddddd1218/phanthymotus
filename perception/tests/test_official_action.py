"""CPU contracts: exact labels, causality and explicit artifact failures."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plugins.official_action import (ActionBackendError, CausalWindow, FORMAT, INPUT_SHAPE,
    OfficialActionBackend, OfficialOnnxSessionCache, canonical_labels, causal_blob, file_hash, pose_blob,
    schema_hash, validate_manifest)
from plugins.pose_stgcn import SkeletonActionBackend, pre_normalize_2d, resample_clip


class FakeEngine:
    input_shape = INPUT_SHAPE

    def __init__(self, values):
        self.values = values

    def infer(self, blob):
        return [self.values]


class OfficialActionTests(unittest.TestCase):
    def test_retired_node_keeps_no_tracking_state_in_reused_session(self):
        from test_pose_plugin import _plugin
        plugin, _ = _plugin({"darvin_candidate": "official126_streaming", "action_backend": "official126"})
        paths = {"artifact.json": str(self.path / "artifact.json")}
        spec = {"backend": "official126", "revision": "fixture", "files": {"model.onnx": self.manifest["engine"]}}
        engine = FakeEngine(np.zeros((1, 126), np.float32))
        with patch("plugins.darvin_action.candidate_spec", return_value=spec), \
                patch("plugins.darvin_assets.ensure_candidate_bundle", return_value=paths), \
                patch("plugins.official_action._OnnxEngine", return_value=engine) as load:
            for instance in ("__warmup__", "/clip", "/clip"):
                plugin.dispatch("pose", {"action": "start", "input_topic": "/input", "instance_id": instance})
                try:
                    node = plugin._nodes[instance]
                    self.assertEqual(node._tracker.tracks, [])
                    self.assertIsNone(node._classifier.last_error)
                    self.assertIs(node._classifier._ensure_engine(), engine)
                    node._tracker.update([[0., 0., 100., 200.]], self.raw[:1], 1., (640, 480))
                    self.assertEqual(len(node._tracker.tracks), 1)
                    node._classifier._load_error = "retired clip error"
                finally:
                    plugin.dispatch("pose", {"action": "stop", "instance_id": instance})
                self.assertFalse(plugin._nodes)
            load.assert_called_once()

    def test_session_reuse_still_validates_and_isolates_per_clip_settings(self):
        cache = OfficialOnnxSessionCache()
        path = self.path / 'artifact.json'
        engine = FakeEngine(np.zeros((1, 126), np.float32))
        with patch('plugins.official_action._OnnxEngine', return_value=engine) as load:
            first = OfficialActionBackend(manifest_path=path, window_s=1., min_score=0.)
            second = OfficialActionBackend(manifest_path=path, window_s=2.5, min_score=.9)
            self.assertIs(first._ensure_engine(cache), second._ensure_engine(cache))
            first._load_error = 'previous clip'
            self.assertIsNone(second.last_error)
            self.assertNotEqual(first.history_s, second.history_s)
            self.assertNotEqual(first.min_score, second.min_score)
            (self.path / 'model.onnx').write_bytes(b'wrong model')
            with self.assertRaises(ActionBackendError):
                OfficialActionBackend(manifest_path=path)._ensure_engine(cache)
            load.assert_called_once()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        (self.path / 'model.onnx').write_bytes(b'fake engine for contract tests')
        labels = canonical_labels()
        self.manifest = {'format': FORMAT, 'classes': labels, 'taxonomy_sha256': schema_hash(labels),
            'input_shape': list(INPUT_SHAPE), 'output_shape': [1, 126], 'output_kind': 'logits',
            'preprocessing': {'normalization': 'fix', 'order': 'normalize_then_resample',
                              'joint_channels': 'COCO17 x/y/confidence', 'person_slots': 'primary_then_zero'},
            'engine': {'kind': 'onnx_cpu', 'file': 'model.onnx', 'sha256': file_hash(self.path / 'model.onnx')}}
        (self.path / 'artifact.json').write_text(json.dumps(self.manifest))
        self.raw = np.zeros((5, 17, 3), dtype=np.float32)
        self.raw[..., :2] = np.arange(170).reshape(5, 17, 2)
        self.raw[..., 2] = .9

    def backend(self, values):
        return OfficialActionBackend(manifest_path=self.path / 'artifact.json', engine=FakeEngine(values))

    def test_all_126_rows_decode_exactly_including_none_and_static(self):
        for row in canonical_labels():
            logits = np.zeros((1, 126), dtype=np.float32)
            logits[0, row['output_index']] = 10
            backend = self.backend(logits)
            frame = SimpleNamespace(t=0., keypoints=self.raw[0], image_size=(640, 480))
            result = backend.classify([frame])['activity']
            self.assertEqual((result['name'], result['name_zh'], result['official_id']),
                             (row['activity'], row['activity_zh'], row['official_id']))
            self.assertGreater(result['score'], .99)

    def test_wrong_shapes_nonfinite_and_multiple_outputs_fail(self):
        for shape in ((1, 60), (1, 127), (126,), (2, 126)):
            with self.assertRaises(ActionBackendError):
                self.backend(np.zeros(shape)).predict_blob(pose_blob(self.raw, (640, 480)))
        with self.assertRaises(ActionBackendError):
            self.backend(np.full((1, 126), np.nan)).predict_blob(pose_blob(self.raw, (640, 480)))

    def calibrated_manifest(self):
        parameters = dict(history_boundary_s=1., early_temperature=2., mature_temperature=1.)
        receipt = self.path / 'calibration.json'
        receipt.write_text(json.dumps(dict(fit_split='train', parameters=parameters,
            engine_sha256=self.manifest['engine']['sha256'])))
        self.manifest['temporal_calibration'] = dict(format='darvin-temporal-temperature-v1',
            **parameters, fit_split='train', fit_receipt=dict(file=receipt.name, sha256=file_hash(receipt)))
        (self.path / 'artifact.json').write_text(json.dumps(self.manifest))

    def test_temporal_temperature_changes_confidence_and_preserves_labels(self):
        logits = np.zeros((1,126),np.float32);logits[0,125]=10
        raw=self.backend(logits).predict_blob(pose_blob(self.raw,(640,480)))
        self.calibrated_manifest();backend=self.backend(logits)
        early=backend.classify([SimpleNamespace(t=0.,keypoints=self.raw[0],image_size=(640,480))])
        mature=backend.classify([SimpleNamespace(t=t,keypoints=self.raw[i],image_size=(640,480))
                                 for i,t in enumerate((0.,.5,1.))])
        self.assertEqual(early['activity']['name'],raw['name'])
        self.assertLess(early['activity']['score'],raw['score'])
        self.assertEqual(mature['activity'],raw)
        self.assertEqual(early['evidence']['temperature'],2.)
        self.assertEqual(mature['evidence']['temperature'],1.)

    def test_invalid_or_tampered_temporal_calibration_is_refused(self):
        self.calibrated_manifest()
        for key,value in [('early_temperature',0.),('mature_temperature',float('nan')),
                          ('history_boundary_s',0.),('fit_split','val')]:
            bad=copy.deepcopy(self.manifest);bad['temporal_calibration'][key]=value
            with self.assertRaises(ActionBackendError):validate_manifest(bad)
        (self.path/'calibration.json').write_text('{}')
        with self.assertRaises(ActionBackendError):self.backend(np.zeros((1,126),np.float32))

    def test_manifest_temperature_must_match_training_receipt(self):
        self.calibrated_manifest()
        self.manifest['temporal_calibration']['early_temperature']=3.
        (self.path/'artifact.json').write_text(json.dumps(self.manifest))
        with self.assertRaises(ActionBackendError):self.backend(np.zeros((1,126),np.float32))

    def test_replacing_weights_requires_matching_calibration_receipt(self):
        self.calibrated_manifest()
        (self.path/'model.onnx').write_bytes(b'different model')
        self.manifest['engine']['sha256']=file_hash(self.path/'model.onnx')
        (self.path/'artifact.json').write_text(json.dumps(self.manifest))
        with self.assertRaises(ActionBackendError):self.backend(np.zeros((1,126),np.float32))

    def test_manifest_order_and_engine_tampering_fail(self):
        bad = copy.deepcopy(self.manifest)
        bad['classes'][0], bad['classes'][1] = bad['classes'][1], bad['classes'][0]
        with self.assertRaises(ActionBackendError):
            validate_manifest(bad)
        (self.path / 'model.onnx').write_bytes(b'tampered')
        with self.assertRaises(ActionBackendError):
            self.backend(np.zeros((1, 126)))

    def test_normalize_before_resample_matches_training(self):
        result = pose_blob(self.raw, (640, 480))
        expected = resample_clip(pre_normalize_2d(self.raw, (640, 480), 'fix'), 100)
        np.testing.assert_array_equal(result[0, 0], expected)
        self.assertFalse(result[0, 1].any())
        np.testing.assert_allclose(result[0, 0, :, :, 2], .9)

    def test_future_frames_cannot_change_causal_input(self):
        times = np.arange(5, dtype=float)
        original = causal_blob(times, self.raw, (640, 480), 2., 1.)
        changed = self.raw.copy()
        changed[3:, :, :2] += 10000
        np.testing.assert_array_equal(original, causal_blob(times, changed, (640, 480), 2., 1.))
        np.testing.assert_array_equal(original, pose_blob(self.raw[1:3], (640, 480)))

    def test_stream_reset_missing_and_backward_time(self):
        window = CausalWindow(1.)
        window.push(0., self.raw[0], (640, 480))
        with self.assertRaises(ActionBackendError):
            window.push(0., self.raw[1], (640, 480))
        window.push(2., None, (640, 480))
        self.assertIsNone(window.blob())
        window.reset()
        window.push(0., self.raw[1], (640, 480))
        np.testing.assert_array_equal(window.blob(), pose_blob(self.raw[1:2], (640, 480)))

    def test_old_decoder_rejects_126_logits(self):
        backend = SkeletonActionBackend.__new__(SkeletonActionBackend)
        backend._ensure_engine = lambda: FakeEngine(np.zeros((1, 126)))
        with self.assertRaises(ActionBackendError):
            backend._run(np.zeros(INPUT_SHAPE))

    def test_failure_is_reported_without_geometry_fallback(self):
        backend = self.backend(np.zeros((1, 60)))
        frame = SimpleNamespace(t=0., keypoints=self.raw[0], image_size=(640, 480))
        result = backend.classify([frame])
        self.assertIsNone(result['activity'])
        self.assertTrue(result['backend_error'])
        self.assertIn('126', backend.last_error)

    def test_onnx_metadata_rejects_preprocessing_drift(self):
        from unittest.mock import patch
        from plugins.official_action import _OnnxEngine
        import onnxruntime
        inputs = [SimpleNamespace(shape=list(INPUT_SHAPE), type='tensor(float)', name='keypoint')]
        outputs = [SimpleNamespace(shape=[1, 126], type='tensor(float)', name='logits')]
        metadata = {'format': FORMAT, 'taxonomy_sha256': self.manifest['taxonomy_sha256'],
                    'normalization': 'fix', 'preprocessing_order': 'normalize_then_resample', 'output_kind': 'logits'}
        session = SimpleNamespace(get_inputs=lambda: inputs, get_outputs=lambda: outputs,
                                  get_modelmeta=lambda: SimpleNamespace(custom_metadata_map=metadata))
        with patch.object(onnxruntime, 'InferenceSession', return_value=session):
            _OnnxEngine(self.path / 'model.onnx', metadata['taxonomy_sha256'], 'fix')
            with self.assertRaises(ActionBackendError):
                _OnnxEngine(self.path / 'model.onnx', metadata['taxonomy_sha256'], 'auto')

    def test_linear_time_resample_and_linear_beta_schedule(self):
        parameters = dict(history_boundary_s=1.5, early_temperature=4.0,
                          mature_temperature=1.25, schedule='linear_beta',
                          maturity_min_scale=0.08)
        receipt = self.path / 'calibration.json'
        receipt.write_text(json.dumps(dict(fit_split='train', parameters=parameters,
            engine_sha256=self.manifest['engine']['sha256'])))
        self.manifest['preprocessing']['resample'] = 'linear_time'
        self.manifest['temporal_calibration'] = dict(
            format='darvin-temporal-temperature-v1', **parameters, fit_split='train',
            fit_receipt=dict(file=receipt.name, sha256=file_hash(receipt)))
        (self.path / 'artifact.json').write_text(json.dumps(self.manifest))
        logits = np.zeros((1, 126), np.float32)
        logits[0, 10] = 8.0
        backend = OfficialActionBackend(manifest_path=self.path / 'artifact.json',
                                        window_s=12.0, engine=FakeEngine(logits))
        f0 = SimpleNamespace(t=0.0, keypoints=self.raw[0], image_size=(640, 480))
        f_mid = SimpleNamespace(t=0.75, keypoints=self.raw[1], image_size=(640, 480))
        f_mat = SimpleNamespace(t=1.5, keypoints=self.raw[2], image_size=(640, 480))
        r0 = backend.classify([f0])
        rmid = backend.classify([f0, f_mid])
        rmat = backend.classify([f0, f_mid, f_mat])
        self.assertAlmostEqual(r0['evidence']['temperature'], 4.0, places=6)
        self.assertAlmostEqual(rmat['evidence']['temperature'], 1.25, places=6)
        self.assertLess(r0['activity']['score'], rmid['activity']['score'])
        self.assertLess(rmid['activity']['score'], rmat['activity']['score'])


if __name__ == '__main__':
    unittest.main()
