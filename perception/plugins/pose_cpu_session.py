"""plugins/pose_cpu_session.py: CPU ONNX Runtime pose session, explicit opt-in alternative to TRT."""
import os

import numpy as np
import onnxruntime as ort

from plugins.vision_runtime import letterbox, to_blob

_IN_SHAPE, _OUT_SHAPE = [1, 3, 640, 640], [1, 300, 57]


class PoseCpuSession:
    """Fixed-shape [1,3,640,640] -> [1,300,57] pose model on CPUExecutionProvider, one thread."""

    def __init__(self, path):
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"pose CPU ONNX not found: {path!r}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
        self._session = ort.InferenceSession(
            path, sess_options=opts, providers=["CPUExecutionProvider"])
        inputs, outputs = self._session.get_inputs(), self._session.get_outputs()
        shapes = [list(i.shape) for i in inputs], [list(o.shape) for o in outputs]
        if (shapes != ([_IN_SHAPE], [_OUT_SHAPE]) or inputs[0].type != "tensor(float)"
                or outputs[0].type != "tensor(float)"):
            self._session = None
            raise ValueError(f"{path}: need float32 {_IN_SHAPE} -> {_OUT_SHAPE}, got {shapes}")
        self._input_name, self._output_name = inputs[0].name, outputs[0].name
        self.path, self.input_size = path, (640, 640)

    def infer(self, frame):
        image, meta = letterbox(frame, *self.input_size)
        blob = to_blob(image, np.float32)
        out = self._session.run([self._output_name], {self._input_name: blob})[0]
        return [out], meta

    def close(self):
        self._session = None
