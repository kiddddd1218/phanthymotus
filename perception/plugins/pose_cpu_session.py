"""plugins/pose_cpu_session.py: CPU ONNX Runtime pose session, explicit opt-in alternative to TRT."""
import os

import numpy as np
import onnxruntime as ort

from plugins.vision_runtime import letterbox, to_blob

_ALLOWED_SIZES = (256, 320, 384, 416, 480, 512, 640)
_OUT_SHAPE = [1, 300, 57]


class PoseCpuSession:
    """Fixed-shape [1,3,S,S] -> [1,300,57] pose model on CPUExecutionProvider, one thread."""

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
        in_ok = (
            len(inputs) == 1
            and len(shapes[0][0]) == 4
            and shapes[0][0][:2] == [1, 3]
            and shapes[0][0][2] == shapes[0][0][3]
            and shapes[0][0][2] in _ALLOWED_SIZES
        )
        if (not in_ok or shapes[1] != [_OUT_SHAPE]
                or inputs[0].type != "tensor(float)"
                or outputs[0].type != "tensor(float)"):
            self._session = None
            raise ValueError(
                f"{path}: need float32 [1, 3, S, S] (S in {_ALLOWED_SIZES}) -> {_OUT_SHAPE}, got {shapes}"
            )
        side = int(shapes[0][0][2])
        self._input_name, self._output_name = inputs[0].name, outputs[0].name
        self.path, self.input_size = path, (side, side)

    def infer(self, frame):
        image, meta = letterbox(frame, *self.input_size)
        blob = to_blob(image, np.float32)
        out = self._session.run([self._output_name], {self._input_name: blob})[0]
        return [out], meta

    def close(self):
        self._session = None
