# Darvin314 streaming candidates

This change selects a fine-tuned action model explicitly, downloads its public ModelScope bundle with pinned sizes/SHA256 hashes, and loads it through the measured preprocessing and decoder. It adds one optional `plugins.pose.darvin_candidate` setting. Empty selection preserves the original backend behavior. A selected package must agree with `action_backend`; load/configuration failures do not silently select the original weights or geometry-only behavior.

| Candidate | Backend | Local macro-F1, 333 clips / 123 supported classes |
|---|---|---:|
| native60_streaming | hybrid, original NTU60 decoding and motion/fall gates | 0.19843729 |
| official126_streaming | official126, exact 126-label vocabulary | 0.52650723 |

These are reused local development results from cached YOLO11s poses, not official Darvin scores. The native60 head has 60 outputs; the existing single-person decoder uses 49 learned labels. Its shared-class subset score must not be compared with the full-taxonomy score. Whole-clip models are separate artifacts and are not selected by these submissions.

Both candidates use a 2.5-second causal window, 0.5-second action interval, label_hold=1, fps=12, confidence=0.4, max_persons=1, and hands=false. The 60-class model uses fix normalization before resampling, retaining its original geometry/hybrid behavior. The 126-class model retains its model-bound, training-fitted temporal calibration and minimum score 0. Native and official output spaces cannot be interchanged.

The complete immutable URLs and hashes are in `perception/data/darvin_candidates.json`. Only inference files are fetched: ONNX, artifact.json and, for the 126-class candidate, calibration.json. No weights belong in this code repository. Cache directories under ACTION_MODEL_DIR (default /models/action) include the selected model revision; existing valid bundles are reused by the standard verified-bundle downloader.

The action head runs explicitly through CPUExecutionProvider with one intra-op and one inter-op thread. Pose remains on the supplied runtime's TensorRT path. The exported graphs use ONNX IR8/opset16. Local checks use ONNX Runtime1.20.1 and cannot establish Jetson ARM/TensorRT performance, total memory, GPU utilization or official acceptance.

The isolated integration reproduced all serialized action predictions from the saved 333-case native60 and 333-case official126 replays: 4,534 and 8,327 emissions respectively, with unchanged final clip labels. ROS/cv2 test stubs supplied the cached-pose ingress; no MP4/pose extraction, GPU execution or target-device test was performed by that check.

Code provenance is the supplied upstream revision 7b16c433463dcb7017eb95b5bb40ef99297d3d34. The submission commit must be the new immutable code commit containing the selected configuration. ModelScope artifact commits and the upstream reference revision do not substitute for that commit. MCP_PORT and WS_PORT environment overrides are preserved.
