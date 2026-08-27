#!/usr/bin/env python3
"""Local ONNX motion trajectory inspector."""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


WEB_ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
MAX_POINTS = 700
BODY_NAMES = (
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)


def yaw_from_wxyz(quaternion: np.ndarray) -> float:
    """Return yaw in radians for an ONNX WXYZ quaternion."""
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def clip_length(model_path: str | Path) -> int | None:
    """Read the exported motion frame count from the ONNX Clip constant."""
    model = onnx.load(str(model_path), load_external_data=False)
    constants: dict[str, int | float] = {}
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                values = numpy_helper.to_array(attr.t)
                if values.size == 1 and np.issubdtype(values.dtype, np.number):
                    constants[node.output[0]] = float(values.reshape(-1)[0])
    for node in model.graph.node:
        if node.op_type != "Clip" or len(node.input) < 3:
            continue
        maximum = constants.get(node.input[2])
        if maximum is not None and math.isfinite(float(maximum)):
            return int(maximum) + 1
    return None


def _output_map(session: ort.InferenceSession) -> dict[str, int]:
    return {item.name: index for index, item in enumerate(session.get_outputs())}


def extract_trajectory(payload: bytes, filename: str) -> dict[str, object]:
    if not payload:
        raise ValueError("ONNX 文件为空。")
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=True) as handle:
        handle.write(payload)
        handle.flush()
        session = ort.InferenceSession(handle.name, providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        if len(inputs) < 2 or inputs[0].name != "obs" or inputs[1].name != "time_step":
            raise ValueError("需要包含 obs 和 time_step 两个输入的 WBT Motion ONNX。")
        names = _output_map(session)
        required = ("body_pos_w", "body_quat_w")
        missing = [name for name in required if name not in names]
        if missing:
            raise ValueError(f"ONNX 缺少轨迹输出: {', '.join(missing)}。")
        obs_shape = inputs[0].shape
        if len(obs_shape) != 2 or not isinstance(obs_shape[1], int):
            raise ValueError("obs 输入形状不是固定的 [1, observation_size]。")
        observation = np.zeros((1, obs_shape[1]), dtype=np.float32)
        length = clip_length(handle.name) or 1
        length = min(length, 100_000)
        frames: list[dict[str, object]] = []
        stride = max(1, math.ceil(length / MAX_POINTS))
        for index in range(0, length, stride):
            outputs = session.run(None, {"obs": observation, "time_step": np.array([[index]], dtype=np.float32)})
            positions = np.asarray(outputs[names["body_pos_w"]][0], dtype=np.float64)
            quaternions = np.asarray(outputs[names["body_quat_w"]][0], dtype=np.float64)
            if positions.shape != (14, 3) or quaternions.shape != (14, 4):
                raise ValueError("body_pos_w/body_quat_w 不是预期的 [1, 14, 3]/[1, 14, 4]。")
            pelvis = positions[0]
            left_foot = positions[3]
            right_foot = positions[6]
            frames.append({
                "frame": index,
                "time": round(index / 50.0, 4),
                "position": [round(float(value), 6) for value in pelvis],
                "yaw": round(yaw_from_wxyz(quaternions[0]), 6),
                "leftFoot": [round(float(value), 6) for value in left_foot],
                "rightFoot": [round(float(value), 6) for value in right_foot],
            })
        if not frames:
            raise ValueError("没有可用的 ONNX 轨迹帧。")
        start, end = frames[0], frames[-1]
        dx = float(end["position"][0]) - float(start["position"][0])
        dy = float(end["position"][1]) - float(start["position"][1])
        return {
            "filename": Path(filename).name,
            "frameCount": length,
            "sampledFrameCount": len(frames),
            "fps": 50,
            "bodyNames": BODY_NAMES,
            "frames": frames,
            "start": start,
            "end": end,
            "displacement": {"x": round(dx, 6), "y": round(dy, 6), "distance": round(math.hypot(dx, dy), 6)},
            "headingChange": round(float(end["yaw"]) - float(start["yaw"]), 6),
        }


class Handler(BaseHTTPRequestHandler):
    server_version = "ONNXTrajectory/1.0"

    def _send(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        relative = "index.html" if path == "/" else path.removeprefix("/")
        target = (WEB_ROOT / relative).resolve()
        if WEB_ROOT not in target.parents or not target.is_file():
            self._send(HTTPStatus.NOT_FOUND, b"Not found", "text/plain; charset=utf-8")
            return
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self._send(HTTPStatus.OK, target.read_bytes(), content_type)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/trajectory":
            self._send(HTTPStatus.NOT_FOUND, b"Not found", "text/plain; charset=utf-8")
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > MAX_UPLOAD_BYTES:
                raise ValueError("文件为空或超过 512 MiB 限制。")
            result = extract_trajectory(self.rfile.read(size), self.headers.get("X-Filename", "motion.onnx"))
            body = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode()
            self._send(HTTPStatus.OK, body, "application/json; charset=utf-8")
        except Exception as exc:  # keep browser errors actionable
            body = json.dumps({"error": str(exc)}, ensure_ascii=False).encode()
            self._send(HTTPStatus.BAD_REQUEST, body, "application/json; charset=utf-8")

    def log_message(self, format: str, *args: object) -> None:
        print(f"[trajectory-web] {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"ONNX trajectory inspector: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
