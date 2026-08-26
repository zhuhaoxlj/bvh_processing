import json
import logging
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import pytest
import respx
from fastapi.testclient import TestClient
from httpx import Response

from bvh_processing.main import create_app
from bvh_processing.retargeting.exporter import RetargetArtifacts
from bvh_processing.training.service import TrainingArtifact

SOURCE_URL = "https://minio.example.com/motions/walk.bvh"
CALLBACK_URL = "https://backend.example.com/callbacks/bvh"
PROGRESS_CALLBACK_URL = "https://backend.example.com/progress-callbacks/bvh"
RETARGET_NPZ = b"tracking-npz-content"
RETARGET_JSON = b'{"robot":"unitree_g1","frames":[]}'
TRAIN_NPZ_URL = "https://minio.example.com/motions/walk_g1_tracking.npz"
TRAIN_NPZ = b"retargeted-npz-content"
TRAIN_MP4 = b"generated-mp4-content"
TRAIN_ONNX = b"generated-onnx-content"
BVH_CONTENT = b"""HIERARCHY
ROOT Hips
{
  OFFSET 0 0 0
  CHANNELS 1 Xposition
  JOINT LeftToe
  {
  }
  JOINT RightToe
  {
  }
}
MOTION
Frames: 1
Frame Time: 0.0333333
0
"""
UNSUPPORTED_BVH_CONTENT = b"HIERARCHY\nROOT Hips\nMOTION\nFrames: 1\n"
MERGE_SOURCE_URL_1 = "https://minio.example.com/motions/first.bvh"
MERGE_SOURCE_URL_2 = "https://minio.example.com/motions/second.bvh"
MERGE_BVH_1 = b"""HIERARCHY
ROOT Hips
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Yrotation Xrotation Zrotation
  JOINT LeftToe
  {
    OFFSET -1 -1 0
    CHANNELS 3 Yrotation Xrotation Zrotation
  }
  JOINT RightToe
  {
    OFFSET 1 -1 0
    CHANNELS 3 Yrotation Xrotation Zrotation
  }
}
MOTION
Frames: 2
Frame Time: 0.05
0 0 0 0 0 0 0 0 0 0 0 0
1 0 0 0 0 0 0 0 0 0 0 0
"""
MERGE_BVH_2 = b"""HIERARCHY
ROOT Hips
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Yrotation Xrotation Zrotation
  JOINT LeftToe
  {
    OFFSET -1 -1 0
    CHANNELS 3 Yrotation Xrotation Zrotation
  }
  JOINT RightToe
  {
    OFFSET 1 -1 0
    CHANNELS 3 Yrotation Xrotation Zrotation
  }
}
MOTION
Frames: 2
Frame Time: 0.05
2 0 0 0 0 0 0 0 0 0 0 0
3 0 0 0 0 0 0 0 0 0 0 0
"""
MERGE_BVH_HIGH_FPS = b"""HIERARCHY
ROOT Hips
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Yrotation Xrotation Zrotation
  JOINT LeftToe
  {
    OFFSET -1 -1 0
    CHANNELS 3 Yrotation Xrotation Zrotation
  }
  JOINT RightToe
  {
    OFFSET 1 -1 0
    CHANNELS 3 Yrotation Xrotation Zrotation
  }
}
MOTION
Frames: 4
Frame Time: 0.025
0 0 0 0 0 0 0 0 0 0 0 0
1 0 0 0 0 0 0 0 0 0 0 0
2 0 0 0 0 0 0 0 0 0 0 0
3 0 0 0 0 0 0 0 0 0 0 0
"""


def _request_body() -> dict[str, object]:
    return {
        "actionId": "action-42",
        "originalFileUrl": SOURCE_URL,
        "handleOptions": [1, 2, 3],
        "callbackUrl": CALLBACK_URL,
    }


def test_process_accepts_task_and_callbacks_with_file() -> None:
    app = create_app()
    with respx.mock:
        respx.get(SOURCE_URL).mock(
            return_value=Response(
                200,
                content=BVH_CONTENT,
                headers={"Content-Length": str(len(BVH_CONTENT))},
            )
        )
        callback = respx.post(CALLBACK_URL).mock(return_value=Response(204))
        progress_callback = respx.post(PROGRESS_CALLBACK_URL).mock(
            return_value=Response(204)
        )

        with TestClient(app) as client:
            response = client.post("/api/v1/bvh/process", json=_request_body())

    body = response.json()
    assert response.status_code == 200
    assert body["success"] is True
    assert body["message"] == "任务已接收"
    UUID(body["taskId"])

    callback_body = callback.calls.last.request.content
    assert b'name="actionId"' in callback_body
    assert b"action-42" in callback_body
    assert b"\r\n\r\ntrue\r\n" in callback_body
    assert b'name="file"' in callback_body
    assert b'filename="walk_processed.bvh"' in callback_body
    assert BVH_CONTENT in callback_body
    assert b"callbackToken" not in callback_body
    assert len(callback.calls) == 1
    assert len(progress_callback.calls) == 3

    progress_bodies = [
        json.loads(call.request.content) for call in progress_callback.calls
    ]
    assert [body["progress"] for body in progress_bodies] == [35, 65, 95]
    assert [body["step"] for body in progress_bodies] == [2, 3, 4]
    assert [body["stepCode"] for body in progress_bodies] == [
        "DENOISE",
        "SMOOTH_FRAME",
        "FOOT_LOCK",
    ]
    assert all(body["actionId"] == "action-42" for body in progress_bodies)
    assert all(body["originalFileUrl"] == SOURCE_URL for body in progress_bodies)


def test_retarget_accepts_task_and_callbacks_with_npz_and_json(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(
        logging.INFO,
        logger="bvh_processing.retargeting.task",
    )
    payload = {
        "originalFileUrl": SOURCE_URL,
        "robotType": 1,
        "callbackUrl": CALLBACK_URL,
    }
    artifacts = RetargetArtifacts(
        npz=BytesIO(RETARGET_NPZ),
        npz_filename="walk_g1_tracking.npz",
        preview=BytesIO(RETARGET_JSON),
        preview_filename="walk_g1_preview.json",
    )

    with (
        patch(
            "bvh_processing.retargeting.task.retarget_downloaded_bvh",
            return_value=artifacts,
        ),
        respx.mock,
    ):
        respx.get(SOURCE_URL).mock(return_value=Response(200, content=BVH_CONTENT))
        callback = respx.post(CALLBACK_URL).mock(return_value=Response(204))

        with TestClient(create_app()) as client:
            response = client.post("/api/v1/bvh/retarget", json=payload)

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["message"] == "重定向任务已接收"
    UUID(response.json()["taskId"])
    assert len(callback.calls) == 1

    callback_body = callback.calls.last.request.content
    assert b'name="actionId"' not in callback_body
    assert b'name="npzFile"' in callback_body
    assert b'name="jsonFile"' in callback_body
    assert b'filename="walk_g1_tracking.npz"' in callback_body
    assert b'filename="walk_g1_preview.json"' in callback_body
    assert b"application/json" in callback_body
    assert RETARGET_NPZ in callback_body
    assert RETARGET_JSON in callback_body
    assert "robotType=1 robot=G1" in caplog.text


def test_retarget_download_failure_callbacks_without_file() -> None:
    payload = {
        "originalFileUrl": SOURCE_URL,
        "robotType": 1,
        "callbackUrl": CALLBACK_URL,
    }
    with respx.mock:
        respx.get(SOURCE_URL).mock(return_value=Response(404))
        callback = respx.post(CALLBACK_URL).mock(return_value=Response(204))

        with TestClient(create_app()) as client:
            response = client.post("/api/v1/bvh/retarget", json=payload)

    assert response.status_code == 200
    assert response.json()["success"] is True
    callback_body = callback.calls.last.request.content
    assert b'name="actionId"' not in callback_body
    assert b"false" in callback_body
    assert b'name="file"' not in callback_body


def test_retarget_rejects_invalid_robot_type() -> None:
    payload = {
        "originalFileUrl": SOURCE_URL,
        "robotType": 4,
        "callbackUrl": CALLBACK_URL,
    }

    with TestClient(create_app()) as client:
        response = client.post("/api/v1/bvh/retarget", json=payload)

    assert response.status_code == 422
    assert response.json()["success"] is False


@pytest.mark.parametrize(
    ("return_type", "content", "filename", "content_type"),
    [
        (1, TRAIN_MP4, "walk_g1_tracking_trained.mp4", "video/mp4"),
        (
            2,
            TRAIN_ONNX,
            "walk_g1_tracking_trained.onnx",
            "application/octet-stream",
        ),
    ],
)
def test_train_accepts_task_and_callbacks_selected_artifact(
    return_type: int,
    content: bytes,
    filename: str,
    content_type: str,
) -> None:
    payload = {
        "actionId": "training-42",
        "robotType": 1,
        "algorithmType": 1,
        "npzFileUrl": TRAIN_NPZ_URL,
        "domainRandomization": 2,
        "returnType": return_type,
        "callbackUrl": CALLBACK_URL,
    }
    artifact = TrainingArtifact(
        content=BytesIO(content),
        filename=filename,
        content_type=content_type,
    )

    with (
        patch(
            "bvh_processing.training.task.run_training_program",
            return_value=artifact,
        ) as train_program,
        respx.mock,
    ):
        respx.get(TRAIN_NPZ_URL).mock(return_value=Response(200, content=TRAIN_NPZ))
        callback = respx.post(CALLBACK_URL).mock(return_value=Response(204))

        with TestClient(create_app()) as client:
            response = client.post("/api/v1/bvh/train", json=payload)

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["message"] == "训练任务已接收"
    UUID(response.json()["taskId"])
    train_payload = train_program.call_args.args[1]
    assert train_payload.robot_type == 1
    assert train_payload.algorithm_type == 1
    assert train_payload.domain_randomization == 2
    assert train_payload.return_type == return_type

    callback_body = callback.calls.last.request.content
    assert b"training-42" in callback_body
    assert f'filename="{filename}"'.encode() in callback_body
    assert content_type.encode() in callback_body
    assert content in callback_body


def test_train_return_type_two_callbacks_demo_policy() -> None:
    payload = {
        "actionId": "training-policy-demo",
        "robotType": 1,
        "algorithmType": 1,
        "npzFileUrl": TRAIN_NPZ_URL,
        "domainRandomization": 2,
        "returnType": 2,
        "callbackUrl": CALLBACK_URL,
    }
    policy_path = (
        Path(__file__).parents[1]
        / "src"
        / "bvh_processing"
        / "assets"
        / "policy_demo"
        / "1a2_34000.onnx"
    )

    with respx.mock:
        respx.get(TRAIN_NPZ_URL).mock(return_value=Response(200, content=TRAIN_NPZ))
        callback = respx.post(CALLBACK_URL).mock(return_value=Response(204))

        with TestClient(create_app()) as client:
            response = client.post("/api/v1/bvh/train", json=payload)

    assert response.status_code == 200
    assert len(callback.calls) == 1
    callback_body = callback.calls.last.request.content
    assert b'filename="1a2_34000.onnx"' in callback_body
    assert b"application/octet-stream" in callback_body
    assert policy_path.read_bytes() in callback_body


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("algorithmType", 4),
        ("domainRandomization", 0),
        ("returnType", 3),
    ],
)
def test_train_rejects_invalid_enum_values(field: str, value: int) -> None:
    payload = {
        "robotType": 1,
        "algorithmType": 1,
        "npzFileUrl": TRAIN_NPZ_URL,
        "domainRandomization": 2,
        "returnType": 1,
        "callbackUrl": CALLBACK_URL,
    }
    payload[field] = value

    with TestClient(create_app()) as client:
        response = client.post("/api/v1/bvh/train", json=payload)

    assert response.status_code == 422
    assert response.json()["success"] is False


def test_download_failure_callbacks_without_file() -> None:
    app = create_app()
    with respx.mock:
        respx.get(SOURCE_URL).mock(return_value=Response(404))
        callback = respx.post(CALLBACK_URL).mock(return_value=Response(204))

        with TestClient(app) as client:
            response = client.post("/api/v1/bvh/process", json=_request_body())

    assert response.status_code == 200
    assert response.json()["success"] is True

    callback_body = callback.calls.last.request.content
    assert b"\r\n\r\nfalse\r\n" in callback_body
    assert b"MinIO" in callback_body
    assert b'name="file"' not in callback_body


def test_unsupported_bvh_format_callbacks_without_processing() -> None:
    app = create_app()
    with respx.mock:
        respx.get(SOURCE_URL).mock(
            return_value=Response(200, content=UNSUPPORTED_BVH_CONTENT)
        )
        callback = respx.post(CALLBACK_URL).mock(return_value=Response(204))
        progress_callback = respx.post(PROGRESS_CALLBACK_URL).mock(
            return_value=Response(204)
        )

        with TestClient(app) as client:
            response = client.post("/api/v1/bvh/process", json=_request_body())

    assert response.status_code == 200
    callback_body = callback.calls.last.request.content
    assert "只支持LAFAN1格式和Nokov格式的 BVH 文件".encode() in callback_body
    assert b"\r\n\r\nfalse\r\n" in callback_body
    assert b'name="file"' not in callback_body
    assert len(progress_callback.calls) == 0


def test_process_rejects_invalid_handle_options() -> None:
    payload = _request_body()
    payload["handleOptions"] = "1,2,3"

    with TestClient(create_app()) as client:
        response = client.post("/api/v1/bvh/process", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert body["success"] is False
    assert body["code"] == "request_validation_error"
    assert body["message"] == "请求参数校验失败"
    assert body["errors"][0]["field"] == "handleOptions"
    assert body["errors"][0]["type"] == "list_type"


def test_process_rejects_callback_token() -> None:
    payload = _request_body()
    payload["callbackToken"] = "unused-token"

    with TestClient(create_app()) as client:
        response = client.post("/api/v1/bvh/process", json=payload)

    assert response.status_code == 422
    assert response.json()["success"] is False


def test_merge_accepts_task_and_callbacks_with_merged_file() -> None:
    payload = {
        "actionId": "action-merge-42",
        "fileUrls": [MERGE_SOURCE_URL_1, MERGE_SOURCE_URL_2],
        "intervalsSeconds": [0.1],
        "bvhMotionDuration": [0.05, 0.05],
        "callbackUrl": CALLBACK_URL,
    }
    with respx.mock:
        respx.get(MERGE_SOURCE_URL_1).mock(
            return_value=Response(200, content=MERGE_BVH_1)
        )
        respx.get(MERGE_SOURCE_URL_2).mock(
            return_value=Response(200, content=MERGE_BVH_2)
        )
        callback = respx.post(CALLBACK_URL).mock(return_value=Response(204))

        with TestClient(create_app()) as client:
            response = client.post("/api/v1/bvh/merge", json=payload)

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["message"] == "合并任务已接收"
    UUID(response.json()["taskId"])
    assert len(callback.calls) == 1

    callback_body = callback.calls.last.request.content
    assert b"actionId" in callback_body
    assert b"action-merge-42" in callback_body
    assert b'filename="first_merged.bvh"' in callback_body
    assert b"Frames: 6" in callback_body
    assert b"Frame Time: 0.05" in callback_body


def test_merge_normalizes_all_files_to_lowest_frame_rate() -> None:
    payload = {
        "actionId": "action-merge-fps",
        "fileUrls": [MERGE_SOURCE_URL_1, MERGE_SOURCE_URL_2],
        "intervalsSeconds": [0],
        "bvhMotionDuration": [0.1, 0.05],
        "callbackUrl": CALLBACK_URL,
    }
    with respx.mock:
        respx.get(MERGE_SOURCE_URL_1).mock(
            return_value=Response(200, content=MERGE_BVH_HIGH_FPS)
        )
        respx.get(MERGE_SOURCE_URL_2).mock(
            return_value=Response(200, content=MERGE_BVH_2)
        )
        callback = respx.post(CALLBACK_URL).mock(return_value=Response(204))

        with TestClient(create_app()) as client:
            response = client.post("/api/v1/bvh/merge", json=payload)

    assert response.status_code == 200
    callback_body = callback.calls.last.request.content
    assert b"Frames: 5" in callback_body
    assert b"Frame Time: 0.05" in callback_body


def test_merge_rejects_wrong_interval_count() -> None:
    payload = {
        "actionId": "action-merge-42",
        "fileUrls": [MERGE_SOURCE_URL_1, MERGE_SOURCE_URL_2],
        "intervalsSeconds": [],
        "bvhMotionDuration": [3.21, 10.73],
        "callbackUrl": CALLBACK_URL,
    }

    with TestClient(create_app()) as client:
        response = client.post("/api/v1/bvh/merge", json=payload)

    assert response.status_code == 422
    assert response.json()["success"] is False


def test_merge_rejects_wrong_motion_duration_count() -> None:
    payload = {
        "actionId": "action-merge-42",
        "fileUrls": [MERGE_SOURCE_URL_1, MERGE_SOURCE_URL_2],
        "intervalsSeconds": [0.1],
        "bvhMotionDuration": [],
        "callbackUrl": CALLBACK_URL,
    }

    with TestClient(create_app()) as client:
        response = client.post("/api/v1/bvh/merge", json=payload)

    assert response.status_code == 422
    assert response.json()["success"] is False


def test_merge_reports_incompatible_skeleton() -> None:
    incompatible = MERGE_BVH_2.replace(b"ROOT Hips", b"ROOT Pelvis")
    payload = {
        "actionId": "action-merge-42",
        "fileUrls": [MERGE_SOURCE_URL_1, MERGE_SOURCE_URL_2],
        "intervalsSeconds": [0],
        "bvhMotionDuration": [3.21, 10.73],
        "callbackUrl": CALLBACK_URL,
    }
    with respx.mock:
        respx.get(MERGE_SOURCE_URL_1).mock(
            return_value=Response(200, content=MERGE_BVH_1)
        )
        respx.get(MERGE_SOURCE_URL_2).mock(
            return_value=Response(200, content=incompatible)
        )
        callback = respx.post(CALLBACK_URL).mock(return_value=Response(204))

        with TestClient(create_app()) as client:
            response = client.post("/api/v1/bvh/merge", json=payload)

    assert response.status_code == 200
    callback_body = callback.calls.last.request.content
    assert b"false" in callback_body
    assert b'name="file"' not in callback_body


def test_health() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_service_error_returns_code_and_message() -> None:
    payload = _request_body()
    payload["callbackUrl"] = "https://user:password@backend.example.com/callbacks/bvh"

    with TestClient(create_app()) as client:
        response = client.post("/api/v1/bvh/process", json=payload)

    assert response.status_code == 400
    assert response.json() == {
        "success": False,
        "taskId": None,
        "code": "invalid_callback_url",
        "message": "回调地址不能在 URL 中包含用户名或密码",
    }


def test_http_error_returns_useful_details() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/missing-endpoint")

    assert response.status_code == 404
    assert response.json() == {
        "success": False,
        "taskId": None,
        "code": "http_error",
        "message": "Not Found",
        "detail": "Not Found",
    }


def test_unexpected_error_logs_traceback_and_returns_error_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = create_app()

    @app.get("/test/unexpected")
    async def unexpected() -> None:
        raise RuntimeError("测试异常详情")

    with (
        caplog.at_level(logging.ERROR, logger="bvh_processing.main"),
        TestClient(app, raise_server_exceptions=False) as client,
    ):
        response = client.get("/test/unexpected")

    body = response.json()
    assert response.status_code == 500
    assert body["code"] == "internal_server_error"
    assert body["message"] == "服务器内部错误"
    UUID(body["errorId"])
    assert "测试异常详情" in caplog.text
    assert "Traceback" in caplog.text
