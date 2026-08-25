import json
import logging
from importlib.resources import files
from uuid import UUID

import pytest
import respx
from fastapi.testclient import TestClient
from httpx import Response

from bvh_processing.main import create_app

SOURCE_URL = "https://minio.example.com/motions/walk.bvh"
CALLBACK_URL = "https://backend.example.com/callbacks/bvh"
PROGRESS_CALLBACK_URL = "https://backend.example.com/progress-callbacks/bvh"
PREVIEW_JSON_NAME = "Take_007_049_Skeleton7_g1_preview.json"
BVH_CONTENT = b"""HIERARCHY
ROOT Hips
{
  JOINT LeftToe
  {
  }
  JOINT RightToe
  {
  }
}
MOTION
Frames: 1
"""
UNSUPPORTED_BVH_CONTENT = b"HIERARCHY\nROOT Hips\nMOTION\nFrames: 1\n"
MERGE_SOURCE_URL_1 = "https://minio.example.com/motions/first.bvh"
MERGE_SOURCE_URL_2 = "https://minio.example.com/motions/second.bvh"
MERGE_BVH_1 = b"""HIERARCHY
ROOT Hips
{
  OFFSET 0 0 0
  CHANNELS 3 Xposition Yposition Zposition
  JOINT LeftToe
  {
  }
  JOINT RightToe
  {
  }
}
MOTION
Frames: 2
Frame Time: 0.05
0 0 0
1 1 1
"""
MERGE_BVH_2 = b"""HIERARCHY
ROOT Hips
{
  OFFSET 0 0 0
  CHANNELS 3 Xposition Yposition Zposition
  JOINT LeftToe
  {
  }
  JOINT RightToe
  {
  }
}
MOTION
Frames: 1
Frame Time: 0.05
2 2 2
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


def test_retarget_accepts_task_and_callbacks_with_json_file(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="bvh_processing.services.tasks")
    payload = {
        "originalFileUrl": SOURCE_URL,
        "robotType": 1,
        "callbackUrl": CALLBACK_URL,
    }
    expected_json = (
        files("bvh_processing").joinpath("assets", PREVIEW_JSON_NAME).read_bytes()
    )

    with respx.mock:
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
    assert b"application/json" in callback_body
    assert f'filename="{PREVIEW_JSON_NAME}"'.encode() in callback_body
    assert expected_json in callback_body
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
    assert response.json() == {
        "success": False,
        "taskId": None,
        "message": "请求参数不正确",
    }


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
    assert b"Frames: 5" in callback_body
    assert b"0 0 0\n1 1 1\n1 1 1\n1 1 1\n2 2 2" in callback_body


def test_merge_rejects_wrong_interval_count() -> None:
    payload = {
        "actionId": "action-merge-42",
        "fileUrls": [MERGE_SOURCE_URL_1, MERGE_SOURCE_URL_2],
        "intervalsSeconds": [],
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
