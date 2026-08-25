from uuid import UUID

import respx
from fastapi.testclient import TestClient
from httpx import Response

from bvh_processing.main import create_app

SOURCE_URL = "https://minio.example.com/motions/walk.bvh"
CALLBACK_URL = "https://backend.example.com/callbacks/bvh"
BVH_CONTENT = b"HIERARCHY\nROOT Hips\nMOTION\nFrames: 1\n"
MERGE_SOURCE_URL_1 = "https://minio.example.com/motions/first.bvh"
MERGE_SOURCE_URL_2 = "https://minio.example.com/motions/second.bvh"
MERGE_BVH_1 = b"""HIERARCHY
ROOT Hips
{
  OFFSET 0 0 0
  CHANNELS 3 Xposition Yposition Zposition
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
    assert len(callback.calls) == 4
    progress_bodies = [call.request.content for call in callback.calls[:-1]]
    assert all(b'name="handleOption"' in body for body in progress_bodies)
    assert all(b'name="optionStatus"' in body for body in progress_bodies)


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
