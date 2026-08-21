from uuid import UUID

import respx
from fastapi.testclient import TestClient
from httpx import Response

from bvh_processing.main import create_app

SOURCE_URL = "https://minio.example.com/motions/walk.bvh"
CALLBACK_URL = "https://backend.example.com/callbacks/bvh"
BVH_CONTENT = b"HIERARCHY\nROOT Hips\nMOTION\nFrames: 1\n"


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


def test_health() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
