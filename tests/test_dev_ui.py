"""本地自测页面（BVH_DEV_UI）的端点与全链路测试。"""

import hashlib
import logging
import sys
import threading
import time
from uuid import uuid4

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bvh_processing.api import dev_ui
from bvh_processing.api.dev_ui import reset_store
from bvh_processing.config import Settings, get_settings
from bvh_processing.main import create_app

BVH_CONTENT = b"""HIERARCHY
ROOT Hips
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT LeftToe
  {
    OFFSET -10 0 0
    CHANNELS 3 Zrotation Xrotation Yrotation
  }
  JOINT RightToe
  {
    OFFSET 10 0 0
    CHANNELS 3 Zrotation Xrotation Yrotation
  }
}
MOTION
Frames: 5
Frame Time: 0.0333333
0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 90 0 0 0 0 0 0 0 0
0 0 0 180 0 0 0 0 0 0 0 0
0 0 0 270 0 0 0 0 0 0 0 0
0 0 0 360 0 0 0 0 0 0 0 0
"""
BVH_SHA256 = hashlib.sha256(BVH_CONTENT).hexdigest()


@pytest.fixture(autouse=True)
def _clean_dev_store():
    yield
    reset_store()


def _dev_app(settings: Settings) -> FastAPI:
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    return app


def _dev_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, dev_ui=True, **overrides)


def _upload(client: httpx.Client, content: bytes = BVH_CONTENT, name: str = "walk.bvh"):
    return client.post(
        "/api/v1/dev/bvh/uploads",
        files={"file": (name, content, "application/octet-stream")},
    )


def test_dev_ui_page_and_assets_are_served_when_enabled() -> None:
    with TestClient(_dev_app(_dev_settings())) as client:
        page = client.get("/dev/bvh")
        three = client.get("/dev/bvh/static/three.module.min.js")
        loader = client.get("/dev/bvh/static/BVHLoader.js")

    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert "'/api/v1/bvh/process'" in page.text
    assert 'id="canvasSource"' in page.text
    assert 'id="canvasResult"' in page.text
    # 共享进度条 + 视角同步开关
    assert 'id="timeline"' in page.text
    assert 'id="readout"' in page.text
    assert 'id="syncCam"' in page.text
    assert three.status_code == 200
    assert loader.status_code == 200


def test_dev_ui_is_not_mounted_by_default() -> None:
    with TestClient(create_app(Settings(_env_file=None))) as client:
        assert client.get("/dev/bvh").status_code == 404
        assert client.get("/", follow_redirects=False).status_code == 404
        assert client.post("/api/v1/dev/bvh/uploads").status_code == 404
        assert client.get("/api/v1/dev/bvh/tasks/devtask-1234").status_code == 404


def test_dev_ui_root_redirects_to_the_page() -> None:
    with TestClient(_dev_app(_dev_settings())) as client:
        redirect = client.get("/", follow_redirects=False)
        followed = client.get("/")

    assert redirect.status_code == 302
    assert redirect.headers["location"] == "/dev/bvh"
    assert followed.status_code == 200
    assert 'id="canvasSource"' in followed.text


@pytest.mark.parametrize(
    ("argv", "env", "expected"),
    [
        (["uvicorn"], {}, "http://127.0.0.1:9001/dev/bvh"),
        (
            [
                "uvicorn",
                "bvh_processing.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8123",
            ],
            {},
            "http://127.0.0.1:8123/dev/bvh",
        ),
        (
            ["uvicorn", "--host=192.168.1.5", "--port=7000"],
            {},
            "http://192.168.1.5:7000/dev/bvh",
        ),
        (
            ["uvicorn"],
            {"UVICORN_HOST": "10.0.0.7", "UVICORN_PORT": "9500"},
            "http://10.0.0.7:9500/dev/bvh",
        ),
        (["uvicorn", "--port", "oops"], {}, "http://127.0.0.1:9001/dev/bvh"),
    ],
)
def test_dev_ui_url_follows_host_and_port(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    env: dict[str, str],
    expected: str,
) -> None:
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.delenv("UVICORN_HOST", raising=False)
    monkeypatch.delenv("UVICORN_PORT", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    assert dev_ui.dev_ui_url() == expected


def test_announce_logs_url_and_opens_browser_only_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(
        dev_ui.webbrowser, "open", lambda url: opened.append(url) or True
    )
    monkeypatch.setattr(dev_ui, "_BROWSER_DELAY_SECONDS", 0.0)

    with caplog.at_level(logging.WARNING):
        dev_ui.announce(_dev_settings())
    time.sleep(0.05)

    assert opened == []
    assert "本地自测页面已启用" in caplog.text
    assert dev_ui.dev_ui_url() in caplog.text

    dev_ui.announce(_dev_settings(dev_ui_open_browser=True))
    deadline = time.monotonic() + 2
    while not opened and time.monotonic() < deadline:
        time.sleep(0.01)

    assert opened == [dev_ui.dev_ui_url()]


def test_dev_upload_returns_sha256_and_serves_the_same_bytes() -> None:
    with TestClient(_dev_app(_dev_settings())) as client:
        response = _upload(client)

        assert response.status_code == 200
        body = response.json()
        assert body["filename"] == "walk.bvh"
        assert body["size"] == len(BVH_CONTENT)
        assert body["sha256"] == BVH_SHA256
        assert body["sourcePath"] == f"/api/v1/dev/bvh/uploads/{body['token']}/walk.bvh"

        source = client.get(body["sourcePath"])

    assert source.status_code == 200
    assert source.content == BVH_CONTENT


def test_dev_upload_rejects_wrong_suffix_empty_and_oversized_files() -> None:
    with TestClient(_dev_app(_dev_settings())) as client:
        wrong_suffix = _upload(client, name="walk.txt")
        empty = _upload(client, content=b"")
        bad_id = client.get("/api/v1/dev/bvh/uploads/short/walk.bvh")
        missing_token = client.get("/api/v1/dev/bvh/uploads/deadbeefdeadbeef/walk.bvh")

    assert wrong_suffix.status_code == 400
    assert wrong_suffix.json()["code"] == "invalid_upload_filename"
    assert empty.status_code == 422
    assert empty.json()["code"] == "empty_source_file"
    assert bad_id.status_code == 400
    assert bad_id.json()["code"] == "invalid_dev_task_id"
    assert missing_token.status_code == 404
    assert missing_token.json()["code"] == "dev_upload_not_found"

    with TestClient(_dev_app(_dev_settings(max_file_size_mb=1))) as client:
        oversized = _upload(client, content=b"X" * (1024 * 1024 + 1))

    assert oversized.status_code == 413
    assert oversized.json()["code"] == "source_file_too_large"


def test_dev_upload_allows_its_own_host_in_restricted_allowlists() -> None:
    settings = _dev_settings(
        minio_allowed_hosts="minio.example.com",
        callback_allowed_hosts="backend.example.com",
    )
    with TestClient(_dev_app(settings)) as client:
        assert _upload(client).status_code == 200

    assert "minio.example.com" in settings.allowed_hosts
    assert "testserver" in settings.allowed_hosts
    assert "backend.example.com" in settings.allowed_callback_hosts
    assert "testserver" in settings.allowed_callback_hosts


def test_dev_task_records_progress_and_result_callback() -> None:
    task_id = f"devtask-{uuid4().hex}"
    result_bytes = b"HIERARCHY\nROOT Hips\n{}\nMOTION\nFrames: 1\nFrame Time: 0.05\n0\n"

    with TestClient(_dev_app(_dev_settings())) as client:
        pending = client.get(f"/api/v1/dev/bvh/tasks/{task_id}")
        progress = client.post(
            f"/api/v1/dev/bvh/progress-callback/{task_id}",
            json={
                "actionId": "action-1",
                "originalFileUrl": "http://127.0.0.1:9001/x.bvh",
                "progress": 95,
                "step": 2,
                "stepCode": "DENOISE",
                "message": "正在处理整体去噪",
            },
        )
        callback = client.post(
            f"/api/v1/dev/bvh/callback/{task_id}",
            data={"success": "true", "message": "处理成功", "actionId": "action-1"},
            files={
                "file": (
                    "walk_processed.bvh",
                    result_bytes,
                    "application/octet-stream",
                )
            },
        )
        status = client.get(f"/api/v1/dev/bvh/tasks/{task_id}")
        result = client.get(f"/api/v1/dev/bvh/tasks/{task_id}/result")

    assert pending.status_code == 200
    assert pending.json()["status"] == "pending"
    assert pending.json()["result"] is None
    assert progress.json() == {"received": True}
    assert callback.json() == {"received": True}

    body = status.json()
    assert body["status"] == "succeeded"
    assert body["success"] is True
    assert body["message"] == "处理成功"
    assert body["progress"] == [
        {
            "actionId": "action-1",
            "originalFileUrl": "http://127.0.0.1:9001/x.bvh",
            "progress": 95,
            "step": 2,
            "stepCode": "DENOISE",
            "message": "正在处理整体去噪",
        }
    ]
    assert body["result"]["filename"] == "walk_processed.bvh"
    assert body["result"]["size"] == len(result_bytes)

    assert result.status_code == 200
    assert result.content == result_bytes


def test_dev_task_records_failure_callback() -> None:
    task_id = f"devtask-{uuid4().hex}"
    with TestClient(_dev_app(_dev_settings())) as client:
        client.post(
            f"/api/v1/dev/bvh/callback/{task_id}",
            data={
                "success": "false",
                "message": "只支持LAFAN1格式和Nokov格式的 BVH 文件",
            },
        )
        status = client.get(f"/api/v1/dev/bvh/tasks/{task_id}")
        result = client.get(f"/api/v1/dev/bvh/tasks/{task_id}/result")

    assert status.json()["status"] == "failed"
    assert status.json()["success"] is False
    assert "LAFAN1" in status.json()["message"]
    assert result.status_code == 404
    assert result.json()["code"] == "dev_result_not_ready"


def _wait_for_task(client: httpx.Client, task_id: str, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/dev/bvh/tasks/{task_id}")
        response.raise_for_status()
        last = response.json()
        if last["status"] != "pending":
            return last
        time.sleep(0.1)
    raise AssertionError(f"自测任务未在 {timeout}s 内收到回调：{last}")


def _start_server(app: FastAPI) -> tuple[uvicorn.Server, threading.Thread, str]:
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if server.started and server.servers and server.servers[0].sockets:
            port = server.servers[0].sockets[0].getsockname()[1]
            return server, thread, f"http://127.0.0.1:{port}"
        time.sleep(0.05)
    raise AssertionError("uvicorn 未在预期时间内启动")


def test_dev_ui_upload_process_callback_round_trip() -> None:
    """真实起服务：上传 → /api/v1/bvh/process 回源下载 → 进度/结果回调 → 取回结果。"""
    settings = _dev_settings()
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    server, thread, base_url = _start_server(app)

    try:
        with httpx.Client(base_url=base_url, timeout=30.0) as client:
            upload = _upload(client)
            assert upload.status_code == 200
            upload_body = upload.json()
            assert upload_body["sha256"] == BVH_SHA256

            dev_task_id = f"devtask-{uuid4().hex}"
            accepted = client.post(
                "/api/v1/bvh/process",
                json={
                    "actionId": "dev-e2e-action",
                    "originalFileUrl": base_url + upload_body["sourcePath"],
                    "originalFileSha256": upload_body["sha256"],
                    "handleOptions": [1],
                    "callbackUrl": (
                        f"{base_url}/api/v1/dev/bvh/callback/{dev_task_id}"
                    ),
                },
            )
            assert accepted.status_code == 200
            assert accepted.json()["success"] is True

            task = _wait_for_task(client, dev_task_id)

            assert task["status"] == "succeeded", task
            assert task["result"]["filename"] == "walk_processed.bvh"
            assert [item["stepCode"] for item in task["progress"]] == ["DENOISE"]
            assert task["progress"][0]["progress"] == 95
            assert task["progress"][0]["actionId"] == "dev-e2e-action"

            result = client.get(task["result"]["url"])
            assert result.status_code == 200
            assert result.text.startswith("HIERARCHY")
            assert "MOTION" in result.text
            assert result.text.split("Frames: ")[1].startswith("5\n")
    finally:
        server.should_exit = True
        thread.join(timeout=15)


def test_dev_ui_round_trip_reports_unsupported_skeleton() -> None:
    """格式识别失败时，失败信息要能回到自测页面。"""
    settings = _dev_settings()
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    server, thread, base_url = _start_server(app)

    unsupported = b"HIERARCHY\nROOT Hips\n{\n  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n  JOINT Head\n  {\n    CHANNELS 3 Zrotation Xrotation Yrotation\n  }\n}\nMOTION\nFrames: 1\nFrame Time: 0.05\n0 0 0 0 0 0 0 0 0\n"

    try:
        with httpx.Client(base_url=base_url, timeout=30.0) as client:
            upload = _upload(client, content=unsupported, name="head.bvh")
            assert upload.status_code == 200
            upload_body = upload.json()

            dev_task_id = f"devtask-{uuid4().hex}"
            accepted = client.post(
                "/api/v1/bvh/process",
                json={
                    "actionId": "dev-e2e-unsupported",
                    "originalFileUrl": base_url + upload_body["sourcePath"],
                    "originalFileSha256": upload_body["sha256"],
                    "handleOptions": [1, 2],
                    "callbackUrl": (
                        f"{base_url}/api/v1/dev/bvh/callback/{dev_task_id}"
                    ),
                },
            )
            assert accepted.status_code == 200

            task = _wait_for_task(client, dev_task_id)

            assert task["status"] == "failed", task
            assert "LAFAN1" in task["message"]
            assert task["result"] is None
    finally:
        server.should_exit = True
        thread.join(timeout=15)
