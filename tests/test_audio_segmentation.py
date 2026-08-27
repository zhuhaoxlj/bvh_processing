import json
from unittest.mock import AsyncMock, patch
from uuid import UUID

import respx
from fastapi.testclient import TestClient
from httpx import Response

from bvh_processing.main import create_app

AUDIO_URL = "https://minio.example.com/audio/song.mp3"
CALLBACK_URL = "https://backend.example.com/callbacks/audio"
AUDIO_CONTENT = b"fake-mp3-content"
SEGMENTS = [
    {
        "start": 0.0,
        "end": 0.459,
        "label": "silence",
        "label_zh": "静音",
    },
    {
        "start": 0.459,
        "end": 17.136,
        "label": "intro",
        "label_zh": "前奏",
    },
]


def test_audio_segment_accepts_task_and_callbacks_json_array() -> None:
    payload = {
        "audioFileUrl": AUDIO_URL,
        "callbackUrl": CALLBACK_URL,
    }
    analyze = AsyncMock(return_value=SEGMENTS)
    with patch("bvh_processing.audio_segmentation.task.analyze_audio", analyze), respx.mock:
        respx.get(AUDIO_URL).mock(return_value=Response(200, content=AUDIO_CONTENT))
        callback = respx.post(CALLBACK_URL).mock(return_value=Response(204))

        with TestClient(create_app()) as client:
            response = client.post("/api/v1/audio/segment", json=payload)

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["message"] == "音频解析任务已接收"
    UUID(response.json()["taskId"])
    assert analyze.await_count == 1
    assert analyze.await_args.args[3] == 7
    assert json.loads(callback.calls.last.request.content) == SEGMENTS
    assert callback.calls.last.request.headers["X-Callback-Token"]


def test_audio_segment_supports_nine_label_model() -> None:
    payload = {
        "audioFileUrl": AUDIO_URL,
        "callbackUrl": CALLBACK_URL,
        "sectionLabels": 9,
    }
    analyze = AsyncMock(return_value=SEGMENTS)
    with patch("bvh_processing.audio_segmentation.task.analyze_audio", analyze), respx.mock:
        respx.get(AUDIO_URL).mock(return_value=Response(200, content=AUDIO_CONTENT))
        respx.post(CALLBACK_URL).mock(return_value=Response(204))

        with TestClient(create_app()) as client:
            response = client.post("/api/v1/audio/segment", json=payload)

    assert response.status_code == 200
    assert analyze.await_args.args[3] == 9


def test_audio_segment_rejects_unknown_label_model() -> None:
    payload = {
        "audioFileUrl": AUDIO_URL,
        "callbackUrl": CALLBACK_URL,
        "sectionLabels": 8,
    }
    with TestClient(create_app()) as client:
        response = client.post("/api/v1/audio/segment", json=payload)

    assert response.status_code == 422
    assert response.json()["success"] is False


def test_audio_download_failure_sends_failure_callback() -> None:
    payload = {
        "audioFileUrl": AUDIO_URL,
        "callbackUrl": CALLBACK_URL,
    }
    with respx.mock:
        respx.get(AUDIO_URL).mock(return_value=Response(404))
        callback = respx.post(CALLBACK_URL).mock(return_value=Response(204))

        with TestClient(create_app()) as client:
            response = client.post("/api/v1/audio/segment", json=payload)

    assert response.status_code == 200
    failure = json.loads(callback.calls.last.request.content)
    assert failure["success"] is False
    assert failure["message"] == "MinIO 返回 HTTP 404"
    UUID(failure["taskId"])