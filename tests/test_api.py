import json
import logging
from io import BytesIO
from unittest.mock import patch
from uuid import UUID

import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response

from bvh_processing.config import Settings, get_settings
from bvh_processing.main import create_app
from bvh_processing.retargeting.exporter import RetargetArtifacts

SOURCE_URL = "https://minio.example.com/motions/walk.bvh"
CALLBACK_URL = "https://backend.example.com/callbacks/bvh"
LOSS_CALLBACK_URL = "https://backend.example.com/callbacks/training-loss"
PROGRESS_CALLBACK_URL = "https://backend.example.com/progress-callbacks/bvh"
RETARGET_NPZ = b"tracking-npz-content"
RETARGET_JSON = b'{"robot":"unitree_g1","frames":[]}'
TRAIN_NPZ_URL = "https://minio.example.com/motions/walk_g1_tracking.npz"
TRAIN_NPZ = b"retargeted-npz-content"
GPU_CONTROL_URL = "https://gpu.example.com"
GPU_TOKEN = "test-gpu-token"
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


def _merge_request_body(
    *,
    first_duration: float = 0.05,
    second_duration: float = 0.05,
    gap: float = 0.1,
) -> dict[str, object]:
    return {
        "actionId": "action-merge-42",
        "timelineOffsetSec": 0,
        "segments": [
            {
                "segmentId": "segment-1",
                "actionId": 101,
                "actionUrl": MERGE_SOURCE_URL_1,
                "sourceInSec": 0,
                "sourceOutSec": 0.05,
                "outputDurationSec": first_duration,
                "gapAfterSec": gap,
            },
            {
                "segmentId": "segment-2",
                "actionId": 102,
                "actionUrl": MERGE_SOURCE_URL_2,
                "sourceInSec": 0,
                "sourceOutSec": 0.05,
                "outputDurationSec": second_duration,
                "gapAfterSec": 0,
            },
        ],
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
    assert callback_body.count(b'name="file"') == 0
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


def _train_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "actionId": "training-42",
        "robotType": 1,
        "algorithmType": 1,
        "npzFileUrl": TRAIN_NPZ_URL,
        "domainRandomization": 2,
        "returnType": 2,
        "numEnvs": 12288,
        "maxIterations": 5000,
        "seed": 7,
        "callbackUrl": CALLBACK_URL,
    }
    payload.update(overrides)
    return payload


def _create_training_test_app() -> FastAPI:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None,
        mock=False,
        gpu_control_api_url=GPU_CONTROL_URL,
        gpu_control_api_token=GPU_TOKEN,
    )
    return app


@pytest.mark.parametrize(
    ("return_type", "expected_filenames"),
    [
        (1, ("1a2_34000.mp4", "1a2_34000.onnx")),
        (2, ("1a2_34000.onnx",)),
    ],
)
def test_train_mock_callbacks_demo_artifacts_without_downloading_npz(
    return_type: int,
    expected_filenames: tuple[str, ...],
) -> None:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None,
        mock=True,
        gpu_control_api_url=GPU_CONTROL_URL,
        gpu_control_api_token=GPU_TOKEN,
    )
    cloud_loss = {
        "job_id": "job_cloud_latest",
        "losses": {"value_function": [{"step": 12, "value": 0.081}]},
    }

    with respx.mock:
        callback = respx.post(CALLBACK_URL).mock(return_value=Response(204))
        jobs = respx.get(f"{GPU_CONTROL_URL}/api/v1/jobs").mock(
            return_value=Response(200, json={"jobs": [{"id": "job_cloud_latest"}]})
        )
        loss = respx.get(
            f"{GPU_CONTROL_URL}/api/v1/jobs/job_cloud_latest/loss?max_points=500"
        ).mock(return_value=Response(200, json=cloud_loss))
        loss_callback = respx.post(LOSS_CALLBACK_URL).mock(return_value=Response(204))
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/bvh/train",
                json=_train_payload(
                    returnType=return_type,
                    lossCallbackUrl=LOSS_CALLBACK_URL,
                ),
            )

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["message"] == "Mock 训练任务已接收"
    UUID(response.json()["taskId"])
    assert len(jobs.calls) == 1
    assert jobs.calls[0].request.headers["authorization"] == f"Bearer {GPU_TOKEN}"
    assert len(loss.calls) == 1
    assert len(loss_callback.calls) == 1
    assert json.loads(loss_callback.calls[0].request.content) == {
        "actionId": "training-42",
        "data": cloud_loss,
    }
    assert len(callback.calls) == 1

    callback_body = callback.calls[0].request.content
    assert callback_body.count(b'name="file"') == len(expected_filenames)
    for filename in expected_filenames:
        assert f'filename="{filename}"'.encode() in callback_body
    assert (b'filename="1a2_34000.mp4"' in callback_body) is (return_type == 1)
    assert b'filename="1a2_34000.onnx"' in callback_body
    assert "Mock 训练成功".encode() in callback_body


def test_train_uploads_npz_selects_first_available_gpu_and_starts_job() -> None:
    with respx.mock:
        download = respx.get(TRAIN_NPZ_URL).mock(
            return_value=Response(200, content=TRAIN_NPZ)
        )
        upload = respx.post(f"{GPU_CONTROL_URL}/api/v1/artifacts/motions").mock(
            return_value=Response(201, json={"id": "motion_abc"})
        )
        gpu_query = respx.get(f"{GPU_CONTROL_URL}/api/v1/gpus/simple").mock(
            return_value=Response(
                200,
                json={
                    "gpus": [
                        {"gpu": 3, "available": True},
                        {"gpu": 0, "available": False},
                        {"gpu": 1, "available": True},
                    ]
                },
            )
        )
        create_job = respx.post(f"{GPU_CONTROL_URL}/api/v1/jobs").mock(
            return_value=Response(202, json={"id": "job_abc"})
        )

        with TestClient(_create_training_test_app()) as client:
            response = client.post(
                "/api/v1/bvh/train",
                json=_train_payload(),
            )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "taskId": "job_abc",
        "message": "训练任务已提交，使用 cuda:1",
    }
    assert len(download.calls) == 1
    assert len(upload.calls) == 1
    assert len(gpu_query.calls) == 1
    assert len(create_job.calls) == 1
    assert b'filename="walk_g1_tracking.npz"' in upload.calls[0].request.content
    assert TRAIN_NPZ in upload.calls[0].request.content
    assert upload.calls[0].request.headers["authorization"] == f"Bearer {GPU_TOKEN}"
    assert json.loads(create_job.calls[0].request.content) == {
        "artifact_id": "motion_abc",
        "devices": ["cuda:1"],
        "num_envs": 12288,
        "max_iterations": 5000,
        "seed": 7,
    }


def test_train_returns_capacity_full_when_no_gpu_is_available() -> None:
    with respx.mock:
        respx.get(TRAIN_NPZ_URL).mock(return_value=Response(200, content=TRAIN_NPZ))
        respx.post(f"{GPU_CONTROL_URL}/api/v1/artifacts/motions").mock(
            return_value=Response(201, json={"id": "motion_abc"})
        )
        respx.get(f"{GPU_CONTROL_URL}/api/v1/gpus/simple").mock(
            return_value=Response(
                200,
                json={"gpus": [{"gpu": 0, "available": False}]},
            )
        )
        create_job = respx.post(f"{GPU_CONTROL_URL}/api/v1/jobs").mock(
            return_value=Response(202, json={"id": "unexpected"})
        )

        with TestClient(_create_training_test_app()) as client:
            response = client.post(
                "/api/v1/bvh/train",
                json=_train_payload(),
            )

    assert response.status_code == 409
    assert response.json() == {
        "success": False,
        "taskId": None,
        "code": "gpu_capacity_full",
        "message": "当前显卡训练任务已满",
    }
    assert len(create_job.calls) == 0


def test_train_tries_next_gpu_when_first_gpu_is_claimed_concurrently() -> None:
    with respx.mock:
        respx.get(TRAIN_NPZ_URL).mock(return_value=Response(200, content=TRAIN_NPZ))
        respx.post(f"{GPU_CONTROL_URL}/api/v1/artifacts/motions").mock(
            return_value=Response(201, json={"id": "motion_abc"})
        )
        respx.get(f"{GPU_CONTROL_URL}/api/v1/gpus/simple").mock(
            return_value=Response(
                200,
                json={
                    "gpus": [
                        {"gpu": 0, "available": True},
                        {"gpu": 2, "available": True},
                    ]
                },
            )
        )
        create_job = respx.post(f"{GPU_CONTROL_URL}/api/v1/jobs").mock(
            side_effect=[
                Response(409, json={"detail": "GPU is unavailable"}),
                Response(202, json={"id": "job_second_gpu"}),
            ]
        )

        with TestClient(_create_training_test_app()) as client:
            response = client.post(
                "/api/v1/bvh/train",
                json=_train_payload(),
            )

    assert response.status_code == 200
    assert response.json()["taskId"] == "job_second_gpu"
    assert response.json()["message"] == "训练任务已提交，使用 cuda:2"
    assert [
        json.loads(call.request.content)["devices"] for call in create_job.calls
    ] == [["cuda:0"], ["cuda:2"]]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("robotType", 2),
        ("algorithmType", 2),
        ("algorithmType", 4),
        ("domainRandomization", 1),
        ("domainRandomization", 0),
        ("returnType", 3),
        ("gpu", -1),
        ("numEnvs", 0),
        ("maxIterations", 0),
        ("seed", -1),
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
    payload = _merge_request_body()
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
    payload = _merge_request_body(first_duration=0.1, gap=0)
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


def test_merge_accepts_single_segment_with_dance_id_and_trims_frames() -> None:
    payload = {
        "danceId": "39",
        "timelineOffsetSec": 0,
        "segments": [
            {
                "segmentId": "segment-only",
                "actionId": 101,
                "actionUrl": MERGE_SOURCE_URL_1,
                "sourceInSec": 0.025,
                "sourceOutSec": 0.075,
                "outputDurationSec": 0.05,
                "gapAfterSec": 0,
            }
        ],
        "callbackUrl": CALLBACK_URL,
    }
    with respx.mock:
        source = respx.get(MERGE_SOURCE_URL_1).mock(
            return_value=Response(200, content=MERGE_BVH_HIGH_FPS)
        )
        callback = respx.post(CALLBACK_URL).mock(return_value=Response(204))

        with TestClient(create_app()) as client:
            response = client.post("/api/v1/bvh/merge", json=payload)

    assert response.status_code == 200
    assert len(source.calls) == 1
    callback_body = callback.calls.last.request.content
    assert b'\r\n\r\n39\r\n' in callback_body
    assert b"Frames: 3" in callback_body
    assert b"1 0 0 0 0 0 0 0 0 0 0 0" in callback_body
    assert b"3 0 0 0 0 0 0 0 0 0 0 0" in callback_body


def test_merge_rejects_empty_segments() -> None:
    payload = _merge_request_body()
    payload["segments"] = []

    with TestClient(create_app()) as client:
        response = client.post("/api/v1/bvh/merge", json=payload)

    assert response.status_code == 422
    assert response.json()["success"] is False


def test_merge_rejects_invalid_source_range() -> None:
    payload = _merge_request_body()
    payload["segments"][0]["sourceInSec"] = 0.05
    payload["segments"][0]["sourceOutSec"] = 0.05

    with TestClient(create_app()) as client:
        response = client.post("/api/v1/bvh/merge", json=payload)

    assert response.status_code == 422
    assert response.json()["success"] is False


def test_merge_reports_incompatible_skeleton() -> None:
    incompatible = MERGE_BVH_2.replace(b"ROOT Hips", b"ROOT Pelvis")
    payload = _merge_request_body(gap=0)
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
