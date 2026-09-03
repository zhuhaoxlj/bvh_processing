"""训练接口的本地 Mock 结果和 loss 回调。"""

from __future__ import annotations

import asyncio
import logging
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import httpx

from bvh_processing.config import Settings
from bvh_processing.schemas import TrainBvhRequest
from bvh_processing.services.callback import (
    CallbackFile,
    log_callback_failure,
    send_callback,
)

logger = logging.getLogger(__name__)

_MOCK_LOSS_INTERVAL_SECONDS = 600.0
_MOCK_LOSS_DATA: dict[str, Any] = {
    "job_id": "mock",
    "losses": {
        "value_function": [
            {"step": 0, "value": 0.143},
            {"step": 1, "value": 0.118},
        ],
        "surrogate": [{"step": 0, "value": -0.004}],
        "entropy": [{"step": 0, "value": 12.31}],
    },
}

_DEMO_ROOT = Path(__file__).resolve().parents[1] / "assets" / "policy_demo"
_DEMO_MP4 = _DEMO_ROOT / "1a2_34000.mp4"
_DEMO_ONNX = _DEMO_ROOT / "1a2_34000.onnx"


def _mock_files(payload: TrainBvhRequest, stack: ExitStack) -> tuple[CallbackFile, ...]:
    onnx = CallbackFile(
        content=stack.enter_context(_DEMO_ONNX.open("rb")),
        filename=_DEMO_ONNX.name,
    )
    if payload.return_type == 2:
        return (onnx,)

    mp4 = CallbackFile(
        content=stack.enter_context(_DEMO_MP4.open("rb")),
        filename=_DEMO_MP4.name,
        content_type="video/mp4",
    )
    return (mp4, onnx)


async def _send_mock_loss_callback(
    client: httpx.AsyncClient,
    settings: Settings,
    task_id: str,
    payload: TrainBvhRequest,
) -> None:
    if payload.loss_callback_url is None:
        return
    headers: dict[str, str] = {}
    if settings.callback_token:
        headers["X-Callback-Token"] = settings.callback_token
    action_id = payload.action_id or f"training-{task_id}"
    response = await client.post(
        str(payload.loss_callback_url),
        json={"actionId": action_id, "data": _MOCK_LOSS_DATA},
        headers=headers,
        timeout=settings.gpu_control_timeout_seconds,
    )
    response.raise_for_status()
    logger.info("已回调 Mock loss：taskId=%s", task_id)


async def _run_mock_loss_polling(
    client: httpx.AsyncClient,
    settings: Settings,
    task_id: str,
    payload: TrainBvhRequest,
) -> None:
    if payload.loss_callback_url is None:
        return
    deadline = asyncio.get_running_loop().time() + settings.train_timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        try:
            await asyncio.sleep(_MOCK_LOSS_INTERVAL_SECONDS)
            if asyncio.get_running_loop().time() >= deadline:
                return
            await _send_mock_loss_callback(client, settings, task_id, payload)
        except asyncio.CancelledError:
            raise
        except (httpx.HTTPError, ValueError, TypeError) as error:
            logger.warning("Mock loss 轮询或回调失败：taskId=%s error=%s", task_id, error)
        except Exception:
            logger.exception("Mock loss 轮询异常：taskId=%s", task_id)
            return


async def run_mock_train_task(
    client: httpx.AsyncClient,
    settings: Settings,
    task_id: str,
    payload: TrainBvhRequest,
) -> None:
    """等待十秒后回调 Mock loss 和演示产物，不访问 GPU 云服务。"""

    try:
        await asyncio.sleep(10.0)
        try:
            await _send_mock_loss_callback(client, settings, task_id, payload)
        except (httpx.HTTPError, ValueError, TypeError) as error:
            logger.warning("Mock 首次 loss 回调失败：taskId=%s error=%s", task_id, error)
        asyncio.create_task(
            _run_mock_loss_polling(client, settings, task_id, payload)
        )
        with ExitStack() as stack:
            await send_callback(
                client,
                callback_url=str(payload.callback_url),
                action_id=payload.action_id,
                success=True,
                message="Mock 训练成功",
                callback_token=settings.callback_token,
                attachments=_mock_files(payload, stack),
            )
    except Exception as error:
        logger.exception("Mock training callback failed: taskId=%s", task_id)
        log_callback_failure(task_id, error)
