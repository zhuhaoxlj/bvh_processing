"""训练接口的本地 Mock 结果和 loss 回调。"""

from __future__ import annotations

import asyncio
import logging
from contextlib import ExitStack
from pathlib import Path

import httpx

from bvh_processing.config import Settings
from bvh_processing.schemas import TrainBvhRequest
from bvh_processing.services.callback import (
    CallbackFile,
    log_callback_failure,
    send_callback,
)
from bvh_processing.training.metrics import (
    latest_training_job_id,
    run_metrics_polling,
    send_loss_callback,
)

logger = logging.getLogger(__name__)

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


async def _start_cloud_loss_callbacks(
    client: httpx.AsyncClient,
    settings: Settings,
    task_id: str,
    payload: TrainBvhRequest,
) -> None:
    """使用云端最近一次训练任务的真实 loss 驱动 Mock 回调。"""

    if payload.loss_callback_url is None:
        return
    try:
        cloud_job_id = await latest_training_job_id(client, settings)
    except (httpx.HTTPError, ValueError, TypeError) as error:
        logger.warning("Mock 获取云端训练任务失败：taskId=%s error=%s", task_id, error)
        return

    try:
        await send_loss_callback(client, settings, cloud_job_id, payload)
    except (httpx.HTTPError, ValueError, TypeError) as error:
        logger.warning(
            "Mock 首次云端 loss 回调失败：taskId=%s cloudJobId=%s error=%s",
            task_id,
            cloud_job_id,
            error,
        )
    asyncio.create_task(run_metrics_polling(client, settings, cloud_job_id, payload))
    logger.info("Mock 已绑定云端 loss：taskId=%s cloudJobId=%s", task_id, cloud_job_id)


async def run_mock_train_task(
    client: httpx.AsyncClient,
    settings: Settings,
    task_id: str,
    payload: TrainBvhRequest,
) -> None:
    """立即回调云端真实 loss，十秒后回调本地演示产物。"""

    try:
        await _start_cloud_loss_callbacks(client, settings, task_id, payload)
        await asyncio.sleep(10.0)
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
