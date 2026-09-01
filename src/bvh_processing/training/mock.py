"""训练接口的本地 Mock 结果回调。"""

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


async def run_mock_train_task(
    client: httpx.AsyncClient,
    settings: Settings,
    task_id: str,
    payload: TrainBvhRequest,
) -> None:
    """按 returnType 将内置演示产物回调给业务后端。"""

    try:
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
