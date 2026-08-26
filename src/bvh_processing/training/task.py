"""BVH 训练后台任务及结果回调。"""

import logging

import httpx

from bvh_processing.config import Settings
from bvh_processing.errors import BvhServiceError
from bvh_processing.schemas import TrainBvhRequest
from bvh_processing.services.callback import log_callback_failure, send_callback
from bvh_processing.services.download import DownloadedBvh, download_npz
from bvh_processing.training.service import TrainingArtifact, run_training_program

logger = logging.getLogger(__name__)


def _failure_message(error: Exception) -> str:
    if isinstance(error, BvhServiceError):
        return error.message
    return "训练任务执行失败"


async def _send_failure_callback(
    client: httpx.AsyncClient,
    settings: Settings,
    task_id: str,
    payload: TrainBvhRequest,
    error: Exception,
) -> None:
    try:
        await send_callback(
            client,
            callback_url=str(payload.callback_url),
            action_id=payload.action_id,
            success=False,
            message=_failure_message(error),
            callback_token=settings.callback_token,
        )
    except Exception as callback_error:  # noqa: BLE001
        log_callback_failure(task_id, callback_error)


async def run_train_task(
    client: httpx.AsyncClient,
    settings: Settings,
    task_id: str,
    payload: TrainBvhRequest,
) -> None:
    source: DownloadedBvh | None = None
    artifact: TrainingArtifact | None = None
    try:
        source = await download_npz(client, str(payload.npz_file_url), settings)
        artifact = await run_training_program(source, payload, settings)
        await send_callback(
            client,
            callback_url=str(payload.callback_url),
            action_id=payload.action_id,
            success=True,
            message="训练成功",
            callback_token=settings.callback_token,
            file=artifact.content,
            filename=artifact.filename,
            file_content_type=artifact.content_type,
        )
    except Exception as error:  # noqa: BLE001
        logger.error(
            "BVH training task %s failed: %s",
            task_id,
            type(error).__name__,
        )
        await _send_failure_callback(client, settings, task_id, payload, error)
    finally:
        if artifact is not None:
            artifact.close()
        if source is not None:
            source.content.close()
