"""BVH 重定向后台任务；该模块不承担普通处理或合并任务。"""

import asyncio
import logging

import httpx

from bvh_processing.config import Settings
from bvh_processing.errors import BvhServiceError
from bvh_processing.retargeting.exporter import RetargetArtifacts
from bvh_processing.retargeting.service import retarget_downloaded_bvh
from bvh_processing.schemas import RetargetBvhRequest
from bvh_processing.services.callback import (
    CallbackFile,
    log_callback_failure,
    send_callback,
)
from bvh_processing.services.download import DownloadedBvh, download_bvh

logger = logging.getLogger(__name__)


def _failure_message(error: Exception) -> str:
    if isinstance(error, BvhServiceError):
        return error.message
    return "BVH 重定向失败"


async def _send_failure_callback(
    client: httpx.AsyncClient,
    settings: Settings,
    task_id: str,
    payload: RetargetBvhRequest,
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


async def run_retarget_task(
    client: httpx.AsyncClient,
    settings: Settings,
    task_id: str,
    payload: RetargetBvhRequest,
) -> None:
    resource: DownloadedBvh | None = None
    artifacts: RetargetArtifacts | None = None
    logger.info(
        "BVH Robot Retargeter task %s robotType=%d robot=G1",
        task_id,
        payload.robot_type,
    )
    try:
        resource = await download_bvh(
            client,
            str(payload.original_file_url),
            settings,
        )
        artifacts = await asyncio.to_thread(retarget_downloaded_bvh, resource)
        await send_callback(
            client,
            callback_url=str(payload.callback_url),
            action_id=payload.action_id,
            success=True,
            message="重定向 NPZ 和 JSON 生成成功",
            callback_token=settings.callback_token,
            attachments=(
                CallbackFile(
                    artifacts.npz,
                    artifacts.npz_filename,
                    "application/octet-stream",
                    "npzFile",
                ),
                CallbackFile(
                    artifacts.preview,
                    artifacts.preview_filename,
                    "application/json",
                    "jsonFile",
                ),
            ),
        )
    except Exception as error:
        logger.exception("BVH Robot Retargeter task failed: taskId=%s", task_id)
        await _send_failure_callback(client, settings, task_id, payload, error)
    finally:
        if artifacts is not None:
            artifacts.close()
        if resource is not None:
            resource.content.close()
