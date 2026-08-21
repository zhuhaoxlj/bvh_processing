import logging

import httpx

from bvh_processing.config import Settings
from bvh_processing.errors import BvhServiceError
from bvh_processing.schemas import ProcessBvhRequest
from bvh_processing.services.callback import log_callback_failure, send_callback
from bvh_processing.services.download import DownloadedBvh, download_bvh
from bvh_processing.services.processing import process_bvh, processed_filename

logger = logging.getLogger(__name__)


def _failure_message(error: Exception) -> str:
    if isinstance(error, BvhServiceError):
        return error.message
    return "BVH 文件处理失败"


# async def _send_failure_callback(
#     client: httpx.AsyncClient,
#     task_id: str,
#     payload: ProcessBvhRequest,
#     error: Exception,
# ) -> None:
#     try:
#         await send_callback(
#             client,
#             callback_url=str(payload.callback_url),
#             action_id=payload.action_id,
#             success=False,
#             message=_failure_message(error),
#         )
#     except Exception as callback_error:  # noqa: BLE001
#         # 回调失败不能逃逸到后台任务运行器。
#         log_callback_failure(task_id, callback_error)

async def _send_failure_callback(
    client: httpx.AsyncClient,
    settings: Settings,
    task_id: str,
    payload: ProcessBvhRequest,
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
        # 回调失败不能逃逸到后台任务运行器。
        log_callback_failure(task_id, callback_error)

async def run_processing_task(
    client: httpx.AsyncClient,
    settings: Settings,
    task_id: str,
    payload: ProcessBvhRequest,
) -> None:
    resource: DownloadedBvh | None = None
    try:
        resource = await download_bvh(
            client,
            str(payload.original_file_url),
            settings,
        )
        result = process_bvh(resource, payload.handle_options)
    except Exception as error:  # noqa: BLE001
        # 算法层可能抛出未知异常，任务边界必须将其转换为失败回调。
        logger.error(
            "BVH task %s processing failed: %s",
            task_id,
            type(error).__name__,
        )
        if resource is not None:
            resource.content.close()
        # await _send_failure_callback(client, task_id, payload, error)
        await _send_failure_callback(client, settings, task_id, payload, error)
        return

    try:
        await send_callback(
            client,
            callback_url=str(payload.callback_url),
            action_id=payload.action_id,
            success=True,
            message="处理成功",
            callback_token=settings.callback_token,
            file=result.content,
            filename=processed_filename(result.source_filename),
        )
    except Exception as callback_error:  # noqa: BLE001
        # 回调失败不能逃逸到后台任务运行器。
        log_callback_failure(task_id, callback_error)
    finally:
        resource.content.close()
