import asyncio
import logging

import httpx

from bvh_processing.config import Settings
from bvh_processing.errors import BvhServiceError
from bvh_processing.schemas import MergeBvhRequest, ProcessBvhRequest
from bvh_processing.services.callback import (
    log_callback_failure,
    send_callback,
    send_progress_callback,
)
from bvh_processing.services.download import DownloadedBvh, download_bvh
from bvh_processing.services.processing import (
    merge_bvh_files,
    process_bvh,
    processed_filename,
)

logger = logging.getLogger(__name__)


def _failure_message(error: Exception) -> str:
    if isinstance(error, BvhServiceError):
        return error.message
    return "BVH 文件处理失败"

async def _send_failure_callback(
    client: httpx.AsyncClient,
    settings: Settings,
    task_id: str,
    payload: ProcessBvhRequest | MergeBvhRequest,
    error: Exception,
    handle_option: int | None = None,
) -> None:
    try:
        await send_callback(
            client,
            callback_url=str(payload.callback_url),
            action_id=payload.action_id,
            success=False,
            message=_failure_message(error),
            callback_token=settings.callback_token,
            handle_option=handle_option,
            option_status="failed" if handle_option is not None else None,
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

    # 中间进度：打到 /progress-callback
    callback_url = str(payload.callback_url)
    progress_url = callback_url.replace("/callback", "/progress-callback", 1)

    option_to_step = {
        1: (2, "DENOISE", "整体去噪"),
        2: (3, "SMOOTH_FRAME", "整体平滑与补帧"),
        3: (4, "FOOT_LOCK", "脚步锁定校正"),
        4: (5, "LOOP_OPTIMIZE", "循环优化（首尾自然过渡）"),
    }

    options = [opt for opt in payload.handle_options if opt in option_to_step]
    total = max(len(options), 1)

    for index, handle_option in enumerate(options, start=1):
        step, step_code, step_name = option_to_step[handle_option]
        progress = 5 + int(90 * index / total)
        try:
            await send_progress_callback(
                client,
                progress_url=progress_url,
                action_id=payload.action_id,
                original_file_url=str(payload.original_file_url),
                progress=progress,
                step=step,
                step_code=step_code,
                message=f"正在处理{step_name}",
                callback_token=settings.callback_token,
            )
        except Exception as callback_error:  # noqa: BLE001
            log_callback_failure(task_id, callback_error)

    # 最终成功回调：只发一次
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


async def run_merge_task(
    client: httpx.AsyncClient,
    settings: Settings,
    task_id: str,
    payload: MergeBvhRequest,
) -> None:
    resources: list[DownloadedBvh] = []
    result: DownloadedBvh | None = None
    try:
        for file_url in payload.file_urls:
            resources.append(await download_bvh(client, str(file_url), settings))
        # BVH 解析和大量帧写入属于阻塞工作，放在线程中避免阻塞事件循环。
        result = await asyncio.to_thread(
            merge_bvh_files,
            resources,
            payload.intervals_seconds,
        )
        await send_callback(
            client,
            callback_url=str(payload.callback_url),
            action_id=payload.action_id,
            success=True,
            message="BVH 合并成功",
            callback_token=settings.callback_token,
            file=result.content,
            filename=result.source_filename,
        )
    except Exception as error:  # noqa: BLE001
        logger.error(
            "BVH merge task %s failed: %s",
            task_id,
            type(error).__name__,
        )
        await _send_failure_callback(client, settings, task_id, payload, error)
    finally:
        if result is not None:
            result.content.close()
        for resource in resources:
            resource.content.close()
