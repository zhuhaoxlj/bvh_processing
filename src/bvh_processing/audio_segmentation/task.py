from __future__ import annotations

import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

from bvh_processing.audio_segmentation.service import analyze_audio, download_audio
from bvh_processing.config import Settings
from bvh_processing.errors import BvhServiceError
from bvh_processing.schemas import SegmentAudioRequest
from bvh_processing.services.callback import log_callback_failure

logger = logging.getLogger(__name__)


async def _post_result(
    client: httpx.AsyncClient,
    settings: Settings,
    payload: SegmentAudioRequest,
    result: object,
) -> None:
    headers = {}
    if settings.callback_token:
        headers["X-Callback-Token"] = settings.callback_token
    response = await client.post(
        str(payload.callback_url),
        json=result,
        headers=headers,
    )
    response.raise_for_status()


async def run_audio_segmentation_task(
    client: httpx.AsyncClient,
    settings: Settings,
    task_id: str,
    payload: SegmentAudioRequest,
) -> None:
    try:
        with TemporaryDirectory(prefix="audio-segmentation-") as directory:
            workspace = Path(directory)
            audio_path = await download_audio(
                client,
                str(payload.audio_file_url),
                workspace,
                settings,
            )
            segments = await analyze_audio(
                audio_path,
                workspace,
                settings,
                payload.section_labels,
            )
        await _post_result(client, settings, payload, segments)
    except Exception as error:
        logger.exception("Audio segmentation task failed: taskId=%s", task_id)
        message = (
            error.message
            if isinstance(error, BvhServiceError)
            else "音频结构解析失败"
        )
        try:
            await _post_result(
                client,
                settings,
                payload,
                {"success": False, "taskId": task_id, "message": message},
            )
        except Exception as callback_error:  # noqa: BLE001
            log_callback_failure(task_id, callback_error)