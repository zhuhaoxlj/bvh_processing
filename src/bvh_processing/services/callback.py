import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import BinaryIO
from urllib.parse import urlsplit

import httpx

from bvh_processing.errors import BvhServiceError

logger = logging.getLogger(__name__)

MultipartValue = tuple[None, str] | tuple[str, BinaryIO, str]
MultipartPart = tuple[str, MultipartValue]


@dataclass(frozen=True, slots=True)
class CallbackFile:
    content: BinaryIO
    filename: str
    content_type: str = "application/octet-stream"


def validate_callback_url(
    callback_url: str,
    allowed_hosts: frozenset[str],
) -> None:
    parsed = urlsplit(callback_url)
    if parsed.username or parsed.password:
        raise BvhServiceError(
            status_code=400,
            code="invalid_callback_url",
            message="回调地址不能在 URL 中包含用户名或密码",
        )

    hostname = parsed.hostname.lower().rstrip(".") if parsed.hostname else ""
    is_allowed = not allowed_hosts or any(
        hostname == allowed or hostname.endswith(f".{allowed}")
        for allowed in allowed_hosts
    )
    if not hostname or not is_allowed:
        raise BvhServiceError(
            status_code=400,
            code="callback_host_not_allowed",
            message="回调地址不在允许的主机列表中",
        )


def _form_fields(
    action_id: str | None,
    success: bool,
    message: str,
    handle_option: int | None = None,
    option_status: str | None = None,
) -> list[MultipartPart]:
    fields = [
        ("success", (None, str(success).lower())),
        ("message", (None, message)),
    ]
    if action_id is not None:
        fields.insert(0, ("actionId", (None, action_id)))
    if handle_option is not None:
        fields.append(("handleOption", (None, str(handle_option))))
    if option_status is not None:
        fields.append(("optionStatus", (None, option_status)))
    return fields


async def send_callback(
    client: httpx.AsyncClient,
    *,
    callback_url: str,
    action_id: str | None,
    success: bool,
    message: str,
    callback_token: str | None = None,
    handle_option: int | None = None,
    option_status: str | None = None,
    file: BinaryIO | None = None,
    filename: str | None = None,
    file_content_type: str = "application/octet-stream",
    attachments: Sequence[CallbackFile] = (),
) -> None:
    parts: list[MultipartPart] = _form_fields(
        action_id, success, message, handle_option, option_status
    )
    if success and handle_option is None:
        callback_files = list(attachments)
        if file is not None and filename is not None:
            callback_files.insert(
                0,
                CallbackFile(file, filename, file_content_type),
            )
        if not callback_files:
            raise ValueError("成功回调必须包含处理后的文件")
        for callback_file in callback_files:
            callback_file.content.seek(0)
            parts.append(
                (
                    "file",
                    (
                        callback_file.filename,
                        callback_file.content,
                        callback_file.content_type,
                    ),
                )
            )

    headers = {}
    if callback_token:
        headers["X-Callback-Token"] = callback_token

    response = await client.post(callback_url, files=parts, headers=headers)
    response.raise_for_status()


def log_callback_failure(task_id: str, error: Exception) -> None:
    logger.exception(
        "BVH callback failed: taskId=%s error=%s",
        task_id,
        error,
    )


async def send_progress_callback(
    client: httpx.AsyncClient,
    *,
    progress_url: str,
    action_id: str,
    original_file_url: str,
    progress: int,
    step: int,
    step_code: str,
    message: str,
    callback_token: str | None = None,
) -> None:
    headers: dict[str, str] = {}
    if callback_token:
        headers["X-Callback-Token"] = callback_token

    payload = {
        "actionId": action_id,
        "originalFileUrl": original_file_url,
        "progress": progress,
        "step": step,
        "stepCode": step_code,
        "message": message,
    }

    response = await client.post(progress_url, json=payload, headers=headers)
    response.raise_for_status()
