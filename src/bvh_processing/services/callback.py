import logging
from typing import BinaryIO
from urllib.parse import urlsplit

import httpx

from bvh_processing.errors import BvhServiceError

logger = logging.getLogger(__name__)

MultipartValue = tuple[None, str] | tuple[str, BinaryIO, str]
MultipartPart = tuple[str, MultipartValue]


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


def _form_fields(action_id: str, success: bool, message: str) -> list[MultipartPart]:
    return [
        ("actionId", (None, action_id)),
        ("success", (None, str(success).lower())),
        ("message", (None, message)),
    ]


# async def send_callback(
#     client: httpx.AsyncClient,
#     *,
#     callback_url: str,
#     action_id: str,
#     success: bool,
#     message: str,
#     file: BinaryIO | None = None,
#     filename: str | None = None,
# ) -> None:
#     parts: list[MultipartPart] = _form_fields(action_id, success, message)
#     if success:
#         if file is None or filename is None:
#             raise ValueError("成功回调必须包含处理后的 BVH 文件")
#         file.seek(0)
#         parts.append(
#             (
#                 "file",
#                 (filename, file, "application/octet-stream"),
#             )
#         )

#     response = await client.post(callback_url, files=parts)
#     response.raise_for_status()


async def send_callback(
    client: httpx.AsyncClient,
    *,
    callback_url: str,
    action_id: str,
    success: bool,
    message: str,
    callback_token: str | None = None,
    file: BinaryIO | None = None,
    filename: str | None = None,
) -> None:
    parts: list[MultipartPart] = _form_fields(action_id, success, message)
    if success:
        if file is None or filename is None:
            raise ValueError("成功回调必须包含处理后的 BVH 文件")
        file.seek(0)
        parts.append(
            (
                "file",
                (filename, file, "application/octet-stream"),
            )
        )

    headers = {}
    if callback_token:
        headers["X-Callback-Token"] = callback_token

    response = await client.post(callback_url, files=parts, headers=headers)
    response.raise_for_status()


def log_callback_failure(task_id: str, error: Exception) -> None:
    logger.error(
        "BVH task %s callback failed: %s",
        task_id,
        type(error).__name__,
    )
