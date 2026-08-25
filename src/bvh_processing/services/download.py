from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from tempfile import SpooledTemporaryFile
from typing import BinaryIO
from urllib.parse import unquote, urlsplit

import httpx

from bvh_processing.config import Settings
from bvh_processing.errors import BvhServiceError
from bvh_processing.services.classify_bvh import (
    BVHClassificationError,
    classify_downloaded_bvh,
)

_CHUNK_SIZE = 64 * 1024
_SPOOL_MEMORY_LIMIT = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class DownloadedBvh:
    content: BinaryIO
    source_filename: str
    size: int


def _is_allowed_host(host: str, allowed_hosts: frozenset[str]) -> bool:
    normalized = host.lower().rstrip(".")
    return not allowed_hosts or any(
        normalized == allowed or normalized.endswith(f".{allowed}")
        for allowed in allowed_hosts
    )


def _source_filename(url: str) -> str:
    name = unquote(PurePosixPath(urlsplit(url).path).name)
    return name if name.lower().endswith(".bvh") else "source.bvh"


async def download_bvh(
    client: httpx.AsyncClient,
    source_url: str,
    settings: Settings,
) -> DownloadedBvh:
    parsed = urlsplit(source_url)
    if parsed.username or parsed.password:
        raise BvhServiceError(
            status_code=400,
            code="invalid_source_url",
            message="下载地址不能在 URL 中包含用户名或密码",
        )
    if not parsed.hostname or not _is_allowed_host(
        parsed.hostname, settings.allowed_hosts
    ):
        raise BvhServiceError(
            status_code=400,
            code="source_host_not_allowed",
            message="该 MinIO 地址不在允许的主机列表中",
        )

    # 文件所有权转交给任务执行器，任务结束后统一关闭。
    content = SpooledTemporaryFile(  # noqa: SIM115
        max_size=_SPOOL_MEMORY_LIMIT,
        mode="w+b",
    )
    size = 0
    try:
        async with client.stream("GET", source_url) as response:
            if response.is_error:
                raise BvhServiceError(
                    status_code=502,
                    code="source_download_failed",
                    message=f"MinIO 返回 HTTP {response.status_code}",
                )

            declared_size = response.headers.get("content-length")
            if declared_size and int(declared_size) > settings.max_file_size_bytes:
                raise BvhServiceError(
                    status_code=413,
                    code="source_file_too_large",
                    message=f"BVH 文件不能超过 {settings.max_file_size_mb} MB",
                )

            async for chunk in response.aiter_bytes(_CHUNK_SIZE):
                size += len(chunk)
                if size > settings.max_file_size_bytes:
                    raise BvhServiceError(
                        status_code=413,
                        code="source_file_too_large",
                        message=f"BVH 文件不能超过 {settings.max_file_size_mb} MB",
                    )
                content.write(chunk)
    except BvhServiceError:
        content.close()
        raise
    except (httpx.HTTPError, OSError, ValueError) as exc:
        content.close()
        raise BvhServiceError(
            status_code=502,
            code="source_download_failed",
            message="无法下载 MinIO 中的 BVH 文件",
        ) from exc

    if size == 0:
        content.close()
        raise BvhServiceError(
            status_code=422,
            code="empty_source_file",
            message="MinIO 中的 BVH 文件为空",
        )

    try:
        classify_downloaded_bvh(content)
    except BVHClassificationError as exc:
        content.close()
        raise BvhServiceError(
            status_code=422,
            code="unsupported_bvh_format",
            message="只支持LAFAN1格式和Nokov格式的 BVH 文件",
        ) from exc

    content.seek(0)
    return DownloadedBvh(
        content=content,
        source_filename=_source_filename(source_url),
        size=size,
    )
