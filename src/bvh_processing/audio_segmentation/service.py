from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

import httpx

from bvh_processing.config import Settings
from bvh_processing.errors import BvhServiceError

_CHUNK_SIZE = 64 * 1024
_SUPPORTED_SUFFIXES = {
    ".mp3",
    ".wav",
    ".wave",
    ".flac",
    ".aiff",
    ".aif",
    ".ogg",
    ".m4a",
}
_RUNNER = Path(__file__).with_name("runner.py")


def _is_allowed_host(host: str, allowed_hosts: frozenset[str]) -> bool:
    normalized = host.lower().rstrip(".")
    return not allowed_hosts or any(
        normalized == allowed or normalized.endswith(f".{allowed}")
        for allowed in allowed_hosts
    )


def _audio_filename(source_url: str) -> str:
    filename = unquote(PurePosixPath(urlsplit(source_url).path).name)
    suffix = Path(filename).suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(_SUPPORTED_SUFFIXES))
        raise BvhServiceError(
            status_code=422,
            code="unsupported_audio_format",
            message=f"不支持该音频格式，当前支持：{supported}",
        )
    return f"source{suffix}"


async def download_audio(
    client: httpx.AsyncClient,
    source_url: str,
    destination: Path,
    settings: Settings,
) -> Path:
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

    audio_path = destination / "audio" / _audio_filename(source_url)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
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
                    message=f"音频文件不能超过 {settings.max_file_size_mb} MB",
                )
            with audio_path.open("wb") as output:
                async for chunk in response.aiter_bytes(_CHUNK_SIZE):
                    size += len(chunk)
                    if size > settings.max_file_size_bytes:
                        raise BvhServiceError(
                            status_code=413,
                            code="source_file_too_large",
                            message=f"音频文件不能超过 {settings.max_file_size_mb} MB",
                        )
                    output.write(chunk)
    except BvhServiceError:
        audio_path.unlink(missing_ok=True)
        raise
    except (httpx.HTTPError, OSError, ValueError) as exc:
        audio_path.unlink(missing_ok=True)
        raise BvhServiceError(
            status_code=502,
            code="source_download_failed",
            message="无法下载 MinIO 中的音频文件",
        ) from exc

    if size == 0:
        audio_path.unlink(missing_ok=True)
        raise BvhServiceError(
            status_code=422,
            code="empty_source_file",
            message="MinIO 中的音频文件为空",
        )
    return audio_path


async def analyze_audio(
    audio_path: Path,
    workspace: Path,
    settings: Settings,
    section_labels: int,
) -> list[dict[str, object]]:
    output_path = workspace / "segments.json"
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(_RUNNER),
        str(audio_path),
        str(output_path),
        "--labels",
        str(section_labels),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=settings.audio_analysis_timeout_seconds,
        )
    except TimeoutError as exc:
        process.kill()
        await process.communicate()
        raise BvhServiceError(
            status_code=504,
            code="audio_analysis_timeout",
            message="音频结构解析超时",
        ) from exc

    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        if not detail:
            detail = stdout.decode("utf-8", errors="replace").strip()
        raise BvhServiceError(
            status_code=500,
            code="audio_analysis_failed",
            message=f"音频结构解析失败：{detail[-500:]}",
        )

    try:
        result = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BvhServiceError(
            status_code=500,
            code="invalid_audio_analysis_result",
            message="LinkSeg 未生成有效的 JSON 结果",
        ) from exc
    if not isinstance(result, list):
        raise BvhServiceError(
            status_code=500,
            code="invalid_audio_analysis_result",
            message="LinkSeg 返回的 JSON 结果不是数组",
        )
    return result