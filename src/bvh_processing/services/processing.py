from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import SpooledTemporaryFile

from bvh_processing.errors import BvhServiceError
from bvh_processing.services.download import DownloadedBvh

_SPOOL_MEMORY_LIMIT = 8 * 1024 * 1024
_FRAMES_PATTERN = re.compile(r"^Frames\s*:\s*(\d+)\s*$", re.IGNORECASE)
_FRAME_TIME_PATTERN = re.compile(
    r"^Frame\s+Time\s*:\s*([0-9]+(?:\.[0-9]*)?|\.[0-9]+)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class _ParsedBvh:
    hierarchy: str
    frame_time: float
    frame_time_text: str
    frames: list[str]
    channel_count: int


def processed_filename(source_filename: str) -> str:
    source = Path(source_filename)
    return f"{source.stem}_processed.bvh"


def merged_filename(source_filename: str) -> str:
    source = Path(source_filename)
    return f"{source.stem}_merged.bvh"


def process_bvh(
    downloaded: DownloadedBvh,
    handle_options: list[int],
) -> DownloadedBvh:
    """平滑算法接入点；联调阶段保持 BVH 内容不变。"""
    del handle_options
    return downloaded


def _invalid_bvh(message: str) -> BvhServiceError:
    return BvhServiceError(
        status_code=422,
        code="invalid_bvh",
        message=message,
    )


def _parse_bvh(downloaded: DownloadedBvh) -> _ParsedBvh:
    downloaded.content.seek(0)
    raw = downloaded.content.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise _invalid_bvh(
            f"{downloaded.source_filename} 不是 UTF-8 编码的 BVH 文件"
        ) from error

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    motion_index = next(
        (index for index, line in enumerate(lines) if line.strip().upper() == "MOTION"),
        None,
    )
    if motion_index is None or motion_index + 2 >= len(lines):
        raise _invalid_bvh(f"{downloaded.source_filename} 缺少 MOTION 数据")

    frames_match = _FRAMES_PATTERN.match(lines[motion_index + 1].strip())
    frame_time_match = _FRAME_TIME_PATTERN.match(lines[motion_index + 2].strip())
    if frames_match is None or frame_time_match is None:
        raise _invalid_bvh(f"{downloaded.source_filename} 的 MOTION 头格式不正确")

    frame_count = int(frames_match.group(1))
    frame_time_text = frame_time_match.group(1)
    frame_time = float(frame_time_text)
    if frame_time <= 0:
        raise _invalid_bvh(f"{downloaded.source_filename} 的 Frame Time 必须大于 0")

    frames = [line.strip() for line in lines[motion_index + 3 :] if line.strip()]
    if len(frames) != frame_count or not frames:
        raise _invalid_bvh(
            f"{downloaded.source_filename} 声明 {frame_count} 帧，实际读取 {len(frames)} 帧"
        )

    channel_count = len(frames[0].split())
    if channel_count == 0 or any(
        len(frame.split()) != channel_count for frame in frames
    ):
        raise _invalid_bvh(f"{downloaded.source_filename} 的帧通道数量不一致")

    hierarchy = "\n".join(line.rstrip() for line in lines[:motion_index]).strip()
    if not hierarchy:
        raise _invalid_bvh(f"{downloaded.source_filename} 缺少 HIERARCHY 数据")
    return _ParsedBvh(
        hierarchy=hierarchy,
        frame_time=frame_time,
        frame_time_text=frame_time_text,
        frames=frames,
        channel_count=channel_count,
    )


def merge_bvh_files(
    downloaded_files: list[DownloadedBvh],
    intervals_seconds: list[float],
) -> DownloadedBvh:
    """合并骨架和采样率一致的 BVH，间隔区间保持前一段的最后一帧。"""
    if len(downloaded_files) < 2 or len(intervals_seconds) != len(downloaded_files) - 1:
        raise ValueError("BVH 文件与间隔数量不匹配")

    parsed_files = [_parse_bvh(downloaded) for downloaded in downloaded_files]
    first = parsed_files[0]
    merged_frames: list[str] = []

    for index, parsed in enumerate(parsed_files):
        if parsed.hierarchy != first.hierarchy:
            raise _invalid_bvh("所有 BVH 文件必须使用完全相同的骨架层级")
        if parsed.channel_count != first.channel_count:
            raise _invalid_bvh("所有 BVH 文件的通道数量必须一致")
        if not math.isclose(
            parsed.frame_time, first.frame_time, rel_tol=1e-7, abs_tol=1e-9
        ):
            raise _invalid_bvh("所有 BVH 文件的 Frame Time 必须一致")

        merged_frames.extend(parsed.frames)
        if index < len(intervals_seconds):
            interval_frame_count = math.floor(
                intervals_seconds[index] / first.frame_time + 0.5
            )
            merged_frames.extend([parsed.frames[-1]] * interval_frame_count)

    output_text = (
        f"{first.hierarchy}\nMOTION\n"
        f"Frames: {len(merged_frames)}\n"
        f"Frame Time: {first.frame_time_text}\n" + "\n".join(merged_frames) + "\n"
    )
    output_bytes = output_text.encode("utf-8")
    output = SpooledTemporaryFile(max_size=_SPOOL_MEMORY_LIMIT, mode="w+b")  # noqa: SIM115
    output.write(output_bytes)
    output.seek(0)
    return DownloadedBvh(
        content=output,
        source_filename=merged_filename(downloaded_files[0].source_filename),
        size=len(output_bytes),
    )
