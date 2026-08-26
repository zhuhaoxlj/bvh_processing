from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from tempfile import SpooledTemporaryFile

from bvh_processing.errors import BvhServiceError
from bvh_processing.services.download import DownloadedBvh
from bvh_processing.services.transition import (
    create_transitions,
    motion_from_parts,
    motion_hierarchy,
)

_SPOOL_MEMORY_LIMIT = 8 * 1024 * 1024
_FRAMES_PATTERN = re.compile(r"^Frames\s*:\s*(\d+)\s*$", re.IGNORECASE)
_FRAME_TIME_PATTERN = re.compile(
    r"^Frame\s+Time\s*:\s*([0-9]+(?:\.[0-9]*)?|\.[0-9]+)\s*$",
    re.IGNORECASE,
)
_CHANNELS_PATTERN = re.compile(r"^\s*CHANNELS\s+(\d+)\s+(.+)$", re.IGNORECASE)


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
    """按照 handleOptions 的顺序处理 BVH 动作数据。"""
    processors = {
        1: denoise_bvh,
        2: smooth_bvh,
        3: lock_bvh_feet,
        4: optimize_bvh_loop,
    }
    current = downloaded
    try:
        for option in handle_options:
            processor = processors.get(option)
            if processor is None:
                raise BvhServiceError(
                    status_code=422,
                    code="invalid_handle_option",
                    message=f"不支持的 BVH 处理选项：{option}",
                )
            processed = processor(current)
            if processed is not current and current is not downloaded:
                current.content.close()
            current = processed
    except Exception:
        if current is not downloaded:
            current.content.close()
        raise
    return current


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


def _resample_frames(parsed: _ParsedBvh, target_frame_time: float) -> list[str]:
    """按目标采样间隔选取最近帧，只降低帧率，不生成插值姿势。"""
    if math.isclose(parsed.frame_time, target_frame_time, rel_tol=1e-7, abs_tol=1e-9):
        return parsed.frames

    output_frame_count = max(
        1,
        math.floor(len(parsed.frames) * parsed.frame_time / target_frame_time + 0.5),
    )
    return [
        parsed.frames[
            min(
                math.floor(index * target_frame_time / parsed.frame_time + 0.5),
                len(parsed.frames) - 1,
            )
        ]
        for index in range(output_frame_count)
    ]


def _build_bvh(
    parsed: _ParsedBvh,
    frames: list[str],
    frame_time_text: str,
    source_filename: str,
) -> DownloadedBvh:
    output_text = (
        f"{parsed.hierarchy}\nMOTION\n"
        f"Frames: {len(frames)}\n"
        f"Frame Time: {frame_time_text}\n" + "\n".join(frames) + "\n"
    )
    output_bytes = output_text.encode("utf-8")
    output = SpooledTemporaryFile(max_size=_SPOOL_MEMORY_LIMIT, mode="w+b")  # noqa: SIM115
    output.write(output_bytes)
    output.seek(0)
    return DownloadedBvh(
        content=output,
        source_filename=source_filename,
        size=len(output_bytes),
    )


def _rotation_channel_indexes(parsed: _ParsedBvh) -> set[int]:
    channel_names: list[str] = []
    for line in parsed.hierarchy.splitlines():
        match = _CHANNELS_PATTERN.match(line)
        if match is None:
            continue
        declared_count = int(match.group(1))
        names = match.group(2).split()
        if len(names) != declared_count:
            return set()
        channel_names.extend(names)
    if len(channel_names) != parsed.channel_count:
        return set()
    return {
        index
        for index, name in enumerate(channel_names)
        if name.lower().endswith("rotation")
    }


def _unwrap_rotations(
    values: list[list[float]],
    rotation_channels: set[int],
) -> None:
    """展开跨越正负 180 度边界的旋转，避免滤波产生错误的中间角度。"""
    for channel in rotation_channels:
        for frame_index in range(1, len(values)):
            previous = values[frame_index - 1][channel]
            current = values[frame_index][channel]
            delta = (current - previous + 180.0) % 360.0 - 180.0
            values[frame_index][channel] = previous + delta


def _motion_values(parsed: _ParsedBvh, source_filename: str) -> list[list[float]]:
    values: list[list[float]] = []
    try:
        for frame in parsed.frames:
            row = [float(value) for value in frame.split()]
            if any(not math.isfinite(value) for value in row):
                raise ValueError
            values.append(row)
    except ValueError as error:
        raise _invalid_bvh(f"{source_filename} 的 MOTION 帧包含无效数值") from error
    return values


def _format_motion_value(value: float) -> str:
    if math.isclose(value, 0.0, abs_tol=1e-12):
        return "0"
    return f"{value:.10g}"


def _build_processed_bvh(
    downloaded: DownloadedBvh,
    parsed: _ParsedBvh,
    values: list[list[float]],
) -> DownloadedBvh:
    frames = [
        " ".join(_format_motion_value(value) for value in frame) for frame in values
    ]
    return _build_bvh(
        parsed,
        frames,
        parsed.frame_time_text,
        downloaded.source_filename,
    )


def denoise_bvh(
    downloaded: DownloadedBvh,
    window_size: int = 3,
) -> DownloadedBvh:
    """使用时间轴中值滤波移除各运动通道的孤立尖峰。"""
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("去噪窗口必须是大于等于 1 的奇数")

    parsed = _parse_bvh(downloaded)
    values = _motion_values(parsed, downloaded.source_filename)
    _unwrap_rotations(values, _rotation_channel_indexes(parsed))
    radius = window_size // 2
    filtered: list[list[float]] = []
    for frame_index, frame in enumerate(values):
        neighbors = [
            min(max(index, 0), len(values) - 1)
            for index in range(frame_index - radius, frame_index + radius + 1)
        ]
        filtered.append(
            [
                statistics.median(values[index][channel] for index in neighbors)
                for channel in range(len(frame))
            ]
        )
    return _build_processed_bvh(downloaded, parsed, filtered)


def smooth_bvh(
    downloaded: DownloadedBvh,
    window_size: int = 5,
) -> DownloadedBvh:
    """使用居中移动平均滤波平滑各运动通道的逐帧抖动。"""
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("平滑窗口必须是大于等于 1 的奇数")

    parsed = _parse_bvh(downloaded)
    values = _motion_values(parsed, downloaded.source_filename)
    _unwrap_rotations(values, _rotation_channel_indexes(parsed))
    radius = window_size // 2
    smoothed: list[list[float]] = []
    for frame_index, frame in enumerate(values):
        neighbors = [
            min(max(index, 0), len(values) - 1)
            for index in range(frame_index - radius, frame_index + radius + 1)
        ]
        smoothed.append(
            [
                statistics.fmean(values[index][channel] for index in neighbors)
                for channel in range(len(frame))
            ]
        )
    return _build_processed_bvh(downloaded, parsed, smoothed)


def lock_bvh_feet(downloaded: DownloadedBvh) -> DownloadedBvh:
    """脚步锁定算法占位；当前保持 BVH 数据不变。"""
    return downloaded


def optimize_bvh_loop(downloaded: DownloadedBvh) -> DownloadedBvh:
    """循环优化算法占位；当前保持 BVH 数据不变。"""
    return downloaded


def normalize_bvh_frame_rates(
    downloaded_files: list[DownloadedBvh],
) -> list[DownloadedBvh]:
    """将多个 BVH 分别降采样到其中的最低帧率。"""
    if not downloaded_files:
        raise ValueError("BVH 文件不能为空")

    parsed_files = [_parse_bvh(downloaded) for downloaded in downloaded_files]
    # 帧率最低的文件拥有最大的 Frame Time；其他文件只做降采样。
    target = max(parsed_files, key=lambda parsed: parsed.frame_time)
    normalized_files: list[DownloadedBvh] = []
    try:
        for downloaded, parsed in zip(downloaded_files, parsed_files, strict=True):
            normalized_files.append(
                _build_bvh(
                    parsed,
                    _resample_frames(parsed, target.frame_time),
                    target.frame_time_text,
                    downloaded.source_filename,
                )
            )
    except Exception:
        for normalized in normalized_files:
            normalized.content.close()
        raise
    return normalized_files


def adjust_bvh_motion_durations(
    downloaded_files: list[DownloadedBvh],
    target_durations_seconds: list[float],
) -> list[DownloadedBvh]:
    """按目标动作时长重采样 BVH，保持统一帧率和首尾姿势。"""
    if len(downloaded_files) != len(target_durations_seconds):
        raise ValueError("BVH 文件与动作时长数量不匹配")

    adjusted_files: list[DownloadedBvh] = []
    try:
        for downloaded, target_duration in zip(
            downloaded_files,
            target_durations_seconds,
            strict=True,
        ):
            parsed = _parse_bvh(downloaded)
            if target_duration < 0:
                raise ValueError("BVH 动作时长不能为负数")

            # BVH 的动作时长是首帧到末帧的时间跨度。
            source_duration = max(0.0, (len(parsed.frames) - 1) * parsed.frame_time)
            if source_duration == 0.0 or target_duration == 0.0:
                output_frames = [parsed.frames[0]]
            else:
                output_frame_count = max(
                    2,
                    math.floor(target_duration / parsed.frame_time + 0.5) + 1,
                )
                output_frames = [
                    parsed.frames[
                        min(
                            math.floor(index * source_duration / target_duration + 0.5),
                            len(parsed.frames) - 1,
                        )
                    ]
                    for index in range(output_frame_count)
                ]

            adjusted_files.append(
                _build_bvh(
                    parsed,
                    output_frames,
                    parsed.frame_time_text,
                    downloaded.source_filename,
                )
            )
    except Exception:
        for adjusted in adjusted_files:
            adjusted.content.close()
        raise
    return adjusted_files


def merge_bvh_files(
    downloaded_files: list[DownloadedBvh],
    intervals_seconds: list[float],
) -> DownloadedBvh:
    """使用动作对齐、旋转插值和脚部锁定过渡来合并多个 BVH。"""
    if len(downloaded_files) < 2 or len(intervals_seconds) != len(downloaded_files) - 1:
        raise ValueError("BVH 文件与过渡时间数量不匹配")

    parsed_files = [_parse_bvh(downloaded) for downloaded in downloaded_files]
    first = parsed_files[0]
    for parsed in parsed_files[1:]:
        if not math.isclose(
            parsed.frame_time, first.frame_time, rel_tol=1e-7, abs_tol=1e-9
        ):
            raise _invalid_bvh("所有 BVH 文件的 Frame Time 必须一致")

    # intervalsSeconds 不再生成静止帧，而是转换成每个接缝的过渡帧数。
    transition_frame_counts = [
        math.floor(seconds / first.frame_time + 0.5) for seconds in intervals_seconds
    ]
    try:
        motions = [
            motion_from_parts(parsed.hierarchy, parsed.frames, parsed.frame_time)
            for parsed in parsed_files
        ]
        merged_motion = create_transitions(motions, transition_frame_counts)
    except ValueError as error:
        raise _invalid_bvh(str(error)) from error

    merged_frames = [
        " ".join(_format_motion_value(float(value)) for value in frame)
        for frame in merged_motion.frames
    ]
    merged = _ParsedBvh(
        hierarchy=motion_hierarchy(merged_motion),
        frame_time=first.frame_time,
        frame_time_text=first.frame_time_text,
        frames=merged_frames,
        channel_count=merged_motion.frames.shape[1],
    )
    return _build_bvh(
        merged,
        merged_frames,
        first.frame_time_text,
        merged_filename(downloaded_files[0].source_filename),
    )
