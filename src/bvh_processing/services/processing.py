from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import SpooledTemporaryFile

import numpy as np
from scipy.spatial.transform import Rotation

from bvh_processing.config import Settings
from bvh_processing.errors import BvhServiceError
from bvh_processing.services.download import DownloadedBvh
from bvh_processing.services.mdm_transition import generate_mdm_merge

_SPOOL_MEMORY_LIMIT = 8 * 1024 * 1024
_FRAMES_PATTERN = re.compile(r"^Frames\s*:\s*(\d+)\s*$", re.IGNORECASE)
_FRAME_TIME_PATTERN = re.compile(
    r"^Frame\s+Time\s*:\s*([0-9]+(?:\.[0-9]*)?|\.[0-9]+)\s*$",
    re.IGNORECASE,
)
_CHANNELS_PATTERN = re.compile(r"^\s*CHANNELS\s+(\d+)\s+(.+)$", re.IGNORECASE)
_EPSILON = 1e-9
# 判定"静止绑定姿势帧"的容差：所有旋转通道都接近 0 就认为是 T-pose 帧。
_REST_POSE_TOLERANCE_DEGREES = 1.0
# 判定人体朝向用的左右大腿根关节名，按优先级排列。
_LEFT_HIP_JOINTS = ("LeftUpLeg", "LeftHip", "LeftUpperLeg", "LeftThigh")
_RIGHT_HIP_JOINTS = ("RightUpLeg", "RightHip", "RightUpperLeg", "RightThigh")


@dataclass(frozen=True, slots=True)
class _ParsedBvh:
    hierarchy: str
    frame_time: float
    frame_time_text: str
    frames: list[str]
    channel_count: int


@dataclass(slots=True)
class _Joint:
    """HIERARCHY 里的一个关节；``parent`` 是它在关节列表中的下标。"""

    name: str
    parent: int | None
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    channels: tuple[str, ...] = field(default_factory=tuple)


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
        5: orient_bvh_facing_to_x,
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


def _joint_names(parsed: _ParsedBvh) -> list[str]:
    return [
        match.group(1)
        for line in parsed.hierarchy.splitlines()
        if (match := re.match(r"^\s*(?:ROOT|JOINT)\s+(\S+)", line))
    ]


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


def _neighbor_indexes(
    frame_index: int,
    frame_count: int,
    radius: int,
    motion_start: int = 0,
) -> list[int]:
    """居中窗口下标；窗口中心始终是该帧本身，碰到边界就对称收缩。

    复制边界帧会让首帧在自己窗口里占 3/5 权重，直接被拉进动作；
    直接截断窗口则让它只占 1/3，同样被拉走。对称收缩保证边界帧的窗口只剩它自己。
    ``motion_start`` 之前是开头的静止绑定姿势帧，它们不属于动作，
    既不参与平均、也不被平均（否则静止帧会被摊进后面几帧，形成"追赶式"跳变）。
    """
    if frame_index < motion_start:
        return [frame_index]
    effective = min(radius, frame_index - motion_start, frame_count - 1 - frame_index)
    return list(range(frame_index - effective, frame_index + effective + 1))


def denoise_bvh(
    downloaded: DownloadedBvh,
    window_size: int = 3,
) -> DownloadedBvh:
    """使用时间轴中值滤波移除各运动通道的孤立尖峰。"""
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("去噪窗口必须是大于等于 1 的奇数")

    parsed = _parse_bvh(downloaded)
    values = _motion_values(parsed, downloaded.source_filename)
    rest_channels = _rotation_channel_indexes(parsed)
    _unwrap_rotations(values, rest_channels)
    motion_start = _motion_start_index(values, rest_channels)
    radius = window_size // 2
    filtered: list[list[float]] = []
    for frame_index, frame in enumerate(values):
        neighbors = _neighbor_indexes(frame_index, len(values), radius, motion_start)
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
    rest_channels = _rotation_channel_indexes(parsed)
    _unwrap_rotations(values, rest_channels)
    motion_start = _motion_start_index(values, rest_channels)
    radius = window_size // 2
    smoothed: list[list[float]] = []
    for frame_index, frame in enumerate(values):
        neighbors = _neighbor_indexes(frame_index, len(values), radius, motion_start)
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


def _parse_joints(hierarchy: str) -> list[_Joint]:
    """解析 HIERARCHY，返回与 MOTION 数据同序的关节列表（先根后子，深度优先）。"""
    joints: list[_Joint] = []
    stack: list[int] = []
    for raw_line in hierarchy.splitlines():
        line = raw_line.strip()
        if line.startswith(("ROOT ", "JOINT ")):
            parent = stack[-1] if stack else None
            joints.append(_Joint(name=line.split(None, 1)[1].strip(), parent=parent))
            stack.append(len(joints) - 1)
        elif line == "End Site":
            joints.append(_Joint(name="End Site", parent=stack[-1] if stack else None))
            stack.append(len(joints) - 1)
        elif line == "}":
            if stack:
                stack.pop()
        elif line.startswith("OFFSET"):
            parts = line.split()
            if len(parts) >= 4 and stack:
                joints[stack[-1]].offset = (
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                )
        elif line.startswith("CHANNELS"):
            parts = line.split()
            if len(parts) >= 3 and stack:
                joints[stack[-1]].channels = tuple(parts[2:])
    return joints


def _joint_index(joints: list[_Joint], names: tuple[str, ...]) -> int | None:
    for name in names:
        for index, joint in enumerate(joints):
            if joint.name == name:
                return index
    return None


def _offset_from_root(joints: list[_Joint], index: int) -> np.ndarray:
    """累加根节点到该关节的 OFFSET；要求这条链上只有平移（左右髋通常直接挂在根上）。"""
    total = np.zeros(3)
    current: int | None = index
    while current is not None:
        joint = joints[current]
        total = total + np.array(joint.offset)
        current = joint.parent
    return total


def _axis_rotation(axis: str, degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    cos, sin = math.cos(radians), math.sin(radians)
    if axis == "X":
        return np.array([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]])
    if axis == "Y":
        return np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def _root_rotation_channels(root: _Joint) -> list[tuple[int, str]]:
    """根节点的旋转通道，返回 (在帧数据中的下标, 轴名)，顺序与文件一致。"""
    return [
        (index, name[0].upper())
        for index, name in enumerate(root.channels)
        if name.lower().endswith("rotation")
    ]


def _compose_rotation(
    frame: list[float],
    rotation_channels: list[tuple[int, str]],
) -> np.ndarray:
    """按通道出现顺序合成旋转矩阵，与 three.js BVHLoader 的乘法顺序一致。"""
    matrix = np.identity(3)
    for index, axis in rotation_channels:
        matrix = matrix @ _axis_rotation(axis, frame[index])
    return matrix


def _cannot_orient(message: str) -> BvhServiceError:
    return BvhServiceError(
        status_code=422,
        code="cannot_orient_bvh",
        message=message,
    )


def _motion_start_index(values: list[list[float]], rest_channels: set[int]) -> int:
    """返回动作真正开始的那一帧下标：跳过开头连续静止（旋转通道≈0）的绑定姿势帧。

    有些素材会把 T-pose 当成第 0 帧写进去（该帧所有旋转通道都是 0，与下一帧相差上百度）。
    这类帧不属于动作：既不能用来判断朝向，也不该被平均进后面的动作帧，
    否则静止帧会被摊到随后几帧上，形成"追赶式"的一跳一跳。
    全片都是静止姿势时退回第 0 帧。
    """
    index = 0
    for frame in values:
        if any(
            abs(frame[channel]) > _REST_POSE_TOLERANCE_DEGREES
            for channel in rest_channels
        ):
            break
        index += 1
    return index if index < len(values) else 0


def orient_bvh_facing_to_x(downloaded: DownloadedBvh) -> DownloadedBvh:
    """绕世界 Y 轴整体旋转动作，使人体朝向对准 X 轴正方向。

    朝向由左右大腿根（髋）关节的连线确定：右髋指向左髋的反方向是身体右侧，
    再取 ``up × right`` 得到面向。旋转角**只由第一帧决定**（跳过开头静止的 T-pose
    帧，见 :func:`_motion_start_index`），后续所有帧跟着这一帧一起转。
    整段动作共用同一个旋转角，因此动作本身不变形，只是整体转向；
    根节点的位移也一起绕原点旋转，轨迹随之对齐到新的坐标系。
    """
    parsed = _parse_bvh(downloaded)
    joints = _parse_joints(parsed.hierarchy)
    root_index = next(
        (index for index, joint in enumerate(joints) if joint.parent is None),
        None,
    )
    if root_index is None:
        raise _invalid_bvh(f"{downloaded.source_filename} 缺少根节点")
    root = joints[root_index]

    rotation_channels = _root_rotation_channels(root)
    if not rotation_channels:
        raise _cannot_orient("根节点没有旋转通道，无法调整人体朝向")

    left_index = _joint_index(joints, _LEFT_HIP_JOINTS)
    right_index = _joint_index(joints, _RIGHT_HIP_JOINTS)
    if left_index is None or right_index is None:
        raise _cannot_orient(
            "找不到左右髋关节（如 LeftUpLeg/RightUpLeg），无法判断人体朝向"
        )

    local_right = _offset_from_root(joints, right_index) - _offset_from_root(
        joints, left_index
    )
    if float(np.linalg.norm(local_right[[0, 2]])) < _EPSILON:
        raise _cannot_orient("左右髋关节连线退化，无法判断人体朝向")

    values = _motion_values(parsed, downloaded.source_filename)

    # 朝向只由第一帧（跳过开头静止的 T-pose 帧）决定，整段动作随之刚性旋转。
    reference_index = _motion_start_index(values, _rotation_channel_indexes(parsed))
    right_vector = (
        _compose_rotation(values[reference_index], rotation_channels) @ local_right
    )
    right_x, right_z = float(right_vector[0]), float(right_vector[2])
    if math.hypot(right_x, right_z) < _EPSILON:
        raise _cannot_orient(
            f"第 {reference_index + 1} 帧的左右髋关节连线退化，无法判断人体朝向"
        )

    # 绕 Y 轴旋转 angle 后朝向落到 +X：朝向 = up × right = (right_z, 0, -right_x)，
    # 其 XZ 极角为 atan2(-right_x, right_z)。
    angle = math.degrees(math.atan2(-right_x, right_z))
    rotation = _axis_rotation("Y", angle)
    euler_order = "".join(axis for _, axis in rotation_channels)

    position_indexes = {
        name[0].upper(): index
        for index, name in enumerate(root.channels)
        if name.lower().endswith("position")
    }
    has_position = {"X", "Y", "Z"}.issubset(position_indexes)

    for frame in values:
        target = rotation @ _compose_rotation(frame, rotation_channels)
        for (index, _), value in zip(
            rotation_channels,
            Rotation.from_matrix(target).as_euler(euler_order, degrees=True),
        ):
            frame[index] = float(value)

        if has_position:
            position = np.array(
                [frame[position_indexes[axis]] for axis in ("X", "Y", "Z")]
            )
            for axis, value in zip(("X", "Y", "Z"), rotation @ position):
                frame[position_indexes[axis]] = float(value)

    return _build_processed_bvh(downloaded, parsed, values)


def trim_bvh(
    downloaded: DownloadedBvh,
    source_in_seconds: float,
    source_out_seconds: float | None,
) -> DownloadedBvh:
    """按最近的已有帧裁剪 BVH，不在裁剪边界生成插值姿态。"""
    parsed = _parse_bvh(downloaded)
    source_duration = (len(parsed.frames) - 1) * parsed.frame_time
    end_seconds = source_duration if source_out_seconds is None else source_out_seconds
    tolerance = parsed.frame_time / 2 + 1e-9

    if source_in_seconds < 0:
        raise _invalid_bvh("sourceInSec 不能为负数")
    if end_seconds <= source_in_seconds:
        raise _invalid_bvh("sourceOutSec 必须大于 sourceInSec")
    if source_in_seconds > source_duration + tolerance:
        raise _invalid_bvh(
            f"{downloaded.source_filename} 的 sourceInSec 超出 BVH 时长"
        )
    if end_seconds > source_duration + tolerance:
        raise _invalid_bvh(
            f"{downloaded.source_filename} 的 sourceOutSec 超出 BVH 时长"
        )

    last_index = len(parsed.frames) - 1
    start_index = min(
        math.floor(source_in_seconds / parsed.frame_time + 0.5),
        last_index,
    )
    end_index = min(
        math.floor(end_seconds / parsed.frame_time + 0.5),
        last_index,
    )
    if end_index < start_index:
        raise _invalid_bvh("裁剪区间没有可用的 BVH 帧")

    return _build_bvh(
        parsed,
        parsed.frames[start_index : end_index + 1],
        parsed.frame_time_text,
        downloaded.source_filename,
    )


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
            if target_duration == 0.0:
                output_frames = [parsed.frames[0]]
            else:
                output_frame_count = max(
                    2,
                    math.floor(target_duration / parsed.frame_time + 0.5) + 1,
                )
                if source_duration == 0.0:
                    output_frames = [parsed.frames[0]] * output_frame_count
                else:
                    output_frames = [
                        parsed.frames[
                            min(
                                math.floor(
                                    index * source_duration / target_duration + 0.5
                                ),
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
    settings: Settings,
) -> DownloadedBvh:
    """使用 MDM 生成相邻动作的中间过渡并合并多个 BVH。"""
    if not downloaded_files or len(intervals_seconds) != len(downloaded_files) - 1:
        raise ValueError("BVH 文件与过渡时间数量不匹配")

    parsed_files = [_parse_bvh(downloaded) for downloaded in downloaded_files]
    first = parsed_files[0]
    if len(parsed_files) == 1:
        return _build_bvh(
            first,
            first.frames,
            first.frame_time_text,
            merged_filename(downloaded_files[0].source_filename),
        )
    for parsed in parsed_files[1:]:
        if not math.isclose(
            parsed.frame_time, first.frame_time, rel_tol=1e-7, abs_tol=1e-9
        ):
            raise _invalid_bvh("所有 BVH 文件的 Frame Time 必须一致")

    try:
        expected_names = _joint_names(parsed_files[0])
        for parsed in parsed_files[1:]:
            if _joint_names(parsed) != expected_names:
                raise ValueError("所有 BVH 必须使用相同的关节名称和顺序")
        merged_bytes = generate_mdm_merge(
            downloaded_files,
            intervals_seconds,
            settings,
        )
    except ValueError as error:
        raise _invalid_bvh(str(error)) from error
    output = SpooledTemporaryFile(max_size=_SPOOL_MEMORY_LIMIT, mode="w+b")  # noqa: SIM115
    output.write(merged_bytes)
    output.seek(0)
    result = DownloadedBvh(
        content=output,
        source_filename=merged_filename(downloaded_files[0].source_filename),
        size=len(merged_bytes),
    )
    try:
        parsed_result = _parse_bvh(result)
        if not math.isclose(
            parsed_result.frame_time,
            first.frame_time,
            rel_tol=1e-7,
            abs_tol=1e-9,
        ):
            raise _invalid_bvh("MDM 输出 BVH 的 Frame Time 与输入不一致")
    except Exception:
        result.content.close()
        raise
    return result
