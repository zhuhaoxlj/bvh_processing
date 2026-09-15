"""Read BVH channels and evaluate FK, including Nokov's repeated offsets.

The hierarchy/channel approach reuses the neighbouring project's
tools/bvh_transition.py; FK here reads each joint's actual channel layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
import re

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = ["BVHMotion", "Joint", "read_bvh", "world_positions"]

MAX_FRAMES = 100_000
MAX_JOINTS = 256
CHANNELS = {f"{axis}{kind}" for axis in "XYZ" for kind in ("position", "rotation")}


@dataclass(frozen=True)
class Joint:
    name: str
    parent: int
    offset: np.ndarray
    channels: tuple[str, ...]
    start: int


@dataclass(frozen=True)
class BVHMotion:
    joints: tuple[Joint, ...]
    frames: np.ndarray
    frame_time: float

    @property
    def fps(self) -> float:
        return 1.0 / self.frame_time

    @property
    def rotation_columns(self) -> list[int]:
        return [j.start + i for j in self.joints for i, c in enumerate(j.channels) if c.endswith("rotation")]


def read_bvh(content: bytes | str) -> BVHMotion:
    """Parse a complete BVH without assuming three channels per child joint."""
    if isinstance(content, bytes):
        try:
            content = content.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise ValueError("BVH 必须是 UTF-8 / ASCII 文本文件") from error
    parts = re.split(r"(?m)^\s*MOTION\s*$", content, maxsplit=1)
    if len(parts) != 2:
        raise ValueError("不是有效的 BVH：缺少 MOTION 段")
    tokens = re.findall(r"[{}]|[^\s{}]+", parts[0])
    cursor = 0
    width = 0
    joints: list[Joint] = []

    def take(expected: str | None = None) -> str:
        nonlocal cursor
        if cursor >= len(tokens):
            raise ValueError("BVH 骨架段不完整")
        value = tokens[cursor]
        cursor += 1
        if expected is not None and value != expected:
            raise ValueError(f"BVH 骨架语法错误：需要 {expected}，实际为 {value}")
        return value

    def parse_joint(parent: int, depth: int = 0) -> None:
        nonlocal width
        if depth > 64 or len(joints) >= MAX_JOINTS:
            raise ValueError("BVH 骨架过大或层级过深")
        name = take()
        if any(j.name == name for j in joints):
            raise ValueError(f"BVH 关节名重复：{name}")
        take("{")
        take("OFFSET")
        try:
            offset = np.asarray([float(take()) for _ in range(3)])
            take("CHANNELS")
            count = int(take())
        except (ValueError, OverflowError) as error:
            raise ValueError(f"关节 {name} 的 OFFSET / CHANNELS 无效") from error
        if not np.isfinite(offset).all() or not 0 < count <= 6:
            raise ValueError(f"关节 {name} 的 OFFSET / CHANNELS 无效")
        channels = tuple(take() for _ in range(count))
        if len(set(channels)) != count or set(channels) - CHANNELS:
            raise ValueError(f"关节 {name} 含有未知或重复通道")
        rotation_axes = [c[0] for c in channels if c.endswith("rotation")]
        if len(rotation_axes) != 3 or set(rotation_axes) != set("XYZ"):
            raise ValueError(f"关节 {name} 必须包含三个旋转通道")
        # The supported exporters put translation before the Euler rotation block.
        if any(c.endswith("position") for c in channels[next(i for i, c in enumerate(channels) if c.endswith("rotation")):]):
            raise ValueError(f"关节 {name} 的位置/旋转通道交错排列，暂不支持此导出格式")
        index = len(joints)
        joints.append(Joint(name, parent, offset, channels, width))
        width += count
        while True:
            token = take()
            if token == "}":
                return
            if token == "JOINT":
                parse_joint(index, depth + 1)
            elif token == "End":
                take("Site")
                take("{")
                take("OFFSET")
                try:
                    end_offset = [float(take()) for _ in range(3)]
                except ValueError as error:
                    raise ValueError("End Site 偏移无效") from error
                if not np.isfinite(end_offset).all():
                    raise ValueError("End Site 偏移必须是有限数值")
                take("}")
            else:
                raise ValueError(f"BVH 骨架中出现未知字段：{token}")

    take("HIERARCHY")
    take("ROOT")
    parse_joint(-1)
    if cursor != len(tokens):
        raise ValueError("BVH 只能包含一个根骨架")
    match = re.fullmatch(
        r"\s*Frames:\s*(\d+)\s+Frame\s+Time:\s*([\d.eE+-]+)\s*\n([\s\S]*)",
        parts[1],
    )
    if not match:
        raise ValueError("BVH 缺少有效的 Frames / Frame Time")
    frame_count, frame_time = int(match[1]), float(match[2])
    if not 3 <= frame_count <= MAX_FRAMES or frame_count * len(joints) > 3_000_000:
        raise ValueError("动作需至少 3 帧，且最多包含 300 万个关节采样点")
    if not np.isfinite(frame_time) or not 0.001 <= frame_time <= 1:
        raise ValueError("BVH 帧率必须在 1–1000 FPS 之间")
    try:
        frames = np.loadtxt(StringIO(match[3]), dtype=np.float64, ndmin=2)
    except ValueError as error:
        raise ValueError("BVH 动作行的列数或数值无效") from error
    if frames.shape != (frame_count, width):
        raise ValueError(f"BVH 声明 {frame_count} 帧、{width} 通道，实际为 {frames.shape}")
    if not np.isfinite(frames).all():
        raise ValueError("BVH 包含 NaN 或无穷大数值")
    return BVHMotion(tuple(joints), frames, frame_time)


def world_positions(motion: BVHMotion) -> tuple[np.ndarray, list[str]]:
    """Evaluate declared intrinsic Euler order; return coordinates and offset detections."""
    count = len(motion.frames)
    positions = np.empty((count, len(motion.joints), 3), dtype=np.float64)
    rotations: list[np.ndarray] = []
    repeated_offsets: list[str] = []
    for index, joint in enumerate(motion.joints):
        values = motion.frames[:, joint.start:joint.start + len(joint.channels)]
        local_positions = np.broadcast_to(joint.offset, (count, 3)).copy()
        translation = np.zeros((count, 3))
        translation_axes: list[int] = []
        rotation_columns: list[int] = []
        order = ""
        for column, channel in enumerate(joint.channels):
            if channel.endswith("position"):
                axis = "XYZ".index(channel[0])
                translation_axes.append(axis)
                translation[:, axis] = values[:, column]
            else:
                rotation_columns.append(column)
                order += channel[0]
        tolerance = max(1e-4, float(np.abs(joint.offset).max()) * 1e-5)
        if joint.parent >= 0 and len(translation_axes) == 3 and np.allclose(translation, joint.offset, atol=tolerance, rtol=0):
            # Nokov six-channel exports repeat the absolute local offset in MOTION.
            # Keep the channel values (including quantisation); don't add OFFSET twice.
            local_positions = translation
            repeated_offsets.append(joint.name)
        else:
            local_positions += translation
        local_rotations = Rotation.from_euler(order, values[:, rotation_columns], degrees=True).as_matrix()
        if joint.parent < 0:
            positions[:, index] = local_positions
            rotations.append(local_rotations)
        else:
            parent_rotation = rotations[joint.parent]
            positions[:, index] = positions[:, joint.parent] + np.einsum("fij,fj->fi", parent_rotation, local_positions)
            rotations.append(parent_rotation @ local_rotations)
    return positions, repeated_offsets
