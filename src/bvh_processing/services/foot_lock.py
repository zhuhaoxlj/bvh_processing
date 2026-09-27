"""把 BVH 支撑脚落到 Y=0。

接触检测沿用 SOMA Retargeter 的 Baseline A：在米制世界坐标上用速度、加速度
幅度和 jerk 做状态机。速度使用 ``位移 / (2 dt)``，这是该检测器阈值的既定约定，
不是普通中心差分。检测之后的贴地修正与 soma 的 BVH 接地流程一致：删掉短于
3 帧的接触，丢掉每只脚接触高度里最高的 25%，在接触帧把最低的支撑点对齐到
Y=0，对其余帧插值并做 sigma=2 的平滑，最后禁止任何脚穿过地面。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

_VELOCITY_CONTACT = 0.1
_JERK_CONTACT = -0.05
_VELOCITY_UNCONTACT = 0.2
_VELOCITY_PROBABLY_LIFT_OFF = 0.05
_POST_CONTACT_JERK_WINDOW = 5
_TRANSITION_FRAMES = 4
_EDGE_PROPAGATION_FRAMES = 8
_MINIMUM_CONTACT_FRAMES = 3
_CONTACT_HEIGHT_PERCENTILE = 75.0
_SMOOTHING_SIGMA = 2.0
_CENTIMETER_OFFSET = 5.0
_CORRECTION_EPSILON = 1e-8

_LEFT_FOOT = ("LeftFoot", "LeftAnkle")
_RIGHT_FOOT = ("RightFoot", "RightAnkle")
_LEFT_TOE = ("LeftToeBase", "LeftToe")
_RIGHT_TOE = ("RightToeBase", "RightToe")
_NAME_KEY = re.compile(r"[\s_]+")


class FootLockError(ValueError):
    """当前 BVH 无法做脚步锁定。"""


@dataclass(slots=True)
class _Node:
    index: int
    name: str
    parent: int | None
    offset: np.ndarray
    channels: tuple[str, ...]
    is_end: bool


def lock_feet_to_ground(
    hierarchy: str,
    motion: np.ndarray,
    frame_time: float,
) -> np.ndarray | None:
    """返回修正后的动作；没有可用接触或修正量为 0 时返回 None。"""
    nodes = _parse_hierarchy(hierarchy)
    if motion.ndim != 2 or motion.shape[0] == 0:
        raise FootLockError("MOTION 帧数据为空")
    if frame_time <= 0:
        raise FootLockError("Frame Time 必须大于 0")

    world = _world_positions(nodes, motion)
    scale = _meter_scale(nodes)
    sides = _foot_sides(nodes)
    contact_nodes: list[_Node] = []
    side_columns: dict[str, list[int]] = {}
    for side, (foot, toe) in sides.items():
        columns: list[int] = []
        for joint in (foot, toe):
            if joint is None:
                continue
            columns.append(len(contact_nodes))
            contact_nodes.append(joint)
        side_columns[side] = columns

    positions = np.stack([world[node.index] for node in contact_nodes], axis=1)
    contacts, speed = _detect_contacts(positions * scale, frame_time)
    side_heights = {
        side: _side_heights(nodes, world, foot, toe)
        for side, (foot, toe) in sides.items()
    }
    side_contacts = _select_contacts(contacts, speed, side_columns, side_heights)
    correction = _height_correction(side_contacts, side_heights)
    if correction is None or float(np.max(np.abs(correction))) <= _CORRECTION_EPSILON:
        return None

    root = nodes[0]
    y_column = next(
        (
            index
            for index, channel in enumerate(root.channels)
            if channel.lower() == "yposition"
        ),
        None,
    )
    if y_column is None:
        raise FootLockError("根节点缺少 Yposition，无法修正脚底高度")

    corrected = np.array(motion, dtype=np.float64, copy=True)
    corrected[:, y_column] += correction
    return corrected


def _parse_hierarchy(hierarchy: str) -> list[_Node]:
    nodes: list[_Node] = []
    stack: list[int] = []
    for raw_line in hierarchy.splitlines():
        text = raw_line.strip()
        if not text or text in {"HIERARCHY", "{"} or text.startswith("{"):
            continue
        if text.startswith(("ROOT ", "JOINT ")):
            parent = stack[-1] if stack else None
            node = _Node(
                index=len(nodes),
                name=text.split()[1],
                parent=parent,
                offset=np.zeros(3, dtype=np.float64),
                channels=(),
                is_end=False,
            )
            nodes.append(node)
            stack.append(node.index)
        elif text.startswith("End Site"):
            if not stack:
                raise FootLockError("End Site 出现在根节点之前")
            parent = stack[-1]
            node = _Node(
                index=len(nodes),
                name=f"{nodes[parent].name}__EndSite",
                parent=parent,
                offset=np.zeros(3, dtype=np.float64),
                channels=(),
                is_end=True,
            )
            nodes.append(node)
            stack.append(node.index)
        elif text.startswith("OFFSET"):
            if not stack:
                raise FootLockError("OFFSET 出现在关节之前")
            parts = text.split()
            try:
                offset = np.array(
                    [float(parts[1]), float(parts[2]), float(parts[3])],
                    dtype=np.float64,
                )
            except (IndexError, ValueError) as error:
                raise FootLockError("OFFSET 格式不正确") from error
            nodes[stack[-1]].offset = offset
        elif text.startswith("CHANNELS"):
            if not stack:
                raise FootLockError("CHANNELS 出现在关节之前")
            parts = text.split()
            try:
                declared = int(parts[1])
            except (IndexError, ValueError) as error:
                raise FootLockError("CHANNELS 格式不正确") from error
            channels = tuple(parts[2:])
            if len(channels) != declared:
                raise FootLockError("CHANNELS 声明的数量与通道名不一致")
            nodes[stack[-1]].channels = channels
        elif text.startswith("}"):
            if stack:
                stack.pop()

    if not nodes or nodes[0].parent is not None:
        raise FootLockError("BVH 缺少根节点")
    return nodes


def _name_key(name: str) -> str:
    return _NAME_KEY.sub("", name.split(":")[-1]).lower()


def _find_joint(nodes: list[_Node], aliases: tuple[str, ...]) -> _Node | None:
    by_name = {}
    for node in nodes:
        if not node.is_end:
            by_name.setdefault(_name_key(node.name), node)
    for alias in aliases:
        found = by_name.get(_name_key(alias))
        if found is not None:
            return found
    return None


def _foot_sides(nodes: list[_Node]) -> dict[str, tuple[_Node | None, _Node | None]]:
    sides = {
        "left": (_find_joint(nodes, _LEFT_FOOT), _find_joint(nodes, _LEFT_TOE)),
        "right": (_find_joint(nodes, _RIGHT_FOOT), _find_joint(nodes, _RIGHT_TOE)),
    }
    missing = [
        side
        for side, joints in sides.items()
        if joints[0] is None and joints[1] is None
    ]
    if missing:
        raise FootLockError(
            "缺少左右脚关节，无法进行脚步锁定。需要 LeftFoot/LeftAnkle"
            " 或 LeftToe/LeftToeBase，右侧同理"
        )
    return sides


def _meter_scale(nodes: list[_Node]) -> float:
    lengths = [float(np.linalg.norm(node.offset)) for node in nodes if not node.is_end]
    longest = max(lengths, default=0.0)
    return 0.01 if longest > _CENTIMETER_OFFSET else 1.0


def _axis_matrices(axis: str, degrees: np.ndarray) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    matrix = np.zeros((degrees.shape[0], 3, 3), dtype=np.float64)
    if axis == "X":
        matrix[:, 0, 0] = 1.0
        matrix[:, 1, 1] = cosine
        matrix[:, 1, 2] = -sine
        matrix[:, 2, 1] = sine
        matrix[:, 2, 2] = cosine
    elif axis == "Y":
        matrix[:, 0, 0] = cosine
        matrix[:, 0, 2] = sine
        matrix[:, 1, 1] = 1.0
        matrix[:, 2, 0] = -sine
        matrix[:, 2, 2] = cosine
    elif axis == "Z":
        matrix[:, 0, 0] = cosine
        matrix[:, 0, 1] = -sine
        matrix[:, 1, 0] = sine
        matrix[:, 1, 1] = cosine
        matrix[:, 2, 2] = 1.0
    else:
        raise FootLockError(f"不支持的旋转通道：{axis}rotation")
    return matrix


def _local_rotation(
    channels: tuple[str, ...],
    columns: np.ndarray,
    frame_count: int,
) -> np.ndarray:
    local = np.broadcast_to(np.eye(3), (frame_count, 3, 3)).copy()
    for index, channel in enumerate(channels):
        if not channel.lower().endswith("rotation"):
            continue
        local = np.einsum(
            "nij,njk->nik",
            local,
            _axis_matrices(channel[0].upper(), columns[:, index]),
        )
    return local


def _local_translation(node: _Node, columns: np.ndarray) -> np.ndarray:
    frame_count = columns.shape[0]
    translation = np.zeros((frame_count, 3), dtype=np.float64)
    axes: list[int] = []
    for index, channel in enumerate(node.channels):
        lowered = channel.lower()
        if not lowered.endswith("position"):
            continue
        axis = "xyz".find(lowered[0])
        if axis < 0:
            raise FootLockError(f"不支持的位置通道：{channel}")
        translation[:, axis] = columns[:, index]
        axes.append(axis)

    if node.parent is None:
        if axes:
            return translation
        return np.broadcast_to(node.offset, (frame_count, 3)).copy()

    offset_scale = max(1e-4, float(np.abs(node.offset).max()) * 1e-5)
    repeats_offset = len(axes) == 3 and np.allclose(
        translation,
        node.offset,
        atol=offset_scale,
        rtol=0.0,
    )
    if repeats_offset:
        return translation
    return translation + node.offset


def _world_positions(nodes: list[_Node], motion: np.ndarray) -> list[np.ndarray]:
    frame_count = motion.shape[0]
    positions = [np.zeros((frame_count, 3), dtype=np.float64) for _ in nodes]
    rotations = [np.broadcast_to(np.eye(3), (frame_count, 3, 3)).copy() for _ in nodes]
    cursor = 0
    for node in nodes:
        width = len(node.channels)
        if cursor + width > motion.shape[1]:
            raise FootLockError("MOTION 通道数量少于 HIERARCHY 声明")
        columns = motion[:, cursor : cursor + width]
        if not node.is_end:
            cursor += width
        local_rotation = _local_rotation(node.channels, columns, frame_count)
        local_translation = _local_translation(node, columns)
        if node.parent is None:
            positions[node.index] = local_translation
            rotations[node.index] = local_rotation
            continue
        parent_rotation = rotations[node.parent]
        positions[node.index] = positions[node.parent] + np.einsum(
            "nij,nj->ni",
            parent_rotation,
            local_translation,
        )
        rotations[node.index] = np.einsum(
            "nij,njk->nik",
            parent_rotation,
            local_rotation,
        )
    if cursor != motion.shape[1]:
        raise FootLockError("MOTION 通道数量与 HIERARCHY 不一致")
    return positions


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.flatnonzero(np.diff(padded))
    return [
        (int(start), int(end))
        for start, end in zip(changes[::2], changes[1::2], strict=True)
    ]


def _remove_short_runs(mask: np.ndarray, minimum_frames: int) -> np.ndarray:
    result = mask.copy()
    for start, end in _true_runs(mask):
        if end - start < minimum_frames:
            result[start:end] = False
    return result


def _ramp_contacts(contacts: np.ndarray, transition_frames: int) -> np.ndarray:
    if transition_frames <= 0:
        return contacts.astype(np.float64, copy=True)
    result = contacts.astype(np.float64, copy=True)
    ramp = 1.0 - np.arange(1, transition_frames + 1, dtype=np.float64) / (
        transition_frames + 1
    )
    frame_count, channel_count = contacts.shape
    for channel in range(channel_count):
        for start, end in _true_runs(contacts[:, channel] > 0.5):
            before = min(transition_frames, start)
            if before:
                result[start - before : start, channel] = np.maximum(
                    result[start - before : start, channel],
                    ramp[:before][::-1],
                )
            after = min(transition_frames, frame_count - end)
            if after:
                result[end : end + after, channel] = np.maximum(
                    result[end : end + after, channel],
                    ramp[:after],
                )
    return result


def _propagate_edges(
    contacts: np.ndarray,
    speed: np.ndarray,
    velocity_threshold: float,
    max_edge_frames: int,
) -> None:
    if max_edge_frames <= 0:
        return
    frame_count, channel_count = contacts.shape
    for channel in range(channel_count):
        active = np.flatnonzero(contacts[:, channel] > 0.5)
        if active.size == 0:
            continue
        first = int(active[0])
        if 0 < first <= max_edge_frames:
            blocked = np.flatnonzero(speed[:first, channel] >= velocity_threshold)
            start = int(blocked[-1] + 1) if blocked.size else 0
            contacts[start:first, channel] = 1.0
        last = int(active[-1])
        if last >= frame_count - 1 - max_edge_frames:
            trailing = speed[last + 1 :, channel] < velocity_threshold
            if trailing.size:
                stop = np.flatnonzero(~trailing)
                end = last + 1 + (int(stop[0]) if stop.size else int(trailing.size))
                contacts[last + 1 : end, channel] = 1.0


def _detect_contacts(
    positions_m: np.ndarray,
    frame_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 (T, C) 接触权重和同形状的半速速度。"""
    frame_count, channel_count, _ = positions_m.shape
    dt = frame_time
    velocities = np.zeros_like(positions_m)
    if frame_count > 1:
        velocities[1:] = np.diff(positions_m, axis=0) / (2.0 * dt)

    accelerations = np.zeros((frame_count, channel_count), dtype=np.float64)
    if frame_count > 2:
        velocity_delta = np.diff(velocities, axis=0)
        accelerations[2:] = np.linalg.norm(velocity_delta[1:], axis=-1) / dt

    jerk = np.zeros((frame_count, channel_count), dtype=np.float64)
    if frame_count > 3:
        jerk[3:] = np.diff(accelerations[2:], axis=0) / dt

    speed = np.linalg.norm(velocities, axis=-1)
    contacts = np.zeros((frame_count, channel_count), dtype=np.float64)
    in_contact = [False] * channel_count
    contact_frame = [-1] * channel_count
    lift_off_frame = [-1] * channel_count
    post_jerk_left = [0] * channel_count

    for frame in range(frame_count):
        for channel in range(channel_count):
            frame_jerk = float(jerk[frame, channel])
            frame_speed = float(speed[frame, channel])
            next_speed = float(speed[min(frame + 1, frame_count - 1), channel])

            if not in_contact[channel] and (
                frame_jerk < 0.0 or post_jerk_left[channel] > 0
            ):
                if frame_jerk < _JERK_CONTACT:
                    if post_jerk_left[channel] == 0:
                        post_jerk_left[channel] = _POST_CONTACT_JERK_WINDOW
                    else:
                        post_jerk_left[channel] -= 1
                if frame_speed < _VELOCITY_CONTACT:
                    in_contact[channel] = True
                    contact_frame[channel] = frame

            if in_contact[channel]:
                if frame_speed >= _VELOCITY_PROBABLY_LIFT_OFF:
                    if lift_off_frame[channel] == -1:
                        lift_off_frame[channel] = frame
                else:
                    lift_off_frame[channel] = -1

                if (
                    frame_speed > _VELOCITY_UNCONTACT
                    and next_speed > _VELOCITY_UNCONTACT
                ):
                    if lift_off_frame[channel] - contact_frame[channel] > 0:
                        contacts[
                            lift_off_frame[channel] : frame,
                            channel,
                        ] = 0.0
                    else:
                        contact_frame[channel] = -1
                    in_contact[channel] = False
                    lift_off_frame[channel] = -1
                    post_jerk_left[channel] = 0

            contacts[frame, channel] = 1.0 if in_contact[channel] else 0.0

    _propagate_edges(
        contacts,
        speed,
        _VELOCITY_CONTACT,
        _EDGE_PROPAGATION_FRAMES,
    )
    return _ramp_contacts(contacts, _TRANSITION_FRAMES), speed


def _side_heights(
    nodes: list[_Node],
    world: list[np.ndarray],
    foot: _Node | None,
    toe: _Node | None,
) -> np.ndarray:
    selected: list[_Node] = []
    for joint in (foot, toe):
        if joint is None:
            continue
        selected.append(joint)
        selected.extend(
            node for node in nodes if node.is_end and node.parent == joint.index
        )
    heights = np.stack([world[node.index][:, 1] for node in selected], axis=1)
    return np.min(heights, axis=1)


def _select_contacts(
    contacts: np.ndarray,
    speed: np.ndarray,
    side_columns: dict[str, list[int]],
    side_heights: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    selected = {}
    for side, columns in side_columns.items():
        mask = _remove_short_runs(
            np.max(contacts[:, columns], axis=1) > 0.5,
            _MINIMUM_CONTACT_FRAMES,
        )
        selected[side] = mask

    if not any(mask.any() for mask in selected.values()):
        all_slow = all(
            float(np.max(speed[:, columns])) < _VELOCITY_CONTACT
            for columns in side_columns.values()
        )
        if all_slow:
            for mask in selected.values():
                mask[:] = True

    for side, mask in selected.items():
        if not mask.any():
            continue
        threshold = float(
            np.percentile(side_heights[side][mask], _CONTACT_HEIGHT_PERCENTILE)
        )
        mask &= side_heights[side] <= threshold
    return selected


def _height_correction(
    side_contacts: dict[str, np.ndarray],
    side_heights: dict[str, np.ndarray],
) -> np.ndarray | None:
    frame_count = next(iter(side_heights.values())).shape[0]
    samples = np.full(frame_count, np.nan, dtype=np.float64)
    for frame in range(frame_count):
        active = [
            side_heights[side][frame]
            for side, mask in side_contacts.items()
            if mask[frame]
        ]
        if active:
            samples[frame] = -min(active)

    valid = np.flatnonzero(np.isfinite(samples))
    if valid.size == 0:
        return None

    timeline = np.arange(frame_count)
    correction = np.interp(timeline, valid, samples[valid])
    correction = gaussian_filter1d(correction, sigma=_SMOOTHING_SIGMA, mode="nearest")
    correction[valid] = samples[valid]
    all_heights = np.stack(list(side_heights.values()), axis=1)
    return np.maximum(correction, -all_heights.min(axis=1))
