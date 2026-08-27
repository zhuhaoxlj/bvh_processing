"""连接同拓扑 BVH 动作，并为相邻动作生成脚部锁定的平滑过渡。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

ROOT_CHANNELS = [
    "Xposition",
    "Yposition",
    "Zposition",
    "Yrotation",
    "Xrotation",
    "Zrotation",
]
JOINT_CHANNELS = ["Yrotation", "Xrotation", "Zrotation"]


@dataclass
class Joint:
    name: str
    offset: np.ndarray
    channels: list[str]
    children: list[Joint] = field(default_factory=list)

    @property
    def is_end_site(self) -> bool:
        return self.name.startswith("EndSite_")


@dataclass
class Motion:
    root: Joint
    frames: np.ndarray
    frame_time: float

    @property
    def joints(self) -> list[Joint]:
        result: list[Joint] = []

        def visit(joint: Joint) -> None:
            if not joint.is_end_site:
                result.append(joint)
            for child in joint.children:
                visit(child)

        visit(self.root)
        return result

    @property
    def channels(self) -> list[tuple[Joint, list[str]]]:
        return [(joint, joint.channels) for joint in self.joints]


def _parse_joint(
    lines: list[str], index: int, end_site_id: list[int]
) -> tuple[Joint, int]:
    line = lines[index].strip()
    match = re.match(r"(?:ROOT|JOINT)\s+(\S+)", line)
    if match:
        name = match.group(1)
    else:
        name = f"EndSite_{end_site_id[0]}"
        end_site_id[0] += 1
    index += 1
    if index >= len(lines) or lines[index].strip() != "{":
        raise ValueError(f"BVH 骨架格式错误：{line}")
    index += 1
    offset: np.ndarray | None = None
    channels: list[str] = []
    children: list[Joint] = []
    while index < len(lines):
        current = lines[index].strip()
        if current == "}":
            if offset is None:
                raise ValueError(f"BVH 关节 {name} 缺少 OFFSET")
            return Joint(name, offset, channels, children), index + 1
        if current.startswith("OFFSET"):
            offset = np.asarray(current.split()[1:4], dtype=float)
            index += 1
        elif current.startswith("CHANNELS"):
            channels = current.split()[2:]
            index += 1
        elif current.startswith(("JOINT", "End Site")):
            child, index = _parse_joint(lines, index, end_site_id)
            children.append(child)
        else:
            index += 1
    raise ValueError("BVH 骨架层级未闭合")


def motion_from_parts(hierarchy: str, frames: list[str], frame_time: float) -> Motion:
    lines = hierarchy.splitlines()
    if not lines or lines[0].strip().upper() != "HIERARCHY":
        raise ValueError("BVH 缺少 HIERARCHY")
    root, _ = _parse_joint(lines, 1, [0])
    try:
        values = np.asarray(
            [[float(value) for value in frame.split()] for frame in frames],
            dtype=float,
        )
    except ValueError as error:
        raise ValueError("BVH MOTION 包含无效数值") from error
    expected = sum(
        len(channels) for _, channels in Motion(root, values, frame_time).channels
    )
    if values.ndim != 2 or values.shape[1] != expected:
        raise ValueError(
            f"BVH 运动通道数量错误：读取 {values.shape}，期望每帧 {expected} 个通道"
        )
    return Motion(root, values, frame_time)


def _canonical_hierarchy(joint: Joint, is_root: bool = True) -> Joint:
    return Joint(
        name=joint.name,
        offset=joint.offset.copy(),
        channels=(ROOT_CHANNELS if is_root else JOINT_CHANNELS).copy()
        if not joint.is_end_site
        else [],
        children=[_canonical_hierarchy(child, False) for child in joint.children],
    )


def hierarchy_text(joint: Joint, is_root: bool = True, depth: int = 0) -> list[str]:
    indent = "\t" * depth
    header = "ROOT" if is_root else ("End Site" if joint.is_end_site else "JOINT")
    name = "" if joint.is_end_site else f" {joint.name}"
    lines = [f"{indent}{header}{name}", f"{indent}{{"]
    offset = " ".join(f"{value:.6f}" for value in joint.offset)
    lines.append(f"{indent}\tOFFSET {offset}")
    if not joint.is_end_site:
        lines.append(
            f"{indent}\tCHANNELS {len(joint.channels)} {' '.join(joint.channels)}"
        )
    for child in joint.children:
        lines.extend(hierarchy_text(child, False, depth + 1))
    lines.append(f"{indent}}}")
    return lines


def motion_hierarchy(motion: Motion) -> str:
    return "\n".join(["HIERARCHY", *hierarchy_text(motion.root)])


def _validate_topology(first: Motion, second: Motion) -> None:
    if [joint.name for joint in first.joints] != [
        joint.name for joint in second.joints
    ]:
        raise ValueError("所有 BVH 必须使用相同的关节名称和顺序")


def _extract_rotations(motion: Motion) -> tuple[np.ndarray, np.ndarray]:
    rotations: list[np.ndarray] = []
    root_positions: np.ndarray | None = None
    cursor = 0
    for index, (_, channels) in enumerate(motion.channels):
        values = motion.frames[:, cursor : cursor + len(channels)]
        cursor += len(channels)
        try:
            rotation_indices = [channels.index(f"{axis}rotation") for axis in "YXZ"]
            if index == 0:
                position_indices = [channels.index(f"{axis}position") for axis in "XYZ"]
                root_positions = values[:, position_indices]
        except ValueError as error:
            raise ValueError(
                "根关节须有 XYZ 位移，所有关节须有 YXZ 旋转通道"
            ) from error
        rotations.append(values[:, rotation_indices])
    if root_positions is None:
        raise ValueError("BVH 缺少根关节位移通道")
    return np.stack(rotations, axis=1), root_positions


def _standard_motion(
    root: Joint, rotations: np.ndarray, positions: np.ndarray, frame_time: float
) -> Motion:
    frames = np.concatenate(
        [
            positions,
            rotations[:, 0, :],
            rotations[:, 1:, :].reshape(len(positions), -1),
        ],
        axis=1,
    )
    return Motion(_canonical_hierarchy(root), frames, frame_time)


def _initial_artifact_frame_count(
    rotations: np.ndarray,
    positions: np.ndarray,
) -> int:
    """识别导出器在动作开头写入的零姿势缩放坡道。"""
    if len(positions) < 12:
        return 0

    position_steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    root_orientations = Rotation.from_euler("YXZ", rotations[:, 0], degrees=True)
    rotation_steps = np.degrees(
        (root_orientations[:-1].inv() * root_orientations[1:]).magnitude()
    )
    baseline_start = min(10, len(position_steps) // 3)
    position_baseline = max(float(np.median(position_steps[baseline_start:])), 1e-6)
    rotation_baseline = max(float(np.median(rotation_steps[baseline_start:])), 1e-6)
    anomalous = (position_steps > position_baseline * 20.0) & (
        rotation_steps > rotation_baseline * 20.0
    )

    # 只处理从首帧开始、最多 8 帧的连续异常；动作中途的快速运动不能被裁掉。
    count = 0
    for is_anomalous in anomalous[:8]:
        if not is_anomalous:
            break
        count += 1
    return count


def _canonicalize(
    motion: Motion,
    *,
    output_root: Joint | None = None,
    drop_initial_rest_frame: bool = False,
) -> Motion:
    rotations, positions = _extract_rotations(motion)
    artifact_count = _initial_artifact_frame_count(rotations, positions)
    if artifact_count:
        rotations = rotations[artifact_count:]
        positions = positions[artifact_count:]
    if len(rotations) == 1:
        # 零时长或单帧动作仍可作为静止边界参与过渡。
        rotations = np.repeat(rotations, 2, axis=0)
        positions = np.repeat(positions, 2, axis=0)
    if (
        drop_initial_rest_frame
        and len(rotations) > 2
        and np.max(np.abs(rotations[0])) < 1e-8
    ):
        rotations = rotations[1:]
        positions = positions[1:]
    if len(rotations) < 2:
        raise ValueError("参与过渡的每个 BVH 至少需要两帧")
    canonical = _standard_motion(
        output_root or motion.root, rotations, positions, motion.frame_time
    )
    _validate_topology(canonical, motion)
    return canonical


def _world_positions(motion: Motion, frame: np.ndarray) -> np.ndarray:
    joints = motion.joints
    positions = np.zeros((len(joints), 3))
    rotations = [Rotation.identity() for _ in joints]
    cursor = 0
    joint_index = {id(joint): index for index, joint in enumerate(joints)}

    def visit(joint: Joint, parent: int | None) -> None:
        nonlocal cursor
        if joint.is_end_site:
            return
        index = joint_index[id(joint)]
        width = 6 if parent is None else 3
        values = frame[cursor : cursor + width]
        cursor += width
        local_rotation = Rotation.from_euler("YXZ", values[-3:], degrees=True)
        if parent is None:
            positions[index] = values[:3]
            rotations[index] = local_rotation
        else:
            positions[index] = positions[parent] + rotations[parent].apply(joint.offset)
            rotations[index] = rotations[parent] * local_rotation
        for child in joint.children:
            visit(child, index)

    visit(motion.root, None)
    return positions


def _support_foot_name(first: Motion, second: Motion) -> str:
    names = [joint.name for joint in first.joints]
    candidates = [
        name
        for name in ("LeftToeBase", "RightToeBase", "LeftToe", "RightToe")
        if name in names
    ]
    if not candidates:
        raise ValueError("BVH 缺少用于过渡脚部锁定的左右脚趾关节")
    first_positions = np.asarray(
        [_world_positions(first, frame) for frame in first.frames[-3:]]
    )
    second_positions = np.asarray(
        [_world_positions(second, frame) for frame in second.frames[:3]]
    )
    scores: dict[str, float] = {}
    for name in candidates:
        index = names.index(name)
        end_speed = np.linalg.norm(
            first_positions[-1, index] - first_positions[-2, index]
        )
        start_speed = np.linalg.norm(
            second_positions[1, index] - second_positions[0, index]
        )
        displacement = np.linalg.norm(
            second_positions[0, index] - first_positions[-1, index]
        )
        scores[name] = float(end_speed + 10.0 * start_speed + 0.01 * displacement)
    return min(scores, key=scores.__getitem__)


def _yaw(rotation: Rotation) -> float:
    """提取 Y-up 坐标系中的水平朝向，忽略躯干俯仰和侧倾。"""
    forward = rotation.apply(np.asarray([0.0, 0.0, 1.0]))
    if np.hypot(forward[0], forward[2]) < 1e-8:
        # 躯干接近竖直时使用右方向，避免水平投影退化。
        right = rotation.apply(np.asarray([1.0, 0.0, 0.0]))
        return float(np.arctan2(-right[2], right[0]))
    return float(np.arctan2(forward[0], forward[2]))


def _align_to_boundary(previous: Motion, following: Motion) -> tuple[Motion, str]:
    _validate_topology(previous, following)
    previous_rotations, previous_positions = _extract_rotations(previous)
    rotations, positions = _extract_rotations(following)
    previous_root = Rotation.from_euler("YXZ", previous_rotations[-1, 0], degrees=True)
    following_root = Rotation.from_euler("YXZ", rotations[0, 0], degrees=True)
    # 只统一水平朝向。后续动作原本的躯干俯仰和侧倾必须保留，交给桥接段过渡。
    yaw_alignment = Rotation.from_euler(
        "Y",
        _yaw(previous_root) - _yaw(following_root),
    )
    aligned_positions = previous_positions[-1] + yaw_alignment.apply(
        positions - positions[0]
    )
    aligned_rotations = rotations.copy()
    root_rotations = Rotation.from_euler("YXZ", aligned_rotations[:, 0], degrees=True)
    aligned_rotations[:, 0] = (yaw_alignment * root_rotations).as_euler(
        "YXZ", degrees=True
    )
    provisional = _standard_motion(
        previous.root, aligned_rotations, aligned_positions, previous.frame_time
    )
    support_name = _support_foot_name(previous, provisional)
    support_index = [joint.name for joint in previous.joints].index(support_name)
    foot_delta = (
        _world_positions(previous, previous.frames[-1])[support_index]
        - _world_positions(provisional, provisional.frames[0])[support_index]
    )
    aligned_positions += foot_delta
    return (
        _standard_motion(
            previous.root, aligned_rotations, aligned_positions, previous.frame_time
        ),
        support_name,
    )


def _cubic_hermite(
    start: np.ndarray,
    end: np.ndarray,
    start_velocity: np.ndarray,
    end_velocity: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    shaped_t = t.reshape((-1,) + (1,) * start.ndim)
    h00 = 2 * shaped_t**3 - 3 * shaped_t**2 + 1
    h10 = shaped_t**3 - 2 * shaped_t**2 + shaped_t
    h01 = -2 * shaped_t**3 + 3 * shaped_t**2
    h11 = shaped_t**3 - shaped_t**2
    return h00 * start + h10 * start_velocity + h01 * end + h11 * end_velocity


def _bridge_rotations(first: np.ndarray, second: np.ndarray, count: int) -> np.ndarray:
    start = Rotation.from_euler("YXZ", first, degrees=True)
    end = Rotation.from_euler("YXZ", second, degrees=True)
    fractions = np.linspace(0.0, 1.0, count + 2)[1:-1]
    eased = fractions**3 * (fractions * (fractions * 6 - 15) + 10)
    return np.asarray(
        [
            (start * (start.inv() * end) ** value).as_euler("YXZ", degrees=True)
            for value in eased
        ]
    )


def _lock_support_foot(
    first: Motion,
    second: Motion,
    bridge: Motion,
    count: int,
    name: str,
) -> Motion:
    joint_index = [joint.name for joint in bridge.joints].index(name)
    start_target = _world_positions(first, first.frames[-1])[joint_index]
    end_target = _world_positions(second, second.frames[0])[joint_index]
    values = np.clip(
        (np.arange(1, count + 1) / (count + 1) - 0.65) / 0.35,
        0.0,
        1.0,
    )
    release = values * values * (3.0 - 2.0 * values)
    targets = start_target + release[:, None] * (end_target - start_target)
    frames = bridge.frames.copy()
    for index, target in enumerate(targets):
        current = _world_positions(bridge, frames[index])[joint_index]
        frames[index, :3] += target - current
    return Motion(bridge.root, frames, bridge.frame_time)


def _create_bridge(
    previous: Motion,
    following: Motion,
    count: int,
    support_name: str,
) -> Motion:
    previous_rotations, previous_positions = _extract_rotations(previous)
    following_rotations, following_positions = _extract_rotations(following)
    bridge_t = np.arange(1, count + 1) / (count + 1)
    bridge_positions = _cubic_hermite(
        previous_positions[-1],
        following_positions[0],
        (previous_positions[-1] - previous_positions[-2]) * count,
        (following_positions[1] - following_positions[0]) * count,
        bridge_t,
    )
    bridge_rotations = _bridge_rotations(
        previous_rotations[-1], following_rotations[0], count
    )
    bridge = _standard_motion(
        previous.root, bridge_rotations, bridge_positions, previous.frame_time
    )
    return _lock_support_foot(previous, following, bridge, count, support_name)


def create_transitions(
    motions: list[Motion], transition_frame_counts: list[int]
) -> Motion:
    """按每个接缝指定的帧数对齐动作，并生成平滑且锁脚的过渡段。"""
    if len(motions) < 2 or len(transition_frame_counts) != len(motions) - 1:
        raise ValueError("BVH 文件与过渡数量不匹配")
    first = _canonicalize(motions[0])
    accumulated = first
    for motion, count in zip(motions[1:], transition_frame_counts, strict=True):
        if count < 0:
            raise ValueError("过渡帧数不能为负数")
        if not np.isclose(first.frame_time, motion.frame_time, rtol=1e-7, atol=1e-9):
            raise ValueError("所有 BVH 的 Frame Time 必须一致")
        following = _canonicalize(
            motion,
            output_root=first.root,
            drop_initial_rest_frame=True,
        )
        aligned, support_name = _align_to_boundary(accumulated, following)
        parts = [accumulated.frames]
        if count > 0:
            parts.append(
                _create_bridge(accumulated, aligned, count, support_name).frames
            )
        parts.append(aligned.frames)
        accumulated = Motion(
            accumulated.root,
            np.concatenate(parts),
            accumulated.frame_time,
        )
    return accumulated
