#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "numpy>=1.26",
#   "scipy>=1.13",
# ]
# ///
"""Retarget a Nokov/ToeBase BVH motion to the Unitree G1 29-DOF CSV format.

The generated CSV is compatible with ``scripts/csv_to_npz.py``:

    root position xyz (metres), root quaternion xyzw, 29 joint angles (radians)

This converter performs position/orientation constrained, temporally warm-started
inverse kinematics against the repository's G1 URDF. It intentionally does not
copy BVH Euler angles directly because the human and G1 joint axes/topologies do
not match.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


G1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

G1_DEFAULT_JOINT_POSITIONS = np.array(
    [
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)

DEFAULT_G1_ROOT_POSITION = np.array([0.0, 0.0, 0.76], dtype=np.float64)
DEFAULT_URDF_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/whole_body_tracking/whole_body_tracking/assets"
    / "unitree_description/urdf/g1/main.urdf"
)

# Nokov Skeleton7: +X left, +Y up, +Z forward.
# G1/Isaac: +X forward, +Y left, +Z up.
NOKOV_TO_G1_BASIS = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)

REQUIRED_BVH_JOINTS = (
    "Hips",
    "Spine3",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
)


@dataclass(frozen=True)
class BvhJoint:
    """One BVH hierarchy node with its channel slice."""

    name: str
    parent_name: str | None
    offset: np.ndarray
    channels: tuple[str, ...]
    channel_start: int


@dataclass(frozen=True)
class BvhMotion:
    """Parsed BVH hierarchy and frame samples."""

    joints: tuple[BvhJoint, ...]
    frame_time: float
    frame_values: np.ndarray

    @property
    def frames_per_second(self) -> float:
        return 1.0 / self.frame_time


@dataclass(frozen=True)
class UrdfJoint:
    """URDF joint information required for forward kinematics."""

    name: str
    joint_type: str
    parent_link: str
    child_link: str
    origin_translation: np.ndarray
    origin_rotation: np.ndarray
    axis: np.ndarray
    lower_limit: float
    upper_limit: float


@dataclass(frozen=True)
class RetargetTargets:
    """Per-frame G1 root states, link targets, and contact labels."""

    root_positions: np.ndarray
    root_rotations: np.ndarray
    positions: dict[str, np.ndarray]
    rotations: dict[str, np.ndarray]
    left_contacts: np.ndarray
    right_contacts: np.ndarray
    scale: float


class UrdfKinematicModel:
    """Small URDF forward-kinematics model for offline constrained IK."""

    def __init__(self, urdf_path: Path, controlled_joint_names: tuple[str, ...]):
        xml_root = ElementTree.parse(urdf_path).getroot()
        self.root_link = self._find_root_link(xml_root)
        self.controlled_joint_names = controlled_joint_names
        self.controlled_joint_indexes = {
            joint_name: index for index, joint_name in enumerate(controlled_joint_names)
        }
        self.joints = self._parse_joints(xml_root)
        self.joints_by_parent: dict[str, list[UrdfJoint]] = {}
        for joint in self.joints:
            self.joints_by_parent.setdefault(joint.parent_link, []).append(joint)

        missing_joint_names = set(controlled_joint_names) - {joint.name for joint in self.joints}
        if missing_joint_names:
            raise ValueError(f"URDF 缺少 G1 关节: {sorted(missing_joint_names)}")

        joint_lookup = {joint.name: joint for joint in self.joints}
        self.lower_limits = np.array(
            [joint_lookup[name].lower_limit for name in controlled_joint_names], dtype=np.float64
        )
        self.upper_limits = np.array(
            [joint_lookup[name].upper_limit for name in controlled_joint_names], dtype=np.float64
        )

        # Match Isaac Lab's soft_joint_pos_limit_factor=0.9 around each range centre.
        limit_centres = 0.5 * (self.lower_limits + self.upper_limits)
        limit_half_ranges = 0.45 * (self.upper_limits - self.lower_limits)
        self.soft_lower_limits = limit_centres - limit_half_ranges
        self.soft_upper_limits = limit_centres + limit_half_ranges

    @staticmethod
    def _find_root_link(xml_root: ElementTree.Element) -> str:
        link_names = {element.attrib["name"] for element in xml_root.findall("link")}
        child_links = {
            element.find("child").attrib["link"]
            for element in xml_root.findall("joint")
            if element.find("child") is not None
        }
        root_links = link_names - child_links
        if len(root_links) != 1:
            raise ValueError(f"URDF 应只有一个根 link，实际为: {sorted(root_links)}")
        return next(iter(root_links))

    @staticmethod
    def _parse_xyz(value: str | None, default: Iterable[float]) -> np.ndarray:
        if value is None:
            return np.asarray(tuple(default), dtype=np.float64)
        return np.fromstring(value, sep=" ", dtype=np.float64)

    def _parse_joints(self, xml_root: ElementTree.Element) -> tuple[UrdfJoint, ...]:
        parsed_joints: list[UrdfJoint] = []
        for joint_element in xml_root.findall("joint"):
            origin_element = joint_element.find("origin")
            origin_translation = self._parse_xyz(
                origin_element.attrib.get("xyz") if origin_element is not None else None,
                (0.0, 0.0, 0.0),
            )
            origin_rpy = self._parse_xyz(
                origin_element.attrib.get("rpy") if origin_element is not None else None,
                (0.0, 0.0, 0.0),
            )
            axis_element = joint_element.find("axis")
            axis = self._parse_xyz(
                axis_element.attrib.get("xyz") if axis_element is not None else None,
                (1.0, 0.0, 0.0),
            )
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 0.0:
                axis = axis / axis_norm

            limit_element = joint_element.find("limit")
            lower_limit = -math.inf
            upper_limit = math.inf
            if limit_element is not None:
                lower_limit = float(limit_element.attrib.get("lower", -math.inf))
                upper_limit = float(limit_element.attrib.get("upper", math.inf))

            parsed_joints.append(
                UrdfJoint(
                    name=joint_element.attrib["name"],
                    joint_type=joint_element.attrib["type"],
                    parent_link=joint_element.find("parent").attrib["link"],
                    child_link=joint_element.find("child").attrib["link"],
                    origin_translation=origin_translation,
                    origin_rotation=Rotation.from_euler("xyz", origin_rpy).as_matrix(),
                    axis=axis,
                    lower_limit=lower_limit,
                    upper_limit=upper_limit,
                )
            )
        return tuple(parsed_joints)

    def forward_kinematics(
        self,
        joint_positions: np.ndarray,
        root_position: np.ndarray,
        root_rotation: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Return homogeneous world transforms keyed by link name."""

        root_transform = np.eye(4, dtype=np.float64)
        root_transform[:3, :3] = root_rotation
        root_transform[:3, 3] = root_position
        link_transforms = {self.root_link: root_transform}

        pending_links = [self.root_link]
        while pending_links:
            parent_link = pending_links.pop()
            parent_transform = link_transforms[parent_link]
            for joint in self.joints_by_parent.get(parent_link, []):
                joint_origin_transform = np.eye(4, dtype=np.float64)
                joint_origin_transform[:3, :3] = joint.origin_rotation
                joint_origin_transform[:3, 3] = joint.origin_translation
                child_transform = parent_transform @ joint_origin_transform

                if joint.joint_type in {"revolute", "continuous"}:
                    controlled_index = self.controlled_joint_indexes.get(joint.name)
                    joint_angle = (
                        joint_positions[controlled_index] if controlled_index is not None else 0.0
                    )
                    child_transform = child_transform.copy()
                    child_transform[:3, :3] = (
                        child_transform[:3, :3]
                        @ Rotation.from_rotvec(joint.axis * joint_angle).as_matrix()
                    )
                elif joint.joint_type == "prismatic":
                    controlled_index = self.controlled_joint_indexes.get(joint.name)
                    joint_distance = (
                        joint_positions[controlled_index] if controlled_index is not None else 0.0
                    )
                    child_transform = child_transform.copy()
                    child_transform[:3, 3] += child_transform[:3, :3] @ (
                        joint.axis * joint_distance
                    )

                link_transforms[joint.child_link] = child_transform
                pending_links.append(joint.child_link)

        return link_transforms


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 Nokov/ToeBase BVH 通过约束 IK 重定向为 G1 29DOF CSV。"
    )
    parser.add_argument("--input-bvh", type=Path, required=True, help="输入 BVH 文件。")
    parser.add_argument("--output-csv", type=Path, required=True, help="输出 36 列 CSV 文件。")
    parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_URDF_PATH,
        help="G1 main.urdf 路径。",
    )
    parser.add_argument(
        "--position-scale",
        type=float,
        default=0.01,
        help="BVH 长度到米的比例；Nokov 默认厘米，因此为 0.01。",
    )
    parser.add_argument("--start-frame", type=int, default=0, help="起始帧，0-based 且包含。")
    parser.add_argument("--end-frame", type=int, help="结束帧，0-based 且不包含。")
    parser.add_argument(
        "--skip-reference-frame",
        choices=("auto", "yes", "no"),
        default="auto",
        help="是否跳过首个不连续的标定/T-pose 帧。",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="仅处理指定帧数，便于快速诊断。",
    )
    parser.add_argument(
        "--max-ik-evaluations",
        type=int,
        default=24,
        help="每帧最小二乘函数评估上限。",
    )
    parser.add_argument(
        "--diagnostics",
        type=Path,
        help="诊断 JSON；默认与 CSV 同名并加 .diagnostics.json。",
    )
    return parser.parse_args()


def load_bvh(path: Path) -> BvhMotion:
    """Parse a BVH while retaining each node's declared channel order."""

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    motion_line_index = next(
        (line_index for line_index, line in enumerate(lines) if line.strip() == "MOTION"),
        None,
    )
    if motion_line_index is None:
        raise ValueError("BVH 缺少 MOTION 段。")

    hierarchy_lines = lines[:motion_line_index]
    joints: list[BvhJoint] = []
    channel_count = 0

    def parse_joint(line_index: int, parent_name: str | None) -> int:
        nonlocal channel_count
        declaration = hierarchy_lines[line_index].strip()
        declaration_match = re.match(r"(?:ROOT|JOINT)\s+(.+)$", declaration)
        if declaration_match is None:
            raise ValueError(f"非法 BVH 节点声明，第 {line_index + 1} 行: {declaration}")
        joint_name = declaration_match.group(1).strip()

        line_index += 1
        while line_index < len(hierarchy_lines) and not hierarchy_lines[line_index].strip():
            line_index += 1
        if hierarchy_lines[line_index].strip() != "{":
            raise ValueError(f"BVH 节点 {joint_name} 后缺少 '{{'。")

        offset = np.zeros(3, dtype=np.float64)
        channels: tuple[str, ...] = ()
        joint_record_index = len(joints)
        joints.append(BvhJoint(joint_name, parent_name, offset, channels, channel_count))
        line_index += 1

        while line_index < len(hierarchy_lines):
            stripped_line = hierarchy_lines[line_index].strip()
            if not stripped_line:
                line_index += 1
                continue
            if stripped_line.startswith("OFFSET "):
                offset = np.fromstring(stripped_line.removeprefix("OFFSET "), sep=" ")
                if offset.shape != (3,):
                    raise ValueError(f"节点 {joint_name} 的 OFFSET 非 3 维。")
                line_index += 1
                continue
            if stripped_line.startswith("CHANNELS "):
                channel_tokens = stripped_line.split()
                declared_count = int(channel_tokens[1])
                channels = tuple(channel_tokens[2:])
                if len(channels) != declared_count:
                    raise ValueError(f"节点 {joint_name} 的 CHANNELS 数量不一致。")
                joints[joint_record_index] = BvhJoint(
                    joint_name, parent_name, offset, channels, channel_count
                )
                channel_count += declared_count
                line_index += 1
                continue
            if stripped_line.startswith("JOINT "):
                line_index = parse_joint(line_index, joint_name)
                continue
            if stripped_line == "End Site":
                line_index += 1
                brace_depth = 0
                while line_index < len(hierarchy_lines):
                    brace_depth += hierarchy_lines[line_index].count("{")
                    brace_depth -= hierarchy_lines[line_index].count("}")
                    line_index += 1
                    if brace_depth == 0:
                        break
                continue
            if stripped_line == "}":
                return line_index + 1
            line_index += 1
        raise ValueError(f"节点 {joint_name} 缺少结束括号。")

    root_line_index = next(
        index for index, line in enumerate(hierarchy_lines) if line.strip().startswith("ROOT ")
    )
    parse_joint(root_line_index, None)

    missing_joint_names = set(REQUIRED_BVH_JOINTS) - {joint.name for joint in joints}
    if missing_joint_names:
        raise ValueError(f"BVH 缺少 Skeleton7 关节: {sorted(missing_joint_names)}")

    motion_header = lines[motion_line_index + 1 :]
    frames_match = re.match(r"Frames:\s*(\d+)", motion_header[0].strip())
    frame_time_match = re.match(
        r"Frame\s+Time:\s*([0-9.eE+-]+)", motion_header[1].strip()
    )
    if frames_match is None or frame_time_match is None:
        raise ValueError("BVH MOTION 头部格式不正确。")
    declared_frame_count = int(frames_match.group(1))
    frame_time = float(frame_time_match.group(1))
    flattened_values = np.fromstring("\n".join(motion_header[2:]), sep=" ", dtype=np.float64)
    expected_value_count = declared_frame_count * channel_count
    if flattened_values.size != expected_value_count:
        raise ValueError(
            f"BVH 动作值数量错误：期望 {expected_value_count}，实际 {flattened_values.size}。"
        )
    frame_values = flattened_values.reshape(declared_frame_count, channel_count)
    return BvhMotion(tuple(joints), frame_time, frame_values)


def compose_channel_rotation(channels: tuple[str, ...], values: np.ndarray) -> np.ndarray:
    """Compose BVH rotations in their declaration order."""

    composed_rotation = np.eye(3, dtype=np.float64)
    for channel_name, channel_value in zip(channels, values, strict=True):
        if not channel_name.endswith("rotation"):
            continue
        axis_name = channel_name[0].lower()
        composed_rotation = composed_rotation @ Rotation.from_euler(
            axis_name, channel_value, degrees=True
        ).as_matrix()
    return composed_rotation


def compute_bvh_forward_kinematics(
    motion: BvhMotion,
    position_scale: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compute world positions and rotation matrices for every BVH joint."""

    frame_count = motion.frame_values.shape[0]
    world_positions: dict[str, np.ndarray] = {}
    world_rotations: dict[str, np.ndarray] = {}

    for joint in motion.joints:
        local_positions = np.repeat((joint.offset * position_scale)[None, :], frame_count, axis=0)
        local_rotations = np.empty((frame_count, 3, 3), dtype=np.float64)
        channel_values = motion.frame_values[
            :, joint.channel_start : joint.channel_start + len(joint.channels)
        ]
        for channel_index, channel_name in enumerate(joint.channels):
            if channel_name.endswith("position"):
                axis_index = {"X": 0, "Y": 1, "Z": 2}[channel_name[0]]
                local_positions[:, axis_index] += channel_values[:, channel_index] * position_scale
        for frame_index in range(frame_count):
            local_rotations[frame_index] = compose_channel_rotation(
                joint.channels, channel_values[frame_index]
            )

        if joint.parent_name is None:
            world_positions[joint.name] = local_positions
            world_rotations[joint.name] = local_rotations
        else:
            parent_positions = world_positions[joint.parent_name]
            parent_rotations = world_rotations[joint.parent_name]
            world_positions[joint.name] = parent_positions + np.einsum(
                "nij,nj->ni", parent_rotations, local_positions
            )
            world_rotations[joint.name] = parent_rotations @ local_rotations

    return world_positions, world_rotations


def transform_nokov_to_g1_coordinates(
    positions: dict[str, np.ndarray],
    rotations: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    transformed_positions = {
        name: values @ NOKOV_TO_G1_BASIS.T for name, values in positions.items()
    }
    transformed_rotations = {
        name: NOKOV_TO_G1_BASIS @ values @ NOKOV_TO_G1_BASIS.T
        for name, values in rotations.items()
    }
    return transformed_positions, transformed_rotations


def should_skip_reference_frame(
    motion: BvhMotion,
    positions: dict[str, np.ndarray],
    mode: str,
) -> bool:
    if mode == "yes":
        return True
    if mode == "no" or motion.frame_values.shape[0] < 2:
        return False
    root_jump = np.linalg.norm(positions["Hips"][1] - positions["Hips"][0])
    angular_jump = np.percentile(np.abs(motion.frame_values[1, 6:] - motion.frame_values[0, 6:]), 90)
    return bool(root_jump > 0.25 or angular_jump > 15.0)


def extract_yaw_rotation(rotation_matrix: np.ndarray) -> np.ndarray:
    forward_direction = rotation_matrix[:, 0].copy()
    forward_direction[2] = 0.0
    forward_norm = np.linalg.norm(forward_direction)
    if forward_norm < 1.0e-8:
        return np.eye(3, dtype=np.float64)
    forward_direction /= forward_norm
    yaw_angle = math.atan2(forward_direction[1], forward_direction[0])
    return Rotation.from_euler("z", yaw_angle).as_matrix()


def detect_foot_contacts(
    foot_positions: np.ndarray,
    toe_positions: np.ndarray,
    frame_time: float,
    ground_height: float,
) -> np.ndarray:
    support_heights = np.minimum(foot_positions[:, 2], toe_positions[:, 2]) - ground_height
    support_centres = 0.5 * (foot_positions + toe_positions)
    support_velocities = np.gradient(support_centres, frame_time, axis=0)
    horizontal_speeds = np.linalg.norm(support_velocities[:, :2], axis=1)
    vertical_speeds = np.abs(support_velocities[:, 2])
    raw_contacts = (
        (support_heights < 0.055)
        & (horizontal_speeds < 0.35)
        & (vertical_speeds < 0.25)
    )

    # A short majority filter removes one-frame contact flicker at 120 Hz.
    padded_contacts = np.pad(raw_contacts.astype(np.int32), (2, 2), mode="edge")
    contact_votes = np.convolve(padded_contacts, np.ones(5, dtype=np.int32), mode="valid")
    return contact_votes >= 3


def lock_contact_targets(target_positions: np.ndarray, contacts: np.ndarray) -> np.ndarray:
    locked_positions = target_positions.copy()
    contact_start: int | None = None
    for frame_index, is_contact in enumerate(contacts):
        if is_contact and contact_start is None:
            contact_start = frame_index
        is_last_frame = frame_index == len(contacts) - 1
        if contact_start is not None and (not is_contact or is_last_frame):
            contact_end = frame_index if not is_contact else frame_index + 1
            segment_position = np.median(
                target_positions[contact_start:contact_end], axis=0
            )
            locked_positions[contact_start:contact_end, :2] = segment_position[:2]
            contact_start = None
    return locked_positions


def build_retarget_targets(
    positions: dict[str, np.ndarray],
    rotations: dict[str, np.ndarray],
    model: UrdfKinematicModel,
    frame_time: float,
) -> RetargetTargets:
    initial_heading = extract_yaw_rotation(rotations["Hips"][0])
    heading_inverse = initial_heading.T
    initial_hips_position = positions["Hips"][0].copy()

    normalized_positions = {
        name: np.einsum("ij,nj->ni", heading_inverse, values - initial_hips_position)
        for name, values in positions.items()
    }
    normalized_rotations = {
        name: heading_inverse @ values for name, values in rotations.items()
    }

    default_transforms = model.forward_kinematics(
        G1_DEFAULT_JOINT_POSITIONS,
        DEFAULT_G1_ROOT_POSITION,
        np.eye(3, dtype=np.float64),
    )
    source_leg_lengths = [
        np.linalg.norm(positions["LeftFoot"][0] - positions["Hips"][0]),
        np.linalg.norm(positions["RightFoot"][0] - positions["Hips"][0]),
    ]
    target_leg_lengths = [
        np.linalg.norm(default_transforms["left_ankle_roll_link"][:3, 3] - DEFAULT_G1_ROOT_POSITION),
        np.linalg.norm(default_transforms["right_ankle_roll_link"][:3, 3] - DEFAULT_G1_ROOT_POSITION),
    ]
    motion_scale = float(np.mean(target_leg_lengths) / np.mean(source_leg_lengths))

    left_ground_samples = np.minimum(
        normalized_positions["LeftFoot"][:, 2], normalized_positions["LeftToeBase"][:, 2]
    )
    right_ground_samples = np.minimum(
        normalized_positions["RightFoot"][:, 2], normalized_positions["RightToeBase"][:, 2]
    )
    ground_height = float(
        np.percentile(np.concatenate([left_ground_samples, right_ground_samples]), 2.0)
    )

    left_contacts = detect_foot_contacts(
        normalized_positions["LeftFoot"],
        normalized_positions["LeftToeBase"],
        frame_time,
        ground_height,
    )
    right_contacts = detect_foot_contacts(
        normalized_positions["RightFoot"],
        normalized_positions["RightToeBase"],
        frame_time,
        ground_height,
    )

    root_positions = DEFAULT_G1_ROOT_POSITION + motion_scale * normalized_positions["Hips"]
    root_rotations = normalized_rotations["Hips"]
    target_positions: dict[str, np.ndarray] = {}
    source_to_target_links = {
        "LeftLeg": "left_knee_link",
        "RightLeg": "right_knee_link",
        "LeftFoot": "left_ankle_roll_link",
        "RightFoot": "right_ankle_roll_link",
        "LeftForeArm": "left_elbow_link",
        "RightForeArm": "right_elbow_link",
        "LeftHand": "left_wrist_yaw_link",
        "RightHand": "right_wrist_yaw_link",
    }
    for source_name, target_name in source_to_target_links.items():
        source_offsets = normalized_positions[source_name] - normalized_positions["Hips"]
        target_positions[target_name] = root_positions + motion_scale * source_offsets

    target_positions["left_ankle_roll_link"] = lock_contact_targets(
        target_positions["left_ankle_roll_link"], left_contacts
    )
    target_positions["right_ankle_roll_link"] = lock_contact_targets(
        target_positions["right_ankle_roll_link"], right_contacts
    )

    target_rotations = {
        "torso_link": normalized_rotations["Spine3"],
        "left_ankle_roll_link": normalized_rotations["LeftFoot"],
        "right_ankle_roll_link": normalized_rotations["RightFoot"],
    }
    return RetargetTargets(
        root_positions=root_positions,
        root_rotations=root_rotations,
        positions=target_positions,
        rotations=target_rotations,
        left_contacts=left_contacts,
        right_contacts=right_contacts,
        scale=motion_scale,
    )


def rotation_error(current_rotation: np.ndarray, target_rotation: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(target_rotation @ current_rotation.T).as_rotvec()


def solve_motion_ik(
    model: UrdfKinematicModel,
    targets: RetargetTargets,
    maximum_function_evaluations: int,
) -> tuple[np.ndarray, dict[str, object]]:
    frame_count = targets.root_positions.shape[0]
    solved_joint_positions = np.empty((frame_count, len(G1_JOINT_NAMES)), dtype=np.float64)
    previous_joint_positions = np.clip(
        G1_DEFAULT_JOINT_POSITIONS,
        model.soft_lower_limits,
        model.soft_upper_limits,
    )
    position_error_means: list[float] = []
    solver_costs: list[float] = []
    solver_failures: list[int] = []

    position_links = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_knee_link",
        "right_knee_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )

    for frame_index in range(frame_count):
        root_position = targets.root_positions[frame_index]
        root_rotation = targets.root_rotations[frame_index]

        def compute_residuals(candidate_joint_positions: np.ndarray) -> np.ndarray:
            transforms = model.forward_kinematics(
                candidate_joint_positions, root_position, root_rotation
            )
            residual_parts: list[np.ndarray] = []
            for link_name in position_links:
                is_left_contact = (
                    link_name == "left_ankle_roll_link" and targets.left_contacts[frame_index]
                )
                is_right_contact = (
                    link_name == "right_ankle_roll_link" and targets.right_contacts[frame_index]
                )
                if "ankle" in link_name:
                    weight = 9.0 if is_left_contact or is_right_contact else 5.0
                elif "knee" in link_name:
                    weight = 2.0
                elif "elbow" in link_name:
                    weight = 1.6
                else:
                    weight = 2.2
                position_error = transforms[link_name][:3, 3] - targets.positions[link_name][frame_index]
                residual_parts.append(weight * position_error)

            torso_error = rotation_error(
                transforms["torso_link"][:3, :3],
                targets.rotations["torso_link"][frame_index],
            )
            residual_parts.append(1.8 * torso_error)

            for side_name in ("left", "right"):
                ankle_link_name = f"{side_name}_ankle_roll_link"
                is_contact = (
                    targets.left_contacts[frame_index]
                    if side_name == "left"
                    else targets.right_contacts[frame_index]
                )
                ankle_error = rotation_error(
                    transforms[ankle_link_name][:3, :3],
                    targets.rotations[ankle_link_name][frame_index],
                )
                # Preserve heading strongly; flatten support-foot roll/pitch through IK regularity.
                ankle_weight = np.array([0.35, 0.35, 1.2 if is_contact else 0.6])
                residual_parts.append(ankle_weight * ankle_error)

            residual_parts.append(0.10 * (candidate_joint_positions - G1_DEFAULT_JOINT_POSITIONS))
            residual_parts.append(0.32 * (candidate_joint_positions - previous_joint_positions))
            return np.concatenate(residual_parts)

        solver_result = least_squares(
            compute_residuals,
            previous_joint_positions,
            bounds=(model.soft_lower_limits, model.soft_upper_limits),
            method="trf",
            max_nfev=maximum_function_evaluations,
            ftol=2.0e-4,
            xtol=2.0e-4,
            gtol=2.0e-4,
        )
        if not solver_result.success:
            solver_failures.append(frame_index)
        solved_joint_positions[frame_index] = solver_result.x
        previous_joint_positions = solver_result.x
        solver_costs.append(float(solver_result.cost))

        solved_transforms = model.forward_kinematics(
            solver_result.x, root_position, root_rotation
        )
        frame_position_errors = [
            np.linalg.norm(
                solved_transforms[link_name][:3, 3]
                - targets.positions[link_name][frame_index]
            )
            for link_name in position_links
        ]
        position_error_means.append(float(np.mean(frame_position_errors)))

        if frame_index == 0 or (frame_index + 1) % 25 == 0 or frame_index + 1 == frame_count:
            print(
                f"[IK] {frame_index + 1:4d}/{frame_count}: "
                f"cost={solver_result.cost:.4f}, mean position error="
                f"{position_error_means[-1] * 1000.0:.1f} mm"
            )

    diagnostics = {
        "frame_count": frame_count,
        "solver_failure_frames": solver_failures,
        "solver_cost_mean": float(np.mean(solver_costs)),
        "solver_cost_max": float(np.max(solver_costs)),
        "position_error_mean_m": float(np.mean(position_error_means)),
        "position_error_p95_m": float(np.percentile(position_error_means, 95.0)),
        "position_error_max_m": float(np.max(position_error_means)),
    }
    return solved_joint_positions, diagnostics


def ensure_quaternion_sign_continuity(quaternions: np.ndarray) -> np.ndarray:
    continuous_quaternions = quaternions.copy()
    for frame_index in range(1, len(continuous_quaternions)):
        if np.dot(continuous_quaternions[frame_index - 1], continuous_quaternions[frame_index]) < 0.0:
            continuous_quaternions[frame_index] *= -1.0
    return continuous_quaternions


def validate_output_motion(
    output_motion: np.ndarray,
    model: UrdfKinematicModel,
    frame_time: float,
) -> dict[str, object]:
    if output_motion.ndim != 2 or output_motion.shape[1] != 36:
        raise ValueError(f"输出必须为 N x 36，实际为 {output_motion.shape}。")
    if not np.isfinite(output_motion).all():
        raise ValueError("输出包含 NaN 或 Inf。")

    quaternion_norms = np.linalg.norm(output_motion[:, 3:7], axis=1)
    joint_positions = output_motion[:, 7:]
    below_limits = joint_positions < model.soft_lower_limits[None, :] - 1.0e-7
    above_limits = joint_positions > model.soft_upper_limits[None, :] + 1.0e-7
    joint_velocities = np.gradient(joint_positions, frame_time, axis=0)
    root_velocities = np.gradient(output_motion[:, :3], frame_time, axis=0)
    return {
        "quaternion_norm_min": float(np.min(quaternion_norms)),
        "quaternion_norm_max": float(np.max(quaternion_norms)),
        "joint_limit_violation_count": int(np.count_nonzero(below_limits | above_limits)),
        "maximum_joint_speed_rad_s": float(np.max(np.abs(joint_velocities))),
        "maximum_root_speed_m_s": float(np.max(np.linalg.norm(root_velocities, axis=1))),
    }


def main() -> None:
    arguments = parse_arguments()
    if not arguments.input_bvh.is_file():
        raise FileNotFoundError(f"输入 BVH 不存在: {arguments.input_bvh}")
    if not arguments.urdf.is_file():
        raise FileNotFoundError(
            f"G1 URDF 不存在: {arguments.urdf}\n"
            "请先按 README 下载 unitree_description 资产。"
        )
    if arguments.position_scale <= 0.0:
        raise ValueError("--position-scale 必须大于 0。")

    motion = load_bvh(arguments.input_bvh)
    print(
        f"[INFO] BVH: {motion.frame_values.shape[0]} frames, "
        f"{motion.frames_per_second:.6f} Hz, {len(motion.joints)} joints"
    )
    bvh_positions, bvh_rotations = compute_bvh_forward_kinematics(
        motion, arguments.position_scale
    )
    g1_positions, g1_rotations = transform_nokov_to_g1_coordinates(
        bvh_positions, bvh_rotations
    )

    skip_reference_frame = should_skip_reference_frame(
        motion, g1_positions, arguments.skip_reference_frame
    )
    frame_start = arguments.start_frame + (1 if skip_reference_frame else 0)
    frame_end = arguments.end_frame or motion.frame_values.shape[0]
    if arguments.max_frames is not None:
        frame_end = min(frame_end, frame_start + arguments.max_frames)
    if not 0 <= frame_start < frame_end <= motion.frame_values.shape[0]:
        raise ValueError(
            f"非法帧范围 [{frame_start}, {frame_end})，总帧数为 {motion.frame_values.shape[0]}。"
        )
    if skip_reference_frame:
        print("[INFO] 检测到不连续标定帧，已跳过 BVH 第 0 帧。")

    sliced_positions = {
        name: values[frame_start:frame_end] for name, values in g1_positions.items()
    }
    sliced_rotations = {
        name: values[frame_start:frame_end] for name, values in g1_rotations.items()
    }
    model = UrdfKinematicModel(arguments.urdf, G1_JOINT_NAMES)
    targets = build_retarget_targets(
        sliced_positions, sliced_rotations, model, motion.frame_time
    )
    print(
        f"[INFO] 人体到 G1 比例: {targets.scale:.4f}; "
        f"左脚接触 {np.mean(targets.left_contacts) * 100.0:.1f}%; "
        f"右脚接触 {np.mean(targets.right_contacts) * 100.0:.1f}%"
    )

    joint_positions, ik_diagnostics = solve_motion_ik(
        model, targets, arguments.max_ik_evaluations
    )
    root_quaternions_xyzw = Rotation.from_matrix(targets.root_rotations).as_quat()
    root_quaternions_xyzw = ensure_quaternion_sign_continuity(root_quaternions_xyzw)
    output_motion = np.column_stack(
        [targets.root_positions, root_quaternions_xyzw, joint_positions]
    )
    validation_diagnostics = validate_output_motion(output_motion, model, motion.frame_time)

    arguments.output_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(arguments.output_csv, output_motion, delimiter=",", fmt="%.9f")
    diagnostics_path = arguments.diagnostics or arguments.output_csv.with_suffix(
        ".diagnostics.json"
    )
    diagnostics = {
        "input_bvh": str(arguments.input_bvh.resolve()),
        "output_csv": str(arguments.output_csv.resolve()),
        "urdf": str(arguments.urdf.resolve()),
        "source_fps": motion.frames_per_second,
        "source_frame_count": int(motion.frame_values.shape[0]),
        "processed_frame_range": [frame_start, frame_end],
        "skipped_reference_frame": skip_reference_frame,
        "position_scale": arguments.position_scale,
        "retarget_scale": targets.scale,
        "left_contact_ratio": float(np.mean(targets.left_contacts)),
        "right_contact_ratio": float(np.mean(targets.right_contacts)),
        "ik": ik_diagnostics,
        "validation": validation_diagnostics,
    }
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[OK] G1 CSV 已保存: {arguments.output_csv}")
    print(f"[OK] 诊断报告已保存: {diagnostics_path}")
    print(
        "[NEXT] 使用 csv_to_npz.py，并设置 "
        f"--input_fps {round(motion.frames_per_second)} --output_fps 50"
    )


if __name__ == "__main__":
    main()
