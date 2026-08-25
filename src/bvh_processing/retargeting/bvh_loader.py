"""加载 LAFAN1/Nokov BVH，并转换为 Robot Retargeter 使用的 Z-up 帧。"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from bvh_processing.retargeting.lafan_vendor import utils
from bvh_processing.retargeting.lafan_vendor.extract import read_bvh

_FRAME_TIME_PATTERN = re.compile(
    r"^\s*Frame\s+Time\s*:\s*([0-9]+(?:\.[0-9]*)?|\.[0-9]+)\s*$",
    re.IGNORECASE,
)

# Xsens/3ds Max 导出的 LAFAN 风格骨架使用解剖学关节名，GMR 则使用
# LeftUpLeg/LeftLeg/LeftFoot 等命名。统一为 GMR 的语义名称后再重定向。
_LAFAN_XSENS_ALIASES = {
    "LeftUpLeg": "LeftHip",
    "LeftLeg": "LeftKnee",
    "LeftFoot": "LeftAnkle",
    "RightUpLeg": "RightHip",
    "RightLeg": "RightKnee",
    "RightFoot": "RightAnkle",
    "LeftArm": "LeftShoulder",
    "LeftForeArm": "LeftElbow",
    "LeftHand": "LeftWrist",
    "RightArm": "RightShoulder",
    "RightForeArm": "RightElbow",
    "RightHand": "RightWrist",
}


def _add_lafan_aliases(frame: dict) -> None:
    for canonical_name, source_name in _LAFAN_XSENS_ALIASES.items():
        if canonical_name not in frame and source_name in frame:
            frame[canonical_name] = frame[source_name]


def read_bvh_fps(path: str | Path) -> float:
    with Path(path).open("r", encoding="utf-8-sig", errors="replace") as stream:
        for line in stream:
            match = _FRAME_TIME_PATTERN.match(line)
            if match:
                frame_time = float(match.group(1))
                if frame_time <= 0:
                    break
                return 1.0 / frame_time
    raise ValueError("BVH 文件缺少有效的 Frame Time")


def load_bvh_frames(path: str | Path, bvh_format: str) -> list[dict]:
    """复用 GMR 的 LAFAN BVH 解析流程生成 Robot Retargeter 输入帧。"""
    animation = read_bvh(str(path))
    global_quaternions, global_positions = utils.quat_fk(
        animation.quats,
        animation.pos,
        animation.parents,
    )

    rotation_matrix = np.array(
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        dtype=np.float64,
    )
    rotation_quaternion_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
    rotation_quaternion_wxyz = rotation_quaternion_xyzw[[3, 0, 1, 2]]

    frames: list[dict] = []
    for frame_index in range(animation.pos.shape[0]):
        frame = {}
        for joint_index, joint_name in enumerate(animation.bones):
            orientation = utils.quat_mul(
                rotation_quaternion_wxyz,
                global_quaternions[frame_index, joint_index],
            )
            position = (
                global_positions[frame_index, joint_index] @ rotation_matrix.T / 100.0
            )
            frame[joint_name] = [position, orientation]

        if bvh_format == "lafan1":
            _add_lafan_aliases(frame)

        toe_names = {
            "lafan1": ("LeftToe", "RightToe"),
            "nokov": ("LeftToeBase", "RightToeBase"),
        }
        try:
            left_toe, right_toe = toe_names[bvh_format]
            frame["LeftFootMod"] = [
                frame["LeftFoot"][0],
                frame[left_toe][1],
            ]
            frame["RightFootMod"] = [
                frame["RightFoot"][0],
                frame[right_toe][1],
            ]
        except KeyError as exc:
            raise ValueError(f"BVH 骨架缺少重定向所需关节：{exc.args[0]}") from exc
        frames.append(frame)

    if not frames:
        raise ValueError("BVH 文件不包含动作帧")
    return frames
