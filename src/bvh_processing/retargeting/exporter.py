"""将 Robot Retargeter 输出导出为 Whole Body Tracking NPZ 和元数据 JSON。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import BinaryIO

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from bvh_processing.retargeting.robot_retargeter import RobotRetargetResult

_OUTPUT_FPS = 50.0
_SPOOL_MEMORY_LIMIT = 8 * 1024 * 1024

G1_MUJOCO_JOINT_NAMES = (
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
G1_ISAACLAB_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)
G1_ISAACLAB_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)


@dataclass(slots=True)
class RetargetArtifacts:
    npz: BinaryIO
    npz_filename: str
    metadata: BinaryIO
    metadata_filename: str

    def close(self) -> None:
        self.npz.close()
        self.metadata.close()


def _spooled(data: bytes) -> BinaryIO:
    # 所有权转交给 RetargetArtifacts，由调用方统一关闭。
    content = SpooledTemporaryFile(  # noqa: SIM115
        max_size=_SPOOL_MEMORY_LIMIT,
        mode="w+b",
    )
    content.write(data)
    content.seek(0)
    return content


def _continuous_quaternions(quaternions: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternions, dtype=np.float64).copy()
    result /= np.linalg.norm(result, axis=-1, keepdims=True)
    for frame_index in range(1, result.shape[0]):
        flip = np.sum(result[frame_index - 1] * result[frame_index], axis=-1) < 0
        result[frame_index][flip] *= -1
    return result


def _resample(
    result: RobotRetargetResult,
    source_fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_count = result.root_position_m.shape[0]
    if frame_count < 2:
        raise ValueError("重定向结果至少需要两帧才能导出 NPZ")
    duration = (frame_count - 1) / source_fps
    output_times = np.arange(0.0, duration, 1.0 / _OUTPUT_FPS)
    if output_times.size < 2:
        raise ValueError("BVH 动作时长过短，无法生成 50 FPS NPZ")
    input_times = np.arange(frame_count, dtype=np.float64) / source_fps

    root_position = np.column_stack(
        [
            np.interp(output_times, input_times, result.root_position_m[:, axis])
            for axis in range(3)
        ]
    )
    joint_position = np.column_stack(
        [
            np.interp(
                output_times,
                input_times,
                np.unwrap(result.joint_position_rad, axis=0)[:, joint_index],
            )
            for joint_index in range(result.joint_position_rad.shape[1])
        ]
    )
    root_quaternion = _continuous_quaternions(result.root_quaternion_xyzw[:, None])[
        :, 0
    ]
    root_quaternion = Slerp(
        input_times,
        Rotation.from_quat(root_quaternion),
    )(output_times).as_quat()
    return root_position, root_quaternion, joint_position


def _linear_velocity(values: np.ndarray) -> np.ndarray:
    return np.gradient(values, 1.0 / _OUTPUT_FPS, axis=0, edge_order=1)


def _angular_velocity(quaternions_wxyz: np.ndarray) -> np.ndarray:
    rotations = Rotation.from_quat(quaternions_wxyz[..., [1, 2, 3, 0]])
    frame_count = quaternions_wxyz.shape[0]
    velocity = np.empty((*quaternions_wxyz.shape[:2], 3), dtype=np.float64)
    for frame_index in range(frame_count):
        previous_index = max(0, frame_index - 1)
        next_index = min(frame_count - 1, frame_index + 1)
        elapsed = (next_index - previous_index) / _OUTPUT_FPS
        relative = rotations[previous_index].inv() * rotations[next_index]
        velocity[frame_index] = relative.as_rotvec() / elapsed
    return velocity


def export_tracking_artifacts(
    result: RobotRetargetResult,
    source_fps: float,
    source_filename: str,
    bvh_format: str,
) -> RetargetArtifacts:
    root_position, root_quaternion_xyzw, joint_position_mujoco = _resample(
        result,
        source_fps,
    )
    joint_indexes = np.asarray(
        [G1_MUJOCO_JOINT_NAMES.index(name) for name in G1_ISAACLAB_JOINT_NAMES]
    )
    joint_position = joint_position_mujoco[:, joint_indexes]
    joint_velocity = _linear_velocity(joint_position)

    body_ids = np.asarray(
        [
            mj.mj_name2id(result.model, mj.mjtObj.mjOBJ_BODY, name)
            for name in G1_ISAACLAB_BODY_NAMES
        ]
    )
    if np.any(body_ids < 0):
        raise ValueError("G1 模型缺少 Whole Body Tracking 所需机身节点")

    data = mj.MjData(result.model)
    body_position = []
    body_quaternion = []
    for frame_index in range(root_position.shape[0]):
        data.qpos[:3] = root_position[frame_index]
        data.qpos[3:7] = root_quaternion_xyzw[frame_index, [3, 0, 1, 2]]
        data.qpos[7:] = joint_position_mujoco[frame_index]
        mj.mj_forward(result.model, data)
        body_position.append(data.xpos[body_ids].copy())
        body_quaternion.append(data.xquat[body_ids].copy())
    body_position_array = np.asarray(body_position)
    body_quaternion_array = _continuous_quaternions(np.asarray(body_quaternion))

    # 所有权转交给 RetargetArtifacts，由调用方统一关闭。
    npz_content = SpooledTemporaryFile(  # noqa: SIM115
        max_size=_SPOOL_MEMORY_LIMIT,
        mode="w+b",
    )
    np.savez(
        npz_content,
        fps=np.asarray([_OUTPUT_FPS], dtype=np.float64),
        joint_pos=joint_position.astype(np.float32),
        joint_vel=joint_velocity.astype(np.float32),
        body_pos_w=body_position_array.astype(np.float32),
        body_quat_w=body_quaternion_array.astype(np.float32),
        body_lin_vel_w=_linear_velocity(body_position_array).astype(np.float32),
        body_ang_vel_w=_angular_velocity(body_quaternion_array).astype(np.float32),
    )
    npz_content.seek(0)

    stem = Path(source_filename).stem
    metadata_filename = f"{stem}_g1_tracking.json"
    metadata = {
        "schema": "whole_body_tracking_motion",
        "schema_version": 1,
        "container": "numpy_npz",
        "robot": "unitree_g1_29dof",
        "fps": _OUTPUT_FPS,
        "frame_count": int(root_position.shape[0]),
        "joint_count": len(G1_ISAACLAB_JOINT_NAMES),
        "body_count": len(G1_ISAACLAB_BODY_NAMES),
        "source_bvh": source_filename,
        "source_bvh_format": bvh_format,
        "world_coordinate_system": "right_handed_z_up",
        "body_quaternion_order": "wxyz",
        "joint_names": list(G1_ISAACLAB_JOINT_NAMES),
        "body_names": list(G1_ISAACLAB_BODY_NAMES),
        "required_npz_keys": [
            "fps",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        ],
        "generator": "GMR Robot Retargeter",
        "retarget_diagnostics": result.diagnostics,
    }
    metadata_content = _spooled(
        (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode()
    )
    return RetargetArtifacts(
        npz=npz_content,
        npz_filename=f"{stem}_g1_tracking.npz",
        metadata=metadata_content,
        metadata_filename=metadata_filename,
    )
