"""BVH adapter for the ccrpRepo/robot_retargeter keypoint IK algorithm.

The upstream project accepts precomputed SMPL-X keypoint files. This adapter
keeps its link-length normalization and single-stage Mink IK solver, while
feeding it the already parsed, Z-up BVH frames used by GMR. It intentionally
uses GMR's standard 29-DoF G1 model so generated motions retain the repository's
joint order and limits.

Upstream: https://github.com/ccrpRepo/robot_retargeter
Integrated from local revision: f1418972319287c1b93af0f7a3b445f613cff5e4
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import mink
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_G1_XML = Path(__file__).with_name("assets") / "g1_mocap_29dof.xml"

# link_name: (source parent, source child, robot parent anchor, robot child body)
# The three virtual anchors reproduce the fixed helper bodies in the upstream
# G1 XML without requiring a second copy of the robot meshes/model.
LINK_CHAINS = (
    ("left_hip", "hips_mean", "left_up_leg", "hips_anchor", "left_hip_roll_link"),
    ("left_thigh", "left_up_leg", "left_leg", "left_hip_roll_link", "left_knee_link"),
    ("left_calf", "left_leg", "left_foot", "left_knee_link", "left_ankle_roll_link"),
    ("right_hip", "hips_mean", "right_up_leg", "hips_anchor", "right_hip_roll_link"),
    (
        "right_thigh",
        "right_up_leg",
        "right_leg",
        "right_hip_roll_link",
        "right_knee_link",
    ),
    (
        "right_calf",
        "right_leg",
        "right_foot",
        "right_knee_link",
        "right_ankle_roll_link",
    ),
    ("neck", "hips_mean", "shoulder_mean", "hips_anchor", "neck_anchor"),
    ("head", "shoulder_mean", "head", "neck_anchor", "head_anchor"),
    (
        "left_shoulder",
        "shoulder_mean",
        "left_arm",
        "neck_anchor",
        "left_shoulder_roll_link",
    ),
    (
        "left_arm",
        "left_arm",
        "left_fore_arm",
        "left_shoulder_roll_link",
        "left_elbow_link",
    ),
    (
        "left_fore_arm",
        "left_fore_arm",
        "left_hand",
        "left_elbow_link",
        "left_wrist_yaw_link",
    ),
    (
        "right_shoulder",
        "shoulder_mean",
        "right_arm",
        "neck_anchor",
        "right_shoulder_roll_link",
    ),
    (
        "right_arm",
        "right_arm",
        "right_fore_arm",
        "right_shoulder_roll_link",
        "right_elbow_link",
    ),
    (
        "right_fore_arm",
        "right_fore_arm",
        "right_hand",
        "right_elbow_link",
        "right_wrist_yaw_link",
    ),
)

# keypoint: (robot body, position cost, orientation cost, source orientation)
IK_MATCH_TABLE = {
    "hips_mean": ("pelvis", 100.0, 0.0, "hips"),
    "left_hip": ("left_hip_roll_link", 30.0, 3.0, "left_up_leg"),
    "left_thigh": ("left_knee_link", 0.0, 3.0, "left_leg"),
    "left_calf": ("left_ankle_roll_link", 30.0, 3.0, "left_foot"),
    "right_hip": ("right_hip_roll_link", 30.0, 3.0, "right_up_leg"),
    "right_thigh": ("right_knee_link", 0.0, 3.0, "right_leg"),
    "right_calf": ("right_ankle_roll_link", 30.0, 3.0, "right_foot"),
    "head": ("torso_link", 0.0, 3.0, "head"),
    "left_shoulder": ("left_shoulder_roll_link", 30.0, 3.0, "left_arm"),
    "left_arm": ("left_elbow_link", 10.0, 1.0, "left_fore_arm"),
    "left_fore_arm": ("left_wrist_yaw_link", 10.0, 1.0, "left_hand"),
    "right_shoulder": ("right_shoulder_roll_link", 30.0, 3.0, "right_arm"),
    "right_arm": ("right_elbow_link", 10.0, 1.0, "right_fore_arm"),
    "right_fore_arm": ("right_wrist_yaw_link", 10.0, 1.0, "right_hand"),
}

SOURCE_BODIES = {
    "hips": "Hips",
    "left_up_leg": "LeftUpLeg",
    "left_leg": "LeftLeg",
    "left_foot": "LeftFootMod",
    "right_up_leg": "RightUpLeg",
    "right_leg": "RightLeg",
    "right_foot": "RightFootMod",
    "head": "Head",
    "left_arm": "LeftArm",
    "left_fore_arm": "LeftForeArm",
    "left_hand": "LeftHand",
    "right_arm": "RightArm",
    "right_fore_arm": "RightForeArm",
    "right_hand": "RightHand",
}

# BVH-to-G1 frame offsets are the same conventions as GMR's Nokov IK config.
COMMON_OFFSET = np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float64)
ORIENTATION_OFFSETS = {
    "hips": COMMON_OFFSET,
    "left_up_leg": COMMON_OFFSET,
    "left_leg": COMMON_OFFSET,
    "left_foot": COMMON_OFFSET,
    "right_up_leg": COMMON_OFFSET,
    "right_leg": COMMON_OFFSET,
    "right_foot": COMMON_OFFSET,
    "head": COMMON_OFFSET,
    "left_arm": np.array([2**-0.5, 0.0, -(2**-0.5), 0.0]),
    "left_fore_arm": np.array([1.0, 0.0, 0.0, 0.0]),
    "left_hand": np.array([1.0, 0.0, 0.0, 0.0]),
    "right_arm": np.array([0.0, 2**-0.5, 0.0, 2**-0.5]),
    "right_fore_arm": np.array([0.0, 0.0, 0.0, 1.0]),
    "right_hand": np.array([0.0, 0.0, 0.0, 1.0]),
}


@dataclass(frozen=True)
class RobotRetargetResult:
    root_position_m: np.ndarray
    root_quaternion_xyzw: np.ndarray
    joint_position_rad: np.ndarray
    model: mj.MjModel
    diagnostics: dict


def _body_position(model: mj.MjModel, data: mj.MjData, name: str) -> np.ndarray:
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(
            f"G1 model is missing body required by robot_retargeter: {name}"
        )
    return data.xpos[body_id].copy()


def _robot_anchors(model: mj.MjModel) -> dict[str, np.ndarray]:
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    anchors = {
        name: _body_position(model, data, name)
        for name in {entry[3] for entry in LINK_CHAINS}
        | {entry[4] for entry in LINK_CHAINS}
        if not name.endswith("_anchor")
    }
    pelvis = _body_position(model, data, "pelvis")
    torso = _body_position(model, data, "torso_link")
    anchors["hips_anchor"] = pelvis + np.array([0.0, 0.0, -0.133165])
    anchors["neck_anchor"] = torso + np.array([0.0, 0.0, 0.247])
    anchors["head_anchor"] = anchors["neck_anchor"] + np.array([0.0, 0.0, 0.16])
    return anchors


def _extract_source(
    frames: Sequence[Mapping[str, Sequence[np.ndarray]]],
) -> tuple[dict, dict]:
    positions = {
        semantic: np.asarray([frame[bvh_name][0] for frame in frames], dtype=np.float64)
        for semantic, bvh_name in SOURCE_BODIES.items()
    }
    quaternions = {
        semantic: np.asarray([frame[bvh_name][1] for frame in frames], dtype=np.float64)
        for semantic, bvh_name in SOURCE_BODIES.items()
    }
    positions["hips_mean"] = 0.5 * (
        positions["left_up_leg"] + positions["right_up_leg"]
    )
    positions["shoulder_mean"] = 0.5 * (positions["left_arm"] + positions["right_arm"])
    return positions, quaternions


def build_scaled_keypoints(
    frames: Sequence[Mapping[str, Sequence[np.ndarray]]],
    model: mj.MjModel,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    """Convert GMR BVH frames into upstream-style robot-scaled keypoints."""

    if not frames:
        raise ValueError("robot_retargeter requires at least one BVH frame")
    source_positions, source_quaternions = _extract_source(frames)
    anchors = _robot_anchors(model)
    scaled_source_positions = {"hips_mean": source_positions["hips_mean"].copy()}
    scaled_positions = {}
    link_lengths = {}

    for (
        link_name,
        source_parent,
        source_child,
        robot_parent,
        robot_child,
    ) in LINK_CHAINS:
        source_vector = source_positions[source_child] - source_positions[source_parent]
        source_length = np.linalg.norm(source_vector, axis=1)
        if np.any(source_length <= 1.0e-8):
            raise ValueError(f"BVH has a near-zero source link: {link_name}")
        robot_length = float(
            np.linalg.norm(anchors[robot_child] - anchors[robot_parent])
        )
        link_lengths[link_name] = robot_length
        scaled_source_positions[source_child] = (
            scaled_source_positions[source_parent]
            + source_vector * (robot_length / source_length)[:, None]
        )
        scaled_positions[link_name] = scaled_source_positions[source_child]

    # The upstream hips task targets a helper body below the pelvis. Shift the
    # target so the standard GMR pelvis body reaches the equivalent world pose.
    scaled_positions["hips_mean"] = scaled_source_positions["hips_mean"] + np.array(
        [0.0, 0.0, 0.133165]
    )

    target_quaternions = {}
    for keypoint_name, (
        _robot_body,
        _pos_cost,
        _rot_cost,
        source_name,
    ) in IK_MATCH_TABLE.items():
        source_xyzw = source_quaternions[source_name][:, [1, 2, 3, 0]]
        offset_xyzw = ORIENTATION_OFFSETS[source_name][[1, 2, 3, 0]]
        adjusted_xyzw = (
            Rotation.from_quat(source_xyzw) * Rotation.from_quat(offset_xyzw)
        ).as_quat()
        target_quaternions[keypoint_name] = adjusted_xyzw[:, [3, 0, 1, 2]]

    return scaled_positions, target_quaternions, link_lengths


def _apply_upstream_limit_offsets(model: mj.MjModel) -> None:
    for joint_id in range(model.njnt):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id) or ""
        offset_degrees = (
            10.0 if "knee_joint" in name else 20.0 if "elbow_joint" in name else 0.0
        )
        if offset_degrees:
            model.jnt_range[joint_id, 0] = min(
                model.jnt_range[joint_id, 1],
                model.jnt_range[joint_id, 0] + np.deg2rad(offset_degrees),
            )


def retarget_bvh_frames(
    frames: Sequence[Mapping[str, Sequence[np.ndarray]]],
    motion_fps: float,
    *,
    model_path: str | Path = DEFAULT_G1_XML,
    solver: str = "daqp",
    damping: float = 1.0,
    max_iterations: int = 50,
    error_tolerance: float = 1.0e-3,
) -> RobotRetargetResult:
    """Run the integrated robot_retargeter keypoint IK on parsed BVH frames."""

    if not np.isfinite(motion_fps) or motion_fps <= 0.0:
        raise ValueError(f"motion_fps must be finite and positive, got {motion_fps}")
    model = mj.MjModel.from_xml_path(str(model_path))
    _apply_upstream_limit_offsets(model)
    configuration = mink.Configuration(model)
    limits = [mink.ConfigurationLimit(model)]
    positions, quaternions, link_lengths = build_scaled_keypoints(frames, model)

    tasks = {}
    for keypoint_name, (
        robot_body,
        position_cost,
        orientation_cost,
        _source_name,
    ) in IK_MATCH_TABLE.items():
        tasks[keypoint_name] = mink.FrameTask(
            frame_name=robot_body,
            frame_type="body",
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            lm_damping=1.0,
        )

    initial_qpos = configuration.q.copy()
    initial_qpos[:3] = positions["hips_mean"][0]
    initial_qpos[3:7] = quaternions["hips_mean"][0]
    configuration.update(initial_qpos)

    qpos_frames = []
    errors = []
    iterations = []
    stabilized_frames = 0
    for frame_index in range(len(frames)):
        for keypoint_name, task in tasks.items():
            task.set_target(
                mink.SE3.from_rotation_and_translation(
                    mink.SO3(quaternions[keypoint_name][frame_index]),
                    positions[keypoint_name][frame_index],
                )
            )

        task_list = list(tasks.values())
        current_error = float(
            np.linalg.norm(
                np.concatenate(
                    [task.compute_error(configuration) for task in task_list]
                )
            )
        )
        iteration_count = 0
        while (
            iteration_count < max(1, int(max_iterations))
            and current_error > error_tolerance
        ):
            dt = configuration.model.opt.timestep
            velocity = mink.solve_ik(
                configuration,
                task_list,
                dt,
                solver,
                damping,
                limits=limits,
            )
            configuration.integrate_inplace(velocity, dt)
            next_error = float(
                np.linalg.norm(
                    np.concatenate(
                        [task.compute_error(configuration) for task in task_list]
                    )
                )
            )
            iteration_count += 1
            if current_error - next_error <= 1.0e-3:
                stabilized_frames += 1
                current_error = next_error
                break
            current_error = next_error

        qpos_frames.append(configuration.q.copy())
        errors.append(current_error)
        iterations.append(iteration_count)

    qpos = np.asarray(qpos_frames, dtype=np.float64)
    diagnostics = {
        "upstream_repository": "https://github.com/ccrpRepo/robot_retargeter",
        "upstream_revision": "f1418972319287c1b93af0f7a3b445f613cff5e4",
        "solver": solver,
        "max_iterations": int(max_iterations),
        "error_tolerance": float(error_tolerance),
        "error_mean": float(np.mean(errors)),
        "error_p95": float(np.percentile(errors, 95)),
        "error_max": float(np.max(errors)),
        "iterations_mean": float(np.mean(iterations)),
        "iterations_max": int(np.max(iterations)),
        "stabilized_frames": int(stabilized_frames),
        "link_lengths_m": link_lengths,
    }
    return RobotRetargetResult(
        root_position_m=qpos[:, :3],
        root_quaternion_xyzw=qpos[:, 3:7][:, [1, 2, 3, 0]],
        joint_position_rad=qpos[:, 7:],
        model=model,
        diagnostics=diagnostics,
    )
