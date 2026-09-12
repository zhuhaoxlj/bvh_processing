from __future__ import annotations

from io import BytesIO

import mujoco as mj
import numpy as np
import pytest

from bvh_processing.retargeting.exporter import export_tracking_artifacts
from bvh_processing.retargeting.robot_retargeter import (
    SOURCE_BODIES,
    retarget_bvh_frames,
)
from bvh_processing.retargeting.robots import ROBOT_PROFILES, robot_profile


def _standing_frames(frame_count: int = 3) -> list[dict[str, list[np.ndarray]]]:
    identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    rest = {
        "Hips": np.array([0.0, 0.0, 0.90]),
        "LeftUpLeg": np.array([0.0, 0.11, 0.90]),
        "LeftLeg": np.array([0.0, 0.11, 0.50]),
        "LeftFootMod": np.array([0.08, 0.11, 0.06]),
        "RightUpLeg": np.array([0.0, -0.11, 0.90]),
        "RightLeg": np.array([0.0, -0.11, 0.50]),
        "RightFootMod": np.array([0.08, -0.11, 0.06]),
        "Head": np.array([0.0, 0.0, 1.52]),
        "LeftArm": np.array([0.0, 0.22, 1.32]),
        "LeftForeArm": np.array([0.0, 0.22, 1.04]),
        "LeftHand": np.array([0.0, 0.22, 0.78]),
        "RightArm": np.array([0.0, -0.22, 1.32]),
        "RightForeArm": np.array([0.0, -0.22, 1.04]),
        "RightHand": np.array([0.0, -0.22, 0.78]),
    }
    frames = []
    for frame_index in range(frame_count):
        drift = np.array([0.01 * frame_index, 0.0, 0.0])
        frame = {
            name: [position + drift, identity.copy()] for name, position in rest.items()
        }
        frames.append(frame)
    missing = set(SOURCE_BODIES.values()) - set(rest)
    assert not missing
    return frames


def test_robot_profile_lookup() -> None:
    assert robot_profile(1).key == "g1"
    assert robot_profile(2).key == "h2"
    assert robot_profile(3).key == "r1"
    with pytest.raises(ValueError, match="不支持的 robotType"):
        robot_profile(4)


@pytest.mark.parametrize("robot_type", sorted(ROBOT_PROFILES))
def test_robot_models_expose_ik_bodies(robot_type: int) -> None:
    robot = robot_profile(robot_type)
    model = mj.MjModel.from_xml_path(str(robot.xml_path))
    required = {body for _key, (body, *_rest) in robot.ik_match_table.items()}
    required.update(
        name
        for entry in robot.link_chains
        for name in entry[3:5]
        if not name.endswith("_anchor")
    )
    missing = [
        name
        for name in sorted(required)
        if mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name) < 0
    ]
    assert missing == []
    hinge_count = sum(
        1
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] == mj.mjtJoint.mjJNT_HINGE
    )
    expected = {1: 29, 2: 31, 3: 26}[robot_type]
    assert hinge_count == expected


@pytest.mark.parametrize("robot_type", sorted(ROBOT_PROFILES))
def test_retarget_dummy_frames_for_each_robot(robot_type: int) -> None:
    robot = robot_profile(robot_type)
    result = retarget_bvh_frames(
        _standing_frames(),
        30.0,
        robot=robot,
        max_iterations=8,
    )
    assert result.robot.key == robot.key
    assert np.isfinite(result.root_position_m).all()
    assert np.isfinite(result.joint_position_rad).all()
    assert result.joint_position_rad.shape[0] == 3

    artifacts = export_tracking_artifacts(
        result,
        30.0,
        "walk.bvh",
        "lafan1",
    )
    try:
        assert artifacts.npz_filename == f"walk_{robot.key}_tracking.npz"
        assert artifacts.preview_filename == f"walk_{robot.key}_preview.json"
        npz = np.load(BytesIO(artifacts.npz.read()))
        assert npz["joint_pos"].shape[0] >= 2
        assert npz["joint_pos"].shape[1] == result.joint_position_rad.shape[1] or (
            robot.tracking_joint_names is not None
            and npz["joint_pos"].shape[1] == len(robot.tracking_joint_names)
        )
    finally:
        artifacts.close()
