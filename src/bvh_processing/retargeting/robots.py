"""Robot Retargeter 的机型配置：G1 / H2 / R1。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_ASSETS = Path(__file__).with_name("assets")

# BVH 语义连杆 → (source_parent, source_child)，三台机器人共用。
_SOURCE_LINKS = {
    "left_hip": ("hips_mean", "left_up_leg"),
    "left_thigh": ("left_up_leg", "left_leg"),
    "left_calf": ("left_leg", "left_foot"),
    "right_hip": ("hips_mean", "right_up_leg"),
    "right_thigh": ("right_up_leg", "right_leg"),
    "right_calf": ("right_leg", "right_foot"),
    "neck": ("hips_mean", "shoulder_mean"),
    "head": ("shoulder_mean", "head"),
    "left_shoulder": ("shoulder_mean", "left_arm"),
    "left_arm": ("left_arm", "left_fore_arm"),
    "left_fore_arm": ("left_fore_arm", "left_hand"),
    "right_shoulder": ("shoulder_mean", "right_arm"),
    "right_arm": ("right_arm", "right_fore_arm"),
    "right_fore_arm": ("right_fore_arm", "right_hand"),
}

_IK_SOURCE = {
    "hips_mean": "hips",
    "left_hip": "left_up_leg",
    "left_thigh": "left_leg",
    "left_calf": "left_foot",
    "right_hip": "right_up_leg",
    "right_thigh": "right_leg",
    "right_calf": "right_foot",
    "head": "head",
    "left_shoulder": "left_arm",
    "left_arm": "left_fore_arm",
    "left_fore_arm": "left_hand",
    "right_shoulder": "right_arm",
    "right_arm": "right_fore_arm",
    "right_fore_arm": "right_hand",
}

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


def _link_chains(
    *robot_links: tuple[str, str, str],
) -> tuple[tuple[str, str, str, str, str], ...]:
    chains = []
    for link_name, robot_parent, robot_child in robot_links:
        source_parent, source_child = _SOURCE_LINKS[link_name]
        chains.append(
            (link_name, source_parent, source_child, robot_parent, robot_child)
        )
    return tuple(chains)


def _ik_table(
    mapping: dict[str, tuple[str, float, float]],
) -> dict[str, tuple[str, float, float, str]]:
    return {
        keypoint: (body, pos_cost, rot_cost, _IK_SOURCE[keypoint])
        for keypoint, (body, pos_cost, rot_cost) in mapping.items()
    }


@dataclass(frozen=True)
class RobotProfile:
    type_id: int
    key: str
    display_name: str
    preview_robot: str
    xml_path: Path
    preview_mesh_geoms_path: Path | None
    link_chains: tuple[tuple[str, str, str, str, str], ...]
    ik_match_table: dict[str, tuple[str, float, float, str]]
    hips_position_offset: tuple[float, float, float]
    tracking_joint_names: tuple[str, ...] | None
    tracking_body_names: tuple[str, ...] | None

    @property
    def artifact_stem(self) -> str:
        return self.key


G1 = RobotProfile(
    type_id=1,
    key="g1",
    display_name="G1",
    preview_robot="unitree_g1",
    xml_path=_ASSETS / "g1_mocap_29dof.xml",
    preview_mesh_geoms_path=_ASSETS / "g1_preview_mesh_geoms.json",
    link_chains=_link_chains(
        ("left_hip", "hips_anchor", "left_hip_roll_link"),
        ("left_thigh", "left_hip_roll_link", "left_knee_link"),
        ("left_calf", "left_knee_link", "left_ankle_roll_link"),
        ("right_hip", "hips_anchor", "right_hip_roll_link"),
        ("right_thigh", "right_hip_roll_link", "right_knee_link"),
        ("right_calf", "right_knee_link", "right_ankle_roll_link"),
        ("neck", "hips_anchor", "neck_anchor"),
        ("head", "neck_anchor", "head_anchor"),
        ("left_shoulder", "neck_anchor", "left_shoulder_roll_link"),
        ("left_arm", "left_shoulder_roll_link", "left_elbow_link"),
        ("left_fore_arm", "left_elbow_link", "left_wrist_yaw_link"),
        ("right_shoulder", "neck_anchor", "right_shoulder_roll_link"),
        ("right_arm", "right_shoulder_roll_link", "right_elbow_link"),
        ("right_fore_arm", "right_elbow_link", "right_wrist_yaw_link"),
    ),
    ik_match_table=_ik_table(
        {
            "hips_mean": ("pelvis", 100.0, 0.0),
            "left_hip": ("left_hip_roll_link", 30.0, 3.0),
            "left_thigh": ("left_knee_link", 0.0, 3.0),
            "left_calf": ("left_ankle_roll_link", 30.0, 3.0),
            "right_hip": ("right_hip_roll_link", 30.0, 3.0),
            "right_thigh": ("right_knee_link", 0.0, 3.0),
            "right_calf": ("right_ankle_roll_link", 30.0, 3.0),
            "head": ("torso_link", 0.0, 3.0),
            "left_shoulder": ("left_shoulder_roll_link", 30.0, 3.0),
            "left_arm": ("left_elbow_link", 10.0, 1.0),
            "left_fore_arm": ("left_wrist_yaw_link", 10.0, 1.0),
            "right_shoulder": ("right_shoulder_roll_link", 30.0, 3.0),
            "right_arm": ("right_elbow_link", 10.0, 1.0),
            "right_fore_arm": ("right_wrist_yaw_link", 10.0, 1.0),
        }
    ),
    hips_position_offset=(0.0, 0.0, 0.133165),
    tracking_joint_names=G1_ISAACLAB_JOINT_NAMES,
    tracking_body_names=G1_ISAACLAB_BODY_NAMES,
)

H2 = RobotProfile(
    type_id=2,
    key="h2",
    display_name="H2",
    preview_robot="unitree_h2",
    xml_path=_ASSETS / "h2_mocap.xml",
    preview_mesh_geoms_path=_ASSETS / "h2_preview_mesh_geoms.json",
    link_chains=_link_chains(
        ("left_hip", "hips_sphere", "left_hip_roll_link"),
        ("left_thigh", "left_hip_roll_link", "left_knee_link"),
        ("left_calf", "left_knee_link", "left_ankle_roll_link"),
        ("right_hip", "hips_sphere", "right_hip_roll_link"),
        ("right_thigh", "right_hip_roll_link", "right_knee_link"),
        ("right_calf", "right_knee_link", "right_ankle_roll_link"),
        ("neck", "hips_sphere", "neck_sphere"),
        ("head", "neck_sphere", "head_yaw_link"),
        ("left_shoulder", "neck_sphere", "left_shoulder_roll_link"),
        ("left_arm", "left_shoulder_roll_link", "left_elbow_link"),
        ("left_fore_arm", "left_elbow_link", "left_wrist_yaw_link"),
        ("right_shoulder", "neck_sphere", "right_shoulder_roll_link"),
        ("right_arm", "right_shoulder_roll_link", "right_elbow_link"),
        ("right_fore_arm", "right_elbow_link", "right_wrist_yaw_link"),
    ),
    ik_match_table=_ik_table(
        {
            "hips_mean": ("hips_sphere", 100.0, 0.0),
            "left_hip": ("left_hip_roll_link", 30.0, 3.0),
            "left_thigh": ("left_knee_link", 0.0, 3.0),
            "left_calf": ("left_ankle_pitch_link", 30.0, 3.0),
            "right_hip": ("right_hip_roll_link", 30.0, 3.0),
            "right_thigh": ("right_knee_link", 0.0, 3.0),
            "right_calf": ("right_ankle_pitch_link", 30.0, 3.0),
            "head": ("head_yaw_link", 0.0, 3.0),
            "left_shoulder": ("left_shoulder_roll_link", 30.0, 3.0),
            "left_arm": ("left_elbow_link", 10.0, 3.0),
            "left_fore_arm": ("left_wrist_yaw_link", 10.0, 1.0),
            "right_shoulder": ("right_shoulder_roll_link", 30.0, 3.0),
            "right_arm": ("right_elbow_link", 10.0, 3.0),
            "right_fore_arm": ("right_wrist_yaw_link", 10.0, 1.0),
        }
    ),
    hips_position_offset=(0.0, 0.0, 0.0),
    tracking_joint_names=None,
    tracking_body_names=None,
)

R1 = RobotProfile(
    type_id=3,
    key="r1",
    display_name="R1",
    preview_robot="unitree_r1",
    xml_path=_ASSETS / "r1_mocap.xml",
    preview_mesh_geoms_path=_ASSETS / "r1_preview_mesh_geoms.json",
    link_chains=_link_chains(
        ("left_hip", "hips_sphere", "left_hip_pitch_link"),
        ("left_thigh", "left_hip_pitch_link", "left_knee_link"),
        ("left_calf", "left_knee_link", "left_ankle_roll_link"),
        ("right_hip", "hips_sphere", "right_hip_pitch_link"),
        ("right_thigh", "right_hip_pitch_link", "right_knee_link"),
        ("right_calf", "right_knee_link", "right_ankle_roll_link"),
        ("neck", "hips_sphere", "neck_sphere"),
        ("head", "neck_sphere", "head_yaw_link"),
        ("left_shoulder", "neck_sphere", "left_shoulder_sphere"),
        ("left_arm", "left_shoulder_sphere", "left_elbow_link"),
        ("left_fore_arm", "left_elbow_link", "left_hand"),
        ("right_shoulder", "neck_sphere", "right_shoulder_sphere"),
        ("right_arm", "right_shoulder_sphere", "right_elbow_link"),
        ("right_fore_arm", "right_elbow_link", "right_hand"),
    ),
    ik_match_table=_ik_table(
        {
            "hips_mean": ("hips_sphere", 100.0, 0.0),
            "left_hip": ("left_hip_pitch_link", 30.0, 3.0),
            "left_thigh": ("left_knee_link", 0.0, 3.0),
            "left_calf": ("left_ankle_roll_link", 30.0, 3.0),
            "right_hip": ("right_hip_pitch_link", 30.0, 3.0),
            "right_thigh": ("right_knee_link", 0.0, 3.0),
            "right_calf": ("right_ankle_roll_link", 30.0, 3.0),
            "head": ("head_yaw_link", 0.0, 3.0),
            "left_shoulder": ("left_shoulder_sphere", 30.0, 3.0),
            "left_arm": ("left_elbow_link", 10.0, 3.0),
            "left_fore_arm": ("left_hand", 10.0, 1.0),
            "right_shoulder": ("right_shoulder_sphere", 30.0, 3.0),
            "right_arm": ("right_elbow_link", 10.0, 3.0),
            "right_fore_arm": ("right_hand", 10.0, 1.0),
        }
    ),
    hips_position_offset=(0.0, 0.0, 0.0),
    tracking_joint_names=None,
    tracking_body_names=None,
)

ROBOT_PROFILES: dict[int, RobotProfile] = {
    G1.type_id: G1,
    H2.type_id: H2,
    R1.type_id: R1,
}


def robot_profile(robot_type: int) -> RobotProfile:
    try:
        return ROBOT_PROFILES[robot_type]
    except KeyError as exc:
        supported = ", ".join(
            f"{item.type_id}={item.display_name}" for item in ROBOT_PROFILES.values()
        )
        raise ValueError(
            f"不支持的 robotType={robot_type}，当前支持 {supported}"
        ) from exc


__all__ = [
    "G1",
    "G1_ISAACLAB_BODY_NAMES",
    "G1_ISAACLAB_JOINT_NAMES",
    "H2",
    "R1",
    "ROBOT_PROFILES",
    "RobotProfile",
    "robot_profile",
]
