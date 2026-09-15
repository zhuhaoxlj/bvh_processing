"""Convert BVH motion to the HumanML3D representation used by the MDM checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from bvh_processing.vendor.mdm.workbench.bvh import BVHMotion, read_bvh, world_positions
from bvh_processing.vendor.mdm.data.humanml.common.quaternion import qrot_np
from bvh_processing.vendor.mdm.data.humanml.common.skeleton import Skeleton
from bvh_processing.vendor.mdm.data.humanml.scripts.motion_process import extract_features, recover_from_ric
from bvh_processing.vendor.mdm.data.humanml.param_util import t2m_kinematic_chain, t2m_raw_offsets
from bvh_processing.vendor.mdm.data.humanml_utils import HML_JOINT_NAMES

__all__ = ["ConversionResult", "HumanMLAssets", "convert_bvh", "joint_mapping", "load_assets"]

DEFAULT_RESOURCE_ROOT = Path(__file__).resolve().parents[3] / "resources/mdm"
TARGET_FPS = 20
FACE_JOINTS = [2, 1, 17, 16]
RAW_OFFSETS = torch.from_numpy(t2m_raw_offsets.astype(np.float32))
PARENTS = [-1] * 22
for _chain in t2m_kinematic_chain:
    for _parent, _child in zip(_chain, _chain[1:]):
        PARENTS[_child] = _parent

BODY_MAP = {
    "pelvis": "Hips", "left_hip": "LeftUpLeg", "right_hip": "RightUpLeg",
    "left_knee": "LeftLeg", "right_knee": "RightLeg",
    "left_ankle": "LeftFoot", "right_ankle": "RightFoot",
    "left_foot": "LeftToeBase", "right_foot": "RightToeBase",
    "neck": "Neck", "head": "Head",
    "left_collar": "LeftShoulder", "right_collar": "RightShoulder",
    "left_shoulder": "LeftArm", "right_shoulder": "RightArm",
    "left_elbow": "LeftForeArm", "right_elbow": "RightForeArm",
    "left_wrist": "LeftHand", "right_wrist": "RightHand",
}


@dataclass(frozen=True)
class HumanMLAssets:
    target_offsets: torch.Tensor
    mean: np.ndarray
    std: np.ndarray
    hashes: dict[str, str]


@dataclass(frozen=True)
class ConversionResult:
    features: np.ndarray
    normalized: np.ndarray
    joints: np.ndarray
    source_joints: np.ndarray
    metadata: dict

    @property
    def mdm_input(self) -> np.ndarray:
        return self.normalized.T[None, :, None, :].copy()


@lru_cache(maxsize=4)
def load_assets(
    data_root: Path = DEFAULT_RESOURCE_ROOT / "humanml",
) -> HumanMLAssets:
    data_root = Path(data_root)
    paths = {name: data_root / name for name in ["Mean.npy", "Std.npy", "new_joints/000021.npy"]}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError("缺少 HumanML3D 参考数据：" + "、".join(missing))
    arrays = {name: np.load(path, allow_pickle=False) for name, path in paths.items()}
    mean, std, reference = arrays.values()
    if mean.shape != (263,) or std.shape != (263,) or not np.all(std > 0):
        raise ValueError("Mean.npy / Std.npy 必须是模型配套的 263 维统计量，且标准差大于零")
    if reference.ndim != 3 or reference.shape[1:] != (22, 3):
        raise ValueError("参考骨架 000021.npy 必须包含 22 个三维关节")
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("HumanML3D 参考文件包含无效数值")
    skeleton = Skeleton(RAW_OFFSETS, t2m_kinematic_chain, "cpu")
    offsets = skeleton.get_offsets_joints(torch.from_numpy(reference[0]).float())
    return HumanMLAssets(offsets, mean, std, {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()})


def joint_mapping(motion: BVHMotion) -> tuple[list[int], dict[str, str], str]:
    # Mixamo namespaces are labels, not different skeletons.
    names: dict[str, int] = {}
    for index, joint in enumerate(motion.joints):
        name = joint.name.rsplit(":", 1)[-1].lower()
        if name in names:
            raise ValueError(f"去掉命名空间后关节名重复：{joint.name}")
        names[name] = index
    if all(name in names for name in HML_JOINT_NAMES):
        mapping = {name: name for name in HML_JOINT_NAMES}
        profile = "HumanML3D"
    else:
        mapping = dict(BODY_MAP)
        if "spine3" in names:
            # Four Nokov spine joints -> three target joints. FK has already
            # propagated Spine1's motion to Spine2 and Spine3 before selection.
            mapping.update(spine1="Spine", spine2="Spine2", spine3="Spine3")
            profile = "Nokov / ToeBase"
        else:
            mapping.update(spine1="Spine", spine2="Spine1", spine3="Spine2")
            profile = "Mixamo / BVH"
    missing = [source for source in mapping.values() if source.lower() not in names]
    if missing:
        raise ValueError("暂不支持此骨架，缺少关节：" + "、".join(missing) + "。支持 Nokov、Mixamo 或 HumanML3D 命名。")
    indices = [names[mapping[name].lower()] for name in HML_JOINT_NAMES]
    if len(set(indices)) != 22:
        raise ValueError("HumanML3D 的 22 个关节不能映射到重复节点")
    for index, parent in enumerate(PARENTS):
        if parent < 0:
            continue
        ancestor = motion.joints[indices[index]].parent
        while ancestor >= 0 and ancestor != indices[parent]:
            ancestor = motion.joints[ancestor].parent
        if ancestor < 0:
            raise ValueError(f"骨架层级不匹配：{mapping[HML_JOINT_NAMES[index]]} 不在预期的父关节下面")
    return indices, {name: motion.joints[index].name for name, index in zip(HML_JOINT_NAMES, indices)}, profile


def _resample(positions: np.ndarray, frame_time: float) -> np.ndarray:
    # A BVH frame covers [t, t + frame_time); this also handles rounded 120 FPS
    # headers without accidentally dropping the exact 2.5-second sample.
    source_t = np.arange(len(positions)) * frame_time
    target_t = np.arange(0, len(positions) * frame_time - 1e-10, 1 / TARGET_FPS)
    if len(target_t) < 3:
        raise ValueError("动作太短：转换到 20 FPS 后至少需要 3 帧坐标")
    flat = positions.reshape(len(positions), -1)
    result = np.stack([np.interp(target_t, source_t, flat[:, column]) for column in range(flat.shape[1])], axis=1)
    return result.reshape(len(target_t), 22, 3)


def _canonicalize(positions: np.ndarray, assets: HumanMLAssets) -> tuple[np.ndarray, dict]:
    skeleton = Skeleton(RAW_OFFSETS, t2m_kinematic_chain, "cpu")
    source_offsets = skeleton.get_offsets_joints(torch.from_numpy(positions[0]).float())
    lengths = torch.linalg.vector_norm(source_offsets, dim=-1)
    if not torch.isfinite(lengths).all() or torch.any(lengths[1:] < 1e-6):
        raise ValueError("骨架包含重合关节，无法计算人体骨长")
    target_lengths = torch.linalg.vector_norm(assets.target_offsets, dim=-1)
    ratio = float((target_lengths[5] + target_lengths[8]) / (lengths[5] + lengths[8]))
    rotations = skeleton.inverse_kinematics_np(positions, FACE_JOINTS, smooth_forward=False)
    if not np.isfinite(rotations).all():
        raise ValueError("输入姿态出现退化方向，无法完成标准骨架转换")
    skeleton.set_offset(assets.target_offsets)
    uniform = skeleton.forward_kinematics_np(rotations, positions[:, 0] * ratio)
    floor = float(uniform[..., 1].min())
    origin = uniform[0, 0] * np.array([1.0, 0.0, 1.0])
    uniform[..., 1] -= floor
    uniform -= origin
    across = (uniform[0, 2] - uniform[0, 1]) + (uniform[0, 17] - uniform[0, 16])
    forward = np.cross([0.0, 1.0, 0.0], across)
    if np.linalg.norm(forward) < 1e-8:
        raise ValueError("无法从髋部和肩部确定朝向，请检查关节映射")
    yaw = float(np.arctan2(forward[0], forward[2]))
    facing = Rotation.from_euler("Y", -yaw).as_matrix()
    uniform = uniform @ facing.T
    return uniform, {"leg_scale": ratio, "floor_height_m": floor, "root_origin_m": origin.tolist(), "facing_matrix": facing.tolist()}


def convert_bvh(
    content: bytes,
    *,
    filename: str = "motion.bvh",
    scale: float = 0.01,
    up_axis: str = "Y",
    remove_initial_rest: bool = True,
    assets: HumanMLAssets | None = None,
) -> ConversionResult:
    start = time.perf_counter()
    if not np.isfinite(scale) or not 1e-5 <= scale <= 10:
        raise ValueError("单位比例必须是 0.00001–10 之间的有限数值")
    if up_axis not in ("Y", "Z"):
        raise ValueError("向上轴必须是 Y 或 Z")
    assets = assets or load_assets()
    motion = read_bvh(content)
    mapping, named_mapping, profile = joint_mapping(motion)
    source, repeated_offsets = world_positions(motion)
    basis = np.eye(3) if up_axis == "Y" else np.asarray([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    source = (source @ basis.T) * scale
    dropped = 0
    if remove_initial_rest:
        angles = motion.frames[:, motion.rotation_columns]
        if np.max(np.abs(angles[0])) < 1e-7 and np.max(np.abs(angles[1])) > 5:
            source = source[1:]
            dropped = 1
    positions = _resample(source[:, mapping], motion.frame_time)
    canonical, transform = _canonicalize(positions, assets)
    # Feature extraction smooths heading. RIC reconstruction starts at heading
    # zero, so compare in that same heading frame instead of mistaking a constant
    # yaw gauge for a conversion error. Preserve the total transform in metadata.
    heading_skeleton = Skeleton(RAW_OFFSETS, t2m_kinematic_chain, "cpu")
    heading = heading_skeleton.inverse_kinematics_np(canonical, FACE_JOINTS, smooth_forward=True)[0, 0]
    expected = qrot_np(np.broadcast_to(heading, (*canonical.shape[:-1], 4)).copy(), canonical)
    features = extract_features(canonical.copy(), 0.002, RAW_OFFSETS, t2m_kinematic_chain, FACE_JOINTS, [8, 11], [7, 10]).astype(np.float32)
    if features.shape != (len(positions) - 1, 263) or not np.isfinite(features).all():
        raise ValueError("转换结果包含无效特征，请检查骨架和动作数据")
    normalized = ((features - assets.mean) / assets.std).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("归一化结果包含无效数值")
    with torch.inference_mode():
        recovered = recover_from_ric(torch.from_numpy(features), 22).numpy()
    roundtrip_error = float(np.max(np.linalg.norm(recovered - expected[:-1], axis=-1)))
    if not np.isfinite(roundtrip_error) or roundtrip_error > 0.005:
        raise ValueError(f"特征恢复检查失败，最大位置误差 {roundtrip_error:.6f} 米")
    target_lengths = torch.linalg.vector_norm(assets.target_offsets, dim=-1).numpy()
    bone_lengths = np.linalg.norm(recovered[:, 1:] - recovered[:, np.asarray(PARENTS[1:])], axis=-1)
    bone_error = float(np.max(np.abs(bone_lengths - target_lengths[1:])))
    # The left preview retains the source bone lengths; only its rigid display
    # frame is aligned with the recovered motion. The original BVH is untouched.
    source_floor = float(source[..., 1].min())
    source_origin = source[0, mapping[0]] * np.asarray([1.0, 0.0, 1.0])
    source -= source_origin
    source[..., 1] -= source_floor
    source = source @ np.asarray(transform["facing_matrix"]).T
    source = qrot_np(np.broadcast_to(heading, (*source.shape[:-1], 4)).copy(), source).astype(np.float32)
    transform["reconstruction_heading_wxyz"] = heading.tolist()
    transform["source_preview_floor_m"] = source_floor
    transform["source_preview_origin_m"] = source_origin.tolist()
    notices = []
    if dropped:
        notices.append("已跳过首帧全零旋转的初始化姿态；原始 BVH 下载文件保持完整。")
    metadata = {
        "source_name": filename,
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "source_frames": len(motion.frames),
        "source_used_frames": len(source),
        "source_fps": motion.fps,
        "source_frame_time": motion.frame_time,
        "source_joint_names": [j.name for j in motion.joints],
        "source_parents": [j.parent for j in motion.joints],
        "source_channels": motion.frames.shape[1],
        "removed_initial_frames": dropped,
        "source_scale_to_meters": scale,
        "source_up_axis": up_axis,
        "source_duration_seconds": len(motion.frames) * motion.frame_time,
        "output_frames": len(features),
        "resampled_position_frames": len(positions),
        "output_duration_seconds": len(features) / TARGET_FPS,
        "target_fps": TARGET_FPS,
        "feature_dimension": 263,
        "joint_names": HML_JOINT_NAMES,
        "parents": PARENTS,
        "joint_map": named_mapping,
        "skeleton_profile": profile,
        "repeated_offset_joints": repeated_offsets,
        "transform": transform,
        "reference_files_sha256": assets.hashes,
        "roundtrip_max_error_m": roundtrip_error,
        "bone_length_max_error_m": bone_error,
        "all_finite": True,
        "feature_shape": list(features.shape),
        "mdm_input_shape": [1, 263, 1, len(features)],
        "feature_slices": {"root": [0, 4], "relative_positions": [4, 67], "rotations_6d": [67, 193], "local_velocities": [193, 259], "foot_contacts": [259, 263]},
        "notes": notices,
        "conversion_seconds": round(time.perf_counter() - start, 3),
        "method": "BVH FK, HumanML3D skeleton retargeting, 20 FPS feature extraction and checkpoint normalization",
        "neural_model_used": False,
    }
    return ConversionResult(features, normalized, recovered.astype(np.float32), source, metadata)
