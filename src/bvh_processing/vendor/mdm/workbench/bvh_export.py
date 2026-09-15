"""Retarget HumanML3D transition results back onto the source-A BVH skeleton."""

from __future__ import annotations

from dataclasses import dataclass
import re
import warnings

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from bvh_processing.vendor.mdm.workbench.bvh import BVHMotion, read_bvh, world_positions
from bvh_processing.vendor.mdm.workbench.conversion import HML_JOINT_NAMES, TARGET_FPS, joint_mapping
from bvh_processing.vendor.mdm.data.humanml.common.quaternion import cont6d_to_matrix_np
from bvh_processing.vendor.mdm.data.humanml.param_util import t2m_raw_offsets

HML_FEATURE_DIM = 263
HML_ROTATION_START = 4 + (len(HML_JOINT_NAMES) - 1) * 3
HML_ROTATION_DIM = (len(HML_JOINT_NAMES) - 1) * 6
BOUNDARY_BLEND_FRAMES = 6


@dataclass(frozen=True)
class BVHExport:
    combined: bytes
    transition: bytes
    metadata: dict


def _validate_matching_skeletons(first: BVHMotion, second: BVHMotion) -> None:
    if [joint.name for joint in first.joints] != [
        joint.name for joint in second.joints
    ]:
        raise ValueError("动作 A/B 的 BVH 关节名称或顺序不同，无法输出同一骨架的拼接文件")
    if [joint.parent for joint in first.joints] != [
        joint.parent for joint in second.joints
    ]:
        raise ValueError("动作 A/B 的 BVH 层级不同，无法输出同一骨架的拼接文件")


def append_bvh_without_transition(first: bytes, second: bytes) -> bytes:
    """Align and append a clip when the caller explicitly requests no gap."""
    template = read_bvh(first)
    following = read_bvh(second)
    _validate_matching_skeletons(template, following)
    if not np.isclose(
        template.frame_time,
        following.frame_time,
        rtol=1e-6,
        atol=1e-9,
    ):
        raise ValueError("动作 A/B 的 BVH 帧率不同，无法直接拼接")

    first_root, first_local = _motion_components(template)
    second_root, second_local = _motion_components(following)
    second_root, second_local = _align_following(
        second_root,
        second_local,
        first_root[-1],
        first_local[-1, 0],
    )
    second_frames = _motion_frames(template, second_root, second_local)
    return _write_with_template(
        first,
        np.concatenate((template.frames, second_frames)),
        template.frame_time,
    )


def _source_to_y_up_basis(metadata: dict) -> np.ndarray:
    return (
        np.eye(3)
        if metadata["source_up_axis"] == "Y"
        else np.asarray([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    )


def _inverse_conversion_rotation(metadata: dict) -> Rotation:
    transform = metadata["transform"]
    heading = np.asarray(transform["reconstruction_heading_wxyz"], dtype=np.float64)
    heading_rotation = Rotation.from_quat(heading[[1, 2, 3, 0]])
    facing_rotation = Rotation.from_matrix(np.asarray(transform["facing_matrix"], dtype=np.float64))
    return (
        Rotation.from_matrix(_source_to_y_up_basis(metadata)).inv()
        * facing_rotation.inv()
        * heading_rotation.inv()
    )


def _source_root_positions(joints: np.ndarray, metadata: dict) -> np.ndarray:
    transform = metadata["transform"]
    heading = np.asarray(transform["reconstruction_heading_wxyz"], dtype=np.float64)
    root = Rotation.from_quat(heading[[1, 2, 3, 0]]).inv().apply(joints[:, 0])
    facing = Rotation.from_matrix(np.asarray(transform["facing_matrix"], dtype=np.float64))
    root = facing.inv().apply(root)
    offset = np.asarray(transform["root_origin_m"], dtype=np.float64)
    offset[1] = transform["floor_height_m"]
    root = (root + offset) / transform["leg_scale"]
    return (
        Rotation.from_matrix(_source_to_y_up_basis(metadata)).inv().apply(root)
        / metadata["source_scale_to_meters"]
    )


def _target_positions(joints: np.ndarray, template: BVHMotion, metadata: dict) -> np.ndarray:
    mapping, _, profile = joint_mapping(template)
    rotated = _inverse_conversion_rotation(metadata).apply(joints.reshape(-1, 3)).reshape(joints.shape)
    root = _source_root_positions(joints, metadata)
    transform = metadata["transform"]
    source_units_per_canonical_meter = 1 / (
        transform["leg_scale"] * metadata["source_scale_to_meters"]
    )
    rotated = (
        rotated - rotated[:, :1]
    ) * source_units_per_canonical_meter + root[:, None]
    target = np.full((len(joints), len(template.joints), 3), np.nan, dtype=np.float64)
    for hml_index, source_index in enumerate(mapping):
        target[:, source_index] = rotated[:, hml_index]

    if profile == "Nokov / ToeBase":
        names = {
            joint.name.rsplit(":", 1)[-1].lower(): index
            for index, joint in enumerate(template.joints)
        }
        missing = names["spine1"]
        lower, upper = names["spine"], names["spine2"]
        lower_length = np.linalg.norm(template.joints[missing].offset)
        upper_length = np.linalg.norm(template.joints[upper].offset)
        fraction = lower_length / (lower_length + upper_length)
        target[:, missing] = target[:, lower] + fraction * (
            target[:, upper] - target[:, lower]
        )
    if not np.isfinite(target).all():
        raise ValueError(
            f"{profile} 骨架含有模型未映射的附加关节，无法反向导出"
        )
    target[:, 0] = root
    return target


def _rotation_from_directions(rest: np.ndarray, posed: np.ndarray) -> Rotation:
    rest = rest / np.linalg.norm(rest, axis=1, keepdims=True)
    posed = posed / np.linalg.norm(posed, axis=1, keepdims=True)
    return Rotation.align_vectors(posed, rest)[0]


def _validate_source_coordinates(source_a_metadata: dict, source_b_metadata: dict) -> None:
    if source_a_metadata["source_up_axis"] != source_b_metadata["source_up_axis"]:
        raise ValueError("动作 A/B 的向上轴设置不同，请使用相同向上轴重新转换")
    if not np.isclose(
        source_a_metadata["source_scale_to_meters"],
        source_b_metadata["source_scale_to_meters"],
        rtol=1e-9,
        atol=0,
    ):
        raise ValueError("动作 A/B 的单位比例不同，请使用相同单位比例重新转换")


def _joint_children(template: BVHMotion) -> list[list[int]]:
    children: list[list[int]] = [[] for _ in template.joints]
    for child, joint in enumerate(template.joints):
        if joint.parent >= 0:
            children[joint.parent].append(child)
    return children


def _normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    result = quaternions.copy()
    norms = np.linalg.norm(result, axis=1)
    result[norms < 1e-8] = [0, 0, 0, 1]
    return result / np.linalg.norm(result, axis=1, keepdims=True)


def _solve_local_rotations(target: np.ndarray, template: BVHMotion) -> np.ndarray:
    children = _joint_children(template)
    local = np.broadcast_to(np.eye(3), (len(target), len(template.joints), 3, 3)).copy()
    for frame in range(len(target)):
        global_rotations = [Rotation.identity() for _ in template.joints]
        for index, joint in enumerate(template.joints):
            if children[index]:
                rest = np.asarray([template.joints[child].offset for child in children[index]])
                posed = np.asarray([target[frame, child] - target[frame, index] for child in children[index]])
                if np.any(np.linalg.norm(posed, axis=1) < 1e-8):
                    raise ValueError(f"生成动作的关节 {joint.name} 与子关节重合")
                global_rotation = _rotation_from_directions(rest, posed)
            elif joint.parent >= 0:
                global_rotation = global_rotations[joint.parent]
            else:
                global_rotation = Rotation.identity()
            global_rotations[index] = global_rotation
            parent_rotation = Rotation.identity() if joint.parent < 0 else global_rotations[joint.parent]
            local[frame, index] = (parent_rotation.inv() * global_rotation).as_matrix()
    return local


def _apply_hml_twist_deltas(
    local: np.ndarray, features: np.ndarray, template: BVHMotion
) -> tuple[np.ndarray, list[str]]:
    if features.shape != (len(local), HML_FEATURE_DIM):
        raise ValueError("MDM 旋转特征必须与过渡帧数一致，且每帧为 263 维")
    hml_local = cont6d_to_matrix_np(
        features[:, HML_ROTATION_START:HML_ROTATION_START + HML_ROTATION_DIM].reshape(
            len(features), -1, 6
        )
    ).astype(np.float64)
    if not np.isfinite(hml_local).all():
        raise ValueError("MDM 过渡包含无效的 6D 旋转")

    mapping, _, _ = joint_mapping(template)
    children = _joint_children(template)
    source_to_hml = {source: hml for hml, source in enumerate(mapping)}

    result = local.copy()
    driven = []
    for source_parent, source_children in enumerate(children):
        if len(source_children) != 1:
            continue
        source_child = source_children[0]
        hml_child = source_to_hml.get(source_child)
        if hml_child is None or hml_child == 0:
            continue
        generated = Rotation.from_matrix(hml_local[:, hml_child - 1])
        hml_axis = t2m_raw_offsets[hml_child].astype(np.float64)
        hml_axis /= np.linalg.norm(hml_axis)
        generated_quaternions = generated.as_quat()
        hml_twist_quaternions = np.concatenate(
            (
                (generated_quaternions[:, :3] @ hml_axis)[:, None] * hml_axis,
                generated_quaternions[:, 3:4],
            ),
            axis=1,
        )
        hml_twist_quaternions = _normalize_quaternions(hml_twist_quaternions)
        hml_twists = Rotation.from_quat(hml_twist_quaternions)
        delta_quaternions = (hml_twists[0].inv() * hml_twists).as_quat()
        signed_sine = delta_quaternions[:, :3] @ hml_axis
        source_axis = template.joints[source_child].offset.astype(np.float64)
        source_axis /= np.linalg.norm(source_axis)
        source_quaternions = np.concatenate(
            (signed_sine[:, None] * source_axis, delta_quaternions[:, 3:4]), axis=1
        )
        source_quaternions = _normalize_quaternions(source_quaternions)
        twist = Rotation.from_quat(source_quaternions)
        result[:, source_parent] = (
            Rotation.from_matrix(result[:, source_parent]) * twist
        ).as_matrix()
        driven.append(template.joints[source_parent].name)
    return result, driven


def _interpolate_leaf_rotations(
    local: np.ndarray, start: np.ndarray, end: np.ndarray, template: BVHMotion
) -> tuple[np.ndarray, list[str]]:
    result = local.copy()
    times = np.linspace(0, 1, len(local) + 2)[1:-1]
    driven = []
    for index in range(len(template.joints)):
        if any(joint.parent == index for joint in template.joints):
            continue
        endpoints = Rotation.from_matrix(np.stack((start[index], end[index])))
        result[:, index] = Slerp([0, 1], endpoints)(times).as_matrix()
        driven.append(template.joints[index].name)
    return result, driven


def _smoothstep(value: np.ndarray) -> np.ndarray:
    return value * value * (3 - 2 * value)


def _stitch_boundaries(
    root: np.ndarray,
    local: np.ndarray,
    transition_range: tuple[int, int] | list[int],
    source_steps_per_output_frame: float,
) -> tuple[np.ndarray, np.ndarray]:
    root, local = root.copy(), local.copy()
    start, end = transition_range
    blend = min(BOUNDARY_BLEND_FRAMES, (end - start) // 2)
    if blend < 2 or start < 2 or end + 1 >= len(root):
        return root, local

    weights = _smoothstep(np.linspace(0, 1, blend))
    incoming_velocity = root[start - 1] - root[start - 2]
    desired_start = root[start - 1] + incoming_velocity
    start_anchor = root[start + blend - 1].copy()
    root[start:start + blend] = (
        (1 - weights[:, None]) * desired_start + weights[:, None] * start_anchor
    )
    for joint in range(local.shape[1]):
        previous = Rotation.from_matrix(local[start - 2, joint])
        boundary = Rotation.from_matrix(local[start - 1, joint])
        velocity = previous.inv() * boundary
        desired_start = boundary * velocity
        start_anchor = Rotation.from_matrix(local[start + blend - 1, joint])
        endpoints = Rotation.from_matrix(
            np.stack((desired_start.as_matrix(), start_anchor.as_matrix()))
        )
        local[start:start + blend, joint] = Slerp([0, 1], endpoints)(weights).as_matrix()

    outgoing_velocity = root[end + 1] - root[end]
    desired_end = root[end] - source_steps_per_output_frame * outgoing_velocity
    end_anchor = root[end - blend].copy()
    root[end - blend:end] = (
        (1 - weights[:, None]) * end_anchor + weights[:, None] * desired_end
    )
    for joint in range(local.shape[1]):
        boundary = Rotation.from_matrix(local[end, joint])
        following = Rotation.from_matrix(local[end + 1, joint])
        velocity = boundary.inv() * following
        desired_end = boundary * Rotation.from_rotvec(
            velocity.as_rotvec() * -source_steps_per_output_frame
        )
        end_anchor = Rotation.from_matrix(local[end - blend, joint])
        endpoints = Rotation.from_matrix(
            np.stack((end_anchor.as_matrix(), desired_end.as_matrix()))
        )
        local[end - blend:end, joint] = Slerp([0, 1], endpoints)(weights).as_matrix()
    return root, local


def _resample_at_times(
    root: np.ndarray, local: np.ndarray, source_times: np.ndarray, target_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    resampled_root = np.stack(
        [np.interp(target_times, source_times, root[:, axis]) for axis in range(3)], axis=1
    )
    resampled_local = np.empty((len(target_times), local.shape[1], 3, 3), dtype=np.float64)
    for joint in range(local.shape[1]):
        resampled_local[:, joint] = Slerp(source_times, Rotation.from_matrix(local[:, joint]))(target_times).as_matrix()
    return resampled_root, resampled_local


def _resample_transition(
    root_with_b0: np.ndarray, local_with_b0: np.ndarray, frame_time: float
) -> tuple[np.ndarray, np.ndarray]:
    transition_frames = len(root_with_b0) - 1
    frame_count = max(3, int(round((transition_frames / TARGET_FPS) / frame_time)))
    source_times = np.arange(len(root_with_b0), dtype=np.float64) / TARGET_FPS
    target_times = np.arange(frame_count, dtype=np.float64) * frame_time
    return _resample_at_times(root_with_b0, local_with_b0, source_times, target_times)


def _motion_components(motion: BVHMotion) -> tuple[np.ndarray, np.ndarray]:
    root = np.zeros((len(motion.frames), 3), dtype=np.float64)
    local = np.empty((len(motion.frames), len(motion.joints), 3, 3), dtype=np.float64)
    for index, joint in enumerate(motion.joints):
        values = motion.frames[:, joint.start:joint.start + len(joint.channels)]
        order = ""
        rotation_columns = []
        for column, channel in enumerate(joint.channels):
            if channel.endswith("position") and joint.parent < 0:
                root[:, "XYZ".index(channel[0])] = values[:, column]
            elif channel.endswith("rotation"):
                order += channel[0]
                rotation_columns.append(column)
        local[:, index] = Rotation.from_euler(order, values[:, rotation_columns], degrees=True).as_matrix()
    return root, local


def _resample_motion(
    root: np.ndarray, local: np.ndarray, source_frame_time: float, target_frame_time: float
) -> tuple[np.ndarray, np.ndarray]:
    if np.isclose(source_frame_time, target_frame_time, rtol=1e-6, atol=1e-9):
        return root, local
    source_times = np.arange(len(root), dtype=np.float64) * source_frame_time
    frame_count = max(3, int(round(len(root) * source_frame_time / target_frame_time)))
    target_times = np.minimum(np.arange(frame_count, dtype=np.float64) * target_frame_time, source_times[-1])
    return _resample_at_times(root, local, source_times, target_times)


def _align_following(
    root: np.ndarray, local: np.ndarray, target_root: np.ndarray, target_root_rotation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    alignment = Rotation.from_matrix(target_root_rotation) * Rotation.from_matrix(local[0, 0]).inv()
    aligned_root = target_root + alignment.apply(root - root[0])
    aligned_local = local.copy()
    aligned_local[:, 0] = (alignment * Rotation.from_matrix(local[:, 0])).as_matrix()
    return aligned_root, aligned_local


def _motion_frames(template: BVHMotion, root: np.ndarray, local: np.ndarray) -> np.ndarray:
    frames = np.zeros((len(root), template.frames.shape[1]), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for index, joint in enumerate(template.joints):
            order = "".join(channel[0] for channel in joint.channels if channel.endswith("rotation"))
            angles = Rotation.from_matrix(local[:, index]).as_euler(order, degrees=True)
            angles = np.rad2deg(np.unwrap(np.deg2rad(angles), axis=0))
            rotation_column = 0
            for column, channel in enumerate(joint.channels):
                destination = joint.start + column
                if channel.endswith("position"):
                    axis = "XYZ".index(channel[0])
                    frames[:, destination] = root[:, axis] if joint.parent < 0 else joint.offset[axis]
                else:
                    frames[:, destination] = angles[:, rotation_column]
                    rotation_column += 1
    return frames


def _write_with_template(source: bytes, frames: np.ndarray, frame_time: float) -> bytes:
    text = source.decode("utf-8-sig")
    hierarchy = re.split(r"(?m)^\s*MOTION\s*$", text, maxsplit=1)[0].rstrip()
    lines = [hierarchy, "MOTION", f"Frames: {len(frames)}", f"Frame Time: {frame_time:.8f}"]
    lines.extend(" ".join(f"{value:.6f}" for value in frame) for frame in frames)
    return ("\n".join(lines) + "\n").encode("utf-8")


def export_transition_bvh(
    joints: np.ndarray,
    features: np.ndarray,
    transition_range: tuple[int, int] | list[int],
    source_a: bytes,
    source_a_metadata: dict,
    source_b: bytes,
    source_b_metadata: dict,
) -> BVHExport:
    template = read_bvh(source_a)
    following = read_bvh(source_b)
    _validate_source_coordinates(source_a_metadata, source_b_metadata)
    _validate_matching_skeletons(template, following)

    target = _target_positions(joints, template, source_a_metadata)
    generated_local = _solve_local_rotations(target, template)
    hml_start, hml_end = transition_range
    bridge_root = target[hml_start:hml_end, 0]
    bridge_local = generated_local[hml_start:hml_end]
    if len(bridge_root) < 1:
        raise ValueError("MDM 过渡区间无法转换到源 BVH 帧率")
    bridge_local, twist_driven_joints = _apply_hml_twist_deltas(
        bridge_local, features[hml_start:hml_end], template
    )

    a_frames = template.frames.copy()
    a_root, a_local = _motion_components(template)

    b_drop = source_b_metadata["removed_initial_frames"]
    b_root, b_local = _motion_components(following)
    b_root, b_local = b_root[b_drop:], b_local[b_drop:]
    b_root, b_local = _resample_motion(b_root, b_local, following.frame_time, template.frame_time)
    b_root, b_local = _align_following(
        b_root, b_local, target[hml_end, 0], generated_local[hml_end, 0]
    )
    bridge_local, interpolated_leaf_joints = _interpolate_leaf_rotations(
        bridge_local, a_local[-1], b_local[0], template
    )
    b_frames = _motion_frames(template, b_root, b_local)

    stitched_root = np.concatenate((a_root[-2:], bridge_root, b_root[:2]))
    stitched_local = np.concatenate((a_local[-2:], bridge_local, b_local[:2]))
    stitched_root, stitched_local = _stitch_boundaries(
        stitched_root,
        stitched_local,
        (2, len(stitched_root) - 2),
        1 / (TARGET_FPS * template.frame_time),
    )
    bridge_root, bridge_local = _resample_transition(
        stitched_root[2:-1], stitched_local[2:-1], template.frame_time
    )
    bridge_frames = _motion_frames(template, bridge_root, bridge_local)

    frames = np.concatenate((a_frames, bridge_frames, b_frames))
    start, end = len(a_frames), len(a_frames) + len(bridge_frames)
    combined = _write_with_template(source_a, frames, template.frame_time)
    transition = _write_with_template(source_a, bridge_frames, template.frame_time)

    verified = read_bvh(combined)
    positions, _ = world_positions(verified)
    seam_indices = [index for index in (start, end) if 0 < index < len(positions)]
    seam_steps = [float(np.linalg.norm(positions[index] - positions[index - 1], axis=1).max()) for index in seam_indices]
    seam_velocity_changes = []
    for index in seam_indices:
        if index >= 2:
            before = (positions[index - 1] - positions[index - 2]) / verified.frame_time
            after = (positions[index] - positions[index - 1]) / verified.frame_time
            seam_velocity_changes.append(float(np.linalg.norm(after - before, axis=1).max()))
    bone_length_errors = []
    for child, joint in enumerate(verified.joints):
        if joint.parent >= 0:
            actual = np.linalg.norm(positions[:, child] - positions[:, joint.parent], axis=1)
            bone_length_errors.append(np.abs(actual - np.linalg.norm(joint.offset)))
    max_bone_length_error = float(max((errors.max() for errors in bone_length_errors), default=0.0))
    _, _, skeleton_profile = joint_mapping(template)
    return BVHExport(combined, transition, {
        "skeleton_source": "A",
        "skeleton_profile": skeleton_profile,
        "joint_count": len(template.joints),
        "frame_time": template.frame_time,
        "fps": template.fps,
        "frame_count": len(frames),
        "source_a_frames_preserved": len(a_frames),
        "source_b_frames_retargeted": len(b_frames),
        "segments": {"a": [0, start], "transition": [start, end], "b": [end, len(frames)]},
        "transition_frames": end - start,
        "max_seam_joint_step_source_units": max(seam_steps, default=0.0),
        "max_seam_joint_step_m": max(seam_steps, default=0.0) * source_a_metadata["source_scale_to_meters"],
        "max_seam_velocity_change_source_units_per_s": max(seam_velocity_changes, default=0.0),
        "max_seam_velocity_change_m_per_s": max(seam_velocity_changes, default=0.0) * source_a_metadata["source_scale_to_meters"],
        "max_bone_length_error_source_units": max_bone_length_error,
        "max_bone_length_error_m": max_bone_length_error * source_a_metadata["source_scale_to_meters"],
        "hml6d_twist_driven_joints": twist_driven_joints,
        "interpolated_leaf_joints": interpolated_leaf_joints,
        "boundary_blend_frames_20fps": min(BOUNDARY_BLEND_FRAMES, (hml_end - hml_start) // 2),
        "method": "Original A channels, MDM positions plus bone-axis 6D twists retargeted to source-A skeleton, interpolated leaf orientations, endpoint-constrained boundaries and quaternion SLERP resampling",
    })
