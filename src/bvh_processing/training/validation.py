"""严格校验 BeyondMimic 训练动作 NPZ。"""

from __future__ import annotations

import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

from bvh_processing.errors import BvhServiceError

_REQUIRED_ARRAYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
_MAX_ARCHIVE_MEMBERS = 64
_MAX_COMPRESSION_RATIO = 200


def _invalid(message: str) -> BvhServiceError:
    return BvhServiceError(
        status_code=422,
        code="invalid_training_npz",
        message=f"训练 NPZ 校验失败：{message}",
    )


def _validate_archive(path: Path, max_uncompressed_bytes: int) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if not members or len(members) > _MAX_ARCHIVE_MEMBERS:
                raise _invalid("ZIP 条目数量异常")

            total_size = 0
            for member in members:
                member_path = PurePosixPath(member.filename)
                if (
                    member.is_dir()
                    or member.flag_bits & 0x1
                    or member_path.is_absolute()
                    or ".." in member_path.parts
                    or len(member_path.parts) != 1
                    or member_path.suffix != ".npy"
                ):
                    raise _invalid(f"包含不安全的 ZIP 条目：{member.filename}")
                total_size += member.file_size
                if total_size > max_uncompressed_bytes:
                    raise _invalid("解压后数据超过大小限制")
                if member.file_size and (
                    member.compress_size == 0
                    or member.file_size / member.compress_size
                    > _MAX_COMPRESSION_RATIO
                ):
                    raise _invalid(f"ZIP 条目压缩比异常：{member.filename}")

            corrupt_member = archive.testzip()
            if corrupt_member is not None:
                raise _invalid(f"ZIP CRC 校验失败：{corrupt_member}")
    except BvhServiceError:
        raise
    except (OSError, zipfile.BadZipFile) as exc:
        raise _invalid("文件不是有效的 NPZ/ZIP") from exc


def validate_training_npz(path: Path, max_uncompressed_bytes: int) -> None:
    """验证当前 G1 BeyondMimic loader 所需的数组和形状。"""

    _validate_archive(path, max_uncompressed_bytes)
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = [name for name in _REQUIRED_ARRAYS if name not in data.files]
            if missing:
                raise _invalid(f"缺少数组：{', '.join(missing)}")
            arrays = {name: np.asarray(data[name]) for name in _REQUIRED_ARRAYS}
    except BvhServiceError:
        raise
    except Exception as exc:
        raise _invalid("NumPy 无法安全读取全部数组") from exc

    for name, array in arrays.items():
        if not np.issubdtype(array.dtype, np.number):
            raise _invalid(f"数组 {name} 不是数值类型")
        if not np.isfinite(array).all():
            raise _invalid(f"数组 {name} 包含 NaN 或 Inf")

    fps_array = arrays["fps"]
    if fps_array.size != 1:
        raise _invalid("fps 必须是单个数值")
    fps = float(fps_array.reshape(-1)[0])
    if not 1.0 <= fps <= 240.0:
        raise _invalid("fps 必须在 1 到 240 之间")

    joint_pos = arrays["joint_pos"]
    joint_vel = arrays["joint_vel"]
    if joint_pos.ndim != 2 or joint_pos.shape[0] < 2:
        raise _invalid("joint_pos 必须为至少两帧的二维数组")
    if joint_pos.shape[1] != 29 or joint_vel.shape != joint_pos.shape:
        raise _invalid("G1 关节数组必须为 [frames, 29] 且位置速度形状一致")

    frame_count = joint_pos.shape[0]
    expected_shapes = {
        "body_pos_w": (frame_count, 30, 3),
        "body_quat_w": (frame_count, 30, 4),
        "body_lin_vel_w": (frame_count, 30, 3),
        "body_ang_vel_w": (frame_count, 30, 3),
    }
    for name, expected_shape in expected_shapes.items():
        if arrays[name].shape != expected_shape:
            raise _invalid(f"数组 {name} 必须为 {expected_shape}")

    quaternion_norms = np.linalg.norm(arrays["body_quat_w"], axis=-1)
    if float(np.max(np.abs(quaternion_norms - 1.0))) > 1e-2:
        raise _invalid("body_quat_w 四元数未正确归一化")