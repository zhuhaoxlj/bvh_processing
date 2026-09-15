import math
from io import BytesIO

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from bvh_processing.errors import BvhServiceError
from bvh_processing.services.download import DownloadedBvh
from bvh_processing.services.processing import (
    denoise_bvh,
    lock_bvh_feet,
    optimize_bvh_loop,
    orient_bvh_facing_to_x,
    process_bvh,
    smooth_bvh,
    trim_bvh,
)


def _downloaded(frames: list[list[float]]) -> DownloadedBvh:
    motion = "\n".join(" ".join(map(str, frame)) for frame in frames)
    content = f"""HIERARCHY
ROOT Hips
{{
  OFFSET 0 0 0
  CHANNELS 2 Xposition Yrotation
}}
MOTION
Frames: {len(frames)}
Frame Time: 0.0333333
{motion}
""".encode()
    return DownloadedBvh(BytesIO(content), "motion.bvh", len(content))


_ROOT_CHANNELS = "Xposition Yposition Zposition Zrotation Xrotation Yrotation"
_CHILD_CHANNELS = "Zrotation Xrotation Yrotation"
# LAFAN1 风格：左大腿根在 +X，右大腿根在 -X，因此身体右侧指向 -X、初始面向 +Z。
_LEFT_HIP_OFFSET = (10.5, -1.3, 0.0)
_RIGHT_HIP_OFFSET = (-10.5, -1.3, 0.0)


def _humanoid(
    root_frames: list[list[float]],
    *,
    root_channels: str = _ROOT_CHANNELS,
    left_offset: tuple[float, float, float] = _LEFT_HIP_OFFSET,
    right_offset: tuple[float, float, float] = _RIGHT_HIP_OFFSET,
) -> DownloadedBvh:
    """构造带左右大腿根的骨架；root_frames 只给根通道，子关节补 0。"""
    child_count = len(_CHILD_CHANNELS.split())
    padding = " ".join(["0.0"] * (child_count * 4))
    motion = "\n".join(
        " ".join([*(str(value) for value in frame), padding]) for frame in root_frames
    )
    content = f"""HIERARCHY
ROOT Hips
{{
  OFFSET 0 0 0
  CHANNELS {len(root_channels.split())} {root_channels}
  JOINT LeftUpLeg
  {{
    OFFSET {left_offset[0]} {left_offset[1]} {left_offset[2]}
    CHANNELS {child_count} {_CHILD_CHANNELS}
    JOINT LeftLeg
    {{
      OFFSET 0 -14 0
      CHANNELS {child_count} {_CHILD_CHANNELS}
    }}
  }}
  JOINT RightUpLeg
  {{
    OFFSET {right_offset[0]} {right_offset[1]} {right_offset[2]}
    CHANNELS {child_count} {_CHILD_CHANNELS}
    JOINT RightLeg
    {{
      OFFSET 0 -14 0
      CHANNELS {child_count} {_CHILD_CHANNELS}
    }}
  }}
}}
MOTION
Frames: {len(root_frames)}
Frame Time: 0.0333333
{motion}
""".encode()
    return DownloadedBvh(BytesIO(content), "humanoid.bvh", len(content))


def _frames(downloaded: DownloadedBvh) -> list[list[float]]:
    downloaded.content.seek(0)
    lines = downloaded.content.read().decode().splitlines()
    motion_index = lines.index("MOTION")
    return [
        [float(value) for value in line.split()]
        for line in lines[motion_index + 3 :]
        if line.strip()
    ]


def test_denoise_bvh_removes_isolated_motion_spike() -> None:
    # 首帧旋转非 0，避免被当成开头的静止绑定姿势帧（那种情况见下面的前缀测试）。
    source = _downloaded([[0, 5], [0, 1], [100, 90], [0, 3], [0, 4]])

    result = denoise_bvh(source)

    # 第 90 度的孤立尖峰被中值抹掉；首尾帧窗口收缩到只剩自己，保持原值。
    assert _frames(result) == [
        [0, 5],
        [0, 5],
        [0, 3],
        [0, 4],
        [0, 4],
    ]
    result.content.close()


def test_smooth_bvh_uses_centered_moving_average() -> None:
    source = _downloaded([[0, 10], [0, 0], [0, 10], [0, 0], [0, 10], [0, 0], [0, 10]])

    result = smooth_bvh(source)

    # 首尾帧窗口收缩到只剩自己，保持原值；靠近边界处窗口降为 3 帧。
    values = [frame[1] for frame in _frames(result)]
    assert values == pytest.approx([10, 20 / 3, 6, 4, 6, 20 / 3, 10])
    result.content.close()


def test_smooth_bvh_handles_rotation_wraparound() -> None:
    source = _downloaded([[0, 179], [0, -179], [0, -178]])

    result = smooth_bvh(source)

    rotations = [frame[1] for frame in _frames(result)]
    assert rotations == pytest.approx([179, 542 / 3, 182])
    result.content.close()


def test_processors_leave_boundary_frames_untouched() -> None:
    """边界窗口收缩：静止首帧（全 0）后第 1 帧直接起跳时，首尾帧必须原样保留。

    这正是复制边界帧会出错的场景——首帧在自己窗口里占 3/5 权重，会被拉进动作。
    """
    source = _downloaded([[0, 0], [100, 100], [100, 100], [100, 100], [100, 100]])

    denoised = _frames(denoise_bvh(source))
    smoothed = _frames(smooth_bvh(source))
    processed = _frames(process_bvh(source, [1, 2]))

    for frames in (denoised, smoothed, processed):
        assert frames[0] == pytest.approx([0.0, 0.0])
        assert frames[-1] == pytest.approx([100.0, 100.0])


def test_rest_prefix_is_not_averaged_into_following_frames() -> None:
    """开头的静止帧不属于动作，不能被平均进后面的动作帧。

    否则静止帧会被摊到随后几帧上：真实素材上实测 2→3、3→4 的帧间位移会从 ~1
    跳到 30~45，看起来就是"一跳一跳像瞬移"。
    """
    source = _downloaded(
        [[0, 0], [0, 100], [0, 100], [0, 100], [0, 100], [0, 100], [0, 100]]
    )

    for processor in (denoise_bvh, smooth_bvh):
        values = [frame[1] for frame in _frames(processor(source))]
        assert values == pytest.approx([0, 100, 100, 100, 100, 100, 100])


def test_process_bvh_runs_selected_processors_in_order() -> None:
    source = _downloaded([[0, 5], [0, 0], [100, 100], [0, 0], [0, 0]])

    result = process_bvh(source, [1, 2])

    # 去噪(中值3) 先抹掉 100 的尖峰，平滑(均值5) 再抹平残留。
    frames = _frames(result)
    assert [frame[0] for frame in frames] == pytest.approx([0] * 5)
    assert [frame[1] for frame in frames] == pytest.approx([5, 10 / 3, 2, 0, 0])
    result.content.close()


def test_unimplemented_processors_keep_original_stream() -> None:
    source = _downloaded([[0, 0]])

    assert lock_bvh_feet(source) is source
    assert optimize_bvh_loop(source) is source
    assert process_bvh(source, [3, 4]) is source


def test_process_bvh_rejects_unknown_option() -> None:
    source = _downloaded([[0, 0]])

    with pytest.raises(BvhServiceError) as error:
        process_bvh(source, [9])

    assert error.value.code == "invalid_handle_option"


def test_trim_bvh_uses_nearest_existing_frames_without_interpolation() -> None:
    source = _downloaded([[0, 0], [1, 10], [2, 20], [3, 30], [4, 40]])

    result = trim_bvh(source, 0.04, 0.095)

    assert _frames(result) == [[1, 10], [2, 20], [3, 30]]
    result.content.close()


# --------------------------------------------------------------- 朝向对齐


def _world_hip_joints(
    downloaded: DownloadedBvh,
) -> tuple[np.ndarray, np.ndarray]:
    """独立复算左右大腿根的世界坐标，用于验证朝向处理结果。"""
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    for frame in _frames(downloaded):
        rotation = Rotation.from_euler(
            "ZXY", [frame[3], frame[4], frame[5]], degrees=True
        )
        position = np.array(frame[:3])
        left.append(rotation.apply(_LEFT_HIP_OFFSET) + position)
        right.append(rotation.apply(_RIGHT_HIP_OFFSET) + position)
    return np.array(left), np.array(right)


def _world_facing(downloaded: DownloadedBvh, index: int = 0) -> tuple[float, float]:
    """某一帧人体面向的地面投影：right = 右髋-左髋，facing = up × right。"""
    left, right = _world_hip_joints(downloaded)
    axis = right - left
    facing = np.array([axis[index][2], -axis[index][0]])
    facing = facing / np.linalg.norm(facing)
    return float(facing[0]), float(facing[1])


def test_orient_bvh_facing_to_x_turns_identity_pose_to_positive_x() -> None:
    # 根旋转为 0 时，该骨架面向 +Z，需要绕 Y 轴转 90° 才对上 +X。
    source = _humanoid([[0, 100, 0, 0, 0, 0], [0, 100, 0, 0, 0, 0]])

    result = orient_bvh_facing_to_x(source)

    assert _world_facing(result) == pytest.approx((1.0, 0.0), abs=1e-9)
    rotations = [frame[3:6] for frame in _frames(result)]
    assert rotations[0][2] == pytest.approx(90.0, abs=1e-6)
    result.content.close()


def test_orient_bvh_facing_to_x_aligns_regardless_of_input_rotation() -> None:
    for y_rotation in (-135.0, -35.0, 47.0, 180.0):
        source = _humanoid([[0, 100, 0, 0, 0, y_rotation]])

        result = orient_bvh_facing_to_x(source)

        assert _world_facing(result) == pytest.approx((1.0, 0.0), abs=1e-9), y_rotation
        result.content.close()


def test_orient_bvh_facing_to_x_keeps_motion_shape() -> None:
    source = _humanoid(
        [
            [0, 100, 0, 0, 0, 0],
            [10, 105, -20, 15, -8, 30],
            [-5, 95, 12, -20, 4, -12],
        ]
    )

    result = orient_bvh_facing_to_x(source)

    before_left, before_right = _world_hip_joints(source)
    after_left, after_right = _world_hip_joints(result)

    # 刚体旋转：高度不变、两髋间距离不变。
    assert after_left[:, 1] == pytest.approx(before_left[:, 1], abs=1e-6)
    assert after_right[:, 1] == pytest.approx(before_right[:, 1], abs=1e-6)
    before_axis = before_right - before_left
    after_axis = after_right - after_left
    assert np.linalg.norm(after_axis, axis=1) == pytest.approx(
        np.linalg.norm(before_axis, axis=1), abs=1e-6
    )

    # 髋轴是水平向量：各帧的偏航变化量相同，说明整段动作只转了一次、没有逐帧变形。
    yaw_delta = np.arctan2(after_axis[:, 2], after_axis[:, 0]) - np.arctan2(
        before_axis[:, 2], before_axis[:, 0]
    )
    yaw_delta = (yaw_delta + np.pi) % (2 * np.pi) - np.pi
    assert yaw_delta == pytest.approx(np.full(len(yaw_delta), yaw_delta[0]), abs=1e-6)
    result.content.close()


def test_orient_bvh_facing_to_x_rotates_root_translation_around_origin() -> None:
    # 面向 +Z、站在 (0, 100, 50)：转到 +X 后位置应变成 (50, 100, 0)。
    source = _humanoid([[0, 100, 50, 0, 0, 0]])

    result = orient_bvh_facing_to_x(source)

    assert _frames(result)[0][:3] == pytest.approx([50.0, 100.0, 0.0], abs=1e-6)
    result.content.close()


def test_orient_bvh_facing_to_x_uses_first_pose_frame() -> None:
    """朝向只由第一帧决定：后面几帧各自怎么转都不影响旋转角。"""
    facing_30 = _humanoid([[0, 100, 0, 0, 0, 30], [0, 100, 0, 0, 0, -60]])
    facing_30_other = _humanoid([[0, 100, 0, 0, 0, 30], [0, 100, 0, 0, 0, 170]])

    first = _frames(orient_bvh_facing_to_x(facing_30))[0][3:6]
    second = _frames(orient_bvh_facing_to_x(facing_30_other))[0][3:6]

    # 第 0 帧对齐到 +X；第 1 帧只是跟着转，不参与决定旋转角。
    assert first == pytest.approx(second, abs=1e-9)
    assert _world_facing(orient_bvh_facing_to_x(facing_30), 0) == pytest.approx(
        (1.0, 0.0), abs=1e-9
    )
    assert _world_facing(facing_30, 1) != pytest.approx(
        _world_facing(orient_bvh_facing_to_x(facing_30), 1), abs=1e-3
    )


def test_orient_bvh_facing_to_x_skips_leading_t_pose_frame() -> None:
    """开头是静止 T-pose（旋转全 0）时，基准帧顺延到第 1 帧；T-pose 帧只是跟着转。"""
    source = _humanoid([[0, 100, 0, 0, 0, 0], [0, 100, 0, 0, 0, 30]])

    result = orient_bvh_facing_to_x(source)

    # 第 1 帧（真正摆姿势的那帧）对齐到 +X：它原本是 60°，所以整段转了 60°。
    assert _world_facing(result, 1) == pytest.approx((1.0, 0.0), abs=1e-9)
    # T-pose 帧原本面向 +Z（90°），跟着转 60° 后变成 30°，而不是自己也被对齐。
    assert _world_facing(result, 0) == pytest.approx(
        (math.cos(math.radians(30)), math.sin(math.radians(30))), abs=1e-9
    )
    result.content.close()


def test_orient_bvh_facing_to_x_requires_hips_and_rotation_channels() -> None:
    without_hips = _downloaded([[0, 0]])
    with pytest.raises(BvhServiceError) as missing_hips:
        orient_bvh_facing_to_x(without_hips)
    assert missing_hips.value.code == "cannot_orient_bvh"
    assert "髋" in missing_hips.value.message

    without_rotation = _humanoid(
        [[0, 100, 0]], root_channels="Xposition Yposition Zposition"
    )
    with pytest.raises(BvhServiceError) as missing_rotation:
        orient_bvh_facing_to_x(without_rotation)
    assert missing_rotation.value.code == "cannot_orient_bvh"


def test_process_bvh_applies_orient_as_option_five() -> None:
    source = _humanoid([[0, 100, 0, 0, 0, 0], [0, 100, 0, 0, 0, 0]])

    result = process_bvh(source, [1, 5])

    assert _world_facing(result) == pytest.approx((1.0, 0.0), abs=1e-6)
    result.content.close()
