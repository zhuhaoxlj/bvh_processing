from io import BytesIO

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from bvh_processing.errors import BvhServiceError
from bvh_processing.services.download import DownloadedBvh
from bvh_processing.services.processing import (
    _STANDARD_POSE_PATH,
    _build_bvh,
    _motion_values,
    _parse_bvh,
    _parse_joints,
    optimize_bvh_loop,
    process_bvh,
)


@pytest.fixture
def reference():
    raw = _STANDARD_POSE_PATH.read_bytes()
    source = DownloadedBvh(BytesIO(raw), "reference.bvh", len(raw))
    yield _parse_bvh(source)
    source.content.close()


def _clip(reference, frames, frame_time="0.008333333333333333"):
    return _build_bvh(
        reference,
        [" ".join(map(str, frame)) for frame in frames],
        frame_time,
        "motion.bvh",
    )


def _values(downloaded):
    return np.asarray(_motion_values(_parse_bvh(downloaded), "motion.bvh"))


def _assert_same_rotations(hierarchy, actual, expected):
    cursor = 0
    for joint in _parse_joints(hierarchy):
        indexes = [
            cursor + index
            for index, channel in enumerate(joint.channels)
            if channel.endswith("rotation")
        ]
        if indexes:
            order = "".join(
                channel[0] for channel in joint.channels if channel.endswith("rotation")
            )
            actual_rotation = Rotation.from_euler(order, actual[indexes], degrees=True)
            expected_rotation = Rotation.from_euler(
                order, expected[indexes], degrees=True
            )
            assert (expected_rotation.inv() * actual_rotation).magnitude() < 1e-8
        cursor += len(joint.channels)


@pytest.mark.parametrize(
    ("frame_time", "extra_frames"), [("0.008333333333333333", 120), ("0.0333333", 30)]
)
def test_loop_appends_transition_to_fixed_pose(reference, frame_time, extra_frames):
    standard = np.asarray(_motion_values(reference, "reference.bvh"))[0]
    last = standard.copy()
    last[:3] = [125, 110, -240]
    last[3:] += 20
    source = _clip(reference, [last, last], frame_time)
    result = process_bvh(source, [4])
    parsed = _parse_bvh(result)
    values = _values(result)

    assert len(values) == 2 + extra_frames
    assert parsed.frames[:2] == _parse_bvh(source).frames
    assert parsed.hierarchy == reference.hierarchy
    assert parsed.frame_time_text == frame_time
    assert np.isfinite(values).all()
    assert values[:, 0] == pytest.approx(np.full(len(values), 125))
    assert values[:, 2] == pytest.approx(np.full(len(values), -240))
    assert values[-1, 1] == pytest.approx(standard[1])
    _assert_same_rotations(reference.hierarchy, values[-1], standard)
    height_steps = np.abs(np.diff(values[1:, 1]))
    assert height_steps[0] < height_steps.max() / 20
    assert height_steps[-1] < height_steps.max() / 20
    assert not source.content.closed
    source.content.close()
    result.content.close()


def test_loop_static_pose_stays_static(reference):
    standard = np.asarray(_motion_values(reference, "reference.bvh"))[0]
    source = _clip(reference, [standard] * 360)
    result = optimize_bvh_loop(source)
    np.testing.assert_allclose(_values(result), np.tile(standard, (480, 1)), atol=1e-8)
    source.content.close()
    result.content.close()


@pytest.mark.parametrize("yaw_delta", [358, -358, 90])
def test_loop_uses_shortest_rotation_path(reference, yaw_delta):
    standard = np.asarray(_motion_values(reference, "reference.bvh"))[0]
    last = standard.copy()
    last[3] += yaw_delta
    source = _clip(reference, [last])
    result = optimize_bvh_loop(source)
    values = _values(result)
    rotations = Rotation.from_euler("YXZ", values[:, 3:6], degrees=True)
    steps = (rotations[:-1].inv() * rotations[1:]).magnitude()
    expected_distance = abs((yaw_delta + 180) % 360 - 180)
    assert np.rad2deg(steps.sum()) == pytest.approx(expected_distance, abs=1e-6)
    assert np.max(np.abs(np.diff(values[:, 3:6], axis=0))) < 2
    _assert_same_rotations(reference.hierarchy, values[-1], standard)
    source.content.close()
    result.content.close()


def test_loop_keeps_noncanonical_euler_branch(reference):
    standard = np.asarray(_motion_values(reference, "reference.bvh"))[0]
    last = standard.copy()
    last[3:6] = [standard[3] + 180, 180 - standard[4], standard[5] + 180]
    source = _clip(reference, [last])
    result = optimize_bvh_loop(source)
    assert np.max(np.abs(np.diff(_values(result)[:, 3:6], axis=0))) < 1e-7
    source.content.close()
    result.content.close()


def test_loop_supports_other_euler_channel_order(reference):
    standard = np.asarray(_motion_values(reference, "reference.bvh"))[0]
    reordered = standard.copy()
    for cursor in range(3, len(standard), 3):
        reordered[cursor : cursor + 3] = Rotation.from_euler(
            "YXZ", standard[cursor : cursor + 3], degrees=True
        ).as_euler("ZXY", degrees=True)
    source = _clip(reference, [reordered])
    raw = source.content.read().replace(
        b"Yrotation Xrotation Zrotation", b"Zrotation Xrotation Yrotation"
    )
    source.content.close()
    source = DownloadedBvh(BytesIO(raw), "reordered.bvh", len(raw))
    result = optimize_bvh_loop(source)
    _assert_same_rotations(_parse_bvh(source).hierarchy, _values(result)[-1], reordered)
    assert np.max(np.abs(np.diff(_values(result), axis=0))) < 1e-7
    source.content.close()
    result.content.close()


@pytest.mark.parametrize(
    ("before", "after"),
    [(b"8.43016", b"9.43016"), (b"JOINT Spine1", b"JOINT OtherSpine")],
)
def test_loop_rejects_different_bone_lengths_and_names(reference, before, after):
    raw = _STANDARD_POSE_PATH.read_bytes().replace(before, after)
    source = DownloadedBvh(BytesIO(raw), "incompatible.bvh", len(raw))
    with pytest.raises(BvhServiceError, match="不匹配"):
        optimize_bvh_loop(source)
    source.content.close()


def test_loop_rejects_nonfinite_motion(reference):
    standard = np.asarray(_motion_values(reference, "reference.bvh"))[0]
    standard[3] = float("nan")
    source = _clip(reference, [standard])
    with pytest.raises(BvhServiceError, match="无效数值"):
        optimize_bvh_loop(source)
    source.content.close()
