import re
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from bvh_processing.errors import BvhServiceError
from bvh_processing.services.download import DownloadedBvh
from bvh_processing.services.foot_lock import (
    _foot_sides,
    _parse_hierarchy,
    _side_heights,
    _world_positions,
)
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


def test_loop_rejects_different_joint_names():
    raw = _STANDARD_POSE_PATH.read_bytes().replace(b"JOINT Spine1", b"JOINT OtherSpine")
    source = DownloadedBvh(BytesIO(raw), "incompatible.bvh", len(raw))
    with pytest.raises(BvhServiceError, match="不匹配"):
        optimize_bvh_loop(source)
    source.content.close()


def _assert_grounded(hierarchy, values):
    nodes = _parse_hierarchy(hierarchy)
    positions = _world_positions(nodes, values[-1:])
    heights = [
        float(_side_heights(nodes, positions, foot, toe)[0])
        for foot, toe in _foot_sides(nodes).values()
    ]
    assert min(heights) == pytest.approx(0, abs=1e-6)
    assert all(height >= -1e-6 for height in heights)


def test_loop_accepts_real_soma_offsets_and_grounds_target(reference):
    raw = (Path(__file__).parent / "fixtures/soma_different_offsets.bvh").read_bytes()
    source = DownloadedBvh(BytesIO(raw), "soma.bvh", len(raw))
    parsed = _parse_bvh(source)
    result = process_bvh(source, [4])
    output = _parse_bvh(result)
    values = _values(result)
    standard = np.asarray(_motion_values(reference, "reference.bvh"))[0]

    assert output.hierarchy == parsed.hierarchy
    assert output.frames[:2] == parsed.frames
    assert len(output.frames) == 32
    assert output.frame_time_text == parsed.frame_time_text
    assert np.isfinite(values).all()
    np.testing.assert_allclose(values[-1, [0, 2]], _values(source)[-1, [0, 2]])
    _assert_same_rotations(parsed.hierarchy, values[-1], standard)
    _assert_grounded(parsed.hierarchy, values)
    assert abs(values[-1, 1] - standard[1]) > 1
    source.content.close()
    result.content.close()


@pytest.mark.parametrize("scale", [0.01, 1.2])
def test_loop_adapts_target_height_to_skeleton_scale(reference, scale):
    text = _STANDARD_POSE_PATH.read_text()
    text = re.sub(
        r"OFFSET\s+([^\n]+)",
        lambda match: (
            "OFFSET "
            + " ".join(str(float(value) * scale) for value in match.group(1).split())
        ),
        text,
    )
    raw = text.encode()
    source = DownloadedBvh(BytesIO(raw), "scaled.bvh", len(raw))
    result = optimize_bvh_loop(source)
    values = _values(result)
    standard = np.asarray(_motion_values(reference, "reference.bvh"))[0]
    assert values[-1, 1] == pytest.approx(standard[1] * scale, abs=1e-6)
    _assert_same_rotations(_parse_bvh(source).hierarchy, values[-1], standard)
    _assert_grounded(_parse_bvh(source).hierarchy, values)
    source.content.close()
    result.content.close()


def test_loop_rejects_nonfinite_motion(reference):
    standard = np.asarray(_motion_values(reference, "reference.bvh"))[0]
    standard[3] = float("nan")
    source = _clip(reference, [standard])
    with pytest.raises(BvhServiceError, match="无效数值"):
        optimize_bvh_loop(source)
    source.content.close()


@pytest.mark.parametrize("value", [b"nan", b"inf"])
def test_loop_rejects_nonfinite_offsets(value):
    raw = _STANDARD_POSE_PATH.read_bytes().replace(b"8.43016", value)
    source = DownloadedBvh(BytesIO(raw), "invalid-offset.bvh", len(raw))
    with pytest.raises(BvhServiceError, match="OFFSET 包含无效数值"):
        optimize_bvh_loop(source)
    source.content.close()
