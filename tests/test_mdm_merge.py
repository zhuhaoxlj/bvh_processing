from io import BytesIO
from typing import ClassVar

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from bvh_processing.vendor.mdm.data.humanml.param_util import (
    t2m_raw_offsets,
)
from bvh_processing.vendor.mdm.merge import (
    GUIDANCE_PARAM,
    TEXT_CONDITION,
    BvhClip,
    merge_bvh_clips,
)
from bvh_processing.vendor.mdm.workbench.bvh import read_bvh, world_positions
from bvh_processing.vendor.mdm.workbench.bvh_export import _write_with_template
from bvh_processing.vendor.mdm.workbench.conversion import (
    DEFAULT_RESOURCE_ROOT,
    HML_JOINT_NAMES,
    PARENTS,
    load_assets,
)
from bvh_processing.vendor.mdm.workbench.transitions import (
    assemble_transition,
    prepare_transition,
)


@pytest.fixture
def source_bvh() -> bytes:
    children = [[] for _ in HML_JOINT_NAMES]
    for child, parent in enumerate(PARENTS):
        if parent >= 0:
            children[parent].append(child)

    def hierarchy(index: int, depth: int = 0) -> list[str]:
        indent = "  " * depth
        kind = "ROOT" if index == 0 else "JOINT"
        offset = t2m_raw_offsets[index].astype(float) * 10
        channels = (
            "6 Xposition Yposition Zposition Yrotation Xrotation Zrotation"
            if index == 0
            else "3 Yrotation Xrotation Zrotation"
        )
        lines = [
            f"{indent}{kind} {HML_JOINT_NAMES[index]}",
            f"{indent}{{",
            f"{indent}  OFFSET {offset[0]:g} {offset[1]:g} {offset[2]:g}",
            f"{indent}  CHANNELS {channels}",
        ]
        for child in children[index]:
            lines.extend(hierarchy(child, depth + 1))
        lines.append(f"{indent}}}")
        return lines

    frames = []
    for frame in range(24):
        root = [frame * 0.25, 100, 0, frame * 0.4, 1, 0]
        angles = [
            value
            for joint in range(1, len(HML_JOINT_NAMES))
            for value in (np.sin(frame / 5 + joint) * 2, 0.5, 0)
        ]
        frames.append(" ".join(f"{value:.6f}" for value in [*root, *angles]))
    return "\n".join(
        [
            "HIERARCHY",
            *hierarchy(0),
            "MOTION",
            f"Frames: {len(frames)}",
            "Frame Time: 0.05",
            *frames,
            "",
        ]
    ).encode()


class FakeEngine:
    data_root = DEFAULT_RESOURCE_ROOT / "humanml"
    calls = 0
    conditions: ClassVar[list[tuple[str, float]]] = []

    def generate(
        self,
        motion_a,
        motion_b,
        seconds,
        seed,
        progress,
        *,
        text_condition="",
        guidance_param=0.0,
    ):
        type(self).calls += 1
        type(self).conditions.append((text_condition, guidance_param))
        assets = load_assets(self.data_root)
        prepared = prepare_transition(motion_a, motion_b, seconds, assets)
        sample = prepared.condition.copy()
        start = prepared.left_frames
        sample[..., start : start + prepared.gap_frames] = sample[
            ..., start - 1 : start
        ]
        return assemble_transition(
            prepared,
            sample,
            assets,
            {"checkpoint": "fake"},
            seed,
        )


class HoveringEngine(FakeEngine):
    def generate(self, *args, **kwargs):
        result = super().generate(*args, **kwargs)
        start, end = result.metadata["transition_range"]
        result.joints[start:end, :, 1] += 0.08
        return result


class SteppingEngine(FakeEngine):
    def generate(self, *args, **kwargs):
        result = super().generate(*args, **kwargs)
        start, end = result.metadata["transition_range"]
        leg = result.joints[start:end, [4, 7, 10]]
        hip = result.joints[start:end, 1:2]
        result.joints[start:end, [4, 7, 10]] = hip + Rotation.from_euler(
            "X", 30, degrees=True
        ).apply((leg - hip).reshape(-1, 3)).reshape(leg.shape)
        return result


def _clips(content: bytes, count: int) -> list[BvhClip]:
    return [BvhClip(f"motion-{index}.bvh", BytesIO(content)) for index in range(count)]


def test_zero_gap_aligns_without_loading_mdm_assets(
    source_bvh: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "bvh_processing.vendor.mdm.merge.load_assets",
        lambda *_args: pytest.fail("zero-gap merge loaded MDM assets"),
    )

    result = merge_bvh_clips(
        _clips(source_bvh, 2),
        [0],
        engine=object(),
        seed=10,
        scale=0.01,
        up_axis="Y",
    )

    assert len(read_bvh(result).frames) == 2 * len(read_bvh(source_bvh).frames)


@pytest.mark.parametrize("clip_count", [2, 3])
def test_generated_single_and_multi_seam_merge(
    clip_count: int,
    source_bvh: bytes,
) -> None:
    FakeEngine.calls = 0
    FakeEngine.conditions = []

    result = merge_bvh_clips(
        _clips(source_bvh, clip_count),
        [0.5] * (clip_count - 1),
        engine=FakeEngine(),
        seed=21,
        scale=0.01,
        up_axis="Y",
    )

    source = read_bvh(source_bvh)
    merged = read_bvh(result)
    assert FakeEngine.calls == clip_count - 1
    assert FakeEngine.conditions == [
        (TEXT_CONDITION, GUIDANCE_PARAM)
    ] * (clip_count - 1)
    assert len(merged.frames) > clip_count * len(source.frames)
    assert merged.frame_time == source.frame_time
    np.testing.assert_allclose(merged.frames[: len(source.frames)], source.frames, atol=1e-6)
    np.testing.assert_allclose(
        merged.frames[-len(source.frames) :, source.rotation_columns],
        source.frames[:, source.rotation_columns],
        atol=1e-3,
    )


def test_generated_merge_preserves_original_segment_rotations(source_bvh: bytes) -> None:
    source = read_bvh(source_bvh)
    merged = read_bvh(
        merge_bvh_clips(
            _clips(source_bvh, 2),
            [0.5],
            engine=FakeEngine(),
            seed=21,
            scale=0.01,
            up_axis="Y",
        )
    )

    np.testing.assert_allclose(merged.frames[: len(source.frames)], source.frames, atol=1e-6)
    np.testing.assert_allclose(
        merged.frames[-len(source.frames) :, source.rotation_columns],
        source.frames[:, source.rotation_columns],
        atol=1e-3,
    )


@pytest.mark.parametrize("seconds", [0.05, 0.5])
@pytest.mark.parametrize("second_frame_time", [0.05, 0.0333333])
def test_generated_merge_preserves_following_clip_heading(
    source_bvh: bytes, seconds: float, second_frame_time: float
) -> None:
    first = read_bvh(source_bvh)
    second_frames = first.frames.copy()
    root = first.joints[0]
    yaw_column = root.start + root.channels.index("Yrotation")
    second_frames[:, yaw_column] += 25
    second_bvh = _write_with_template(source_bvh, second_frames, second_frame_time)

    merged = read_bvh(
        merge_bvh_clips(
            [
                BvhClip("first.bvh", BytesIO(source_bvh)),
                BvhClip("second.bvh", BytesIO(second_bvh)),
            ],
            [seconds],
            engine=FakeEngine(),
            seed=21,
            scale=0.01,
            up_axis="Y",
        )
    )

    np.testing.assert_allclose(merged.frames[: len(first.frames)], first.frames, atol=1e-6)
    second_output_frames = max(
        3, round(len(second_frames) * second_frame_time / first.frame_time)
    )
    np.testing.assert_allclose(
        merged.frames[-second_output_frames, first.rotation_columns],
        second_frames[0, first.rotation_columns],
        atol=1e-3,
    )
    if second_frame_time == first.frame_time:
        np.testing.assert_allclose(
            merged.frames[-len(first.frames) :, first.rotation_columns],
            second_frames[:, first.rotation_columns],
            atol=1e-3,
        )


def test_generated_merge_preserves_following_initial_rest_frame(source_bvh: bytes) -> None:
    source = read_bvh(source_bvh)
    second_frames = source.frames.copy()
    second_frames[0, source.rotation_columns] = 0
    root = source.joints[0]
    second_frames[1, root.start + root.channels.index("Yrotation")] = 10
    second_bvh = _write_with_template(source_bvh, second_frames, source.frame_time)

    merged = read_bvh(
        merge_bvh_clips(
            [
                BvhClip("first.bvh", BytesIO(source_bvh)),
                BvhClip("second.bvh", BytesIO(second_bvh)),
            ],
            [0.5],
            engine=FakeEngine(),
            seed=21,
            scale=0.01,
            up_axis="Y",
        )
    )

    np.testing.assert_allclose(
        merged.frames[-len(source.frames) :, source.rotation_columns],
        second_frames[:, source.rotation_columns],
        atol=1e-3,
    )


def test_generated_merge_preserves_120_fps_clip_edges(source_bvh: bytes) -> None:
    source = read_bvh(source_bvh)
    first_frames = np.repeat(source.frames, 5, axis=0)
    second_frames = first_frames.copy()
    root = source.joints[0]
    second_frames[:, root.start + root.channels.index("Yrotation")] += 25
    frame_time = 0.00833333
    first_bvh = _write_with_template(source_bvh, first_frames, frame_time)
    second_bvh = _write_with_template(source_bvh, second_frames, frame_time)

    merged = read_bvh(
        merge_bvh_clips(
            [
                BvhClip("first.bvh", BytesIO(first_bvh)),
                BvhClip("second.bvh", BytesIO(second_bvh)),
            ],
            [0.05],
            engine=FakeEngine(),
            seed=21,
            scale=0.01,
            up_axis="Y",
        )
    )

    assert len(merged.frames) == 2 * len(first_frames) + 6
    np.testing.assert_allclose(merged.frames[: len(first_frames)], first_frames, atol=1e-6)
    np.testing.assert_allclose(
        merged.frames[-len(second_frames) :, source.rotation_columns],
        second_frames[:, source.rotation_columns],
        atol=1e-3,
    )
    np.testing.assert_allclose(
        merged.frames[-len(second_frames) :, :3]
        - merged.frames[-len(second_frames), :3],
        second_frames[:, :3] - second_frames[0, :3],
        atol=1e-3,
    )


def test_generated_merge_keeps_support_foot_on_ground(source_bvh: bytes) -> None:
    source = read_bvh(source_bvh)
    merged = read_bvh(
        merge_bvh_clips(
            _clips(source_bvh, 2),
            [0.5],
            engine=HoveringEngine(),
            seed=21,
            scale=0.01,
            up_axis="Y",
        )
    )
    positions, _ = world_positions(merged)
    feet = [
        index for index, joint in enumerate(merged.joints)
        if joint.name in ("left_foot", "right_foot")
    ]
    start, end = len(source.frames), len(merged.frames) - len(source.frames)
    support_height = positions[:, feet, 1].min(axis=1)
    floor = np.linspace(support_height[start - 1], support_height[end], end - start + 2)[1:-1]

    np.testing.assert_allclose(merged.frames[:start], source.frames, atol=1e-6)
    np.testing.assert_allclose(
        merged.frames[end:, source.rotation_columns],
        source.frames[:, source.rotation_columns],
        atol=1e-3,
    )
    assert np.max(np.abs(support_height[start:end] - floor)) < 0.5


def test_generated_merge_keeps_both_stationary_feet_grounded(source_bvh: bytes) -> None:
    source = read_bvh(source_bvh)
    merged = read_bvh(
        merge_bvh_clips(
            _clips(source_bvh, 2),
            [0.5],
            engine=SteppingEngine(),
            seed=21,
            scale=0.01,
            up_axis="Y",
        )
    )
    positions, _ = world_positions(merged)
    feet = [
        index for index, joint in enumerate(merged.joints)
        if joint.name in ("left_foot", "right_foot")
    ]
    start, end = len(source.frames), len(merged.frames) - len(source.frames)
    ground = positions[start - 1, feet, 1].min()
    assert np.max(positions[start:end, feet, 1] - ground) < 1.5
    np.testing.assert_allclose(merged.frames[:start], source.frames, atol=1e-6)
    np.testing.assert_allclose(
        merged.frames[end:, source.rotation_columns], source.frames[:, source.rotation_columns],
        atol=1e-3,
    )


def test_generated_merge_keeps_leg_motion_for_moving_clips(source_bvh: bytes) -> None:
    source = read_bvh(source_bvh)
    moving_frames = source.frames.copy()
    moving_frames[:, 0] += np.arange(len(moving_frames)) * 4
    moving_bvh = _write_with_template(source_bvh, moving_frames, source.frame_time)
    merged = read_bvh(
        merge_bvh_clips(
            _clips(moving_bvh, 2),
            [0.5],
            engine=SteppingEngine(),
            seed=21,
            scale=0.01,
            up_axis="Y",
        )
    )
    positions, _ = world_positions(merged)
    feet = [
        index for index, joint in enumerate(merged.joints)
        if joint.name in ("left_foot", "right_foot")
    ]
    start, end = len(source.frames), len(merged.frames) - len(source.frames)
    support_height = positions[start:end, feet, 1].min(axis=1)
    assert np.max(positions[start:end, feet, 1] - support_height[:, None]) > 1.5


def test_zero_gap_rejects_incompatible_skeleton(source_bvh: bytes) -> None:
    incompatible = source_bvh.replace(b"ROOT pelvis", b"ROOT other_pelvis", 1)

    with pytest.raises(ValueError, match="关节名称"):
        merge_bvh_clips(
            [
                BvhClip("a.bvh", BytesIO(source_bvh)),
                BvhClip("b.bvh", BytesIO(incompatible)),
            ],
            [0],
            engine=object(),
            seed=10,
            scale=0.01,
            up_axis="Y",
        )
