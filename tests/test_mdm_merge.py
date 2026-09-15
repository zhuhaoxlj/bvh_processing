from io import BytesIO
from typing import ClassVar

import numpy as np
import pytest

from bvh_processing.vendor.mdm.data.humanml.param_util import (
    t2m_raw_offsets,
)
from bvh_processing.vendor.mdm.merge import (
    GUIDANCE_PARAM,
    TEXT_CONDITION,
    BvhClip,
    merge_bvh_clips,
)
from bvh_processing.vendor.mdm.workbench.bvh import read_bvh
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
