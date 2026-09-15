"""Direct in-process BVH merging backed by the vendored MDM model."""

from __future__ import annotations

from dataclasses import dataclass

from bvh_processing.vendor.mdm.workbench.bvh_export import (
    append_bvh_without_transition,
    export_transition_bvh,
)
from bvh_processing.vendor.mdm.workbench.conversion import convert_bvh, load_assets
from bvh_processing.vendor.mdm.workbench.transitions import MDMEngine

TEXT_CONDITION = "A short, direct interpolation between the two poses."
GUIDANCE_PARAM = 2.5


@dataclass(frozen=True)
class BvhClip:
    filename: str
    content: bytes


def merge_bvh_clips(
    clips: list[BvhClip],
    transition_seconds: list[float],
    *,
    engine: MDMEngine,
    seed: int,
    scale: float,
    up_axis: str,
) -> bytes:
    if not clips or len(transition_seconds) != len(clips) - 1:
        raise ValueError("BVH 文件数量与过渡时长数量不匹配")

    assets = None
    current = clips[0].content
    for index, (clip, seconds) in enumerate(
        zip(clips[1:], transition_seconds, strict=True)
    ):
        following = clip.content
        if seconds == 0:
            current = append_bvh_without_transition(current, following)
            continue

        if assets is None:
            assets = load_assets(engine.data_root)
        converted_a = convert_bvh(
            current,
            filename=f"merged-{index}.bvh",
            scale=scale,
            up_axis=up_axis,
            assets=assets,
        )
        converted_b = convert_bvh(
            following,
            filename=clip.filename,
            scale=scale,
            up_axis=up_axis,
            assets=assets,
        )
        result = engine.generate(
            converted_a.features,
            converted_b.features,
            seconds,
            seed + index,
            lambda _phase, _step, _total: None,
            text_condition=TEXT_CONDITION,
            guidance_param=GUIDANCE_PARAM,
        )
        current = export_transition_bvh(
            result.joints,
            result.features,
            result.metadata["transition_range"],
            current,
            converted_a.metadata,
            following,
            converted_b.metadata,
        ).combined
    return current
