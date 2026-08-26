from io import BytesIO

import pytest

from bvh_processing.errors import BvhServiceError
from bvh_processing.services.download import DownloadedBvh
from bvh_processing.services.processing import (
    denoise_bvh,
    lock_bvh_feet,
    optimize_bvh_loop,
    process_bvh,
    smooth_bvh,
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
    source = _downloaded(
        [[0, 0], [0, 1], [100, 90], [0, 3], [0, 4]]
    )

    result = denoise_bvh(source)

    assert _frames(result) == [
        [0, 0],
        [0, 1],
        [0, 3],
        [0, 4],
        [0, 4],
    ]
    result.content.close()


def test_smooth_bvh_uses_centered_moving_average() -> None:
    source = _downloaded(
        [[0, 0], [0, 10], [0, 0], [0, 10], [0, 0], [0, 10], [0, 0]]
    )

    result = smooth_bvh(source)

    assert _frames(result) == [
        [0, 2],
        [0, 4],
        [0, 4],
        [0, 6],
        [0, 4],
        [0, 4],
        [0, 2],
    ]
    result.content.close()


def test_smooth_bvh_handles_rotation_wraparound() -> None:
    source = _downloaded([[0, 179], [0, -179], [0, -178]])

    result = smooth_bvh(source)

    rotations = [frame[1] for frame in _frames(result)]
    assert rotations == pytest.approx([180, 180.6, 181.2])
    result.content.close()


def test_process_bvh_runs_selected_processors_in_order() -> None:
    source = _downloaded(
        [[0, 0], [0, 0], [100, 100], [0, 0], [0, 0]]
    )

    result = process_bvh(source, [1, 2])

    assert _frames(result) == [[0, 0]] * 5
    result.content.close()


def test_unimplemented_processors_keep_original_stream() -> None:
    source = _downloaded([[0, 0]])

    assert lock_bvh_feet(source) is source
    assert optimize_bvh_loop(source) is source
    assert process_bvh(source, [3, 4]) is source


def test_process_bvh_rejects_unknown_option() -> None:
    source = _downloaded([[0, 0]])

    with pytest.raises(BvhServiceError) as error:
        process_bvh(source, [5])

    assert error.value.code == "invalid_handle_option"