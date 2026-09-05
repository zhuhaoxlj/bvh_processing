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
    process_bvh,
    smooth_bvh,
    trim_bvh,
)
from bvh_processing.services.transition import (
    Joint,
    Motion,
    _align_to_boundary,
    _canonicalize,
    _extract_rotations,
    _world_positions,
)


def _transition_motion(
    positions: list[list[float]],
    root_rotations: list[list[float]],
) -> Motion:
    root = Joint(
        "Hips",
        np.zeros(3),
        [
            "Xposition",
            "Yposition",
            "Zposition",
            "Yrotation",
            "Xrotation",
            "Zrotation",
        ],
        [
            Joint(
                "LeftToe",
                np.asarray([-1.0, -1.0, 0.0]),
                ["Yrotation", "Xrotation", "Zrotation"],
            ),
            Joint(
                "RightToe",
                np.asarray([1.0, -1.0, 0.0]),
                ["Yrotation", "Xrotation", "Zrotation"],
            ),
        ],
    )
    frames = np.concatenate(
        [
            np.asarray(positions, dtype=float),
            np.asarray(root_rotations, dtype=float),
            np.zeros((len(positions), 6)),
        ],
        axis=1,
    )
    return Motion(root, frames, 1.0 / 120.0)


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


def test_trim_bvh_uses_nearest_existing_frames_without_interpolation() -> None:
    source = _downloaded([[0, 0], [1, 10], [2, 20], [3, 30], [4, 40]])

    result = trim_bvh(source, 0.04, 0.095)

    assert _frames(result) == [[1, 10], [2, 20], [3, 30]]
    result.content.close()


def test_transition_discards_exporter_startup_ramp() -> None:
    positions = [
        [-25.0, 90.0, -10.0],
        [-50.0, 85.0, -20.0],
        [-75.0, 80.0, -30.0],
        [-100.0, 75.0, -40.0],
        *[[-100.0 + index * 0.02, 75.0, -40.0] for index in range(12)],
    ]
    rotations = [
        [25.0, 5.0, 1.0],
        [50.0, 10.0, 2.0],
        [75.0, 15.0, 3.0],
        [100.0, 20.0, 4.0],
        *[[100.0 + index * 0.02, 20.0, 4.0] for index in range(12)],
    ]

    canonical = _canonicalize(_transition_motion(positions, rotations))

    assert canonical.frames[0, :3] == pytest.approx(positions[3])
    steps = np.linalg.norm(np.diff(canonical.frames[:, :3], axis=0), axis=1)
    assert np.max(steps) < 0.1


def test_boundary_alignment_preserves_following_torso_tilt_and_locks_foot() -> None:
    previous = _transition_motion(
        [[0.0, 1.0, 0.0], [0.1, 1.0, 0.0]],
        [[70.0, -8.0, 3.0], [80.0, -8.0, 3.0]],
    )
    following = _transition_motion(
        [[5.0, 2.0, 7.0], [5.1, 2.0, 7.0]],
        [[20.0, 25.0, -12.0], [22.0, 25.0, -12.0]],
    )

    aligned, support_name = _align_to_boundary(previous, following)
    aligned_rotations, _ = _extract_rotations(aligned)
    source_rotation = Rotation.from_euler("YXZ", [20.0, 25.0, -12.0], degrees=True)
    aligned_rotation = Rotation.from_euler(
        "YXZ", aligned_rotations[0, 0], degrees=True
    )
    yaw_only_delta = aligned_rotation * source_rotation.inv()
    delta_euler = yaw_only_delta.as_euler("YXZ", degrees=True)

    assert delta_euler[1:] == pytest.approx([0.0, 0.0], abs=1e-8)
    joint_names = [joint.name for joint in previous.joints]
    support_index = joint_names.index(support_name)
    previous_foot = _world_positions(previous, previous.frames[-1])[support_index]
    following_foot = _world_positions(aligned, aligned.frames[0])[support_index]
    assert following_foot == pytest.approx(previous_foot)
