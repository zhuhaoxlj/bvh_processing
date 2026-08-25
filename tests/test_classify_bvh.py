from io import BytesIO

import pytest

from bvh_processing.services.classify_bvh import (
    BVHClassificationError,
    classify_downloaded_bvh,
)


def _bvh_with_joints(*joint_names: str) -> BytesIO:
    joints = "\n".join(f"JOINT {name}" for name in joint_names)
    return BytesIO(f"HIERARCHY\nROOT Hips\n{joints}\nMOTION\n".encode())


@pytest.mark.parametrize(
    ("joint_names", "expected_format"),
    [
        (("LeftToe", "RightToe"), "lafan1"),
        (("LeftToeBase", "RightToeBase"), "nokov"),
    ],
)
def test_classifies_supported_bvh_formats(
    joint_names: tuple[str, str],
    expected_format: str,
) -> None:
    content = _bvh_with_joints(*joint_names)
    content.seek(3)

    assert classify_downloaded_bvh(content) == expected_format
    assert content.tell() == 3


def test_rejects_unsupported_bvh_format() -> None:
    content = _bvh_with_joints("LeftFoot", "RightFoot")

    with pytest.raises(BVHClassificationError, match="Unsupported BVH skeleton"):
        classify_downloaded_bvh(content)
