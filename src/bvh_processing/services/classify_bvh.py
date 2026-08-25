"""将 BVH 骨架识别为 LAFAN1 或 Nokov 格式。"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import BinaryIO

JOINT_PATTERN = re.compile(r"^\s*(?:ROOT|JOINT)\s+([^\s{]+)")
LAFAN1_TOE_JOINTS = frozenset({"LeftToe", "RightToe"})
NOKOV_TOE_JOINTS = frozenset({"LeftToeBase", "RightToeBase"})


class BVHClassificationError(ValueError):
    """BVH 骨架无法被可靠分类时抛出。"""


def _read_joint_names(lines: Iterable[str]) -> set[str]:
    joint_names: set[str] = set()
    has_hierarchy = False

    for line in lines:
        stripped_line = line.strip()
        if stripped_line == "HIERARCHY":
            has_hierarchy = True
        elif stripped_line == "MOTION":
            break

        match = JOINT_PATTERN.match(line)
        if match:
            joint_names.add(match.group(1))

    if not has_hierarchy or not joint_names:
        raise BVHClassificationError(
            "Invalid BVH file: no HIERARCHY section or skeleton joints found"
        )
    return joint_names


def _classify_joint_names(joint_names: set[str]) -> str:
    is_lafan1 = LAFAN1_TOE_JOINTS.issubset(joint_names)
    is_nokov = NOKOV_TOE_JOINTS.issubset(joint_names)

    if is_lafan1 and is_nokov:
        raise BVHClassificationError(
            "Ambiguous BVH skeleton: both LeftToe/RightToe and "
            "LeftToeBase/RightToeBase are present"
        )
    if is_lafan1:
        return "lafan1"
    if is_nokov:
        return "nokov"

    found_toe_joints = sorted(name for name in joint_names if "toe" in name.lower())
    found_description = ", ".join(found_toe_joints) or "none"
    raise BVHClassificationError(
        "Unsupported BVH skeleton: expected LeftToe and RightToe for LAFAN1, "
        "or LeftToeBase and RightToeBase for Nokov; "
        f"found toe joints: {found_description}"
    )


def read_bvh_joint_names(bvh_file: str | Path) -> set[str]:
    """读取 BVH 文件 HIERARCHY 区域中的 ROOT 和 JOINT 名称。"""
    path = Path(bvh_file)
    if not path.is_file():
        raise BVHClassificationError(f"BVH file does not exist: {path}")

    try:
        with path.open("r", encoding="utf-8-sig", errors="replace") as file:
            return _read_joint_names(file)
    except OSError as exc:
        raise BVHClassificationError(f"Unable to read BVH file: {path}") from exc


def classify_bvh(bvh_file: str | Path) -> str:
    """根据脚趾关节名称返回 ``lafan1`` 或 ``nokov``。"""
    return _classify_joint_names(read_bvh_joint_names(bvh_file))


def classify_downloaded_bvh(content: BinaryIO) -> str:
    """识别已下载到二进制流中的 BVH，并保持调用前的流位置。"""
    original_position = content.tell()
    try:
        content.seek(0)
        text = content.read().decode("utf-8-sig", errors="replace")
        return _classify_joint_names(_read_joint_names(text.splitlines()))
    except OSError as exc:
        raise BVHClassificationError("Unable to read downloaded BVH file") from exc
    finally:
        content.seek(original_position)
