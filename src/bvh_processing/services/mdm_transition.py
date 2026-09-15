"""Generate BVH transitions directly with the bundled MDM module."""

from __future__ import annotations

from pathlib import Path
from threading import Lock

from bvh_processing.config import Settings
from bvh_processing.services.download import DownloadedBvh
from bvh_processing.vendor.mdm.merge import BvhClip, merge_bvh_clips
from bvh_processing.vendor.mdm.workbench.transitions import MDMEngine

_MDM_LOCK = Lock()
_MDM_ENGINE: MDMEngine | None = None
_MDM_RESOURCE_ROOT: Path | None = None


def _engine_for(resource_root: Path) -> MDMEngine:
    global _MDM_ENGINE, _MDM_RESOURCE_ROOT
    resource_root = resource_root.resolve()
    if _MDM_ENGINE is None or _MDM_RESOURCE_ROOT != resource_root:
        _MDM_ENGINE = MDMEngine(
            checkpoint=resource_root / "checkpoint/model000600000.pt",
            data_root=resource_root / "humanml",
            text_encoder=resource_root / "text_encoder",
        )
        _MDM_RESOURCE_ROOT = resource_root
    return _MDM_ENGINE


def generate_mdm_merge(
    downloaded_files: list[DownloadedBvh],
    intervals_seconds: list[float],
    settings: Settings,
) -> bytes:
    if len(downloaded_files) < 2:
        raise ValueError("MDM 合并至少需要两个 BVH 文件")
    if len(intervals_seconds) != len(downloaded_files) - 1:
        raise ValueError("BVH 文件数量与过渡时长数量不匹配")

    clips = []
    for downloaded in downloaded_files:
        downloaded.content.seek(0)
        clips.append(BvhClip(downloaded.source_filename, downloaded.content.read()))

    with _MDM_LOCK:
        result = merge_bvh_clips(
            clips,
            intervals_seconds,
            engine=_engine_for(Path(settings.mdm_resource_root).expanduser()),
            seed=settings.mdm_seed,
            scale=settings.mdm_source_scale,
            up_axis=settings.mdm_source_up_axis,
        )
    if not result:
        raise ValueError("MDM 模块生成了空 BVH 文件")
    return result
