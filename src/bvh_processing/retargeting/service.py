"""Robot Retargeter 的 BVH 到 G1 Whole Body Tracking 转换入口。"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from bvh_processing.errors import BvhServiceError
from bvh_processing.retargeting.bvh_loader import load_bvh_frames, read_bvh_fps
from bvh_processing.retargeting.exporter import (
    RetargetArtifacts,
    export_tracking_artifacts,
)
from bvh_processing.retargeting.robot_retargeter import retarget_bvh_frames
from bvh_processing.services.classify_bvh import classify_downloaded_bvh
from bvh_processing.services.download import DownloadedBvh


def retarget_downloaded_bvh(downloaded: DownloadedBvh) -> RetargetArtifacts:
    """执行 Robot Retargeter，并生成训练 NPZ 与元数据 JSON。"""
    bvh_format = classify_downloaded_bvh(downloaded.content)
    try:
        with TemporaryDirectory(prefix="bvh-retarget-") as directory:
            bvh_path = Path(directory) / downloaded.source_filename
            downloaded.content.seek(0)
            bvh_path.write_bytes(downloaded.content.read())
            frames = load_bvh_frames(bvh_path, bvh_format)
            source_fps = read_bvh_fps(bvh_path)
            result = retarget_bvh_frames(
                frames,
                source_fps,
                solver="daqp",
                max_iterations=50,
            )
            return export_tracking_artifacts(
                result,
                source_fps,
                downloaded.source_filename,
                bvh_format,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        raise BvhServiceError(
            status_code=422,
            code="retarget_failed",
            message=f"BVH 重定向失败：{exc}",
        ) from exc
    finally:
        downloaded.content.seek(0)
