from pathlib import Path

from bvh_processing.services.download import DownloadedBvh


def processed_filename(source_filename: str) -> str:
    source = Path(source_filename)
    return f"{source.stem}_processed.bvh"


def process_bvh(
    downloaded: DownloadedBvh,
    handle_options: list[int],
) -> DownloadedBvh:
    """平滑算法接入点；联调阶段保持 BVH 内容不变。"""
    del handle_options
    return downloaded
