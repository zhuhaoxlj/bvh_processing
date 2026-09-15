"""Run the local MDM project in its own Python environment."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

from bvh_processing.config import Settings
from bvh_processing.services.download import DownloadedBvh

_MDM_LOCK = Lock()


def generate_mdm_merge(
    downloaded_files: list[DownloadedBvh],
    intervals_seconds: list[float],
    settings: Settings,
) -> bytes:
    if len(downloaded_files) < 2:
        raise ValueError("MDM 合并至少需要两个 BVH 文件")
    if len(intervals_seconds) != len(downloaded_files) - 1:
        raise ValueError("BVH 文件数量与过渡时长数量不匹配")

    project_root = Path(settings.mdm_project_root).expanduser().resolve()
    # Resolving the executable symlink would bypass the virtual environment and
    # run its base interpreter without the MDM dependencies.
    python = Path(settings.mdm_python).expanduser().absolute()
    if not project_root.is_dir():
        raise ValueError(f"MDM 项目目录不存在：{project_root}")
    if not python.is_file():
        raise ValueError(f"MDM Python 不存在：{python}")

    with TemporaryDirectory(prefix="bvh-mdm-") as directory:
        workspace = Path(directory)
        inputs: list[Path] = []
        for index, downloaded in enumerate(downloaded_files):
            path = workspace / f"input-{index}.bvh"
            downloaded.content.seek(0)
            path.write_bytes(downloaded.content.read())
            inputs.append(path)

        output = workspace / "merged.bvh"
        command = [
            str(python),
            "-m",
            "bvh_workbench.merge_cli",
            "--output",
            str(output),
            "--seed",
            str(settings.mdm_seed),
            "--scale",
            str(settings.mdm_source_scale),
            "--up-axis",
            settings.mdm_source_up_axis,
        ]
        for seconds in intervals_seconds:
            command.extend(("--transition-seconds", str(seconds)))
        command.extend(str(path) for path in inputs)

        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        try:
            with _MDM_LOCK:
                completed = subprocess.run(
                    command,
                    cwd=project_root,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=settings.mdm_timeout_seconds,
                    check=False,
                )
        except subprocess.TimeoutExpired as error:
            raise ValueError(
                f"MDM 生成超过 {settings.mdm_timeout_seconds:g} 秒，任务已终止"
            ) from error
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            detail = detail[-4000:]
            raise ValueError(f"MDM 生成失败：{detail or '子进程未返回错误信息'}")
        if not output.is_file():
            raise ValueError("MDM 子进程成功结束但未生成 BVH 文件")
        result = output.read_bytes()
        if not result:
            raise ValueError("MDM 子进程生成了空 BVH 文件")
        return result
