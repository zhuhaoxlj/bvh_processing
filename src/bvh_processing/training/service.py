"""调用 Whole Body Tracking 训练并收集 MP4 或 ONNX 产物。"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import SpooledTemporaryFile, TemporaryDirectory
from typing import BinaryIO
from uuid import uuid4

from bvh_processing.config import Settings
from bvh_processing.errors import BvhServiceError
from bvh_processing.schemas import TrainBvhRequest
from bvh_processing.services.download import DownloadedBvh
from bvh_processing.training.validation import validate_training_npz

_SPOOL_MEMORY_LIMIT = 8 * 1024 * 1024
_DEMO_POLICY_DIRECTORY = Path(__file__).resolve().parents[1] / "assets" / "policy_demo"
_DEMO_POLICY_PATH = _DEMO_POLICY_DIRECTORY / "nytwm_70500.onnx"
_DEMO_VIDEO_PATH = _DEMO_POLICY_DIRECTORY / "1a2_34000.mp4"
_SEMAPHORE_LOOP: asyncio.AbstractEventLoop | None = None
_TRAINING_SEMAPHORE: asyncio.Semaphore | None = None
_TRAINING_SEMAPHORE_SIZE = 0

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrainingArtifact:
    content: BinaryIO
    filename: str
    content_type: str

    def close(self) -> None:
        self.content.close()


def _get_training_semaphore(limit: int) -> asyncio.Semaphore:
    global _SEMAPHORE_LOOP, _TRAINING_SEMAPHORE, _TRAINING_SEMAPHORE_SIZE

    loop = asyncio.get_running_loop()
    size = max(1, limit)
    if (
        _TRAINING_SEMAPHORE is None
        or _SEMAPHORE_LOOP is not loop
        or _TRAINING_SEMAPHORE_SIZE != size
    ):
        _SEMAPHORE_LOOP = loop
        _TRAINING_SEMAPHORE = asyncio.Semaphore(size)
        _TRAINING_SEMAPHORE_SIZE = size
    return _TRAINING_SEMAPHORE


def _wbt_paths(settings: Settings) -> tuple[Path, Path, Path]:
    project_root = Path(settings.wbt_project_root).expanduser().resolve()
    python = Path(settings.wbt_python).expanduser().resolve()
    xml = Path(settings.wbt_mujoco_xml).expanduser()
    if not xml.is_absolute():
        xml = project_root / xml

    required = (
        (project_root / "scripts" / "rsl_rl" / "train.py", "训练脚本"),
        (project_root / "scripts" / "mujoco_sim2sim.py", "渲染脚本"),
        (python, "Isaac Lab Python"),
        (xml, "G1 MuJoCo XML"),
    )
    missing = [label for path, label in required if not path.is_file()]
    if not missing:
        try:
            probe = subprocess.run(
                [str(python), "-c", "import isaaclab"],
                cwd=project_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            probe = None
        if probe is None or probe.returncode != 0:
            missing.append("Isaac Lab Python 模块")
    if missing:
        raise BvhServiceError(
            status_code=503,
            code="training_environment_not_configured",
            message=f"Whole Body Tracking 环境未配置完整：{', '.join(missing)}",
        )
    return project_root, python, xml.resolve()


def _training_command(
    project_root: Path,
    python: Path,
    motion_path: Path,
    run_name: str,
    payload: TrainBvhRequest,
    settings: Settings,
) -> list[str]:
    return [
        str(python),
        str(project_root / "scripts" / "rsl_rl" / "train.py"),
        "--task",
        settings.wbt_task,
        "--motion_file",
        str(motion_path),
        "--num_envs",
        str(payload.num_envs),
        "--max_iterations",
        str(payload.max_iterations),
        "--save_interval",
        str(min(500, payload.max_iterations)),
        "--seed",
        str(payload.seed),
        "--run_name",
        run_name,
        "--logger",
        "tensorboard",
        "--headless",
        "--device",
        f"cuda:{payload.gpu}",
    ]


def _render_command(
    project_root: Path,
    python: Path,
    xml: Path,
    model_path: Path,
    video_path: Path,
) -> list[str]:
    return [
        str(python),
        str(project_root / "scripts" / "mujoco_sim2sim.py"),
        "--model",
        str(model_path),
        "--xml",
        str(xml),
        "--headless",
        "--record-video",
        "--record-one-motion",
        "--video-path",
        str(video_path),
        "--video-width",
        "1280",
        "--video-height",
        "720",
        "--video-fps",
        "50",
    ]


def _process_environment(project_root: Path, python: Path) -> dict[str, str]:
    environment = os.environ.copy()
    source_path = str(project_root / "source" / "whole_body_tracking")
    current_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_path}{os.pathsep}{current_pythonpath}"
        if current_pythonpath
        else source_path
    )
    environment["VIRTUAL_ENV"] = str(python.parent.parent)
    environment.setdefault("OMP_NUM_THREADS", "8")
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return environment


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=10)
    except TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()


async def _run_process(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    timeout: float,
    log_path: Path,
    operation: str,
) -> None:
    with log_path.open("wb") as log_file:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            env=environment,
            stdout=log_file,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return_code = await asyncio.wait_for(process.wait(), timeout=timeout)
        except TimeoutError as exc:
            await _stop_process(process)
            raise BvhServiceError(
                status_code=504,
                code=f"{operation}_timeout",
                message=f"{operation}任务执行超时",
            ) from exc

    if return_code != 0:
        detail = log_path.read_text(encoding="utf-8", errors="replace")[-1000:].strip()
        message = f"{operation}程序执行失败"
        if detail:
            message = f"{message}：{detail}"
        raise BvhServiceError(
            status_code=500,
            code=f"{operation}_failed",
            message=message,
        )


def _find_onnx(project_root: Path, run_name: str) -> Path:
    experiment_root = project_root / "logs" / "rsl_rl" / "g1_flat"
    run_directories = [
        path for path in experiment_root.glob(f"*_{run_name}") if path.is_dir()
    ]
    if not run_directories:
        raise BvhServiceError(
            status_code=500,
            code="training_artifact_missing",
            message="训练完成但未找到本次训练输出目录",
        )
    run_directory = max(run_directories, key=lambda path: path.stat().st_mtime_ns)
    models = [
        path
        for path in run_directory.glob("model_*.onnx")
        if path.is_file() and path.stat().st_size > 0
    ]
    if not models:
        raise BvhServiceError(
            status_code=500,
            code="training_artifact_missing",
            message="训练完成但未生成 ONNX 模型",
        )
    return max(models, key=lambda path: path.stat().st_mtime_ns)


def _copy_artifact(path: Path, filename: str, content_type: str) -> TrainingArtifact:
    if not path.is_file() or path.stat().st_size == 0:
        raise BvhServiceError(
            status_code=500,
            code="training_artifact_missing",
            message="训练程序未生成有效回传文件",
        )
    content = SpooledTemporaryFile(  # noqa: SIM115
        max_size=_SPOOL_MEMORY_LIMIT,
        mode="w+b",
    )
    with path.open("rb") as source:
        shutil.copyfileobj(source, content)
    content.seek(0)
    return TrainingArtifact(content, filename, content_type)


def _demo_artifact(return_type: int) -> TrainingArtifact:
    if return_type == 1:
        return _copy_artifact(_DEMO_VIDEO_PATH, _DEMO_VIDEO_PATH.name, "video/mp4")
    return _copy_artifact(
        _DEMO_POLICY_PATH,
        _DEMO_POLICY_PATH.name,
        "application/octet-stream",
    )


async def run_training_program(
    source: DownloadedBvh,
    payload: TrainBvhRequest,
    settings: Settings,
) -> TrainingArtifact:
    """有 Isaac Lab 时训练，否则回传内置的演示产物。"""

    try:
        project_root, python, xml = _wbt_paths(settings)
    except BvhServiceError as error:
        if error.code != "training_environment_not_configured":
            raise
        logger.warning(
            "Isaac Lab environment unavailable; returning bundled demo artifact: %s",
            error.message,
        )
        return _demo_artifact(payload.return_type)

    workspace_root = Path(settings.train_workspace_root).expanduser().resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)
    run_name = f"bvh_{uuid4().hex}"
    environment = _process_environment(project_root, python)

    with TemporaryDirectory(prefix=f"{run_name}-", dir=workspace_root) as directory:
        workdir = Path(directory)
        motion_path = workdir / "motion.npz"
        video_path = workdir / "preview.mp4"
        source.content.seek(0)
        await asyncio.to_thread(_copy_stream, source.content, motion_path)
        await asyncio.to_thread(
            validate_training_npz,
            motion_path,
            settings.npz_max_uncompressed_bytes,
        )

        semaphore = _get_training_semaphore(settings.train_max_concurrency)
        async with semaphore:
            await _run_process(
                _training_command(
                    project_root,
                    python,
                    motion_path,
                    run_name,
                    payload,
                    settings,
                ),
                cwd=project_root,
                environment=environment,
                timeout=settings.train_timeout_seconds,
                log_path=workdir / "training.log",
                operation="training",
            )
            model_path = _find_onnx(project_root, run_name)

            source_stem = Path(source.source_filename).stem
            if payload.return_type == 2:
                return _copy_artifact(
                    model_path,
                    f"{source_stem}_trained.onnx",
                    "application/octet-stream",
                )

            render_environment = environment | {"MUJOCO_GL": "egl"}
            await _run_process(
                _render_command(
                    project_root,
                    python,
                    xml,
                    model_path,
                    video_path,
                ),
                cwd=project_root,
                environment=render_environment,
                timeout=settings.render_timeout_seconds,
                log_path=workdir / "render.log",
                operation="render",
            )
            return _copy_artifact(
                video_path,
                f"{source_stem}_trained.mp4",
                "video/mp4",
            )


def _copy_stream(source: BinaryIO, destination: Path) -> None:
    with destination.open("wb") as destination_file:
        shutil.copyfileobj(source, destination_file)
