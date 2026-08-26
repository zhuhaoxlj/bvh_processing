"""调用外部训练程序并收集 MP4 或 ONNX 训练产物。"""

import asyncio
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import SpooledTemporaryFile, TemporaryDirectory
from typing import BinaryIO

from bvh_processing.config import Settings
from bvh_processing.errors import BvhServiceError
from bvh_processing.schemas import TrainBvhRequest
from bvh_processing.services.download import DownloadedBvh

_SPOOL_MEMORY_LIMIT = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class TrainingArtifact:
    content: BinaryIO
    filename: str
    content_type: str

    def close(self) -> None:
        self.content.close()


def _artifact_details(return_type: int) -> tuple[str, str]:
    if return_type == 1:
        return "training_preview.mp4", "video/mp4"
    return "trained_policy.onnx", "application/octet-stream"


async def run_training_program(
    source: DownloadedBvh,
    payload: TrainBvhRequest,
    settings: Settings,
) -> TrainingArtifact:
    """运行 BVH_TRAIN_COMMAND，固定参数协议见 README。"""
    command = shlex.split(settings.train_command)
    if not command:
        raise BvhServiceError(
            status_code=503,
            code="train_command_not_configured",
            message="训练程序尚未配置",
        )

    output_filename, content_type = _artifact_details(payload.return_type)
    with TemporaryDirectory(prefix="bvh-train-") as temporary_directory:
        workdir = Path(temporary_directory)
        input_path = workdir / "retargeted_motion.npz"
        output_path = workdir / output_filename

        source.content.seek(0)
        await asyncio.to_thread(_copy_stream, source.content, input_path)

        process = await asyncio.create_subprocess_exec(
            *command,
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--robot-type",
            str(payload.robot_type),
            "--algorithm-type",
            str(payload.algorithm_type),
            "--domain-randomization",
            str(payload.domain_randomization),
            "--return-type",
            str(payload.return_type),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(
                process.communicate(), timeout=settings.train_timeout_seconds
            )
        except TimeoutError as exc:
            process.kill()
            await process.communicate()
            raise BvhServiceError(
                status_code=504,
                code="training_timeout",
                message="训练任务执行超时",
            ) from exc

        if process.returncode != 0:
            error_detail = stderr.decode(errors="replace").strip()[-500:]
            message = "训练程序执行失败"
            if error_detail:
                message = f"{message}：{error_detail}"
            raise BvhServiceError(
                status_code=500,
                code="training_failed",
                message=message,
            )
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise BvhServiceError(
                status_code=500,
                code="training_artifact_missing",
                message="训练程序未生成回传文件",
            )

        content = SpooledTemporaryFile(  # noqa: SIM115
            max_size=_SPOOL_MEMORY_LIMIT,
            mode="w+b",
        )
        with output_path.open("rb") as output_file:
            shutil.copyfileobj(output_file, content)
        content.seek(0)

    source_stem = Path(source.source_filename).stem
    suffix = ".mp4" if payload.return_type == 1 else ".onnx"
    return TrainingArtifact(
        content=content,
        filename=f"{source_stem}_trained{suffix}",
        content_type=content_type,
    )


def _copy_stream(source: BinaryIO, destination: Path) -> None:
    with destination.open("wb") as destination_file:
        shutil.copyfileobj(source, destination_file)
