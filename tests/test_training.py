import asyncio
import sys
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from bvh_processing.config import Settings
from bvh_processing.errors import BvhServiceError
from bvh_processing.schemas import TrainBvhRequest
from bvh_processing.services.download import DownloadedBvh
from bvh_processing.training.service import (
    _run_process,
    _training_command,
    run_training_program,
)
from bvh_processing.training.validation import validate_training_npz


def _valid_arrays() -> dict[str, np.ndarray]:
    frames = 3
    quaternions = np.zeros((frames, 30, 4), dtype=np.float32)
    quaternions[..., 0] = 1.0
    return {
        "fps": np.asarray([50.0]),
        "joint_pos": np.zeros((frames, 29), dtype=np.float32),
        "joint_vel": np.zeros((frames, 29), dtype=np.float32),
        "body_pos_w": np.zeros((frames, 30, 3), dtype=np.float32),
        "body_quat_w": quaternions,
        "body_lin_vel_w": np.zeros((frames, 30, 3), dtype=np.float32),
        "body_ang_vel_w": np.zeros((frames, 30, 3), dtype=np.float32),
    }


def _npz_bytes(arrays: dict[str, np.ndarray] | None = None) -> bytes:
    content = BytesIO()
    np.savez(content, **(arrays or _valid_arrays()))
    return content.getvalue()


def _write_npz(path: Path, arrays: dict[str, np.ndarray] | None = None) -> None:
    path.write_bytes(_npz_bytes(arrays))


def _payload(**overrides: object) -> TrainBvhRequest:
    values: dict[str, object] = {
        "robotType": 1,
        "algorithmType": 1,
        "npzFileUrl": "https://minio.example.com/motion.npz",
        "domainRandomization": 2,
        "returnType": 2,
        "callbackUrl": "https://backend.example.com/callback",
    }
    values.update(overrides)
    return TrainBvhRequest.model_validate(values)


def test_default_settings_use_embedded_training_project() -> None:
    settings = Settings(_env_file=None)
    project_root = Path(settings.wbt_project_root)

    assert project_root.name == "whole_body_tracking"
    assert "bvh_processing/vendor" in project_root.as_posix()
    assert (project_root / "scripts" / "rsl_rl" / "train.py").is_file()
    assert (project_root / "scripts" / "mujoco_sim2sim.py").is_file()
    assert (
        project_root
        / "source"
        / "whole_body_tracking"
        / "whole_body_tracking"
        / "assets"
        / "unitree_description"
        / "mjcf"
        / "g1.xml"
    ).is_file()


def test_validate_training_npz_accepts_exporter_shape(tmp_path: Path) -> None:
    path = tmp_path / "motion.npz"
    _write_npz(path)

    validate_training_npz(path, 10 * 1024 * 1024)


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    [
        (lambda arrays: arrays.pop("joint_vel"), "缺少数组：joint_vel"),
        (
            lambda arrays: arrays.__setitem__(
                "joint_pos", np.zeros((3, 28), dtype=np.float32)
            ),
            "[frames, 29]",
        ),
        (
            lambda arrays: arrays["body_pos_w"].__setitem__((0, 0, 0), np.nan),
            "包含 NaN 或 Inf",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "body_quat_w", np.zeros((3, 30, 4), dtype=np.float32)
            ),
            "四元数未正确归一化",
        ),
    ],
)
def test_validate_training_npz_rejects_invalid_motion(
    tmp_path: Path,
    mutation: Callable[[dict[str, np.ndarray]], object],
    expected_message: str,
) -> None:
    arrays = _valid_arrays()
    mutation(arrays)
    path = tmp_path / "invalid.npz"
    _write_npz(path, arrays)

    with pytest.raises(BvhServiceError) as error:
        validate_training_npz(path, 10 * 1024 * 1024)

    assert error.value.code == "invalid_training_npz"
    assert expected_message in error.value.message


def test_training_command_passes_request_controls(tmp_path: Path) -> None:
    payload = _payload(gpu=2, numEnvs=4096, maxIterations=1234, seed=7)
    settings = Settings(wbt_task="Tracking-Test-v0")

    command = _training_command(
        tmp_path,
        Path("/isaac/bin/python"),
        tmp_path / "motion.npz",
        "bvh_run",
        payload,
        settings,
    )

    assert command[0] == "/isaac/bin/python"
    assert command[command.index("--task") + 1] == "Tracking-Test-v0"
    assert command[command.index("--device") + 1] == "cuda:2"
    assert command[command.index("--num_envs") + 1] == "4096"
    assert command[command.index("--max_iterations") + 1] == "1234"
    assert command[command.index("--seed") + 1] == "7"


def test_run_process_reports_external_failure(tmp_path: Path) -> None:
    log_path = tmp_path / "failed.log"

    with pytest.raises(BvhServiceError) as error:
        asyncio.run(
            _run_process(
                [sys.executable, "-c", "import sys; print('failure'); sys.exit(3)"],
                cwd=tmp_path,
                environment={},
                timeout=10,
                log_path=log_path,
                operation="training",
            )
        )

    assert error.value.code == "training_failed"
    assert "failure" in error.value.message


def test_run_process_terminates_on_timeout(tmp_path: Path) -> None:
    with pytest.raises(BvhServiceError) as error:
        asyncio.run(
            _run_process(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                cwd=tmp_path,
                environment={},
                timeout=0.01,
                log_path=tmp_path / "timeout.log",
                operation="render",
            )
        )

    assert error.value.code == "render_timeout"


@pytest.mark.parametrize(
    ("return_type", "asset_name", "content_type", "content"),
    [
        (1, "demo.mp4", "video/mp4", b"demo-video"),
        (2, "demo.onnx", "application/octet-stream", b"demo-policy"),
    ],
)
def test_missing_isaac_lab_returns_bundled_demo_artifact(
    tmp_path: Path,
    return_type: int,
    asset_name: str,
    content_type: str,
    content: bytes,
) -> None:
    asset_path = tmp_path / asset_name
    asset_path.write_bytes(content)
    source = DownloadedBvh(
        content=BytesIO(b"downloaded-npz-is-unused-in-fallback"),
        source_filename="dance_g1_tracking.npz",
        size=1,
    )
    settings = Settings(
        _env_file=None,
        wbt_project_root=str(tmp_path / "missing-wbt"),
        wbt_python=str(tmp_path / "missing-isaac" / "python"),
    )

    constant = "_DEMO_VIDEO_PATH" if return_type == 1 else "_DEMO_POLICY_PATH"
    with patch(f"bvh_processing.training.service.{constant}", asset_path):
        artifact = asyncio.run(
            run_training_program(source, _payload(returnType=return_type), settings)
        )

    try:
        assert artifact.content.read() == content
        assert artifact.filename == asset_name
        assert artifact.content_type == content_type
    finally:
        artifact.close()
        source.content.close()


@pytest.mark.parametrize(
    ("return_type", "expected_content", "expected_suffix", "expected_calls"),
    [
        (2, b"onnx-output", ".onnx", 1),
        (1, b"mp4-output", ".mp4", 2),
    ],
)
def test_run_training_program_returns_real_generated_artifact(
    tmp_path: Path,
    return_type: int,
    expected_content: bytes,
    expected_suffix: str,
    expected_calls: int,
) -> None:
    project_root = tmp_path / "wbt"
    python = tmp_path / "isaac" / "bin" / "python"
    xml = project_root / "g1.xml"
    project_root.mkdir()
    payload = _payload(returnType=return_type, maxIterations=5)
    settings = Settings(train_workspace_root=str(tmp_path / "work"))
    source = DownloadedBvh(
        content=BytesIO(_npz_bytes()),
        source_filename="dance_g1_tracking.npz",
        size=1,
    )
    calls: list[list[str]] = []

    async def fake_run_process(command: list[str], **kwargs: object) -> None:
        calls.append(command)
        if "--run_name" in command:
            run_name = command[command.index("--run_name") + 1]
            output = (
                project_root
                / "logs"
                / "rsl_rl"
                / "g1_flat"
                / f"2026-01-01_00-00-00_{run_name}"
                / "model_5.onnx"
            )
            output.parent.mkdir(parents=True)
            output.write_bytes(b"onnx-output")
        else:
            video_path = Path(command[command.index("--video-path") + 1])
            video_path.write_bytes(b"mp4-output")

    with (
        patch(
            "bvh_processing.training.service._wbt_paths",
            return_value=(project_root, python, xml),
        ),
        patch(
            "bvh_processing.training.service._run_process",
            side_effect=fake_run_process,
        ),
    ):
        artifact = asyncio.run(run_training_program(source, payload, settings))

    try:
        assert artifact.content.read() == expected_content
        assert artifact.filename == f"dance_g1_tracking_trained{expected_suffix}"
        assert len(calls) == expected_calls
    finally:
        artifact.close()
        source.content.close()
