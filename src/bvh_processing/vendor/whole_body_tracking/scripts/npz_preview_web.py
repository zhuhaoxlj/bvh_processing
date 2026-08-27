#!/usr/bin/env python3
"""Local web UI for converting HHTools PKL, validating motion NPZ, and rendering previews.

Run this script with the Isaac Lab virtual environment. The HTTP server itself
only needs NumPy; Isaac Sim is launched in child processes for PKL conversion
and preview rendering.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import pickle
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECOVERY_UTILS_ROOT = (
    PROJECT_ROOT / "source" / "whole_body_tracking" / "whole_body_tracking" / "utils"
)
if str(RECOVERY_UTILS_ROOT) not in sys.path:
    sys.path.insert(0, str(RECOVERY_UTILS_ROOT))

from training_recovery_package import (  # noqa: E402
    MAX_PACKAGE_BYTES,
    RecoveryPackageError,
    build_recovery_package,
    validate_and_extract_recovery_package,
)

WEB_ROOT = PROJECT_ROOT / "tools" / "npz_preview_web"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "npz_preview_web"
MANUAL_LOG_ROOT = PROJECT_ROOT / "logs" / "manual"
REPLAY_SCRIPT = PROJECT_ROOT / "scripts" / "replay_npz.py"
CSV_TO_NPZ_SCRIPT = PROJECT_ROOT / "scripts" / "csv_to_npz.py"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "rsl_rl" / "train.py"
TRAIN_TASK = "Tracking-Flat-G1-Wo-State-Estimation-v0"
TRAIN_MOTION_FPS = 50
# RTX 4090 presets ordered from aggressive dual-GPU utilization to safer fallbacks.
TRAIN_NUM_ENVS = (24576, 22528, 20480, 18432, 16384, 12288, 7168)
TENSORBOARD_HOST = "127.0.0.1"
TENSORBOARD_DEFAULT_PORT = 6006
TENSORBOARD_CANDIDATE_PORTS = tuple(range(TENSORBOARD_DEFAULT_PORT, TENSORBOARD_DEFAULT_PORT + 10))
TENSORBOARD_SESSION = "wbt_tensorboard"
TENSORBOARD_LOG_ROOT = PROJECT_ROOT / "logs" / "rsl_rl" / "g1_flat"
TRAINING_STOP_GRACE_SECONDS = 120.0
RENDER_PROCESS_TIMEOUT_SECONDS = 300.0
PKL_CONVERSION_TIMEOUT_SECONDS = 900.0
REQUIRED_ARRAYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
SYSTEM_INFO_CACHE_SECONDS = 2.5
NVIDIA_SMI_TIMEOUT_SECONDS = 20.0
G1_TRAINING_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


system_info_cache_lock = threading.Lock()
system_info_cache: dict[str, Any] | None = None
system_info_cache_updated_at = 0.0
system_info_refreshing = False


def strict_integer_value(value: Any, label: str) -> int:
    """Parse an integer without silently truncating fractional request values."""

    if isinstance(value, bool):
        raise ValueError(f"{label}必须是整数。")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return int(value)
        raise ValueError(f"{label}必须是整数。")
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value)
    raise ValueError(f"{label}必须是整数。")


@dataclass(frozen=True)
class PPOTrainingSettings:
    """Validated runner, rollout, and optimizer controls supplied by the web UI."""

    num_steps_per_env: int = 24
    num_mini_batches: int = 4
    num_learning_epochs: int = 5
    learning_rate: float = 1.0e-3
    desired_kl: float = 0.01
    save_interval: int = 500

    @classmethod
    def from_mapping(cls, values: dict[str, Any], *, num_envs: int) -> PPOTrainingSettings:
        def finite_float_value(key: str, default: float, label: str) -> float:
            try:
                value = float(values.get(key, default))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label}必须是有效数字。") from exc
            if not math.isfinite(value):
                raise ValueError(f"{label}必须是有限数字。")
            return value

        settings = cls(
            num_steps_per_env=strict_integer_value(
                values.get("num_steps_per_env", cls.num_steps_per_env), "每环境采样步数"
            ),
            num_mini_batches=strict_integer_value(
                values.get("num_mini_batches", cls.num_mini_batches), "Mini-batches"
            ),
            num_learning_epochs=strict_integer_value(
                values.get("num_learning_epochs", cls.num_learning_epochs), "Learning epochs"
            ),
            learning_rate=finite_float_value("learning_rate", cls.learning_rate, "学习率"),
            desired_kl=finite_float_value("desired_kl", cls.desired_kl, "Desired KL"),
            save_interval=strict_integer_value(
                values.get("save_interval", cls.save_interval), "模型保存间隔"
            ),
        )

        if not 1 <= settings.num_steps_per_env <= 256:
            raise ValueError("每环境采样步数必须在 1 到 256 之间。")
        if not 1 <= settings.num_mini_batches <= 128:
            raise ValueError("Mini-batches 必须在 1 到 128 之间。")
        if not 1 <= settings.num_learning_epochs <= 20:
            raise ValueError("Learning epochs 必须在 1 到 20 之间。")
        if not 1.0e-6 <= settings.learning_rate <= 0.1:
            raise ValueError("学习率必须在 0.000001 到 0.1 之间。")
        if not 1.0e-5 <= settings.desired_kl <= 1.0:
            raise ValueError("Desired KL 必须在 0.00001 到 1.0 之间。")
        if not 1 <= settings.save_interval <= 100000:
            raise ValueError("模型保存间隔必须在 1 到 100000 之间。")
        rollout_size = num_envs * settings.num_steps_per_env
        if rollout_size % settings.num_mini_batches:
            raise ValueError(
                "每张 GPU 的环境数 × 每环境采样步数必须能被 Mini-batches 整除。"
            )
        return settings

    def as_dict(self) -> dict[str, int | float]:
        return {
            "num_steps_per_env": self.num_steps_per_env,
            "num_mini_batches": self.num_mini_batches,
            "num_learning_epochs": self.num_learning_epochs,
            "learning_rate": self.learning_rate,
            "desired_kl": self.desired_kl,
            "save_interval": self.save_interval,
        }


@dataclass(frozen=True)
class TrainingCheckpoint:
    """A checkpoint owned by one Motion Inspector training job."""

    run_directory: str
    checkpoint_name: str
    iteration: int
    path: Path

    @property
    def checkpoint_id(self) -> str:
        return f"{self.run_directory}/{self.checkpoint_name}"

    def public(self) -> dict[str, Any]:
        file_stat = self.path.stat()
        return {
            "id": self.checkpoint_id,
            "run_directory": self.run_directory,
            "checkpoint_name": self.checkpoint_name,
            "iteration": self.iteration,
            "size_bytes": file_stat.st_size,
            "modified_at": file_stat.st_mtime,
        }


def parse_training_request_settings(values: dict[str, Any]) -> tuple[int, int, PPOTrainingSettings]:
    """Parse the numeric training controls submitted to the HTTP API."""

    num_envs = strict_integer_value(values.get("num_envs", 7168), "每张 GPU 环境数")
    max_iterations = strict_integer_value(values.get("max_iterations", 10000), "训练迭代数")
    return num_envs, max_iterations, PPOTrainingSettings.from_mapping(values, num_envs=num_envs)


class MotionInspectorHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server sized for several open dashboard tabs."""

    request_queue_size = 128
    daemon_threads = True


def tensorboard_session_name(port: int) -> str:
    """Return the project-owned tmux session name for a TensorBoard port."""

    if port == TENSORBOARD_DEFAULT_PORT:
        return TENSORBOARD_SESSION
    return f"{TENSORBOARD_SESSION}_{port}"


def discover_project_tensorboard_port() -> int:
    """Find a live TensorBoard port owned by this project's tmux session."""

    completed = subprocess.run(
        ["tmux", "list-sessions", "-F", "#{session_name}"],
        capture_output=True,
        text=True,
        check=False,
    )
    session_names = set(completed.stdout.splitlines()) if completed.returncode == 0 else set()
    for candidate_port in TENSORBOARD_CANDIDATE_PORTS:
        if tensorboard_session_name(candidate_port) not in session_names:
            continue
        try:
            with socket.create_connection((TENSORBOARD_HOST, candidate_port), timeout=0.25):
                return candidate_port
        except OSError:
            continue
    return TENSORBOARD_DEFAULT_PORT


tensorboard_port = discover_project_tensorboard_port()


def gpu_inventory() -> list[dict[str, Any]]:
    """Return an nvidia-smi snapshot for every GPU visible to this host."""

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []
    try:
        gpu_query = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
            check=False,
        )
        process_query = subprocess.run(
            [
                nvidia_smi,
                "--query-compute-apps=pid,gpu_uuid,used_memory,process_name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if gpu_query.returncode != 0:
        return []

    def parse_number(value: str, number_type: type[int] | type[float]) -> int | float | None:
        try:
            return number_type(value)
        except ValueError:
            return None

    gpus: list[dict[str, Any]] = []
    gpu_by_uuid: dict[str, dict[str, Any]] = {}
    for line in gpu_query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 9)]
        if len(fields) != 10:
            continue
        index = parse_number(fields[0], int)
        memory_total_mib = parse_number(fields[3], int)
        if index is None or memory_total_mib is None:
            continue
        gpu = {
            "index": index,
            "uuid": fields[1],
            "name": fields[2],
            "memory_mib": memory_total_mib,
            "memory_used_mib": parse_number(fields[4], int),
            "memory_free_mib": parse_number(fields[5], int),
            "utilization_percent": parse_number(fields[6], int),
            "temperature_celsius": parse_number(fields[7], int),
            "power_draw_watts": parse_number(fields[8], float),
            "power_limit_watts": parse_number(fields[9], float),
            "processes": [],
        }
        gpus.append(gpu)
        gpu_by_uuid[fields[1]] = gpu

    if process_query.returncode == 0:
        for line in process_query.stdout.splitlines():
            fields = [field.strip() for field in line.split(",", 3)]
            if len(fields) != 4 or fields[1] not in gpu_by_uuid:
                continue
            process_id = parse_number(fields[0], int)
            used_memory_mib = parse_number(fields[2], int)
            if process_id is None:
                continue
            gpu_by_uuid[fields[1]]["processes"].append(
                {
                    "pid": process_id,
                    "used_memory_mib": used_memory_mib,
                    "name": fields[3],
                }
            )
    return gpus


def _training_process_lines() -> list[str]:
    """Return command lines for training processes visible in this container."""

    completed = subprocess.run(
        ["pgrep", "-af", "[s]cripts/rsl_rl/train.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.splitlines() if completed.returncode == 0 else []


def _training_devices_from_process_line(process_line: str) -> list[str]:
    physical_gpu_ids_match = re.search(
        r"--physical_gpu_ids(?:=|\s+)([0-9]+(?:,[0-9]+)*)", process_line
    )
    if physical_gpu_ids_match:
        return [f"cuda:{gpu_index}" for gpu_index in physical_gpu_ids_match.group(1).split(",")]
    device_match = re.search(r"--device(?:=|\s+)(cuda:\d+)", process_line)
    return [device_match.group(1) if device_match else "cuda:0"]


def _training_argument(process_line: str, argument_name: str) -> str | None:
    match = re.search(rf"{re.escape(argument_name)}(?:=|\s+)([^\s]+)", process_line)
    return match.group(1) if match else None


def _last_logged_training_iteration(run_name: str) -> tuple[int | None, int | None]:
    """Read the current iteration from the final log window without loading the full log."""

    log_path = MANUAL_LOG_ROOT / f"{run_name}.out"
    try:
        with log_path.open("rb") as log_file:
            log_file.seek(0, os.SEEK_END)
            log_file.seek(max(0, log_file.tell() - 64 * 1024))
            tail = log_file.read().decode("utf-8", errors="replace")
    except OSError:
        return None, None

    matches = list(re.finditer(r"Learning iteration\s+(\d+)/(\d+)", tail))
    if not matches:
        return None, None
    latest_match = matches[-1]
    return int(latest_match.group(1)), int(latest_match.group(2))


def active_training_runs() -> list[dict[str, Any]]:
    """Return active training metadata keyed by each occupied CUDA device."""

    runs_by_device_and_name: dict[tuple[str, str], dict[str, Any]] = {}
    for process_line in _training_process_lines():
        run_name = _training_argument(process_line, "--run_name") or "unknown"
        iteration, max_iterations = _last_logged_training_iteration(run_name)
        for device in _training_devices_from_process_line(process_line):
            runs_by_device_and_name[(device, run_name)] = {
                "device": device,
                "run_name": run_name,
                "iteration": iteration,
                "max_iterations": max_iterations,
            }
    return list(runs_by_device_and_name.values())


def active_training_devices() -> set[str]:
    """Return CUDA devices used by active training processes."""

    return {run["device"] for run in active_training_runs()}


def _refresh_system_info_cache() -> None:
    """Refresh GPU telemetry outside HTTP request threads."""

    global system_info_cache, system_info_cache_updated_at, system_info_refreshing

    try:
        gpus = gpu_inventory()
        training_runs = active_training_runs()
        training_devices = sorted({run["device"] for run in training_runs})
        for gpu in gpus:
            gpu["training_runs"] = [
                run for run in training_runs if run["device"] == f"cuda:{gpu['index']}"
            ]
        current_time = time.monotonic()
        with system_info_cache_lock:
            if gpus:
                system_info_cache = {
                    "gpus": gpus,
                    "active_training_devices": training_devices,
                    "stale": False,
                    "refreshing": False,
                }
            elif system_info_cache is not None and system_info_cache.get("gpus"):
                # CUDA context changes can briefly make nvidia-smi unavailable.
                system_info_cache = {
                    **system_info_cache,
                    "active_training_devices": training_devices,
                    "stale": True,
                    "refreshing": False,
                    "warning": "GPU telemetry refresh failed; showing the last successful snapshot.",
                }
            else:
                system_info_cache = {
                    "gpus": [],
                    "active_training_devices": training_devices,
                    "stale": True,
                    "refreshing": False,
                    "warning": "GPU telemetry is temporarily unavailable.",
                }
            system_info_cache_updated_at = current_time
    finally:
        with system_info_cache_lock:
            system_info_refreshing = False


def system_info_snapshot() -> dict[str, Any]:
    """Return immediately and refresh stale GPU telemetry in a background thread."""

    global system_info_refreshing

    should_start_refresh = False
    with system_info_cache_lock:
        current_time = time.monotonic()
        cache_age_seconds = current_time - system_info_cache_updated_at
        if system_info_cache is not None and cache_age_seconds < SYSTEM_INFO_CACHE_SECONDS:
            return system_info_cache

        if not system_info_refreshing:
            system_info_refreshing = True
            should_start_refresh = True

        snapshot = {
            **(
                system_info_cache
                or {
                    "gpus": [],
                    "active_training_devices": [],
                    "warning": "GPU telemetry is loading.",
                }
            ),
            "stale": system_info_cache is not None,
            "refreshing": True,
        }

    if should_start_refresh:
        refresh_thread = threading.Thread(
            target=_refresh_system_info_cache,
            name="gpu-telemetry-refresh",
            daemon=True,
        )
        refresh_thread.start()
    return snapshot


def _safe_float(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _array_summary(array: np.ndarray) -> dict[str, Any]:
    numeric = np.issubdtype(array.dtype, np.number)
    finite = bool(np.isfinite(array).all()) if numeric else False
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": finite,
        "min": _safe_float(np.min(array)) if numeric and array.size else None,
        "max": _safe_float(np.max(array)) if numeric and array.size else None,
    }


class RestrictedNumpyUnpickler(pickle.Unpickler):
    """Load NumPy arrays from HHTools without allowing arbitrary pickle globals."""

    _allowed_globals = {
        ("numpy._core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
        ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy", "dtype"): np.dtype,
    }

    def find_class(self, module: str, name: str) -> Any:
        value = self._allowed_globals.get((module, name))
        if value is None:
            raise pickle.UnpicklingError(f"不允许的对象类型：{module}.{name}")
        return value


def extract_hhtools_pkl_to_csv(source_path: Path, csv_path: Path) -> dict[str, Any]:
    """Validate an HHTools robot PKL and write the 36-column CSV expected by csv_to_npz.py."""

    try:
        payload = RestrictedNumpyUnpickler(io.BytesIO(source_path.read_bytes())).load()
    except (OSError, EOFError, pickle.PickleError, ValueError, TypeError) as exc:
        raise ValueError(f"PKL 安全解析失败：{exc}") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("robot"), dict):
        raise ValueError("不是受支持的 HHTools PKL：缺少 robot 数据。")
    robot = payload["robot"]
    if "joint_q" not in robot or "dof_names" not in robot:
        raise ValueError("HHTools PKL 缺少 robot.joint_q 或 robot.dof_names。")

    joint_q = np.asarray(robot["joint_q"])
    if joint_q.ndim != 2 or joint_q.shape[0] < 2:
        raise ValueError("robot.joint_q 必须是至少两帧的二维数组。")
    if not np.issubdtype(joint_q.dtype, np.number) or not np.isfinite(joint_q).all():
        raise ValueError("robot.joint_q 必须全部是有限数值，不能包含 NaN 或 Inf。")

    dof_names = robot["dof_names"]
    if not isinstance(dof_names, (list, tuple)) or not all(isinstance(name, str) for name in dof_names):
        raise ValueError("robot.dof_names 必须是关节名称列表。")
    if len(set(dof_names)) != len(dof_names):
        raise ValueError("robot.dof_names 包含重复关节名称。")
    expected_columns = 7 + len(dof_names)
    if joint_q.shape[1] != expected_columns:
        raise ValueError(
            f"robot.joint_q 有 {joint_q.shape[1]} 列，但根位姿加 {len(dof_names)} 个关节应为 {expected_columns} 列。"
        )

    missing_joints = [name for name in G1_TRAINING_JOINT_NAMES if name not in dof_names]
    if missing_joints:
        raise ValueError(f"PKL 缺少训练所需关节：{', '.join(missing_joints)}")
    selected_indexes = np.asarray([dof_names.index(name) for name in G1_TRAINING_JOINT_NAMES])

    try:
        source_fps = float(robot["sample_rate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("robot.sample_rate 必须是正数帧率。") from exc
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"robot.sample_rate 必须是正数帧率，当前为 {source_fps!r}。")
    source_duration = (joint_q.shape[0] - 1) / source_fps
    output_frames = int(math.floor(source_duration * TRAIN_MOTION_FPS + 1.0e-6)) + 1
    if output_frames < 3:
        raise ValueError(
            f"动作按 {source_fps:g} Hz → {TRAIN_MOTION_FPS} Hz 转换后只有 {output_frames} 帧；"
            "至少需要 3 帧才能计算线速度和角速度。"
        )

    root_quat_format = str(robot.get("root_quat_format", "")).lower()
    root_quat = joint_q[:, 3:7]
    quat_norm_error = float(np.max(np.abs(np.linalg.norm(root_quat, axis=1) - 1.0)))
    if not math.isfinite(quat_norm_error) or quat_norm_error > 1.0e-2:
        raise ValueError(f"根四元数未归一化，最大模长误差为 {quat_norm_error:.3g}。")
    if root_quat_format == "wxyz":
        root_quat_xyzw = root_quat[:, [1, 2, 3, 0]]
    elif root_quat_format == "xyzw":
        root_quat_xyzw = root_quat
    else:
        raise ValueError(
            f"不支持的 root_quat_format={root_quat_format!r}，当前只支持 wxyz 或 xyzw。"
        )

    training_motion = np.concatenate(
        (
            joint_q[:, :3],
            root_quat_xyzw,
            joint_q[:, 7 + selected_indexes],
        ),
        axis=1,
    )
    if training_motion.shape[1] != 36:
        raise ValueError(f"转换后的训练 CSV 应为 36 列，实际为 {training_motion.shape[1]} 列。")
    np.savetxt(csv_path, training_motion, delimiter=",", fmt="%.9g")

    ignored_joint_names = [name for name in dof_names if name not in G1_TRAINING_JOINT_NAMES]
    objects = payload.get("objects")
    object_count = len(objects) if isinstance(objects, (list, tuple, dict)) else 0
    return {
        "source_format": "hhtools_pkl",
        "source_size_bytes": source_path.stat().st_size,
        "source_frames": int(joint_q.shape[0]),
        "source_fps": source_fps,
        "output_fps": TRAIN_MOTION_FPS,
        "expected_output_frames": output_frames,
        "selected_joint_count": len(G1_TRAINING_JOINT_NAMES),
        "ignored_joint_count": len(ignored_joint_names),
        "ignored_joint_names": ignored_joint_names,
        "ignored_object_count": object_count,
        "root_quat_format": root_quat_format,
        "quaternion_max_error": quat_norm_error,
        "retarget_backend": str(payload.get("retarget_backend") or "unknown"),
    }


def convert_hhtools_pkl_to_npz(source_path: Path, csv_path: Path, npz_path: Path) -> dict[str, Any]:
    """Convert a validated HHTools PKL through the existing Isaac Lab CSV pipeline."""

    conversion = extract_hhtools_pkl_to_csv(source_path, csv_path)
    conversion_log_path = npz_path.parent / "conversion.log"
    source_path_entry = str(PROJECT_ROOT / "source" / "whole_body_tracking")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = source_path_entry + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    environment["VIRTUAL_ENV"] = sys.prefix
    environment.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    command = [
        sys.executable,
        str(CSV_TO_NPZ_SCRIPT),
        "--input_file",
        str(csv_path),
        "--input_fps",
        f"{conversion['source_fps']:g}",
        "--output_fps",
        str(TRAIN_MOTION_FPS),
        "--output_name",
        npz_path.stem,
        "--save_to",
        str(npz_path),
        "--no_wandb",
        "--headless",
        "--device",
        "cuda:0",
        "--shutdown_timeout",
        "30",
    ]
    try:
        with conversion_log_path.open("w", encoding="utf-8") as conversion_log:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=conversion_log,
                stderr=subprocess.STDOUT,
                timeout=PKL_CONVERSION_TIMEOUT_SECONDS,
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"PKL 转 NPZ 超过 {PKL_CONVERSION_TIMEOUT_SECONDS:g} 秒，已停止等待。") from exc
    if completed.returncode != 0 or not npz_path.is_file():
        detail = ""
        if conversion_log_path.is_file():
            detail = conversion_log_path.read_text(encoding="utf-8", errors="replace")[-4000:].strip()
        message = f"Isaac Lab 转换失败，退出码 {completed.returncode}。"
        if detail:
            message += f"\n{detail}"
        raise ValueError(message)
    conversion["csv_name"] = csv_path.name
    conversion["npz_name"] = npz_path.name
    conversion["log_name"] = conversion_log_path.name
    return conversion


def validate_npz(path: Path) -> dict[str, Any]:
    """Fully load and validate a BeyondMimic motion NPZ."""

    checks: list[dict[str, str]] = []

    def add_check(check_id: str, label: str, status: str, detail: str) -> None:
        checks.append({"id": check_id, "label": label, "status": status, "detail": detail})

    result: dict[str, Any] = {
        "valid": False,
        "renderable": False,
        "trainable": False,
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "checks": checks,
        "arrays": {},
        "summary": {},
    }

    try:
        with zipfile.ZipFile(path) as archive:
            corrupt_member = archive.testzip()
        if corrupt_member:
            add_check("container", "NPZ 容器", "fail", "ZIP CRC 校验失败。")
            result["error"] = "NPZ archive CRC check failed"
            return result
        add_check("container", "NPZ 容器", "pass", "文件可解压，ZIP CRC 校验通过。")
    except (OSError, zipfile.BadZipFile) as exc:
        add_check("container", "NPZ 容器", "fail", f"不是有效的 NPZ/ZIP 文件：{exc}")
        result["error"] = str(exc)
        return result

    try:
        with np.load(path, allow_pickle=False) as data:
            keys = list(data.files)
            arrays = {name: data[name] for name in keys}
    except Exception as exc:  # NumPy exposes several format/decompression errors.
        add_check("load", "完整加载", "fail", f"NumPy 无法完整读取所有数组：{exc}")
        result["error"] = str(exc)
        return result

    add_check("load", "完整加载", "pass", f"成功读取 {len(arrays)} 个数组。")
    result["keys"] = keys
    result["arrays"] = {name: _array_summary(array) for name, array in arrays.items()}

    missing = [name for name in REQUIRED_ARRAYS if name not in arrays]
    extra = [name for name in keys if name not in REQUIRED_ARRAYS]
    if missing:
        add_check("keys", "标准字段", "fail", f"缺少：{', '.join(missing)}")
    else:
        detail = "7 个标准字段齐全。"
        if extra:
            detail += f" 另有 {len(extra)} 个扩展字段。"
        add_check("keys", "标准字段", "pass", detail)

    non_numeric = [name for name, array in arrays.items() if not np.issubdtype(array.dtype, np.number)]
    if non_numeric:
        add_check("numeric", "数值类型", "fail", f"非数值数组：{', '.join(non_numeric)}")
    else:
        add_check("numeric", "数值类型", "pass", "所有数组均为数值类型。")

    non_finite = [
        name
        for name, array in arrays.items()
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all()
    ]
    if non_finite:
        add_check("finite", "有限数值", "fail", f"包含 NaN/Inf：{', '.join(non_finite)}")
    else:
        add_check("finite", "有限数值", "pass", "未发现 NaN 或 Inf。")

    if missing:
        result["valid"] = not any(check["status"] == "fail" for check in checks)
        return result

    fps_array = np.asarray(arrays["fps"])
    fps = float(fps_array.reshape(-1)[0]) if fps_array.size else 0.0
    if fps_array.size == 1 and math.isfinite(fps) and fps > 0:
        fps_status = "pass" if 1 <= fps <= 240 else "warn"
        add_check("fps", "动作帧率", fps_status, f"{fps:g} FPS")
    else:
        add_check("fps", "动作帧率", "fail", "fps 必须是一个正数标量。")

    frame_arrays = {name: arrays[name] for name in REQUIRED_ARRAYS if name != "fps"}
    frame_counts = {name: int(array.shape[0]) if array.ndim else 0 for name, array in frame_arrays.items()}
    unique_frame_counts = set(frame_counts.values())
    frames = frame_counts["joint_pos"]
    if len(unique_frame_counts) == 1 and frames > 1:
        add_check("frames", "帧数一致", "pass", f"全部动作数组均为 {frames} 帧。")
    else:
        counts = ", ".join(f"{name}={count}" for name, count in frame_counts.items())
        add_check("frames", "帧数一致", "fail", counts)

    joint_shape_ok = arrays["joint_pos"].ndim == 2 and arrays["joint_vel"].shape == arrays["joint_pos"].shape
    joint_count = int(arrays["joint_pos"].shape[1]) if arrays["joint_pos"].ndim == 2 else 0
    if joint_shape_ok and joint_count == 29:
        add_check("joints", "关节结构", "pass", "29 个 G1 关节，位置与速度形状一致。")
    elif joint_shape_ok:
        add_check("joints", "关节结构", "fail", f"检测到 {joint_count} 个关节，期望 29。")
    else:
        add_check("joints", "关节结构", "fail", "joint_pos/joint_vel 形状不匹配。")

    body_pos = arrays["body_pos_w"]
    body_shape_ok = body_pos.ndim == 3 and body_pos.shape[-1] == 3
    body_count = int(body_pos.shape[1]) if body_shape_ok else 0
    expected_body_shapes = body_shape_ok and (
        arrays["body_quat_w"].shape == body_pos.shape[:-1] + (4,)
        and arrays["body_lin_vel_w"].shape == body_pos.shape
        and arrays["body_ang_vel_w"].shape == body_pos.shape
    )
    if expected_body_shapes and body_count == 30:
        add_check("bodies", "刚体结构", "pass", "30 bodies，与本机当前 G1 模型一致。")
    elif expected_body_shapes and body_count == 37:
        add_check("bodies", "刚体结构", "warn", "37 bodies：旧服务器模型格式，本机 30-body 模型可能不兼容。")
    elif expected_body_shapes:
        add_check("bodies", "刚体结构", "warn", f"检测到 {body_count} bodies，需要确认生成和训练模型一致。")
    else:
        add_check("bodies", "刚体结构", "fail", "刚体位置、姿态或速度数组形状不匹配。")

    quat_norm_min = None
    quat_norm_max = None
    quat_max_error = None
    if arrays["body_quat_w"].ndim == 3 and arrays["body_quat_w"].shape[-1] == 4:
        quat_norms = np.linalg.norm(arrays["body_quat_w"], axis=-1)
        quat_norm_min = float(quat_norms.min())
        quat_norm_max = float(quat_norms.max())
        quat_max_error = float(np.max(np.abs(quat_norms - 1.0)))
        if math.isfinite(quat_max_error) and quat_max_error <= 1e-3:
            add_check("quaternions", "四元数归一化", "pass", f"最大误差 {quat_max_error:.3g}。")
        elif math.isfinite(quat_max_error) and quat_max_error <= 1e-2:
            add_check("quaternions", "四元数归一化", "warn", f"最大误差 {quat_max_error:.3g}，建议重新归一化。")
        else:
            add_check("quaternions", "四元数归一化", "fail", f"最大误差 {quat_max_error!r}。")
    else:
        add_check("quaternions", "四元数归一化", "fail", "body_quat_w 必须为 [frames, bodies, 4]。")

    duration = frames / fps if frames > 0 and fps > 0 else None
    result["summary"] = {
        "fps": fps,
        "frames": frames,
        "duration_seconds": duration,
        "joint_count": joint_count,
        "body_count": body_count,
        "quaternion_norm_min": quat_norm_min,
        "quaternion_norm_max": quat_norm_max,
        "quaternion_max_error": quat_max_error,
    }
    result["valid"] = not any(check["status"] == "fail" for check in checks)
    # replay_npz.py only consumes body index 0 for the robot root state, so a
    # structurally valid 37-body server motion can still be previewed locally.
    # Keep the compatibility warning for training, but do not block rendering.
    result["renderable"] = result["valid"] and body_count > 0
    result["trainable"] = result["valid"] and joint_count == 29 and body_count == 30
    return result


def skipped_validation_report(path: Path) -> dict[str, Any]:
    """Trust a user's prior validation and expose the NPZ only to training."""

    return {
        "valid": None,
        "validation_skipped": True,
        "renderable": False,
        "trainable": True,
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "checks": [
            {
                "id": "validation_skipped",
                "label": "本次未检查",
                "status": "warn",
                "detail": "用户声明该 NPZ 已在此前完成检查；本次上传未读取或验证任何数组。",
            }
        ],
        "arrays": {},
        "summary": {},
    }


def normalize_run_name(value: str, fallback: str) -> str:
    """Return an RSL-RL/tmux-safe run name."""

    normalized = re.sub(r"[^a-z0-9_-]+", "_", value.lower()).strip("_-")
    if not normalized:
        normalized = re.sub(r"[^a-z0-9_-]+", "_", fallback.lower()).strip("_-")
    return normalized[:72] or "motion_training"


def strip_training_attempt_suffix(run_name: str, job_id: str) -> str:
    """Remove a previously generated job/attempt suffix from a run name."""

    return re.sub(rf"_{re.escape(job_id[:8])}_a\d+$", "", run_name)


def build_training_shell_command(
    motion_path: Path,
    run_name: str,
    num_envs: int,
    max_iterations: int,
    devices: list[str],
    log_path: Path,
    exit_path: Path,
    ppo_settings: PPOTrainingSettings | None = None,
    resume_checkpoint: TrainingCheckpoint | None = None,
) -> str:
    """Build the detached tmux payload prescribed by TRAINING_SOP.md."""

    if not devices:
        raise ValueError("至少需要选择一张训练 GPU。")

    ppo_settings = ppo_settings or PPOTrainingSettings()
    physical_gpu_ids = [device.removeprefix("cuda:") for device in devices]
    distributed = len(devices) > 1
    launcher = (
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(len(devices)),
            str(TRAIN_SCRIPT),
        ]
        if distributed
        else [sys.executable, str(TRAIN_SCRIPT)]
    )
    command = launcher + [
        "--task",
        TRAIN_TASK,
        "--motion_file",
        str(motion_path),
        "--num_envs",
        str(num_envs),
        "--max_iterations",
        str(max_iterations),
        "--num_steps_per_env",
        str(ppo_settings.num_steps_per_env),
        "--num_mini_batches",
        str(ppo_settings.num_mini_batches),
        "--num_learning_epochs",
        str(ppo_settings.num_learning_epochs),
        "--learning_rate",
        f"{ppo_settings.learning_rate:g}",
        "--desired_kl",
        f"{ppo_settings.desired_kl:g}",
        "--save_interval",
        str(ppo_settings.save_interval),
        "--run_name",
        run_name,
        "--logger",
        "tensorboard",
        "--headless",
        "--device",
        "cuda:0" if distributed else devices[0],
    ]
    if distributed:
        command.extend(
            [
                "--distributed",
                "--physical_gpu_ids",
                ",".join(physical_gpu_ids),
            ]
        )
    if resume_checkpoint is not None:
        command.extend(
            [
                "--resume",
                "True",
                "--load_run",
                resume_checkpoint.run_directory,
                "--checkpoint",
                resume_checkpoint.checkpoint_name,
            ]
        )
    source_path = PROJECT_ROOT / "source" / "whole_body_tracking"
    shell_lines = [
        "set -o pipefail",
        f"cd {shlex.quote(str(PROJECT_ROOT))}",
        f"export PYTHONPATH={shlex.quote(str(source_path))}${{PYTHONPATH:+:$PYTHONPATH}}",
        "export OMP_NUM_THREADS=8",
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "export OMNI_KIT_ACCEPT_EULA=YES",
        f"export VIRTUAL_ENV={shlex.quote(sys.prefix)}",
    ]
    if distributed:
        shell_lines.append(f"export CUDA_VISIBLE_DEVICES={shlex.quote(','.join(physical_gpu_ids))}")
    shell_lines.extend(
        [
            f"{shlex.join(command)} 2>&1 | tee -a {shlex.quote(str(log_path))}",
            "training_status=${PIPESTATUS[0]}",
            f"printf '%s\\n' \"$training_status\" > {shlex.quote(str(exit_path))}",
            'exit "$training_status"',
        ]
    )
    return "\n".join(shell_lines)


@dataclass
class PreviewJob:
    job_id: str
    original_name: str
    directory: Path
    motion_path: Path
    report: dict[str, Any]
    status: str = "validated"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    video_path: Path | None = None
    log_path: Path | None = None
    error: str | None = None
    video: dict[str, Any] | None = None
    training_status: str = "idle"
    training_log_path: Path | None = None
    training_exit_path: Path | None = None
    training_error: str | None = None
    training_session: str | None = None
    training_run_name: str | None = None
    training_pane_pid: int | None = None
    training_config: dict[str, Any] | None = None
    training_attempt: int = 0
    training_started_at: float | None = None
    training_finished_at: float | None = None

    def metadata(self) -> dict[str, Any]:
        """Return the durable subset needed to rebuild a job after a restart."""

        return {
            "version": 1,
            "job_id": self.job_id,
            "original_name": self.original_name,
            "motion_name": self.motion_path.name,
            "report": self.report,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "video_name": self.video_path.name if self.video_path else None,
            "log_name": self.log_path.name if self.log_path else None,
            "error": self.error,
            "video": self.video,
            "training_status": self.training_status,
            "training_log_path": str(self.training_log_path) if self.training_log_path else None,
            "training_exit_name": self.training_exit_path.name if self.training_exit_path else None,
            "training_error": self.training_error,
            "training_session": self.training_session,
            "training_run_name": self.training_run_name,
            "training_pane_pid": self.training_pane_pid,
            "training_config": self.training_config,
            "training_attempt": self.training_attempt,
            "training_started_at": self.training_started_at,
            "training_finished_at": self.training_finished_at,
        }

    def public(self, include_log: bool = True) -> dict[str, Any]:
        payload = {
            "job_id": self.job_id,
            "filename": self.original_name,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "report": self.report,
            "error": self.error,
            "video": self.video,
            "video_url": f"/api/jobs/{self.job_id}/video" if self.video_path and self.video_path.exists() else None,
            "training": {
                "status": self.training_status,
                "error": self.training_error,
                "session": self.training_session,
                "run_name": self.training_run_name,
                "pane_pid": self.training_pane_pid,
                "config": self.training_config,
                "attempt": self.training_attempt,
                "started_at": self.training_started_at,
                "finished_at": self.training_finished_at,
                "tensorboard_url": f"http://{TENSORBOARD_HOST}:{tensorboard_port}/",
            },
        }
        if include_log and self.log_path and self.log_path.exists():
            payload["log"] = self.log_path.read_text(encoding="utf-8", errors="replace")[-30000:]
        else:
            payload["log"] = ""
        if include_log and self.training_log_path and self.training_log_path.exists():
            payload["training"]["log"] = self.training_log_path.read_text(
                encoding="utf-8", errors="replace"
            )[-50000:]
        else:
            payload["training"]["log"] = ""
        return payload

    def active_training_summary(self) -> dict[str, Any]:
        """Return actionable metadata for the active-training task picker."""

        training_config = self.training_config or {}
        devices = training_config.get("devices")
        if not isinstance(devices, list):
            devices = [training_config["device"]] if training_config.get("device") else []
        iteration, logged_max_iterations = _last_logged_training_iteration(
            self.training_run_name or ""
        )
        return {
            "job_id": self.job_id,
            "filename": self.original_name,
            "status": self.training_status,
            "run_name": self.training_run_name,
            "session": self.training_session,
            "devices": devices,
            "iteration": iteration,
            "max_iterations": logged_max_iterations or training_config.get("max_iterations"),
            "num_envs": training_config.get("num_envs"),
            "started_at": self.training_started_at,
        }

    def history_summary(self) -> dict[str, Any]:
        """Return a lightweight summary for the historical training picker."""

        training_config = self.training_config or {}
        devices = training_config.get("devices")
        if not isinstance(devices, list):
            devices = [training_config["device"]] if training_config.get("device") else []
        return {
            "job_id": self.job_id,
            "filename": self.original_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "training": {
                "status": self.training_status,
                "run_name": self.training_run_name,
                "attempt": self.training_attempt,
                "started_at": self.training_started_at,
                "finished_at": self.training_finished_at,
                "devices": devices,
                "num_envs": training_config.get("num_envs"),
                "resume_iteration": training_config.get("resume_iteration"),
            },
        }


class JobStore:
    def __init__(self, output_root: Path):
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, PreviewJob] = {}
        self.lock = threading.RLock()
        self.device_locks: dict[str, threading.Lock] = {}
        self.tensorboard_lock = threading.Lock()

    def _device_lock(self, device: str) -> threading.Lock:
        """Return the process-local lock for one compute device."""

        with self.lock:
            return self.device_locks.setdefault(device, threading.Lock())

    def _acquire_device_locks(self, devices: list[str]) -> list[threading.Lock]:
        """Acquire all requested GPU locks atomically in a stable order."""

        ordered_devices = sorted(devices, key=lambda device: int(device.split(":", 1)[1]))
        acquired_locks: list[threading.Lock] = []
        for device in ordered_devices:
            device_lock = self._device_lock(device)
            if device_lock.acquire(blocking=False):
                acquired_locks.append(device_lock)
                continue
            for acquired_lock in reversed(acquired_locks):
                acquired_lock.release()
            raise ValueError(f"{device} 正在执行预览渲染或其他训练任务，请选择其他 GPU。")
        return acquired_locks

    @staticmethod
    def _training_run_motion_path(run_directory: Path) -> Path | None:
        environment_config_path = run_directory / "params" / "env.yaml"
        if not environment_config_path.is_file():
            return None
        try:
            for line in environment_config_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                motion_match = re.match(r"^\s*motion_file:\s*(.+?)\s*$", line)
                if motion_match:
                    return Path(motion_match.group(1).strip("'\"")).resolve()
        except OSError:
            return None
        return None

    @staticmethod
    def _job_for_training_run(
        run_directory: Path,
        motion_path: Path | None,
        jobs: list[PreviewJob],
    ) -> PreviewJob | None:
        if motion_path is not None:
            exact_matches = [job for job in jobs if job.motion_path.resolve() == motion_path]
            if len(exact_matches) == 1:
                return exact_matches[0]

            parent_matches = [job for job in jobs if job.job_id == motion_path.parent.name]
            if len(parent_matches) == 1:
                return parent_matches[0]

        run_job_match = re.search(r"_([0-9a-f]{8})_a\d+(?:_|$)", run_directory.name)
        if run_job_match:
            prefix_matches = [job for job in jobs if job.job_id.startswith(run_job_match.group(1))]
            if len(prefix_matches) == 1:
                return prefix_matches[0]

        if motion_path is None or motion_path.is_file():
            return None
        filename_matches = [job for job in jobs if job.motion_path.name == motion_path.name]
        motion_stem = motion_path.stem.lower()
        if len(filename_matches) == 1 and motion_stem in run_directory.name.lower():
            return filename_matches[0]
        return None

    def list_training_checkpoints(self, job: PreviewJob) -> list[TrainingCheckpoint]:
        """Return checkpoints produced by this job, newest iteration first."""

        if not TENSORBOARD_LOG_ROOT.is_dir():
            return []

        persisted_jobs = self.list_persisted_jobs()
        if all(candidate.job_id != job.job_id for candidate in persisted_jobs):
            persisted_jobs.append(job)
        checkpoints: list[TrainingCheckpoint] = []
        for run_directory in TENSORBOARD_LOG_ROOT.iterdir():
            if not run_directory.is_dir():
                continue
            motion_path = self._training_run_motion_path(run_directory)
            owner = self._job_for_training_run(run_directory, motion_path, persisted_jobs)
            if owner is None or owner.job_id != job.job_id:
                continue
            for checkpoint_path in run_directory.iterdir():
                checkpoint_match = re.fullmatch(r"model_(\d+)\.pt", checkpoint_path.name)
                if not checkpoint_match or not checkpoint_path.is_file():
                    continue
                checkpoints.append(
                    TrainingCheckpoint(
                        run_directory=run_directory.name,
                        checkpoint_name=checkpoint_path.name,
                        iteration=int(checkpoint_match.group(1)),
                        path=checkpoint_path,
                    )
                )
        return sorted(
            checkpoints,
            key=lambda checkpoint: (checkpoint.iteration, checkpoint.path.stat().st_mtime),
            reverse=True,
        )

    def resolve_training_checkpoint(self, job: PreviewJob, checkpoint_id: str) -> TrainingCheckpoint:
        """Resolve a client checkpoint identifier without accepting arbitrary paths."""

        checkpoint = next(
            (
                candidate
                for candidate in self.list_training_checkpoints(job)
                if candidate.checkpoint_id == checkpoint_id
            ),
            None,
        )
        if checkpoint is None:
            raise ValueError("续训 checkpoint 不存在或不属于当前动作任务。")
        return checkpoint

    def build_training_recovery_package(
        self,
        job: PreviewJob,
        run_directory_name: str,
        destination: Path,
    ) -> TrainingCheckpoint:
        """Export the newest checkpoint and portable inputs for one owned run."""

        if Path(run_directory_name).name != run_directory_name or not run_directory_name:
            raise ValueError("训练 Run 名称无效。")
        checkpoints = [
            checkpoint
            for checkpoint in self.list_training_checkpoints(job)
            if checkpoint.run_directory == run_directory_name
        ]
        if not checkpoints:
            raise ValueError("训练 Run 不存在、没有 checkpoint，或不属于当前任务。")
        checkpoint = checkpoints[0]
        run_directory = checkpoint.path.parent
        environment_config_path = run_directory / "params" / "env.yaml"
        agent_config_path = run_directory / "params" / "agent.yaml"
        try:
            build_recovery_package(
                destination,
                motion_path=job.motion_path,
                checkpoint_path=checkpoint.path,
                environment_config_path=environment_config_path,
                agent_config_path=agent_config_path,
                source_job_id=job.job_id,
                source_run_directory=run_directory_name,
                original_motion_name=job.original_name,
                training_config=job.training_config,
            )
        except RecoveryPackageError as exc:
            raise ValueError(str(exc)) from exc
        return checkpoint

    @staticmethod
    def _rewrite_motion_file(environment_config_path: Path, motion_path: Path) -> None:
        lines = environment_config_path.read_text(encoding="utf-8").splitlines(keepends=True)
        matching_indexes = [
            index for index, line in enumerate(lines) if re.match(r"^\s*motion_file\s*:", line)
        ]
        if len(matching_indexes) != 1:
            raise ValueError("env.yaml 必须且只能包含一个 motion_file 配置。")
        index = matching_indexes[0]
        prefix_match = re.match(r"^(\s*motion_file\s*:\s*)", lines[index])
        assert prefix_match is not None
        newline = "\r\n" if lines[index].endswith("\r\n") else "\n"
        if not lines[index].endswith(("\n", "\r")):
            newline = ""
        lines[index] = f"{prefix_match.group(1)}{motion_path}{newline}"
        environment_config_path.write_text("".join(lines), encoding="utf-8")

    def _imported_run_name(self, source_run_directory: str, job_id: str) -> str:
        source_name = re.sub(r"[^A-Za-z0-9._-]+", "_", source_run_directory).strip("._-")
        source_name = source_name[:80] or "training"
        base_name = f"{source_name}_import_{job_id[:8]}"
        candidate = base_name
        suffix = 2
        while (TENSORBOARD_LOG_ROOT / candidate).exists():
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        return candidate

    def import_training_recovery_package(self, package_path: Path) -> tuple[PreviewJob, str]:
        """Install a validated package as a new job and a new training run."""

        TENSORBOARD_LOG_ROOT.mkdir(parents=True, exist_ok=True)
        installed_job_directory: Path | None = None
        installed_run_directory: Path | None = None
        with tempfile.TemporaryDirectory(prefix=".recovery-import-", dir=self.output_root) as temporary:
            temporary_root = Path(temporary)
            try:
                extracted = validate_and_extract_recovery_package(
                    package_path,
                    temporary_root / "extracted",
                )
                report = validate_npz(extracted.motion_path)
                if not report.get("trainable"):
                    raise ValueError("恢复包动作文件未通过当前 G1 训练兼容性检查。")

                job_id = uuid.uuid4().hex
                clean_stem = re.sub(
                    r"[^A-Za-z0-9._-]+",
                    "_",
                    Path(extracted.original_motion_name).stem,
                ).strip("._") or "motion"
                staged_job_directory = temporary_root / "job"
                staged_run_directory = temporary_root / "run"
                staged_job_directory.mkdir()
                (staged_run_directory / "params").mkdir(parents=True)

                final_job_directory = self.output_root / job_id
                motion_path = final_job_directory / f"{clean_stem}.npz"
                staged_motion_path = staged_job_directory / motion_path.name
                shutil.copy2(extracted.motion_path, staged_motion_path)
                report["filename"] = motion_path.name

                with self.lock:
                    run_name = self._imported_run_name(extracted.source_run_directory, job_id)
                    final_run_directory = TENSORBOARD_LOG_ROOT / run_name
                    staged_environment_config = staged_run_directory / "params" / "env.yaml"
                    shutil.copy2(extracted.environment_config_path, staged_environment_config)
                    shutil.copy2(
                        extracted.agent_config_path,
                        staged_run_directory / "params" / "agent.yaml",
                    )
                    checkpoint_name = f"model_{extracted.checkpoint_iteration}.pt"
                    shutil.copy2(
                        extracted.checkpoint_path,
                        staged_run_directory / checkpoint_name,
                    )
                    self._rewrite_motion_file(staged_environment_config, motion_path)

                    now = time.time()
                    training_config = {
                        **extracted.training_config,
                        "resume_iteration": extracted.checkpoint_iteration,
                        "imported_from_run": extracted.source_run_directory,
                    }
                    staged_job = PreviewJob(
                        job_id=job_id,
                        original_name=extracted.original_motion_name,
                        directory=staged_job_directory,
                        motion_path=staged_motion_path,
                        report=report,
                        training_status="completed",
                        training_run_name=run_name,
                        training_config=training_config,
                        training_attempt=1,
                        training_finished_at=now,
                    )
                    self._persist_job(staged_job)

                    try:
                        staged_run_directory.replace(final_run_directory)
                        installed_run_directory = final_run_directory
                        staged_job_directory.replace(final_job_directory)
                        installed_job_directory = final_job_directory
                    except Exception:
                        if installed_job_directory is not None:
                            shutil.rmtree(installed_job_directory, ignore_errors=True)
                        if installed_run_directory is not None:
                            shutil.rmtree(installed_run_directory, ignore_errors=True)
                        raise

                    staged_job.directory = final_job_directory
                    staged_job.motion_path = motion_path
                    self.jobs[job_id] = staged_job
                    return staged_job, run_name
            except RecoveryPackageError as exc:
                raise ValueError(str(exc)) from exc

    def create(self, filename: str, payload: bytes, skip_validation: bool = False) -> PreviewJob:
        job_id = uuid.uuid4().hex
        directory = self.output_root / job_id
        directory.mkdir(parents=True)
        clean_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).stem).strip("._") or "motion"
        source_suffix = Path(filename).suffix.lower()
        if source_suffix == ".pkl":
            if skip_validation:
                raise ValueError("PKL 必须先安全解析并转换，不能使用快速训练通道跳过检查。")
            source_path = directory / f"{clean_stem}.pkl"
            csv_path = directory / f"{clean_stem}_g1_29dof.csv"
            motion_path = directory / f"{clean_stem}_{TRAIN_MOTION_FPS}hz.npz"
            source_path.write_bytes(payload)
            conversion_device = "cuda:0"
            device_locks = self._acquire_device_locks([conversion_device])
            try:
                if conversion_device in active_training_devices():
                    raise ValueError(f"{conversion_device} 正在训练，暂时不能执行 PKL 转换。")
                conversion = convert_hhtools_pkl_to_npz(source_path, csv_path, motion_path)
            finally:
                for device_lock in reversed(device_locks):
                    device_lock.release()
            report = validate_npz(motion_path)
            report["conversion"] = conversion
            report["checks"].insert(
                0,
                {
                    "id": "pkl_conversion",
                    "label": "PKL 自动转换",
                    "status": "pass",
                    "detail": (
                        f"HHTools {conversion['source_fps']:g} Hz / {conversion['source_frames']} 帧，"
                        f"按名称提取 {conversion['selected_joint_count']} 个 G1 关节并生成 "
                        f"{conversion['output_fps']} Hz NPZ；忽略 {conversion['ignored_joint_count']} 个非训练关节。"
                    ),
                },
            )
        else:
            motion_path = directory / f"{clean_stem}.npz"
            motion_path.write_bytes(payload)
            report = skipped_validation_report(motion_path) if skip_validation else validate_npz(motion_path)
        job = PreviewJob(job_id, Path(filename).name, directory, motion_path, report)
        with self.lock:
            self.jobs[job_id] = job
            self._persist_job(job)
        return job

    def get(self, job_id: str) -> PreviewJob | None:
        if not JOB_ID_PATTERN.fullmatch(job_id):
            return None
        with self.lock:
            job = self.jobs.get(job_id)
        if job is None:
            job = self._recover_job(job_id)
        if job is not None:
            self._refresh_training_state(job)
        return job

    def list_persisted_jobs(self) -> list[PreviewJob]:
        """Return every recoverable upload job, including jobs without training history."""

        persisted_jobs: list[PreviewJob] = []
        for directory in self.output_root.iterdir():
            if directory.is_symlink() or not directory.is_dir() or not JOB_ID_PATTERN.fullmatch(directory.name):
                continue
            try:
                job = self.get(directory.name)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            if job is not None:
                persisted_jobs.append(job)
        return persisted_jobs

    def list_historical_training_jobs(self, limit: int = 50) -> list[PreviewJob]:
        """Return recent persisted training jobs, including terminal states."""

        historical_jobs = [
            job
            for job in self.list_persisted_jobs()
            if job.training_attempt > 0
            or job.training_status != "idle"
            or bool(job.training_run_name)
        ]

        active_statuses = {"starting", "running", "stopping"}

        def history_sort_key(job: PreviewJob) -> tuple[int, float, str]:
            last_activity = (
                job.training_finished_at
                or job.training_started_at
                or job.updated_at
                or job.created_at
            )
            return (int(job.training_status in active_statuses), float(last_activity), job.job_id)

        historical_jobs.sort(key=history_sort_key, reverse=True)
        return historical_jobs[:limit]

    def list_training_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return one history entry for every RSL-RL run directory."""

        if not TENSORBOARD_LOG_ROOT.is_dir():
            return []

        persisted_jobs = self.list_persisted_jobs()

        training_runs: list[dict[str, Any]] = []
        for run_directory in TENSORBOARD_LOG_ROOT.iterdir():
            if run_directory.is_symlink() or not run_directory.is_dir():
                continue

            checkpoint_entries: list[tuple[int, Path]] = []
            for checkpoint_path in run_directory.iterdir():
                checkpoint_match = re.fullmatch(r"model_(\d+)\.pt", checkpoint_path.name)
                if checkpoint_match and checkpoint_path.is_file():
                    checkpoint_entries.append((int(checkpoint_match.group(1)), checkpoint_path))
            checkpoint_entries.sort(key=lambda entry: (entry[0], entry[1].stat().st_mtime), reverse=True)

            motion_path = self._training_run_motion_path(run_directory)
            job = self._job_for_training_run(run_directory, motion_path, persisted_jobs)

            latest_iteration = checkpoint_entries[0][0] if checkpoint_entries else None
            latest_checkpoint_name = checkpoint_entries[0][1].name if checkpoint_entries else None
            latest_activity = max(
                [run_directory.stat().st_mtime]
                + [checkpoint_path.stat().st_mtime for _, checkpoint_path in checkpoint_entries]
            )
            training_runs.append(
                {
                    "run_directory": run_directory.name,
                    "job_id": job.job_id if job else None,
                    "filename": job.original_name if job else (motion_path.name if motion_path else None),
                    "missing_motion_filename": (
                        motion_path.name if job is None and motion_path is not None and not motion_path.is_file() else None
                    ),
                    "resumable": bool(job and checkpoint_entries),
                    "checkpoint_count": len(checkpoint_entries),
                    "latest_checkpoint_name": latest_checkpoint_name,
                    "latest_iteration": latest_iteration,
                    "updated_at": latest_activity,
                }
            )

        training_runs.sort(key=lambda run: (float(run["updated_at"]), run["run_directory"]), reverse=True)
        return training_runs[:limit]

    def _persist_job(self, job: PreviewJob) -> None:
        metadata_path = job.directory / "job.json"
        temporary_path = job.directory / "job.json.tmp"
        temporary_path.write_text(
            json.dumps(job.metadata(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_path.replace(metadata_path)

    def _job_from_metadata(self, directory: Path, metadata: dict[str, Any]) -> PreviewJob | None:
        motion_name = Path(str(metadata.get("motion_name", ""))).name
        motion_path = directory / motion_name
        if not motion_name or not motion_path.is_file():
            return None
        video_name = Path(str(metadata.get("video_name") or "")).name
        log_name = Path(str(metadata.get("log_name") or "")).name
        training_exit_name = Path(str(metadata.get("training_exit_name") or "training.exit")).name
        training_log_value = metadata.get("training_log_path")
        return PreviewJob(
            job_id=directory.name,
            original_name=str(metadata.get("original_name") or motion_path.name),
            directory=directory,
            motion_path=motion_path,
            report=metadata.get("report") or skipped_validation_report(motion_path),
            status=str(metadata.get("status") or "validated"),
            created_at=float(metadata.get("created_at") or directory.stat().st_mtime),
            updated_at=float(metadata.get("updated_at") or directory.stat().st_mtime),
            video_path=directory / video_name if video_name else None,
            log_path=directory / log_name if log_name else None,
            error=metadata.get("error"),
            video=metadata.get("video"),
            training_status=str(metadata.get("training_status") or "idle"),
            training_log_path=Path(training_log_value) if training_log_value else None,
            training_exit_path=directory / training_exit_name,
            training_error=metadata.get("training_error"),
            training_session=metadata.get("training_session"),
            training_run_name=metadata.get("training_run_name"),
            training_pane_pid=metadata.get("training_pane_pid"),
            training_config=metadata.get("training_config"),
            training_attempt=int(metadata.get("training_attempt") or 0),
            training_started_at=metadata.get("training_started_at"),
            training_finished_at=metadata.get("training_finished_at"),
        )

    def _tmux_sessions(self) -> list[str]:
        completed = subprocess.run(
            ["tmux", "list-sessions", "-F", "#{session_name}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return []
        return [line.strip() for line in completed.stdout.splitlines() if line.strip()]

    def _tmux_pane_pid(self, session_name: str) -> int | None:
        completed = subprocess.run(
            ["tmux", "display-message", "-p", "-t", session_name, "#{pane_pid}"],
            capture_output=True,
            text=True,
            check=False,
        )
        value = completed.stdout.strip()
        return int(value) if completed.returncode == 0 and value.isdigit() else None

    def _tmux_session_exists(self, session_name: str) -> bool:
        return subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0

    def _tcp_port_open(self, host: str, port: int, timeout: float = 0.25) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    def _wait_for_tcp_port(self, host: str, port: int, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._tcp_port_open(host, port):
                return True
            time.sleep(0.25)
        return False

    def ensure_tensorboard(self) -> None:
        """Start the persistent TensorBoard tmux and wait until it accepts connections."""

        global tensorboard_port

        with self.tensorboard_lock:
            tensorboard_executable = shutil.which("tensorboard")
            if not tensorboard_executable:
                prefix_executable = Path(sys.prefix) / "bin" / "tensorboard"
                if prefix_executable.is_file():
                    tensorboard_executable = str(prefix_executable)
            if not tensorboard_executable:
                raise ValueError("当前 Python 环境未安装 TensorBoard 可执行文件。")

            # Reuse only a live tmux session owned by this project. An open
            # port by itself may belong to an unrelated TensorBoard process.
            for candidate_port in TENSORBOARD_CANDIDATE_PORTS:
                candidate_session = (
                    TENSORBOARD_SESSION
                    if candidate_port == TENSORBOARD_DEFAULT_PORT
                    else f"{TENSORBOARD_SESSION}_{candidate_port}"
                )
                if self._tmux_session_exists(candidate_session) and self._tcp_port_open(
                    TENSORBOARD_HOST, candidate_port
                ):
                    tensorboard_port = candidate_port
                    return

            tensorboard_session = ""
            for candidate_port in TENSORBOARD_CANDIDATE_PORTS:
                candidate_session = (
                    TENSORBOARD_SESSION
                    if candidate_port == TENSORBOARD_DEFAULT_PORT
                    else f"{TENSORBOARD_SESSION}_{candidate_port}"
                )
                port_is_occupied = self._tcp_port_open(TENSORBOARD_HOST, candidate_port)
                session_name_is_occupied = self._tmux_session_exists(candidate_session)
                if not port_is_occupied and not session_name_is_occupied:
                    tensorboard_port = candidate_port
                    tensorboard_session = candidate_session
                    break
            else:
                candidate_range = f"{TENSORBOARD_CANDIDATE_PORTS[0]}-{TENSORBOARD_CANDIDATE_PORTS[-1]}"
                raise ValueError(f"TensorBoard 候选端口 {candidate_range} 均被占用。")

            MANUAL_LOG_ROOT.mkdir(parents=True, exist_ok=True)
            TENSORBOARD_LOG_ROOT.mkdir(parents=True, exist_ok=True)
            log_path = MANUAL_LOG_ROOT / f"tensorboard_{tensorboard_port}.out"
            if not self._tmux_session_exists(tensorboard_session):
                shell_payload = "\n".join(
                    (
                        "set -o pipefail",
                        f"cd {shlex.quote(str(PROJECT_ROOT))}",
                        f"export VIRTUAL_ENV={shlex.quote(sys.prefix)}",
                        f"export PATH={shlex.quote(str(Path(sys.prefix) / 'bin'))}:$PATH",
                        f"{shlex.quote(tensorboard_executable)} "
                        f"--logdir {shlex.quote(str(TENSORBOARD_LOG_ROOT))} "
                        f"--host {TENSORBOARD_HOST} --port {tensorboard_port} "
                        f"> {shlex.quote(str(log_path))} 2>&1",
                    )
                )
                completed = subprocess.run(
                    [
                        "tmux",
                        "new-session",
                        "-d",
                        "-s",
                        tensorboard_session,
                        "bash",
                        "-lc",
                        shell_payload,
                    ],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if completed.returncode != 0:
                    raise ValueError(completed.stderr.strip() or "无法启动 TensorBoard tmux 会话。")
            if self._wait_for_tcp_port(TENSORBOARD_HOST, tensorboard_port, timeout=15.0):
                return
            detail = ""
            if log_path.is_file():
                detail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:].strip()
            message = "TensorBoard 启动超时，训练尚未启动。"
            if detail:
                message += f"\n{detail}"
            raise ValueError(message)

    def _training_log_fields(self, log_path: Path) -> dict[str, str]:
        fields: dict[str, str] = {}
        if not log_path.is_file():
            return fields
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()[:20]:
            if ": " in line:
                key, value = line.split(": ", 1)
                fields[key.strip().lower()] = value.strip()
        return fields

    def _recover_job(self, job_id: str) -> PreviewJob | None:
        directory = self.output_root / job_id
        if not directory.is_dir():
            return None
        metadata_path = directory / "job.json"
        job = None
        if metadata_path.is_file():
            try:
                job = self._job_from_metadata(
                    directory, json.loads(metadata_path.read_text(encoding="utf-8"))
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                job = None
        if job is None:
            motion_paths = sorted(directory.glob("*.npz"))
            if not motion_paths:
                return None
            motion_path = motion_paths[0]
            job = PreviewJob(
                job_id=job_id,
                original_name=motion_path.name,
                directory=directory,
                motion_path=motion_path,
                report=validate_npz(motion_path),
                created_at=directory.stat().st_mtime,
                updated_at=directory.stat().st_mtime,
            )
            videos = sorted(directory.glob("*.mp4"), key=lambda path: path.stat().st_mtime)
            if videos:
                job.video_path = videos[-1]
                job.video = probe_video(job.video_path)
                job.status = "completed"
            render_log = directory / "render.log"
            job.log_path = render_log if render_log.is_file() else None

        sessions = self._tmux_sessions()
        prefix = f"wbt_{job_id[:8]}_"
        session = next((name for name in sessions if name.startswith(prefix)), None)
        log_candidates = sorted(
            MANUAL_LOG_ROOT.glob(f"*_{job_id[:8]}_a*.out"),
            key=lambda path: path.stat().st_mtime,
        ) if MANUAL_LOG_ROOT.is_dir() else []
        log_path = job.training_log_path if job.training_log_path and job.training_log_path.is_file() else None
        if log_path is None and log_candidates:
            log_path = log_candidates[-1]
        fields = self._training_log_fields(log_path) if log_path else {}
        session = session or fields.get("tmux session") or job.training_session
        if log_path or session:
            run_name = fields.get("run name") or (log_path.stem if log_path else None)
            attempt_match = re.search(r"_a(\d+)$", run_name or "")
            job.training_log_path = log_path
            job.training_exit_path = directory / "training.exit"
            job.training_session = session
            job.training_run_name = run_name or job.training_run_name
            job.training_attempt = int(attempt_match.group(1)) if attempt_match else max(job.training_attempt, 1)
            session_alive = session is not None and session in sessions
            job.training_pane_pid = self._tmux_pane_pid(session) if session_alive and session is not None else None
            try:
                num_envs = int(fields.get("environments", ""))
            except ValueError:
                num_envs = None
            try:
                max_iterations = int(fields.get("max iterations", ""))
            except ValueError:
                max_iterations = None
            try:
                recovered_ppo_settings = PPOTrainingSettings.from_mapping(
                    {
                        "num_steps_per_env": fields.get("num steps per env", PPOTrainingSettings.num_steps_per_env),
                        "num_mini_batches": fields.get("num mini batches", PPOTrainingSettings.num_mini_batches),
                        "num_learning_epochs": fields.get(
                            "num learning epochs", PPOTrainingSettings.num_learning_epochs
                        ),
                        "learning_rate": fields.get("learning rate", PPOTrainingSettings.learning_rate),
                        "desired_kl": fields.get("desired kl", PPOTrainingSettings.desired_kl),
                    },
                    num_envs=num_envs or TRAIN_NUM_ENVS[-1],
                )
            except ValueError:
                recovered_ppo_settings = PPOTrainingSettings()
            recovered_config = {
                "task": TRAIN_TASK,
                "num_envs": num_envs,
                "max_iterations": max_iterations,
                **recovered_ppo_settings.as_dict(),
                "device": fields.get("device") or "cuda:0",
                "logger": "tensorboard",
                "headless": True,
            }
            job.training_config = {**recovered_config, **(job.training_config or {})}
            if log_path and not job.training_started_at:
                job.training_started_at = log_path.stat().st_ctime
            job.training_status = "running" if session_alive else job.training_status

        with self.lock:
            existing = self.jobs.get(job_id)
            if existing is not None:
                return existing
            self.jobs[job_id] = job
            self._persist_job(job)
        return job

    def _refresh_training_state(self, job: PreviewJob) -> None:
        if not job.training_session or job.training_status not in {"starting", "running", "stopping"}:
            return
        session_alive = job.training_session in self._tmux_sessions()
        with self.lock:
            if session_alive:
                if job.training_status != "stopping":
                    job.training_status = "running"
                job.training_pane_pid = self._tmux_pane_pid(job.training_session)
            else:
                exit_code = None
                if job.training_exit_path and job.training_exit_path.is_file():
                    value = job.training_exit_path.read_text(encoding="utf-8", errors="replace").strip()
                    if re.fullmatch(r"-?\d+", value):
                        exit_code = int(value)
                job.training_finished_at = job.training_finished_at or time.time()
                if job.training_status == "stopping" and exit_code in {None, 0, 130}:
                    job.training_status = "stopped"
                elif exit_code == 0:
                    job.training_status = "completed"
                else:
                    job.training_status = "failed"
                    job.training_error = (
                        f"训练进程退出码：{exit_code}" if exit_code is not None else "训练 tmux 会话意外结束。"
                    )
            job.updated_at = time.time()
            self._persist_job(job)

    def list_active_training_jobs(self) -> list[PreviewJob]:
        """Return every active project-owned training job exactly once."""

        active_statuses = {"starting", "running", "stopping"}
        candidate_ids: set[str] = set()
        with self.lock:
            candidate_ids.update(
                job.job_id for job in self.jobs.values() if job.training_status in active_statuses
            )

        active_job_prefixes = {
            match.group(1)
            for session_name in self._tmux_sessions()
            if (match := re.match(r"^wbt_([0-9a-f]{8})_", session_name))
        }
        if active_job_prefixes:
            for directory in self.output_root.iterdir():
                if (
                    not directory.is_symlink()
                    and directory.is_dir()
                    and JOB_ID_PATTERN.fullmatch(directory.name)
                    and directory.name[:8] in active_job_prefixes
                ):
                    candidate_ids.add(directory.name)

        active_jobs: list[PreviewJob] = []
        for job_id in candidate_ids:
            try:
                job = self.get(job_id)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            if job is not None and job.training_status in active_statuses:
                active_jobs.append(job)

        active_jobs.sort(
            key=lambda job: (float(job.training_started_at or job.updated_at), job.job_id),
            reverse=True,
        )
        return active_jobs

    def find_active_job(self) -> PreviewJob | None:
        active_jobs = self.list_active_training_jobs()
        return active_jobs[0] if active_jobs else None

    def start_render(
        self,
        job: PreviewJob,
        width: int,
        height: int,
        device: str,
        camera_layout: str,
        focal_length: float,
    ) -> None:
        if device in active_training_devices():
            raise ValueError(f"{device} 正在训练；请选择其他空闲 GPU 渲染。")
        with self.lock:
            if job.status in {"queued", "rendering"}:
                raise ValueError("该任务已经在渲染。")
            if not job.report.get("renderable"):
                raise ValueError("NPZ 未通过预览渲染所需的完整性检查。")
            job.status = "queued"
            job.error = None
            job.updated_at = time.time()
            self._persist_job(job)
        threading.Thread(
            target=self._render,
            args=(job, width, height, device, camera_layout, focal_length),
            daemon=True,
        ).start()

    def _render(
        self,
        job: PreviewJob,
        width: int,
        height: int,
        device: str,
        camera_layout: str,
        focal_length: float,
    ) -> None:
        with self._device_lock(device):
            layout_suffix = "front_rear" if camera_layout == "front_rear" else "oblique"
            video_path = job.directory / f"{job.motion_path.stem}_{layout_suffix}_{focal_length:g}mm_preview.mp4"
            log_path = job.directory / "render.log"
            with self.lock:
                job.status = "rendering"
                job.video_path = video_path
                job.log_path = log_path
                job.updated_at = time.time()
                self._persist_job(job)

            env = os.environ.copy()
            source_path = str(PROJECT_ROOT / "source" / "whole_body_tracking")
            env["PYTHONPATH"] = source_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            command = [
                sys.executable,
                str(REPLAY_SCRIPT),
                "--motion_file",
                str(job.motion_path),
                "--output_video",
                str(video_path),
                "--video_width",
                str(width),
                "--video_height",
                str(height),
                "--camera_layout",
                camera_layout,
                "--camera_focal_length",
                f"{focal_length:g}",
                "--enable_cameras",
                "--headless",
                "--device",
                device,
                "--shutdown_timeout",
                "30",
            ]
            try:
                with log_path.open("w", encoding="utf-8") as log_file:
                    log_file.write(f"Physical render device: {device}\n")
                    log_file.write("$ " + " ".join(command) + "\n\n")
                    log_file.flush()
                    completed = subprocess.run(
                        command,
                        cwd=PROJECT_ROOT,
                        env=env,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        timeout=RENDER_PROCESS_TIMEOUT_SECONDS,
                        check=False,
                    )
                if completed.returncode != 0:
                    raise RuntimeError(f"Isaac Sim 渲染进程退出码：{completed.returncode}")
                if not video_path.exists() or video_path.stat().st_size == 0:
                    raise RuntimeError("渲染进程结束，但没有生成有效 MP4。")
                video_info = probe_video(video_path)
                video_info["camera_layout"] = camera_layout
                video_info["camera_count"] = 2 if camera_layout == "front_rear" else 1
                video_info["focal_length_mm"] = focal_length
                video_info["horizontal_fov_degrees"] = math.degrees(2.0 * math.atan(24.0 / (2.0 * focal_length)))
                with self.lock:
                    job.video = video_info
                    job.status = "completed"
                    job.updated_at = time.time()
                    self._persist_job(job)
            except subprocess.TimeoutExpired:
                with self.lock:
                    job.status = "failed"
                    job.error = (
                        f"Isaac Sim 渲染超过 {RENDER_PROCESS_TIMEOUT_SECONDS:g} 秒，已自动终止。"
                    )
                    job.updated_at = time.time()
                    self._persist_job(job)
            except Exception as exc:
                with self.lock:
                    job.status = "failed"
                    job.error = str(exc)
                    job.updated_at = time.time()
                    self._persist_job(job)

    def start_training(
        self,
        job: PreviewJob,
        num_envs: int,
        max_iterations: int,
        run_name: str,
        devices: list[str],
        ppo_settings: PPOTrainingSettings | None = None,
        resume_checkpoint: TrainingCheckpoint | None = None,
    ) -> None:
        ppo_settings = ppo_settings or PPOTrainingSettings()
        if not shutil.which("tmux"):
            raise ValueError("系统未安装 tmux，无法按训练 SOP 启动持久会话。")
        with self.lock:
            if not job.report.get("trainable"):
                raise ValueError("NPZ 与本机 30-body G1 训练配置不兼容，不能开始训练。")
            if job.training_status in {"starting", "running", "stopping"}:
                raise ValueError("该动作已经有训练任务在运行。")
            if job.status in {"queued", "rendering"}:
                raise ValueError("该动作正在渲染预览，请等待渲染完成。")

        self.ensure_tensorboard()
        device_locks = self._acquire_device_locks(devices)

        try:
            active_devices = active_training_devices()
            conflicting_devices = [device for device in devices if device in active_devices]
            if conflicting_devices:
                raise ValueError(
                    f"{', '.join(conflicting_devices)} 已有训练进程，请选择其他空闲 GPU。"
                )
            with self.lock:
                if not job.report.get("trainable"):
                    raise ValueError("NPZ 与本机 30-body G1 训练配置不兼容，不能开始训练。")
                if job.training_status in {"starting", "running", "stopping"}:
                    raise ValueError("该动作已经有训练任务在运行。")
                if job.status in {"queued", "rendering"}:
                    raise ValueError("该动作正在渲染预览，请等待渲染完成。")

                gpu_count = len(devices)
                fallback_name = f"{job.motion_path.stem}_local_{gpu_count}gpu_{num_envs}"
                job.training_attempt += 1
                requested_run_name = strip_training_attempt_suffix(run_name, job.job_id)
                base_run_name = normalize_run_name(requested_run_name, fallback_name)
                safe_run_name = normalize_run_name(
                    f"{base_run_name}_{job.job_id[:8]}_a{job.training_attempt}", fallback_name
                )
                session_name = normalize_run_name(
                    f"wbt_{job.job_id[:8]}_{safe_run_name[:28]}_train",
                    f"wbt_{job.job_id[:8]}_train",
                )
                MANUAL_LOG_ROOT.mkdir(parents=True, exist_ok=True)
                log_path = MANUAL_LOG_ROOT / f"{safe_run_name}.out"
                exit_path = job.directory / "training.exit"
                log_path.unlink(missing_ok=True)
                exit_path.unlink(missing_ok=True)

                job.training_status = "starting"
                job.training_error = None
                job.training_session = session_name
                job.training_run_name = safe_run_name
                job.training_log_path = log_path
                job.training_exit_path = exit_path
                job.training_started_at = time.time()
                job.training_finished_at = None
                job.training_pane_pid = None
                job.training_config = {
                    "task": TRAIN_TASK,
                    "num_envs": num_envs,
                    "max_iterations": max_iterations,
                    **ppo_settings.as_dict(),
                    "requested_run_name": base_run_name,
                    "device": devices[0],
                    "devices": devices,
                    "distributed": gpu_count > 1,
                    "world_size": gpu_count,
                    "logger": "tensorboard",
                    "headless": True,
                }
                if resume_checkpoint is not None:
                    job.training_config.update(
                        {
                            "resume": True,
                            "resume_checkpoint_id": resume_checkpoint.checkpoint_id,
                            "resume_run_directory": resume_checkpoint.run_directory,
                            "resume_checkpoint": resume_checkpoint.checkpoint_name,
                            "resume_iteration": resume_checkpoint.iteration,
                        }
                    )
                job.updated_at = time.time()
                self._persist_job(job)

            threading.Thread(
                target=self._launch_training,
                args=(
                    job,
                    num_envs,
                    max_iterations,
                    safe_run_name,
                    devices,
                    ppo_settings,
                    device_locks,
                    resume_checkpoint,
                ),
                daemon=True,
            ).start()
        except Exception:
            for device_lock in reversed(device_locks):
                device_lock.release()
            raise

    def _launch_training(
        self,
        job: PreviewJob,
        num_envs: int,
        max_iterations: int,
        run_name: str,
        devices: list[str],
        ppo_settings: PPOTrainingSettings,
        device_locks: list[threading.Lock],
        resume_checkpoint: TrainingCheckpoint | None,
    ) -> None:
        try:
            assert job.training_log_path is not None
            assert job.training_exit_path is not None
            assert job.training_session is not None
            shell_command = build_training_shell_command(
                job.motion_path,
                run_name,
                num_envs,
                max_iterations,
                devices,
                job.training_log_path,
                job.training_exit_path,
                ppo_settings,
                resume_checkpoint,
            )
            job.training_log_path.write_text(
                "Motion Inspector training launch\n"
                f"tmux session: {job.training_session}\n"
                f"motion: {job.motion_path}\n"
                f"run name: {run_name}\n"
                f"environments: {num_envs}\n"
                f"max iterations: {max_iterations}\n"
                f"num steps per env: {ppo_settings.num_steps_per_env}\n"
                f"num mini batches: {ppo_settings.num_mini_batches}\n"
                f"num learning epochs: {ppo_settings.num_learning_epochs}\n"
                f"learning rate: {ppo_settings.learning_rate:g}\n"
                f"desired kl: {ppo_settings.desired_kl:g}\n"
                f"device: {devices[0]}\n"
                f"devices: {','.join(devices)}\n"
                f"distributed: {len(devices) > 1}\n"
                f"world size: {len(devices)}\n"
                + (
                    f"resume: true\n"
                    f"resume run: {resume_checkpoint.run_directory}\n"
                    f"resume checkpoint: {resume_checkpoint.checkpoint_name}\n"
                    f"resume iteration: {resume_checkpoint.iteration}\n"
                    if resume_checkpoint is not None
                    else "resume: false\n"
                )
                + "\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                ["tmux", "new-session", "-d", "-s", job.training_session, "bash", "-lc", shell_command],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or "tmux 无法创建训练会话。")
            pane = subprocess.run(
                ["tmux", "display-message", "-p", "-t", job.training_session, "#{pane_pid}"],
                capture_output=True,
                text=True,
                check=False,
            )
            with self.lock:
                job.training_pane_pid = int(pane.stdout.strip()) if pane.stdout.strip().isdigit() else None
                job.training_status = "running"
                job.updated_at = time.time()
                self._persist_job(job)

            while subprocess.run(
                ["tmux", "has-session", "-t", job.training_session],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode == 0:
                time.sleep(2)

            exit_code = None
            if job.training_exit_path.exists():
                exit_text = job.training_exit_path.read_text(encoding="utf-8", errors="replace").strip()
                if re.fullmatch(r"-?\d+", exit_text):
                    exit_code = int(exit_text)
            with self.lock:
                job.training_finished_at = time.time()
                if job.training_status == "stopping" and exit_code in {None, 0, 130}:
                    job.training_status = "stopped"
                elif exit_code == 0:
                    job.training_status = "completed"
                else:
                    job.training_status = "failed"
                    job.training_error = (
                        f"训练进程退出码：{exit_code}" if exit_code is not None else "训练 tmux 会话意外结束。"
                    )
                job.updated_at = time.time()
                self._persist_job(job)
        except Exception as exc:
            with self.lock:
                job.training_status = "failed"
                job.training_error = str(exc)
                job.training_finished_at = time.time()
                job.updated_at = time.time()
                self._persist_job(job)
        finally:
            for device_lock in reversed(device_locks):
                device_lock.release()

    def stop_training(self, job: PreviewJob) -> None:
        with self.lock:
            if job.training_status not in {"starting", "running"} or not job.training_session:
                raise ValueError("当前没有可停止的训练任务。")
            previous_status = job.training_status
            job.training_status = "stopping"
            job.training_error = None
            job.updated_at = time.time()
            session_name = job.training_session
            self._persist_job(job)
        completed = subprocess.run(
            ["tmux", "send-keys", "-t", session_name, "C-c"],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            with self.lock:
                job.training_status = previous_status
                job.training_error = completed.stderr.strip() or "无法向训练 tmux 会话发送 Ctrl+C。"
                job.updated_at = time.time()
                self._persist_job(job)
            raise ValueError(completed.stderr.strip() or "无法向训练 tmux 会话发送 Ctrl+C。")
        threading.Thread(
            target=self._enforce_training_stop,
            args=(job, session_name),
            daemon=True,
        ).start()

    def _enforce_training_stop(self, job: PreviewJob, session_name: str) -> None:
        """Close only this job's tmux session if graceful shutdown stalls."""

        deadline = time.monotonic() + TRAINING_STOP_GRACE_SECONDS
        while time.monotonic() < deadline:
            if not self._tmux_session_exists(session_name):
                return
            time.sleep(1.0)

        with self.lock:
            still_stopping_same_session = (
                job.training_status == "stopping" and job.training_session == session_name
            )
        if still_stopping_same_session and self._tmux_session_exists(session_name):
            subprocess.run(
                ["tmux", "kill-session", "-t", session_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )


def probe_video(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"size_bytes": path.stat().st_size}
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return info
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration,size",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode == 0:
        parsed = json.loads(completed.stdout)
        stream = (parsed.get("streams") or [{}])[0]
        fmt = parsed.get("format") or {}
        info.update(
            {
                "codec": stream.get("codec_name"),
                "width": int(stream["width"]) if stream.get("width") else None,
                "height": int(stream["height"]) if stream.get("height") else None,
                "fps": stream.get("r_frame_rate"),
                "frames": int(stream["nb_frames"]) if stream.get("nb_frames") else None,
                "duration_seconds": float(fmt["duration"]) if fmt.get("duration") else None,
            }
        )
    return info


class PreviewRequestHandler(BaseHTTPRequestHandler):
    server_version = "MotionInspector/1.0"

    @property
    def job_store(self) -> JobStore:
        return self.server.job_store  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}")

    def _json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _error(self, message: str, status: HTTPStatus) -> None:
        self._json({"error": message}, status)

    def _serve_file(self, path: Path, content_type: str, download_name: str | None = None) -> None:
        if not path.is_file():
            self._error("文件不存在。", HTTPStatus.NOT_FOUND)
            return
        file_size = path.stat().st_size
        start = 0
        end = file_size - 1
        range_header = self.headers.get("Range")
        if range_header and not download_name:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            first, last = match.groups()
            if first:
                start = int(first)
                end = min(int(last), end) if last else end
            elif last:
                suffix_length = int(last)
                start = max(0, file_size - suffix_length)
            if start > end or start >= file_size:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return
        content_length = end - start + 1
        self.send_response(HTTPStatus.PARTIAL_CONTENT if range_header and not download_name else HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.send_header("Accept-Ranges", "bytes")
        if range_header and not download_name:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.end_headers()
        with path.open("rb") as file_handle:
            file_handle.seek(start)
            remaining = content_length
            while remaining:
                chunk = file_handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)

    def do_GET(self) -> None:  # noqa: N802
        parsed_url = urlparse(self.path)
        route = unquote(parsed_url.path)
        if route == "/":
            self._serve_file(WEB_ROOT / "index.html", "text/html; charset=utf-8")
            return
        if route == "/app.js":
            self._serve_file(WEB_ROOT / "app.js", "text/javascript; charset=utf-8")
            return
        if route == "/styles.css":
            self._serve_file(WEB_ROOT / "styles.css", "text/css; charset=utf-8")
            return
        if route == "/api/system-info":
            self._json(system_info_snapshot())
            return
        if route == "/api/active-jobs":
            jobs = self.job_store.list_active_training_jobs()
            self._json({"jobs": [job.active_training_summary() for job in jobs]})
            return
        if route == "/api/active-job":
            job = self.job_store.find_active_job()
            if not job:
                self._error("当前没有正在运行的训练任务。", HTTPStatus.NOT_FOUND)
                return
            self._json(job.public())
            return
        if route == "/api/jobs":
            jobs = self.job_store.list_historical_training_jobs(limit=50)
            self._json({"jobs": [job.history_summary() for job in jobs]})
            return
        if route == "/api/training-runs":
            self._json({"runs": self.job_store.list_training_runs(limit=100)})
            return
        match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})/training-package", route)
        if match:
            job = self.job_store.get(match.group(1))
            if not job:
                self._error("任务不存在或服务已重启。", HTTPStatus.NOT_FOUND)
                return
            run_directory = (parse_qs(parsed_url.query).get("run_directory") or [""])[0]
            if not run_directory:
                self._error("必须指定要导出的训练 Run。", HTTPStatus.BAD_REQUEST)
                return
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    prefix="training-recovery-",
                    suffix=".zip",
                    dir=self.job_store.output_root,
                    delete=False,
                ) as temporary_file:
                    temporary_path = Path(temporary_file.name)
                checkpoint = self.job_store.build_training_recovery_package(
                    job,
                    run_directory,
                    temporary_path,
                )
                clean_run_name = re.sub(r"[^A-Za-z0-9._-]+", "_", run_directory).strip("._")
                download_name = f"{clean_run_name or 'training'}_model_{checkpoint.iteration}_recovery.zip"
                self._serve_file(temporary_path, "application/zip", download_name)
            except ValueError as exc:
                self._error(str(exc), HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                self._error(f"无法导出训练恢复包：{exc}", HTTPStatus.INTERNAL_SERVER_ERROR)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
            return
        match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})", route)
        if match:
            job = self.job_store.get(match.group(1))
            if not job:
                self._error("任务不存在或服务已重启。", HTTPStatus.NOT_FOUND)
                return
            self._json(job.public())
            return
        match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})/checkpoints", route)
        if match:
            job = self.job_store.get(match.group(1))
            if not job:
                self._error("任务不存在或服务已重启。", HTTPStatus.NOT_FOUND)
                return
            checkpoints = self.job_store.list_training_checkpoints(job)
            self._json(
                {
                    "checkpoints": [checkpoint.public() for checkpoint in checkpoints],
                    "latest": checkpoints[0].public() if checkpoints else None,
                }
            )
            return
        match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})/video", route)
        if match:
            job = self.job_store.get(match.group(1))
            if not job or not job.video_path:
                self._error("视频尚未生成。", HTTPStatus.NOT_FOUND)
                return
            self._serve_file(job.video_path, "video/mp4")
            return
        match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})/download", route)
        if match:
            job = self.job_store.get(match.group(1))
            if not job or not job.video_path:
                self._error("视频尚未生成。", HTTPStatus.NOT_FOUND)
                return
            self._serve_file(job.video_path, "video/mp4", job.video_path.name)
            return
        self._error("页面不存在。", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        route = unquote(urlparse(self.path).path)
        if route == "/api/training-package/import":
            self._import_training_package()
            return
        if route == "/api/upload":
            self._upload()
            return
        match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})/render", route)
        if match:
            self._start_render(match.group(1))
            return
        match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})/train", route)
        if match:
            self._start_training(match.group(1))
            return
        match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})/stop-training", route)
        if match:
            self._stop_training(match.group(1))
            return
        self._error("接口不存在。", HTTPStatus.NOT_FOUND)

    def _import_training_package(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if content_length <= 0:
            self._error("上传内容为空。", HTTPStatus.BAD_REQUEST)
            return
        if content_length > MAX_PACKAGE_BYTES:
            self._error("训练恢复包超过 1 GiB 限制。", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return

        package_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="training-recovery-upload-",
                suffix=".zip",
                dir=self.job_store.output_root,
                delete=False,
            ) as package_file:
                package_path = Path(package_file.name)
                remaining = content_length
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("上传未完整接收。")
                    package_file.write(chunk)
                    remaining -= len(chunk)
            job, run_directory = self.job_store.import_training_recovery_package(package_path)
        except ValueError as exc:
            self._error(str(exc), HTTPStatus.BAD_REQUEST)
            return
        except Exception as exc:
            self._error(f"无法导入训练恢复包：{exc}", HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        finally:
            if package_path is not None:
                package_path.unlink(missing_ok=True)

        payload = job.public(include_log=False)
        payload["imported_run_directory"] = run_directory
        self._json(payload, HTTPStatus.CREATED)

    def _upload(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        filename = unquote(self.headers.get("X-Filename", "motion.npz"))
        skip_validation = self.headers.get("X-Skip-Validation", "false").lower() == "true"
        suffix = Path(filename).suffix.lower()
        if suffix not in {".npz", ".pkl"}:
            self._error("只接受 .npz 或 HHTools .pkl 文件。", HTTPStatus.BAD_REQUEST)
            return
        if suffix == ".pkl" and skip_validation:
            self._error("PKL 必须先转换和检查，不能使用快速训练通道。", HTTPStatus.BAD_REQUEST)
            return
        if content_length <= 0:
            self._error("上传内容为空。", HTTPStatus.BAD_REQUEST)
            return
        if content_length > MAX_UPLOAD_BYTES:
            self._error("文件超过 512 MiB 限制。", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return
        payload = self.rfile.read(content_length)
        if len(payload) != content_length:
            self._error("上传未完整接收。", HTTPStatus.BAD_REQUEST)
            return
        try:
            job = self.job_store.create(filename, payload, skip_validation=skip_validation)
        except ValueError as exc:
            self._error(str(exc), HTTPStatus.BAD_REQUEST)
            return
        except Exception as exc:
            self._error(f"无法保存或检查文件：{exc}", HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._json(job.public(include_log=False), HTTPStatus.CREATED)

    def _start_render(self, job_id: str) -> None:
        job = self.job_store.get(job_id)
        if not job:
            self._error("任务不存在或服务已重启。", HTTPStatus.NOT_FOUND)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            settings = json.loads(self.rfile.read(content_length) or b"{}")
            width = int(settings.get("width", 960))
            height = int(settings.get("height", 720))
            device = str(settings.get("device", "cuda:0"))
            camera_layout = str(settings.get("camera_layout", "front_rear"))
            focal_length = float(settings.get("focal_length", 18.0))
            if width < 320 or width > 3840 or height < 240 or height > 2160:
                raise ValueError("分辨率超出允许范围。")
            if not re.fullmatch(r"cuda:\d+", device):
                raise ValueError("视频渲染设备格式必须是 cuda:N。")
            if camera_layout not in {"oblique", "front_rear"}:
                raise ValueError("机位模式必须是 oblique 或 front_rear。")
            if focal_length not in {18.0, 24.0, 35.0}:
                raise ValueError("镜头焦距必须是 18、24 或 35mm。")
            self.job_store.start_render(job, width, height, device, camera_layout, focal_length)
        except (ValueError, json.JSONDecodeError) as exc:
            self._error(str(exc), HTTPStatus.BAD_REQUEST)
            return
        self._json(job.public(include_log=False), HTTPStatus.ACCEPTED)

    def _start_training(self, job_id: str) -> None:
        job = self.job_store.get(job_id)
        if not job:
            self._error("任务不存在或服务已重启。", HTTPStatus.NOT_FOUND)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            settings = json.loads(self.rfile.read(content_length) or b"{}")
            num_envs, max_iterations, ppo_settings = parse_training_request_settings(settings)
            requested_devices = settings.get("devices")
            if requested_devices is None:
                requested_devices = [settings.get("device", "cuda:0")]
            if not isinstance(requested_devices, list) or not requested_devices:
                raise ValueError("训练设备必须是至少包含一张 GPU 的数组。")
            if not all(isinstance(device, str) for device in requested_devices):
                raise ValueError("训练设备数组中的每一项都必须是 cuda:N 字符串。")
            if len(set(requested_devices)) != len(requested_devices):
                raise ValueError("训练设备不能重复选择。")
            devices = sorted(
                requested_devices,
                key=lambda device: int(device.split(":", 1)[1])
                if re.fullmatch(r"cuda:\d+", device)
                else -1,
            )
            requested_name = str(settings.get("run_name", ""))
            resume_checkpoint_id = settings.get("resume_checkpoint_id")
            if resume_checkpoint_id is not None and not isinstance(resume_checkpoint_id, str):
                raise ValueError("续训 checkpoint 标识必须是字符串。")
            resume_checkpoint = (
                self.job_store.resolve_training_checkpoint(job, resume_checkpoint_id)
                if resume_checkpoint_id
                else None
            )
            if num_envs not in TRAIN_NUM_ENVS:
                allowed_num_envs = "、".join(str(value) for value in TRAIN_NUM_ENVS)
                raise ValueError(f"环境数必须是 {allowed_num_envs} 之一。")
            if max_iterations < 1 or max_iterations > 100000:
                raise ValueError("训练迭代数必须在 1 到 100000 之间。")
            available_gpu_indexes = {int(gpu["index"]) for gpu in gpu_inventory()}
            for device in devices:
                device_match = re.fullmatch(r"cuda:(\d+)", device)
                if not device_match:
                    raise ValueError("训练设备格式必须是 cuda:N。")
                if int(device_match.group(1)) not in available_gpu_indexes:
                    raise ValueError(f"训练设备 {device} 不存在或当前不可见。")
            if len(requested_name) > 100:
                raise ValueError("run name 不能超过 100 个字符。")
            self.job_store.start_training(
                job,
                num_envs,
                max_iterations,
                requested_name,
                devices,
                ppo_settings,
                resume_checkpoint,
            )
        except (ValueError, json.JSONDecodeError) as exc:
            self._error(str(exc), HTTPStatus.BAD_REQUEST)
            return
        self._json(job.public(include_log=False), HTTPStatus.ACCEPTED)

    def _stop_training(self, job_id: str) -> None:
        job = self.job_store.get(job_id)
        if not job:
            self._error("任务不存在或服务已重启。", HTTPStatus.NOT_FOUND)
            return
        try:
            self.job_store.stop_training(job)
        except ValueError as exc:
            self._error(str(exc), HTTPStatus.BAD_REQUEST)
            return
        self._json(job.public(include_log=False), HTTPStatus.ACCEPTED)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate motion NPZ files and render MP4 previews from a local web UI.")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind address (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (default: 8765).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Directory for uploads and videos.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not WEB_ROOT.is_dir():
        raise SystemExit(f"Web assets not found: {WEB_ROOT}")

    job_store = JobStore(args.output_dir.resolve())
    try:
        job_store.ensure_tensorboard()
    except ValueError as exc:
        raise SystemExit(f"TensorBoard startup failed: {exc}") from exc

    server = MotionInspectorHTTPServer((args.host, args.port), PreviewRequestHandler)
    server.job_store = job_store  # type: ignore[attr-defined]
    print("Motion Inspector is ready")
    print(f"Open: http://{args.host}:{args.port}")
    print(f"TensorBoard: http://{TENSORBOARD_HOST}:{tensorboard_port}")
    print(f"Outputs: {args.output_dir.resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
