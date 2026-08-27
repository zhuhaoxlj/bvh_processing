from __future__ import annotations

import http.client
import hashlib
import importlib.util
import io
import json
import pickle
import sys
import tempfile
import threading
import unittest
import urllib.parse
import warnings
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "npz_preview_web.py"
APP_PATH = Path(__file__).resolve().parents[1] / "tools" / "npz_preview_web" / "app.js"
TRAIN_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "rsl_rl" / "train.py"
REPLAY_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "replay_npz.py"
SPEC = importlib.util.spec_from_file_location("npz_preview_web", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ValidateNpzTest(unittest.TestCase):
    def write_motion(self, path: Path, *, bodies: int = 30, joints: int = 29, add_nan: bool = False) -> None:
        frames = 4
        body_quat = np.zeros((frames, bodies, 4), dtype=np.float32)
        body_quat[..., 0] = 1.0
        joint_pos = np.zeros((frames, joints), dtype=np.float32)
        if add_nan:
            joint_pos[1, 0] = np.nan
        np.savez(
            path,
            fps=np.array([50.0]),
            joint_pos=joint_pos,
            joint_vel=np.zeros_like(joint_pos),
            body_pos_w=np.zeros((frames, bodies, 3), dtype=np.float32),
            body_quat_w=body_quat,
            body_lin_vel_w=np.zeros((frames, bodies, 3), dtype=np.float32),
            body_ang_vel_w=np.zeros((frames, bodies, 3), dtype=np.float32),
        )

    def write_hhtools_pkl(
        self, path: Path, *, root_quat_format: str = "wxyz", sample_rate: float = 120.0
    ) -> np.ndarray:
        hand_names = [
            "left_hand_thumb_0_joint",
            "left_hand_index_0_joint",
            "right_hand_thumb_0_joint",
        ]
        dof_names = list(reversed(MODULE.G1_TRAINING_JOINT_NAMES)) + hand_names
        frames = 7
        joint_q = np.zeros((frames, 7 + len(dof_names)), dtype=np.float32)
        frame_offsets = np.arange(frames, dtype=np.float32)[:, None] / 10.0
        joint_q[:, :3] = np.array([1.0, 2.0, 3.0], dtype=np.float32) + frame_offsets
        joint_q[:, 3:7] = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float32)
        for index in range(len(dof_names)):
            joint_q[:, 7 + index] = index + np.arange(frames, dtype=np.float32) / 10.0
        payload = {
            "hhtools_export": "retarget_v1",
            "format": "pkl",
            "retarget_backend": "newton",
            "robot": {
                "joint_q": joint_q,
                "dof_names": dof_names,
                "sample_rate": sample_rate,
                "name": "fixture",
                "root_quat_format": root_quat_format,
                "meta": {"robot": "g1_29dof_with_hand"},
            },
            "objects": [],
        }
        path.write_bytes(pickle.dumps(payload, protocol=4))
        return joint_q

    def test_extracts_hhtools_pkl_to_named_29dof_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "motion.pkl"
            csv_path = Path(directory) / "motion.csv"
            joint_q = self.write_hhtools_pkl(source)

            conversion = MODULE.extract_hhtools_pkl_to_csv(source, csv_path)
            csv_motion = np.loadtxt(csv_path, delimiter=",")

        self.assertEqual(csv_motion.shape, (7, 36))
        np.testing.assert_allclose(csv_motion[0, :3], joint_q[0, :3])
        np.testing.assert_allclose(csv_motion[0, 3:7], [0.70710678, 0.0, 0.0, 0.70710678])
        reversed_names = list(reversed(MODULE.G1_TRAINING_JOINT_NAMES))
        expected_joint_values = [reversed_names.index(name) for name in MODULE.G1_TRAINING_JOINT_NAMES]
        np.testing.assert_allclose(csv_motion[0, 7:], expected_joint_values)
        self.assertEqual(conversion["source_fps"], 120.0)
        self.assertEqual(conversion["source_frames"], 7)
        self.assertEqual(conversion["selected_joint_count"], 29)
        self.assertEqual(conversion["ignored_joint_count"], 3)
        self.assertEqual(conversion["root_quat_format"], "wxyz")

    def test_hhtools_pkl_rejects_unapproved_pickle_globals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "unsafe.pkl"
            csv_path = Path(directory) / "unsafe.csv"
            source.write_bytes(pickle.dumps(Path("unexpected"), protocol=4))

            with self.assertRaisesRegex(ValueError, "不允许的对象类型"):
                MODULE.extract_hhtools_pkl_to_csv(source, csv_path)

        self.assertFalse(csv_path.exists())

    def test_hhtools_pkl_requires_all_training_joints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "missing-joint.pkl"
            csv_path = Path(directory) / "missing-joint.csv"
            self.write_hhtools_pkl(source)
            payload = pickle.loads(source.read_bytes())
            missing_name = MODULE.G1_TRAINING_JOINT_NAMES[0]
            missing_index = payload["robot"]["dof_names"].index(missing_name)
            payload["robot"]["dof_names"].pop(missing_index)
            payload["robot"]["joint_q"] = np.delete(
                payload["robot"]["joint_q"], 7 + missing_index, axis=1
            )
            source.write_bytes(pickle.dumps(payload, protocol=4))

            with self.assertRaisesRegex(ValueError, missing_name):
                MODULE.extract_hhtools_pkl_to_csv(source, csv_path)

    def test_hhtools_pkl_preserves_50hz_source_rate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "motion-50hz.pkl"
            csv_path = Path(directory) / "motion-50hz.csv"
            self.write_hhtools_pkl(source, sample_rate=50.0)

            conversion = MODULE.extract_hhtools_pkl_to_csv(source, csv_path)

        self.assertEqual(conversion["source_fps"], 50.0)
        self.assertEqual(conversion["output_fps"], 50)

    def test_hhtools_pkl_accepts_fractional_source_rate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "motion-59-94hz.pkl"
            csv_path = Path(directory) / "motion-59-94hz.csv"
            self.write_hhtools_pkl(source, sample_rate=59.94)

            conversion = MODULE.extract_hhtools_pkl_to_csv(source, csv_path)

        self.assertEqual(conversion["source_fps"], 59.94)
        self.assertEqual(conversion["output_fps"], 50)

    def test_hhtools_pkl_requires_three_output_frames_for_velocity_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "two-frames.pkl"
            csv_path = Path(directory) / "two-frames.csv"
            self.write_hhtools_pkl(source)
            payload = pickle.loads(source.read_bytes())
            payload["robot"]["joint_q"] = payload["robot"]["joint_q"][:2]
            source.write_bytes(pickle.dumps(payload, protocol=4))

            with self.assertRaisesRegex(ValueError, "至少需要 3 帧"):
                MODULE.extract_hhtools_pkl_to_csv(source, csv_path)

    def test_high_rate_short_pkl_rejects_too_few_resampled_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "short-120hz.pkl"
            csv_path = Path(directory) / "short-120hz.csv"
            self.write_hhtools_pkl(source, sample_rate=120.0)
            payload = pickle.loads(source.read_bytes())
            payload["robot"]["joint_q"] = np.repeat(payload["robot"]["joint_q"][:1], 5, axis=0)
            source.write_bytes(pickle.dumps(payload, protocol=4))

            with self.assertRaisesRegex(ValueError, "转换后只有 2 帧"):
                MODULE.extract_hhtools_pkl_to_csv(source, csv_path)

    def test_hhtools_pkl_passes_50hz_rate_to_isaac_converter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "motion-50hz.pkl"
            csv_path = Path(directory) / "motion-50hz.csv"
            npz_path = Path(directory) / "motion-50hz.npz"
            self.write_hhtools_pkl(source, sample_rate=50.0)

            def fake_run(command: list[str], **kwargs: object) -> mock.Mock:
                input_fps_index = command.index("--input_fps") + 1
                output_fps_index = command.index("--output_fps") + 1
                save_to_index = command.index("--save_to") + 1
                self.assertEqual(command[input_fps_index], "50")
                self.assertEqual(command[output_fps_index], "50")
                self.write_motion(Path(command[save_to_index]))
                return mock.Mock(returncode=0)

            with mock.patch("npz_preview_web.subprocess.run", side_effect=fake_run):
                conversion = MODULE.convert_hhtools_pkl_to_npz(source, csv_path, npz_path)

        self.assertEqual(conversion["source_fps"], 50.0)
        self.assertEqual(conversion["output_fps"], 50)
        self.assertEqual(conversion["npz_name"], "motion-50hz.npz")

    def test_csv_converter_keeps_last_frame_at_matching_fps(self) -> None:
        converter_source = (SCRIPT_PATH.parent / "csv_to_npz.py").read_text(encoding="utf-8")

        self.assertIn(
            "int(np.floor(self.duration / self.output_dt + 1.0e-6)) + 1",
            converter_source,
        )
        self.assertIn("torch.arange(self.output_frames", converter_source)

    def test_job_store_converts_pkl_before_npz_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fixture.pkl"
            self.write_hhtools_pkl(source)
            store = MODULE.JobStore(Path(directory) / "jobs")

            def fake_convert(pkl_path: Path, csv_path: Path, npz_path: Path) -> dict[str, object]:
                self.assertTrue(pkl_path.is_file())
                csv_path.write_text("converted", encoding="utf-8")
                self.write_motion(npz_path)
                return {
                    "source_format": "hhtools_pkl",
                    "source_fps": 120.0,
                    "output_fps": 50,
                    "source_frames": 3,
                    "selected_joint_count": 29,
                    "ignored_joint_count": 3,
                    "root_quat_format": "wxyz",
                }

            with mock.patch("npz_preview_web.active_training_devices", return_value=set()), mock.patch(
                "npz_preview_web.convert_hhtools_pkl_to_npz", side_effect=fake_convert
            ):
                job = store.create("fixture.pkl", source.read_bytes())

        self.assertEqual(job.original_name, "fixture.pkl")
        self.assertEqual(job.motion_path.name, "fixture_50hz.npz")
        self.assertTrue(job.report["trainable"])
        self.assertEqual(job.report["conversion"]["source_format"], "hhtools_pkl")
        self.assertEqual(job.report["checks"][0]["id"], "pkl_conversion")

    def test_job_store_rejects_pkl_conversion_on_busy_cuda_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fixture.pkl"
            self.write_hhtools_pkl(source)
            store = MODULE.JobStore(Path(directory) / "jobs")

            with mock.patch("npz_preview_web.active_training_devices", return_value={"cuda:0"}), mock.patch(
                "npz_preview_web.convert_hhtools_pkl_to_npz"
            ) as convert:
                with self.assertRaisesRegex(ValueError, "cuda:0 正在训练"):
                    store.create("fixture.pkl", source.read_bytes())

        convert.assert_not_called()
        cuda_lock = store._device_lock("cuda:0")
        self.assertTrue(cuda_lock.acquire(blocking=False))
        cuda_lock.release()

    def test_frontend_accepts_pkl_and_explains_automatic_conversion(self) -> None:
        source = APP_PATH.read_text(encoding="utf-8")
        html = (APP_PATH.parent / "index.html").read_text(encoding="utf-8")

        self.assertIn('accept=".npz,.pkl,application/octet-stream"', html)
        self.assertIn("把 NPZ 或 PKL 拖到这里", html)
        self.assertIn('const isPkl = file.name.toLowerCase().endsWith(".pkl")', source)
        self.assertIn("PKL 正在转换为 50 Hz 训练 NPZ", source)
        self.assertIn("快速训练通道只适用于已经检查过的 NPZ", source)

    def test_gpu_inventory_reports_usage_and_processes_for_each_gpu(self) -> None:
        gpu_query = mock.Mock(
            returncode=0,
            stdout=(
                "0, GPU-first, NVIDIA GeForce RTX 4090, 24564, 8192, 16372, 87, 68, 312.4, 450.0\n"
                "1, GPU-second, NVIDIA GeForce RTX 4090, 24564, 1024, 23540, 12, 42, 71.0, 450.0\n"
            ),
        )
        process_query = mock.Mock(
            returncode=0,
            stdout=(
                "1234, GPU-first, 7168, /opt/isaac/python3\n"
                "5678, GPU-second, 512, /usr/bin/python3\n"
            ),
        )

        with mock.patch("npz_preview_web.shutil.which", return_value="/usr/bin/nvidia-smi"), mock.patch(
            "npz_preview_web.subprocess.run", side_effect=[gpu_query, process_query]
        ):
            gpus = MODULE.gpu_inventory()

        self.assertEqual(len(gpus), 2)
        self.assertEqual(gpus[0]["memory_used_mib"], 8192)
        self.assertEqual(gpus[0]["utilization_percent"], 87)
        self.assertEqual(gpus[0]["temperature_celsius"], 68)
        self.assertEqual(gpus[0]["processes"][0]["pid"], 1234)
        self.assertEqual(gpus[0]["processes"][0]["used_memory_mib"], 7168)
        self.assertEqual(gpus[1]["processes"][0]["pid"], 5678)

    def test_system_info_snapshot_reuses_recent_gpu_query(self) -> None:
        gpu_snapshot = [{"index": 0, "name": "NVIDIA GeForce RTX 4090"}]
        cached_snapshot = {
            "gpus": gpu_snapshot,
            "active_training_devices": ["cuda:0"],
            "stale": False,
            "refreshing": False,
        }
        MODULE.system_info_cache = cached_snapshot
        MODULE.system_info_cache_updated_at = 100.0
        MODULE.system_info_refreshing = False

        with mock.patch("npz_preview_web.time.monotonic", side_effect=[101.0, 101.5]), mock.patch(
            "npz_preview_web.threading.Thread"
        ) as refresh_thread:
            first_snapshot = MODULE.system_info_snapshot()
            second_snapshot = MODULE.system_info_snapshot()

        self.assertIs(first_snapshot, second_snapshot)
        self.assertIs(first_snapshot, cached_snapshot)
        refresh_thread.assert_not_called()

    def test_system_info_snapshot_keeps_last_gpu_data_after_transient_failure(self) -> None:
        cached_snapshot = {
            "gpus": [{"index": 0, "name": "NVIDIA GeForce RTX 4090"}],
            "active_training_devices": ["cuda:0"],
            "stale": False,
        }
        MODULE.system_info_cache = cached_snapshot
        MODULE.system_info_cache_updated_at = 100.0
        MODULE.system_info_refreshing = True

        with mock.patch("npz_preview_web.time.monotonic", return_value=110.0), mock.patch(
            "npz_preview_web.gpu_inventory", return_value=[]
        ), mock.patch("npz_preview_web.active_training_runs", return_value=[
            {"device": "cuda:1", "run_name": "fixture", "iteration": None, "max_iterations": None}
        ]):
            MODULE._refresh_system_info_cache()

        self.assertTrue(MODULE.system_info_cache["stale"])
        self.assertFalse(MODULE.system_info_cache["refreshing"])
        self.assertEqual(MODULE.system_info_cache["gpus"], cached_snapshot["gpus"])
        self.assertEqual(MODULE.system_info_cache["active_training_devices"], ["cuda:1"])
        self.assertFalse(MODULE.system_info_refreshing)

    def test_active_training_devices_are_parsed_from_process_arguments(self) -> None:
        completed = mock.Mock(
            returncode=0,
            stdout=(
                "100 python scripts/rsl_rl/train.py --device cuda:0 --num_envs 20480\n"
                "200 python scripts/rsl_rl/train.py --device=cuda:3 --num_envs 18432\n"
                "300 python scripts/rsl_rl/train.py --distributed --physical_gpu_ids 1,2\n"
            ),
        )

        with mock.patch("npz_preview_web.subprocess.run", return_value=completed):
            devices = MODULE.active_training_devices()

        self.assertEqual(devices, {"cuda:0", "cuda:1", "cuda:2", "cuda:3"})

    def test_active_training_runs_include_run_name_and_logged_iteration(self) -> None:
        completed = mock.Mock(
            returncode=0,
            stdout=(
                "100 bash -lc python scripts/rsl_rl/train.py --device cuda:1 --run_name walking_a1\n"
                "101 python scripts/rsl_rl/train.py --device cuda:1 --run_name walking_a1\n"
            ),
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "npz_preview_web.MANUAL_LOG_ROOT", Path(directory)
        ), mock.patch("npz_preview_web.subprocess.run", return_value=completed):
            (Path(directory) / "walking_a1.out").write_text(
                "Learning iteration 123/40000\n", encoding="utf-8"
            )
            runs = MODULE.active_training_runs()

        self.assertEqual(
            runs,
            [{"device": "cuda:1", "run_name": "walking_a1", "iteration": 123, "max_iterations": 40000}],
        )

    def test_lists_all_active_training_jobs_without_duplicate_multi_gpu_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MODULE.JobStore(Path(directory))
            first = store.create("first.npz", b"first", skip_validation=True)
            second = store.create("second.npz", b"second", skip_validation=True)
            inactive = store.create("inactive.npz", b"inactive", skip_validation=True)
            first.training_status = "running"
            first.training_session = f"wbt_{first.job_id[:8]}_a1"
            first.training_run_name = "first_run"
            first.training_started_at = 100.0
            first.training_config = {"devices": ["cuda:0", "cuda:1"], "max_iterations": 1000}
            second.training_status = "starting"
            second.training_session = f"wbt_{second.job_id[:8]}_a1"
            second.training_run_name = "second_run"
            second.training_started_at = 200.0
            second.training_config = {"devices": ["cuda:2"], "max_iterations": 2000}
            inactive.training_status = "completed"

            with mock.patch.object(store, "_tmux_sessions", return_value=[
                first.training_session,
                second.training_session,
            ]), mock.patch.object(store, "_tmux_pane_pid", return_value=123):
                active_jobs = store.list_active_training_jobs()

        self.assertEqual([job.job_id for job in active_jobs], [second.job_id, first.job_id])
        self.assertEqual(active_jobs[1].active_training_summary()["devices"], ["cuda:0", "cuda:1"])

    def test_frontend_can_select_and_stop_any_active_training_job(self) -> None:
        source = APP_PATH.read_text(encoding="utf-8")
        html = (APP_PATH.parent / "index.html").read_text(encoding="utf-8")

        self.assertIn('id="activeTrainingSelect"', html)
        self.assertIn('id="stopSelectedTrainingButton"', html)
        self.assertIn('fetch("/api/active-jobs", { cache: "no-store" })', source)
        self.assertIn("async function stopSelectedTraining()", source)
        self.assertIn("`/api/jobs/${jobId}/stop-training`", source)

    def test_job_store_uses_independent_locks_for_each_device(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MODULE.JobStore(Path(directory))

            first_gpu_lock = store._device_lock("cuda:0")
            same_gpu_lock = store._device_lock("cuda:0")
            second_gpu_lock = store._device_lock("cuda:1")

        self.assertIs(first_gpu_lock, same_gpu_lock)
        self.assertIsNot(first_gpu_lock, second_gpu_lock)

    def test_multi_gpu_lock_acquisition_rolls_back_when_one_gpu_is_busy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MODULE.JobStore(Path(directory))
            busy_gpu_lock = store._device_lock("cuda:1")
            busy_gpu_lock.acquire()
            try:
                with self.assertRaisesRegex(ValueError, "cuda:1"):
                    store._acquire_device_locks(["cuda:1", "cuda:0"])
                first_gpu_lock = store._device_lock("cuda:0")
                self.assertTrue(first_gpu_lock.acquire(blocking=False))
                first_gpu_lock.release()
            finally:
                busy_gpu_lock.release()

    def test_frontend_renders_and_refreshes_gpu_monitor(self) -> None:
        source = APP_PATH.read_text(encoding="utf-8")
        html = (APP_PATH.parent / "index.html").read_text(encoding="utf-8")
        backend_source = SCRIPT_PATH.read_text(encoding="utf-8")
        replay_source = REPLAY_SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn('id="gpuMonitor"', html)
        self.assertIn('id="refreshGpuButton"', html)
        self.assertIn('id="trainingDevice"', html)
        self.assertIn('id="trainingDevice" multiple', html)
        self.assertIn('$("#refreshGpuButton").addEventListener("click", loadSystemInfo)', source)
        self.assertIn("function renderGpuMonitor(gpus, stale = false, refreshing = false)", source)
        self.assertIn("training_runs", source)
        self.assertIn("ACTIVE TRAINING", source)
        self.assertIn("RUN NAME", source)
        self.assertIn("active_training_devices", source)
        self.assertIn("devices: trainingDevices", source)
        self.assertIn("CUDA_VISIBLE_DEVICES", backend_source)
        self.assertIn('"active_gpu": selected_gpu_index', replay_source)
        self.assertIn('"physics_gpu": selected_gpu_index', replay_source)
        self.assertIn("timeout=RENDER_PROCESS_TIMEOUT_SECONDS", backend_source)
        self.assertIn("new AbortController()", source)
        self.assertIn("state.gpuMonitorHasData", source)
        self.assertIn("setInterval(loadSystemInfo, 5000)", source)
        self.assertIn("system_info_snapshot()", backend_source)

    def test_accepts_current_30_body_motion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "valid.npz"
            self.write_motion(path)

            report = MODULE.validate_npz(path)

        self.assertTrue(report["valid"])
        self.assertTrue(report["renderable"])
        self.assertTrue(report["trainable"])
        self.assertEqual(report["summary"]["joint_count"], 29)
        self.assertEqual(report["summary"]["body_count"], 30)
        self.assertEqual(report["summary"]["duration_seconds"], 0.08)

    def test_warns_but_allows_preview_for_37_body_motion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server-format.npz"
            self.write_motion(path, bodies=37)

            report = MODULE.validate_npz(path)

        self.assertTrue(report["valid"])
        self.assertTrue(report["renderable"])
        self.assertFalse(report["trainable"])
        body_check = next(check for check in report["checks"] if check["id"] == "bodies")
        self.assertEqual(body_check["status"], "warn")

    def test_rejects_non_finite_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nan.npz"
            self.write_motion(path, add_nan=True)

            report = MODULE.validate_npz(path)

        self.assertFalse(report["valid"])
        finite_check = next(check for check in report["checks"] if check["id"] == "finite")
        self.assertEqual(finite_check["status"], "fail")

    def test_rejects_missing_required_array(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.npz"
            np.savez(path, fps=np.array([50.0]), joint_pos=np.zeros((2, 29)))

            report = MODULE.validate_npz(path)

        self.assertFalse(report["valid"])
        keys_check = next(check for check in report["checks"] if check["id"] == "keys")
        self.assertEqual(keys_check["status"], "fail")

    def test_builds_training_command_from_sop_defaults(self) -> None:
        command = MODULE.build_training_shell_command(
            Path("/tmp/motion file.npz"),
            "take_007_local_1gpu_7168",
            7168,
            10000,
            ["cuda:0"],
            Path("/tmp/training log.out"),
            Path("/tmp/training.exit"),
        )

        self.assertIn("Tracking-Flat-G1-Wo-State-Estimation-v0", command)
        self.assertIn("--num_envs 7168", command)
        self.assertIn("--max_iterations 10000", command)
        self.assertIn("--logger tensorboard", command)
        self.assertIn("--headless", command)
        self.assertIn("--device cuda:0", command)
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", command)
        self.assertIn("OMNI_KIT_ACCEPT_EULA=YES", command)
        self.assertIn("motion file.npz", command)
        self.assertNotIn("torch.distributed.run", command)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", command)

    def test_builds_distributed_training_command_for_selected_gpus(self) -> None:
        command = MODULE.build_training_shell_command(
            Path("/tmp/motion.npz"),
            "take_007_local_2gpu_7168",
            7168,
            10000,
            ["cuda:1", "cuda:3"],
            Path("/tmp/training.out"),
            Path("/tmp/training.exit"),
        )

        self.assertIn("-m torch.distributed.run", command)
        self.assertIn("--nproc_per_node 2", command)
        self.assertIn("--distributed", command)
        self.assertIn("--physical_gpu_ids 1,3", command)
        self.assertIn("export CUDA_VISIBLE_DEVICES=1,3", command)
        self.assertIn("--device cuda:0", command)

    def test_builds_training_command_that_resumes_a_checkpoint(self) -> None:
        checkpoint = MODULE.TrainingCheckpoint(
            run_directory="2026-07-25_15-00-00_dance_abcdef12_a1",
            checkpoint_name="model_1500.pt",
            iteration=1500,
            path=Path("/tmp/model_1500.pt"),
        )

        command = MODULE.build_training_shell_command(
            Path("/tmp/motion.npz"),
            "dance_resume",
            18432,
            5000,
            ["cuda:0", "cuda:1"],
            Path("/tmp/training.out"),
            Path("/tmp/training.exit"),
            resume_checkpoint=checkpoint,
        )

        self.assertIn("--resume True", command)
        self.assertIn("--load_run 2026-07-25_15-00-00_dance_abcdef12_a1", command)
        self.assertIn("--checkpoint model_1500.pt", command)

    def test_lists_only_checkpoints_owned_by_current_job(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "outputs"
            training_root = Path(directory) / "training"
            job_id = "abcdef1234567890abcdef1234567890"
            job_directory = output_root / job_id
            job_directory.mkdir(parents=True)
            motion_path = job_directory / "dance.npz"
            motion_path.touch()
            job = MODULE.PreviewJob(
                job_id=job_id,
                original_name=motion_path.name,
                directory=job_directory,
                motion_path=motion_path,
                report={"trainable": True},
            )
            owned_run = training_root / "2026-07-25_15-00-00_dance_abcdef12_a1"
            resumed_run = training_root / "2026-07-26_12-00-00_dance_resume"
            other_run = training_root / "2026-07-25_15-00-00_dance_deadbeef_a1"
            owned_run.mkdir(parents=True)
            (resumed_run / "params").mkdir(parents=True)
            other_run.mkdir()
            for filename in ("model_500.pt", "model_1500.pt", "model_1500.onnx", "notes.txt"):
                (owned_run / filename).touch()
            (resumed_run / "params" / "env.yaml").write_text(
                f"commands:\n  motion:\n    motion_file: {motion_path.resolve()}\n",
                encoding="utf-8",
            )
            (resumed_run / "model_2000.pt").touch()
            (other_run / "model_9000.pt").touch()
            store = MODULE.JobStore(output_root)

            with mock.patch.object(MODULE, "TENSORBOARD_LOG_ROOT", training_root):
                checkpoints = store.list_training_checkpoints(job)

        self.assertEqual([checkpoint.iteration for checkpoint in checkpoints], [2000, 1500, 500])
        self.assertEqual(checkpoints[0].checkpoint_name, "model_2000.pt")

    def test_rejects_checkpoint_identifier_from_another_job(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "outputs"
            training_root = Path(directory) / "training"
            job_id = "abcdef1234567890abcdef1234567890"
            job_directory = output_root / job_id
            job_directory.mkdir(parents=True)
            motion_path = job_directory / "dance.npz"
            motion_path.touch()
            job = MODULE.PreviewJob(
                job_id=job_id,
                original_name=motion_path.name,
                directory=job_directory,
                motion_path=motion_path,
                report={"trainable": True},
            )
            training_root.mkdir()
            store = MODULE.JobStore(output_root)

            with mock.patch.object(MODULE, "TENSORBOARD_LOG_ROOT", training_root):
                with self.assertRaisesRegex(ValueError, "不属于当前动作任务"):
                    store.resolve_training_checkpoint(job, "../other/model_500.pt")

    def test_exposes_high_utilization_dual_gpu_environment_presets(self) -> None:
        html = (APP_PATH.parent / "index.html").read_text(encoding="utf-8")

        self.assertIn(22528, MODULE.TRAIN_NUM_ENVS)
        self.assertIn(24576, MODULE.TRAIN_NUM_ENVS)
        self.assertIn('<option value="22528">', html)
        self.assertIn('<option value="24576">', html)

    def test_builds_training_command_with_ppo_tuning_overrides(self) -> None:
        ppo_settings = MODULE.PPOTrainingSettings.from_mapping(
            {
                "num_steps_per_env": 24,
                "num_mini_batches": 16,
                "num_learning_epochs": 5,
                "learning_rate": 0.001,
                "desired_kl": 0.01,
                "save_interval": 250,
            },
            num_envs=18432,
        )

        command = MODULE.build_training_shell_command(
            Path("/tmp/motion.npz"),
            "take_007_local_2gpu_18432",
            18432,
            20000,
            ["cuda:0", "cuda:1"],
            Path("/tmp/training.out"),
            Path("/tmp/training.exit"),
            ppo_settings=ppo_settings,
        )

        self.assertIn("--num_steps_per_env 24", command)
        self.assertIn("--num_mini_batches 16", command)
        self.assertIn("--num_learning_epochs 5", command)
        self.assertIn("--learning_rate 0.001", command)
        self.assertIn("--desired_kl 0.01", command)
        self.assertIn("--save_interval 250", command)

    def test_rejects_ppo_mini_batches_that_do_not_partition_each_gpu_rollout(self) -> None:
        with self.assertRaisesRegex(ValueError, "整除"):
            MODULE.PPOTrainingSettings.from_mapping(
                {
                    "num_steps_per_env": 24,
                    "num_mini_batches": 7,
                    "num_learning_epochs": 5,
                    "learning_rate": 0.001,
                    "desired_kl": 0.01,
                },
                num_envs=18432,
            )

    def test_rejects_fractional_and_non_finite_ppo_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "整数"):
            MODULE.PPOTrainingSettings.from_mapping(
                {"num_steps_per_env": 24.5},
                num_envs=18432,
            )

    def test_rejects_fractional_environment_and_iteration_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "整数"):
            MODULE.parse_training_request_settings({"num_envs": 18432.5})
        with self.assertRaisesRegex(ValueError, "整数"):
            MODULE.parse_training_request_settings({"max_iterations": 10000.5})
        with self.assertRaisesRegex(ValueError, "整数"):
            MODULE.parse_training_request_settings({"save_interval": 12.5})
        with self.assertRaisesRegex(ValueError, "模型保存间隔"):
            MODULE.parse_training_request_settings({"save_interval": 0})
        with self.assertRaisesRegex(ValueError, "有限"):
            MODULE.PPOTrainingSettings.from_mapping(
                {"learning_rate": float("nan")},
                num_envs=18432,
            )

    def test_training_launcher_keeps_runner_on_selected_device(self) -> None:
        source = TRAIN_SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("env_cfg.sim.device = args_cli.device", source)
        self.assertIn("agent_cfg.device = args_cli.device", source)

    def test_training_launcher_applies_ppo_tuning_overrides(self) -> None:
        source = TRAIN_SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn('parser.add_argument("--num_steps_per_env"', source)
        self.assertIn('parser.add_argument("--num_mini_batches"', source)
        self.assertIn('parser.add_argument("--num_learning_epochs"', source)
        self.assertIn('parser.add_argument("--learning_rate"', source)
        self.assertIn('parser.add_argument("--desired_kl"', source)
        self.assertIn('parser.add_argument("--save_interval"', source)
        self.assertIn("agent_cfg.num_steps_per_env = args_cli.num_steps_per_env", source)
        self.assertIn("agent_cfg.algorithm.num_mini_batches = args_cli.num_mini_batches", source)
        self.assertIn("agent_cfg.algorithm.num_learning_epochs = args_cli.num_learning_epochs", source)
        self.assertIn("agent_cfg.algorithm.learning_rate = args_cli.learning_rate", source)
        self.assertIn("agent_cfg.algorithm.desired_kl = args_cli.desired_kl", source)
        self.assertIn("agent_cfg.save_interval = args_cli.save_interval", source)

    def test_frontend_submits_ppo_tuning_controls(self) -> None:
        source = APP_PATH.read_text(encoding="utf-8")
        html = (APP_PATH.parent / "index.html").read_text(encoding="utf-8")

        for control_id in (
            "saveInterval",
            "numStepsPerEnv",
            "numMiniBatches",
            "numLearningEpochs",
            "learningRate",
            "desiredKl",
        ):
            self.assertIn(f'id="{control_id}"', html)
        self.assertIn("num_steps_per_env: numStepsPerEnv", source)
        self.assertIn("num_mini_batches: numMiniBatches", source)
        self.assertIn("num_learning_epochs: numLearningEpochs", source)
        self.assertIn("learning_rate: learningRate", source)
        self.assertIn("desired_kl: desiredKl", source)
        self.assertIn("save_interval: saveInterval", source)

    def test_frontend_can_resume_from_discovered_checkpoint(self) -> None:
        source = APP_PATH.read_text(encoding="utf-8")
        html = (APP_PATH.parent / "index.html").read_text(encoding="utf-8")

        self.assertIn('id="resumeTrainingControls"', html)
        self.assertIn('id="resumeCheckpoint"', html)
        self.assertIn('id="resumeTrainingDevices" multiple', html)
        self.assertIn('id="resumeIterations"', html)
        self.assertIn('id="resumeTrainingButton"', html)
        self.assertIn("/checkpoints`, { cache: \"no-store\" }", source)
        self.assertIn("resume_checkpoint_id: resumeCheckpoint.id", source)
        self.assertIn('resumeCheckpoint ? $("#resumeIterations").value', source)
        self.assertIn('resumeCheckpoint ? $("#resumeTrainingDevices")', source)
        self.assertIn('populateGpuSelect($("#resumeTrainingDevices")', source)
        self.assertIn("function resumeTraining()", source)

    def test_frontend_can_open_persisted_training_history(self) -> None:
        source = APP_PATH.read_text(encoding="utf-8")
        html = (APP_PATH.parent / "index.html").read_text(encoding="utf-8")

        self.assertIn('id="trainingHistorySelect"', html)
        self.assertIn('id="openTrainingHistoryButton"', html)
        self.assertIn('id="refreshTrainingHistoryButton"', html)
        self.assertIn('fetch("/api/training-runs", { cache: "no-store" })', source)
        self.assertIn("function openSelectedHistoricalJob()", source)
        self.assertIn("restoreJob(job);", source)
        self.assertIn("不会停止后台训练", source)

    def test_normalizes_training_run_names(self) -> None:
        self.assertEqual(MODULE.normalize_run_name("Take 007 / Test!", "fallback"), "take_007_test")

    def test_quick_training_report_does_not_read_npz_contents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "previously-checked.npz"
            path.write_bytes(b"intentionally not parsed")

            report = MODULE.skipped_validation_report(path)

        self.assertIsNone(report["valid"])
        self.assertTrue(report["validation_skipped"])
        self.assertTrue(report["trainable"])
        self.assertFalse(report["renderable"])
        self.assertEqual(report["checks"][0]["status"], "warn")

    def test_job_survives_store_recreation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            motion = io.BytesIO()
            body_quat = np.zeros((2, 30, 4), dtype=np.float32)
            body_quat[..., 0] = 1.0
            np.savez(
                motion,
                fps=np.array([50.0]),
                joint_pos=np.zeros((2, 29), dtype=np.float32),
                joint_vel=np.zeros((2, 29), dtype=np.float32),
                body_pos_w=np.zeros((2, 30, 3), dtype=np.float32),
                body_quat_w=body_quat,
                body_lin_vel_w=np.zeros((2, 30, 3), dtype=np.float32),
                body_ang_vel_w=np.zeros((2, 30, 3), dtype=np.float32),
            )
            first_store = MODULE.JobStore(output_root)
            created = first_store.create("remember-me.npz", motion.getvalue())

            restored = MODULE.JobStore(output_root).get(created.job_id)

        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertEqual(restored.original_name, "remember-me.npz")
        self.assertTrue(restored.report["trainable"])

    def test_lists_persisted_training_jobs_with_active_and_recent_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            store = MODULE.JobStore(output_root)
            older_job = store.create("older.npz", b"older", skip_validation=True)
            newer_job = store.create("newer.npz", b"newer", skip_validation=True)
            active_job = store.create("active.npz", b"active", skip_validation=True)
            preview_only_job = store.create("preview-only.npz", b"preview", skip_validation=True)

            older_job.training_status = "completed"
            older_job.training_attempt = 1
            older_job.training_run_name = f"older_{older_job.job_id[:8]}_a1"
            older_job.training_finished_at = 100.0
            newer_job.training_status = "stopped"
            newer_job.training_attempt = 1
            newer_job.training_run_name = f"newer_{newer_job.job_id[:8]}_a1"
            newer_job.training_finished_at = 200.0
            active_job.training_status = "running"
            active_job.training_attempt = 1
            active_job.training_run_name = f"active_{active_job.job_id[:8]}_a1"
            active_job.training_started_at = 50.0
            for job in (older_job, newer_job, active_job, preview_only_job):
                store._persist_job(job)

            restored_store = MODULE.JobStore(output_root)
            with mock.patch.object(restored_store, "_tmux_sessions", return_value=[active_job.training_run_name]):
                historical_jobs = restored_store.list_historical_training_jobs()

        self.assertEqual(
            [job.job_id for job in historical_jobs],
            [active_job.job_id, newer_job.job_id, older_job.job_id],
        )
        self.assertNotIn(preview_only_job.job_id, [job.job_id for job in historical_jobs])

    def test_history_summary_does_not_include_reports_or_logs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            store = MODULE.JobStore(output_root)
            job = store.create("history.npz", b"history", skip_validation=True)
            job.training_status = "completed"
            job.training_attempt = 1
            job.training_config = {
                "device": "cuda:0",
                "devices": ["cuda:0", "cuda:1"],
                "num_envs": 22528,
            }

            summary = job.history_summary()

        self.assertNotIn("report", summary)
        self.assertNotIn("log", summary)
        self.assertEqual(summary["training"]["devices"], ["cuda:0", "cuda:1"])
        self.assertEqual(summary["training"]["num_envs"], 22528)

    def test_lists_each_training_run_and_marks_web_resumability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "outputs"
            training_root = Path(directory) / "training"
            store = MODULE.JobStore(output_root)
            job = store.create("dance.npz", b"dance", skip_validation=True)
            job.training_status = "completed"
            job.training_attempt = 1
            store._persist_job(job)

            linked_run = training_root / "2026-07-28_10-00-00_dance"
            migrated_run = training_root / f"2026-07-28_09-30-00_dance_{job.job_id[:8]}_a1"
            smoke_run = training_root / "2026-07-28_09-00-00_smoke"
            (linked_run / "params").mkdir(parents=True)
            (migrated_run / "params").mkdir(parents=True)
            (smoke_run / "params").mkdir(parents=True)
            (linked_run / "params" / "env.yaml").write_text(
                f"motion_file: {job.motion_path.resolve()}\n",
                encoding="utf-8",
            )
            (migrated_run / "params" / "env.yaml").write_text(
                f"motion_file: /old/server/outputs/npz_preview_web/{job.job_id}/dance.npz\n",
                encoding="utf-8",
            )
            (smoke_run / "params" / "env.yaml").write_text(
                "motion_file: /tmp/external-motion.npz\n",
                encoding="utf-8",
            )
            (linked_run / "model_1500.pt").touch()
            (linked_run / "model_500.pt").touch()
            (migrated_run / "model_2000.pt").touch()
            (smoke_run / "model_1.pt").touch()

            with mock.patch.object(MODULE, "TENSORBOARD_LOG_ROOT", training_root):
                training_runs = store.list_training_runs()

        run_by_name = {training_run["run_directory"]: training_run for training_run in training_runs}
        self.assertEqual(run_by_name[linked_run.name]["job_id"], job.job_id)
        self.assertTrue(run_by_name[linked_run.name]["resumable"])
        self.assertEqual(run_by_name[linked_run.name]["latest_iteration"], 1500)
        self.assertEqual(run_by_name[linked_run.name]["checkpoint_count"], 2)
        self.assertEqual(run_by_name[migrated_run.name]["job_id"], job.job_id)
        self.assertTrue(run_by_name[migrated_run.name]["resumable"])
        self.assertEqual(run_by_name[migrated_run.name]["latest_iteration"], 2000)
        self.assertIsNone(run_by_name[smoke_run.name]["job_id"])
        self.assertFalse(run_by_name[smoke_run.name]["resumable"])

    def test_relinks_migrated_run_after_same_named_motion_is_reuploaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "outputs"
            training_root = Path(directory) / "training"
            store = MODULE.JobStore(output_root)
            replacement_job = store.create("62jtq3_3.npz", b"replacement", skip_validation=True)
            migrated_run = training_root / "2026-08-14_22-36-35_62jtq3_3_local_1gpu_20480"
            (migrated_run / "params").mkdir(parents=True)
            (migrated_run / "params" / "env.yaml").write_text(
                "motion_file: /missing/server/outputs/old-job/62jtq3_3.npz\n",
                encoding="utf-8",
            )
            (migrated_run / "model_58500.pt").touch()

            with mock.patch.object(MODULE, "TENSORBOARD_LOG_ROOT", training_root):
                training_runs = store.list_training_runs()
                checkpoints = store.list_training_checkpoints(replacement_job)

        self.assertEqual(training_runs[0]["job_id"], replacement_job.job_id)
        self.assertTrue(training_runs[0]["resumable"])
        self.assertIsNone(training_runs[0]["missing_motion_filename"])
        self.assertEqual([checkpoint.iteration for checkpoint in checkpoints], [58500])

    def test_reports_missing_motion_filename_for_unlinked_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "outputs"
            training_root = Path(directory) / "training"
            store = MODULE.JobStore(output_root)
            orphaned_run = training_root / "2026-08-14_22-36-35_62jtq3_3_local_1gpu_20480"
            (orphaned_run / "params").mkdir(parents=True)
            (orphaned_run / "params" / "env.yaml").write_text(
                "motion_file: /missing/server/outputs/old-job/62jtq3_3.npz\n",
                encoding="utf-8",
            )
            (orphaned_run / "model_58500.pt").touch()

            with mock.patch.object(MODULE, "TENSORBOARD_LOG_ROOT", training_root):
                training_runs = store.list_training_runs()

        self.assertIsNone(training_runs[0]["job_id"])
        self.assertFalse(training_runs[0]["resumable"])
        self.assertEqual(training_runs[0]["missing_motion_filename"], "62jtq3_3.npz")

    def test_recovers_legacy_active_training_from_tmux_and_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            job_id = "31715625efcc4ce3808451231ddc66c6"
            job_directory = output_root / job_id
            job_directory.mkdir()
            motion_path = job_directory / "dance_fixed.npz"
            self.write_motion(motion_path)
            log_path = Path(directory) / "dance_local_1gpu_7168_31715625_a1.out"
            log_path.write_text(
                "Motion Inspector training launch\n"
                "tmux session: wbt_31715625_dance_train\n"
                f"motion: {motion_path}\n"
                "run name: dance_local_1gpu_7168_31715625_a1\n"
                "environments: 7168\n"
                "max iterations: 10000\n"
                "device: cuda:0\n",
                encoding="utf-8",
            )
            store = MODULE.JobStore(output_root)

            with mock.patch.object(MODULE, "MANUAL_LOG_ROOT", Path(directory)), mock.patch.object(
                store, "_tmux_sessions", return_value=["wbt_31715625_dance_train"]
            ), mock.patch.object(store, "_tmux_pane_pid", return_value=4242):
                restored = store.find_active_job()

        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertEqual(restored.job_id, job_id)
        self.assertEqual(restored.training_status, "running")
        self.assertEqual(restored.training_pane_pid, 4242)
        self.assertEqual(restored.training_config["num_envs"], 7168)
        self.assertEqual(restored.training_run_name, "dance_local_1gpu_7168_31715625_a1")

    def test_recovery_backfills_ppo_defaults_without_overwriting_legacy_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "outputs"
            first_store = MODULE.JobStore(output_root)
            created = first_store.create("legacy.npz", b"legacy", skip_validation=True)
            log_path = Path(directory) / f"legacy_{created.job_id[:8]}_a1.out"
            log_path.write_text(
                "Motion Inspector training launch\n"
                "tmux session: wbt_legacy_train\n"
                "environments: 7168\n"
                "max iterations: 10000\n"
                "device: cuda:0\n",
                encoding="utf-8",
            )
            created.training_config = {
                "task": MODULE.TRAIN_TASK,
                "num_envs": 7168,
                "max_iterations": 10000,
                "device": "cuda:0",
                "custom_legacy_value": "keep-me",
            }
            created.training_log_path = log_path
            created.training_session = "wbt_legacy_train"
            first_store._persist_job(created)
            restored_store = MODULE.JobStore(output_root)

            with mock.patch.object(MODULE, "MANUAL_LOG_ROOT", Path(directory)), mock.patch.object(
                restored_store, "_tmux_sessions", return_value=["wbt_legacy_train"]
            ), mock.patch.object(restored_store, "_tmux_pane_pid", return_value=4242):
                restored = restored_store.get(created.job_id)

        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertEqual(restored.training_config["custom_legacy_value"], "keep-me")
        self.assertEqual(restored.training_config["num_steps_per_env"], 24)
        self.assertEqual(restored.training_config["num_mini_batches"], 4)
        self.assertEqual(restored.training_config["num_learning_epochs"], 5)
        self.assertEqual(restored.training_config["learning_rate"], 0.001)
        self.assertEqual(restored.training_config["desired_kl"], 0.01)

    def test_frontend_bootstrap_restores_and_resumes_saved_job(self) -> None:
        source = APP_PATH.read_text(encoding="utf-8")

        self.assertIn('localStorage.setItem(JOB_STORAGE_KEY, job.job_id)', source)
        self.assertIn('localStorage.getItem(JOB_STORAGE_KEY)', source)
        self.assertIn('fetchJob("/api/active-job")', source)
        self.assertIn("function restoreJob(job)", source)
        self.assertIn("setInterval(pollTraining, 2200)", source)
        self.assertTrue(source.rstrip().endswith("restorePersistedJob();"))

    def test_server_startup_ensures_tensorboard_before_listening(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            arguments = mock.Mock(host="127.0.0.1", port=8766, output_dir=Path(directory))
            job_store = mock.Mock()
            server = mock.Mock()
            server.serve_forever.side_effect = KeyboardInterrupt
            startup_order: list[str] = []
            job_store.ensure_tensorboard.side_effect = lambda: startup_order.append("tensorboard")

            def create_server(*_args: object, **_kwargs: object) -> mock.Mock:
                startup_order.append("frontend")
                return server

            with mock.patch.object(MODULE, "parse_args", return_value=arguments), mock.patch.object(
                MODULE, "JobStore", return_value=job_store
            ), mock.patch.object(
                MODULE, "MotionInspectorHTTPServer", side_effect=create_server
            ), mock.patch("builtins.print"):
                MODULE.main()

        self.assertEqual(startup_order, ["tensorboard", "frontend"])
        self.assertIs(server.job_store, job_store)
        server.server_close.assert_called_once_with()

    def test_tensorboard_is_reused_when_port_is_already_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MODULE.JobStore(Path(directory))
            with mock.patch("npz_preview_web.shutil.which", return_value="/venv/bin/tensorboard"), mock.patch.object(
                store, "_tcp_port_open", side_effect=lambda host, port: port == 6007
            ), mock.patch.object(
                store, "_tmux_session_exists", side_effect=lambda session: session == "wbt_tensorboard_6007"
            ), mock.patch("npz_preview_web.subprocess.run") as run:
                store.ensure_tensorboard()

        run.assert_not_called()

    def test_tensorboard_tmux_is_started_and_ready_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MODULE.JobStore(Path(directory))
            completed = mock.Mock(returncode=0, stderr="")
            with mock.patch("npz_preview_web.shutil.which", return_value="/venv/bin/tensorboard"), mock.patch.object(
                store, "_tcp_port_open", return_value=False
            ), mock.patch.object(
                store, "_tmux_session_exists", return_value=False
            ), mock.patch.object(store, "_wait_for_tcp_port", return_value=True), mock.patch(
                "npz_preview_web.subprocess.run", return_value=completed
            ) as run:
                store.ensure_tensorboard()

        command = run.call_args.args[0]
        self.assertEqual(command[:5], ["tmux", "new-session", "-d", "-s", "wbt_tensorboard"])
        shell_payload = command[-1]
        self.assertIn(str(MODULE.PROJECT_ROOT), shell_payload)
        self.assertIn("tensorboard", shell_payload)
        self.assertIn("--logdir", shell_payload)
        self.assertIn(str(MODULE.PROJECT_ROOT / "logs" / "rsl_rl" / "g1_flat"), shell_payload)
        self.assertIn("--port 6006", shell_payload)
        self.assertIn("/venv/bin/tensorboard", shell_payload)
        self.assertIn("export PATH=", shell_payload)
        self.assertIn(":$PATH", shell_payload)
        self.assertNotIn("-m tensorboard", shell_payload)

    def test_tensorboard_uses_fallback_port_when_default_is_occupied(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MODULE.JobStore(Path(directory))
            completed = mock.Mock(returncode=0, stderr="")

            def port_is_open(host: str, port: int) -> bool:
                return port == MODULE.TENSORBOARD_DEFAULT_PORT

            with mock.patch.object(store, "_tcp_port_open", side_effect=port_is_open), mock.patch.object(
                store, "_tmux_session_exists", return_value=False
            ), mock.patch.object(store, "_wait_for_tcp_port", return_value=True), mock.patch(
                "npz_preview_web.subprocess.run", return_value=completed
            ) as run:
                store.ensure_tensorboard()

        command = run.call_args.args[0]
        self.assertEqual(command[:5], ["tmux", "new-session", "-d", "-s", "wbt_tensorboard_6007"])
        shell_payload = command[-1]
        self.assertIn("--port 6007", shell_payload)

    def test_tensorboard_skips_multiple_ports_owned_by_other_processes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MODULE.JobStore(Path(directory))
            completed = mock.Mock(returncode=0, stderr="")

            with mock.patch("npz_preview_web.shutil.which", return_value="/venv/bin/tensorboard"), mock.patch.object(
                store, "_tcp_port_open", side_effect=lambda host, port: port in {6006, 6007}
            ), mock.patch.object(
                store, "_tmux_session_exists", return_value=False
            ), mock.patch.object(store, "_wait_for_tcp_port", return_value=True), mock.patch(
                "npz_preview_web.subprocess.run", return_value=completed
            ) as run:
                store.ensure_tensorboard()

        command = run.call_args.args[0]
        self.assertEqual(command[:5], ["tmux", "new-session", "-d", "-s", "wbt_tensorboard_6008"])
        self.assertIn("--port 6008", command[-1])

    def test_public_training_payload_uses_selected_tensorboard_port(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MODULE.JobStore(Path(directory))
            job = store.create("motion.npz", b"not-an-npz", skip_validation=True)
            completed = mock.Mock(returncode=0, stderr="")

            with mock.patch.object(
                store, "_tcp_port_open", side_effect=lambda host, port: port == 6006
            ), mock.patch.object(store, "_tmux_session_exists", return_value=False), mock.patch.object(
                store, "_wait_for_tcp_port", return_value=True
            ), mock.patch(
                "npz_preview_web.subprocess.run", return_value=completed
            ):
                store.ensure_tensorboard()

            self.assertEqual(
                job.public(include_log=False)["training"]["tensorboard_url"],
                "http://127.0.0.1:6007/",
            )

    def test_training_waits_for_tensorboard_before_launching_thread(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            job_directory = output_root / ("a" * 32)
            job_directory.mkdir()
            motion_path = job_directory / "dance.npz"
            motion_path.touch()
            job = MODULE.PreviewJob(
                job_id="a" * 32,
                original_name="dance.npz",
                directory=job_directory,
                motion_path=motion_path,
                report={"trainable": True, "renderable": False},
            )
            store = MODULE.JobStore(output_root)
            store.jobs[job.job_id] = job
            order: list[str] = []
            fake_thread = mock.Mock()
            fake_thread.start.side_effect = lambda: order.append("training")
            no_training = mock.Mock(returncode=1, stdout="")
            with mock.patch("npz_preview_web.shutil.which", return_value="/usr/bin/tmux"), mock.patch(
                "npz_preview_web.subprocess.run", return_value=no_training
            ), mock.patch.object(
                store, "ensure_tensorboard", side_effect=lambda: order.append("tensorboard")
            ), mock.patch("npz_preview_web.threading.Thread", return_value=fake_thread):
                store.start_training(job, 7168, 10000, "dance", ["cuda:0"])

        self.assertEqual(order, ["tensorboard", "training"])

    def test_stop_training_targets_only_the_current_job_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            job_directory = output_root / ("b" * 32)
            job_directory.mkdir()
            motion_path = job_directory / "dance.npz"
            motion_path.touch()
            job = MODULE.PreviewJob(
                job_id="b" * 32,
                original_name="dance.npz",
                directory=job_directory,
                motion_path=motion_path,
                report={"trainable": True, "renderable": False},
                training_status="running",
                training_session="wbt_bbbbbbbb_dance_train",
            )
            store = MODULE.JobStore(output_root)
            store.jobs[job.job_id] = job
            completed = mock.Mock(returncode=0, stderr="")
            stop_thread = mock.Mock()

            with mock.patch(
                "npz_preview_web.subprocess.run", return_value=completed
            ) as run, mock.patch(
                "npz_preview_web.threading.Thread", return_value=stop_thread
            ):
                store.stop_training(job)

        self.assertEqual(job.training_status, "stopping")
        self.assertEqual(
            run.call_args.args[0],
            ["tmux", "send-keys", "-t", "wbt_bbbbbbbb_dance_train", "C-c"],
        )
        stop_thread.start.assert_called_once_with()

    def test_stop_training_restores_status_when_tmux_signal_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            job_directory = output_root / ("c" * 32)
            job_directory.mkdir()
            motion_path = job_directory / "dance.npz"
            motion_path.touch()
            job = MODULE.PreviewJob(
                job_id="c" * 32,
                original_name="dance.npz",
                directory=job_directory,
                motion_path=motion_path,
                report={"trainable": True, "renderable": False},
                training_status="running",
                training_session="wbt_cccccccc_dance_train",
            )
            store = MODULE.JobStore(output_root)
            store.jobs[job.job_id] = job
            completed = mock.Mock(returncode=1, stderr="missing session")

            with mock.patch("npz_preview_web.subprocess.run", return_value=completed):
                with self.assertRaisesRegex(ValueError, "missing session"):
                    store.stop_training(job)

        self.assertEqual(job.training_status, "running")
        self.assertEqual(job.training_error, "missing session")

    def test_frontend_stop_button_calls_current_job_endpoint(self) -> None:
        source = APP_PATH.read_text(encoding="utf-8")

        self.assertIn('$("#stopTrainingButton").addEventListener("click", stopTraining)', source)
        self.assertIn("/stop-training`, { method: \"POST\" }", source)
        self.assertIn('stopButton.textContent = "正在停止…"', source)

    def _create_recovery_source(
        self,
        root: Path,
    ) -> tuple[object, object, Path, Path]:
        output_root = root / "outputs"
        training_root = root / "training"
        source_motion = root / "dance.npz"
        self.write_motion(source_motion)
        store = MODULE.JobStore(output_root)
        job = store.create("dance.npz", source_motion.read_bytes())
        job.training_config = {
            "num_envs": 7168,
            "max_iterations": 10000,
            "save_interval": 500,
        }
        job.training_attempt = 1
        job.training_status = "completed"
        run_directory = training_root / "dance_local_1gpu_7168"
        (run_directory / "params").mkdir(parents=True)
        (run_directory / "params" / "env.yaml").write_text(
            f"scene:\n  motion_file: {job.motion_path.resolve()}\n",
            encoding="utf-8",
        )
        (run_directory / "params" / "agent.yaml").write_text(
            "seed: 42\nrunner:\n  save_interval: 500\n",
            encoding="utf-8",
        )
        (run_directory / "model_100.pt").write_bytes(b"older-checkpoint")
        (run_directory / "model_200.pt").write_bytes(b"latest-checkpoint")
        job.training_run_name = run_directory.name
        store._persist_job(job)
        return store, job, training_root, run_directory

    def test_recovery_package_exports_only_latest_checkpoint_with_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store, job, training_root, run_directory = self._create_recovery_source(root)
            package_path = root / "recovery.zip"
            with mock.patch.object(MODULE, "TENSORBOARD_LOG_ROOT", training_root):
                checkpoint = store.build_training_recovery_package(
                    job,
                    run_directory.name,
                    package_path,
                )
            with zipfile.ZipFile(package_path) as archive:
                names = set(archive.namelist())
                manifest = json.loads(archive.read("manifest.json"))
                checkpoint_payload = archive.read("run/model_200.pt")

        self.assertEqual(checkpoint.iteration, 200)
        self.assertEqual(
            names,
            {
                "manifest.json",
                "motion/motion.npz",
                "run/model_200.pt",
                "run/params/env.yaml",
                "run/params/agent.yaml",
            },
        )
        self.assertEqual(manifest["version"], 1)
        self.assertEqual(manifest["checkpoint"]["iteration"], 200)
        self.assertEqual(
            manifest["files"]["run/model_200.pt"]["sha256"],
            hashlib.sha256(checkpoint_payload).hexdigest(),
        )

    def test_recovery_package_import_creates_new_copies_and_rewrites_motion_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store, job, training_root, run_directory = self._create_recovery_source(root)
            package_path = root / "recovery.zip"
            with mock.patch.object(MODULE, "TENSORBOARD_LOG_ROOT", training_root):
                store.build_training_recovery_package(job, run_directory.name, package_path)
                first_job, first_run_name = store.import_training_recovery_package(package_path)
                second_job, second_run_name = store.import_training_recovery_package(package_path)
                first_checkpoints = store.list_training_checkpoints(first_job)

            imported_environment = training_root / first_run_name / "params" / "env.yaml"
            environment_text = imported_environment.read_text(encoding="utf-8")
            restarted_store = MODULE.JobStore(root / "outputs")
            with mock.patch.object(MODULE, "TENSORBOARD_LOG_ROOT", training_root):
                restored = restarted_store.get(first_job.job_id)
                restored_checkpoints = restarted_store.list_training_checkpoints(restored)

        self.assertNotEqual(first_job.job_id, job.job_id)
        self.assertNotEqual(first_job.job_id, second_job.job_id)
        self.assertNotEqual(first_run_name, second_run_name)
        self.assertIn(str(first_job.motion_path), environment_text)
        self.assertNotIn(str(job.motion_path), environment_text)
        self.assertEqual([checkpoint.iteration for checkpoint in first_checkpoints], [200])
        self.assertIsNotNone(restored)
        self.assertEqual([checkpoint.iteration for checkpoint in restored_checkpoints], [200])

    def test_recovery_package_rejects_zip_slip_without_partial_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store, job, training_root, run_directory = self._create_recovery_source(root)
            valid_package = root / "valid.zip"
            malicious_package = root / "malicious.zip"
            with mock.patch.object(MODULE, "TENSORBOARD_LOG_ROOT", training_root):
                store.build_training_recovery_package(job, run_directory.name, valid_package)
            with zipfile.ZipFile(valid_package) as source, zipfile.ZipFile(
                malicious_package,
                "w",
                compression=zipfile.ZIP_DEFLATED,
            ) as destination:
                for info in source.infolist():
                    if info.filename == "run/params/env.yaml":
                        destination.writestr("../env.yaml", source.read(info))
                    else:
                        destination.writestr(info, source.read(info))

            output_directories_before = {path.name for path in (root / "outputs").iterdir() if path.is_dir()}
            training_directories_before = {path.name for path in training_root.iterdir() if path.is_dir()}
            with mock.patch.object(MODULE, "TENSORBOARD_LOG_ROOT", training_root):
                with self.assertRaisesRegex(ValueError, "不安全路径"):
                    store.import_training_recovery_package(malicious_package)
            output_directories_after = {path.name for path in (root / "outputs").iterdir() if path.is_dir()}
            training_directories_after = {path.name for path in training_root.iterdir() if path.is_dir()}

        self.assertEqual(output_directories_after, output_directories_before)
        self.assertEqual(training_directories_after, training_directories_before)
        self.assertFalse((root / "env.yaml").exists())

    def test_recovery_package_rejects_symlink_duplicate_hash_and_version_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store, job, training_root, run_directory = self._create_recovery_source(root)
            valid_package = root / "valid.zip"
            with mock.patch.object(MODULE, "TENSORBOARD_LOG_ROOT", training_root):
                store.build_training_recovery_package(job, run_directory.name, valid_package)
            with zipfile.ZipFile(valid_package) as archive:
                entries = {info.filename: archive.read(info) for info in archive.infolist()}

            hash_package = root / "hash.zip"
            with zipfile.ZipFile(hash_package, "w") as archive:
                for name, payload in entries.items():
                    archive.writestr(
                        name,
                        (b"x" * len(payload)) if name == "run/model_200.pt" else payload,
                    )

            symlink_package = root / "symlink.zip"
            with zipfile.ZipFile(symlink_package, "w") as archive:
                for name, payload in entries.items():
                    if name == "run/params/env.yaml":
                        symlink_info = zipfile.ZipInfo(name)
                        symlink_info.create_system = 3
                        symlink_info.external_attr = 0o120777 << 16
                        archive.writestr(symlink_info, b"motion/motion.npz")
                    else:
                        archive.writestr(name, payload)

            duplicate_package = root / "duplicate.zip"
            with zipfile.ZipFile(duplicate_package, "w") as archive:
                for name, payload in entries.items():
                    archive.writestr(name, payload)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    archive.writestr("run/params/agent.yaml", entries["run/params/agent.yaml"])

            version_package = root / "version.zip"
            invalid_manifest = json.loads(entries["manifest.json"])
            invalid_manifest["version"] = 999
            with zipfile.ZipFile(version_package, "w") as archive:
                for name, payload in entries.items():
                    archive.writestr(
                        name,
                        json.dumps(invalid_manifest).encode("utf-8")
                        if name == "manifest.json"
                        else payload,
                    )

            cases = (
                (hash_package, "SHA-256"),
                (symlink_package, "符号链接或特殊文件"),
                (duplicate_package, "重复文件"),
                (version_package, "版本不受支持"),
            )
            with mock.patch.object(MODULE, "TENSORBOARD_LOG_ROOT", training_root):
                for package_path, message in cases:
                    with self.subTest(package=package_path.name):
                        with self.assertRaisesRegex(ValueError, message):
                            store.import_training_recovery_package(package_path)
            self.assertEqual(
                {path.name for path in (root / "outputs").iterdir() if path.is_dir()},
                {job.job_id},
            )

    def test_training_recovery_http_export_and_import_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store, job, training_root, run_directory = self._create_recovery_source(root)
            server = MODULE.MotionInspectorHTTPServer(
                ("127.0.0.1", 0),
                MODULE.PreviewRequestHandler,
            )
            server.job_store = store
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
            try:
                with mock.patch.object(MODULE, "TENSORBOARD_LOG_ROOT", training_root):
                    query = urllib.parse.urlencode({"run_directory": run_directory.name})
                    connection.request(
                        "GET",
                        f"/api/jobs/{job.job_id}/training-package?{query}",
                    )
                    export_response = connection.getresponse()
                    package_payload = export_response.read()
                    content_type = export_response.getheader("Content-Type")
                    disposition = export_response.getheader("Content-Disposition")

                    connection.request(
                        "POST",
                        "/api/training-package/import",
                        body=package_payload,
                        headers={"Content-Type": "application/zip"},
                    )
                    import_response = connection.getresponse()
                    imported_payload = json.loads(import_response.read())
            finally:
                connection.close()
                server.shutdown()
                server.server_close()
                server_thread.join(timeout=5)

        self.assertEqual(export_response.status, 200)
        self.assertEqual(content_type, "application/zip")
        self.assertIn("attachment", disposition)
        self.assertEqual(import_response.status, 201)
        self.assertNotEqual(imported_payload["job_id"], job.job_id)
        self.assertTrue(imported_payload["imported_run_directory"])

    def test_frontend_exposes_recovery_package_export_and_drop_upload(self) -> None:
        source = APP_PATH.read_text(encoding="utf-8")
        html = (APP_PATH.parent / "index.html").read_text(encoding="utf-8")

        self.assertIn('id="exportTrainingPackageButton"', html)
        self.assertIn('id="trainingPackageDropZone"', html)
        self.assertIn('accept=".zip,application/zip"', html)
        self.assertIn("function exportSelectedTrainingPackage()", source)
        self.assertIn('xhr.open("POST", "/api/training-package/import")', source)
        self.assertIn("event.dataTransfer.files[0]", source)
        self.assertIn("payload.imported_run_directory", source)

    def test_frontend_clears_stale_training_state_before_new_launch(self) -> None:
        source = APP_PATH.read_text(encoding="utf-8")

        submitting_position = source.index('status: "submitting"')
        fetch_position = source.index("/train`, {", submitting_position)
        self.assertLess(submitting_position, fetch_position)
        self.assertIn('log: ""', source[submitting_position:fetch_position])
        self.assertIn('$("#trainingLog").textContent = "正在向本地服务提交训练任务…"', source)
        self.assertIn('status: "launch_failed"', source)
        self.assertIn("训练未启动：${error.message}", source)


if __name__ == "__main__":
    unittest.main()
