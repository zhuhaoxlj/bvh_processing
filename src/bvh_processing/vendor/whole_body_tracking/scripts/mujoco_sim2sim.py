"""Run a whole-body tracking ONNX policy in MuJoCo."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
import onnx
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Motion policy ONNX exported by play.py.")
    parser.add_argument("--xml", type=Path, required=True, help="G1 MuJoCo XML model.")
    parser.add_argument("--headless", action="store_true", help="Run without the MuJoCo viewer.")
    parser.add_argument("--duration", type=float, default=None, help="Stop after this many simulated seconds.")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many control steps.")
    parser.add_argument("--motion-length", type=int, default=None, help="Reference motion length from policy config.")
    parser.add_argument("--record-video", action="store_true", help="Render the simulation to an MP4 file.")
    parser.add_argument(
        "--video-path",
        type=Path,
        default=Path("outputs/mujoco_sim2sim.mp4"),
        help="Output MP4 path (default: outputs/mujoco_sim2sim.mp4).",
    )
    parser.add_argument("--video-width", type=int, default=1280, help="Video width in pixels (default: 1280).")
    parser.add_argument("--video-height", type=int, default=720, help="Video height in pixels (default: 720).")
    parser.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help="Output frame rate. Defaults to the 50 Hz policy control rate.",
    )
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=4.0,
        help="Tracking camera distance from the robot in meters (default: 4.0).",
    )
    parser.add_argument(
        "--camera-azimuth",
        type=float,
        default=135.0,
        help="Tracking camera azimuth in degrees (default: 135).",
    )
    parser.add_argument(
        "--camera-elevation",
        type=float,
        default=-15.0,
        help="Tracking camera elevation in degrees (default: -15).",
    )
    parser.add_argument(
        "--record-one-motion",
        action="store_true",
        help="Stop after one complete reference-motion cycle.",
    )
    parser.add_argument(
        "--reset-on-fall",
        action="store_true",
        help="Restart the motion when the pelvis falls below --fall-height.",
    )
    parser.add_argument(
        "--fall-height",
        type=float,
        default=0.45,
        help="Pelvis height in meters that triggers a reset (default: 0.45).",
    )
    return parser.parse_args()


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quat_matrix(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def metadata(model_path: Path) -> dict[str, str]:
    return {item.key: item.value for item in onnx.load(model_path).metadata_props}


def csv_floats(value: str) -> np.ndarray:
    return np.asarray([float(item) for item in value.split(",")], dtype=np.float64)


class Sim2Sim:
    control_dt = 0.02
    anchor_body_index = 7  # torso_link in G1FlatEnvCfg.commands.motion.body_names

    def __init__(
        self,
        model_path: Path,
        xml_path: Path,
        motion_length: int | None = None,
        reset_on_fall: bool = False,
        fall_height: float = 0.45,
    ):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        meta = metadata(model_path)
        self.observation_names = meta["observation_names"].split(",")
        self.observation_size = self.session.get_inputs()[0].shape[1]
        self.joint_names = meta["joint_names"].split(",")
        self.default_pos = csv_floats(meta["default_joint_pos"])
        self.stiffness = csv_floats(meta["joint_stiffness"])
        self.damping = csv_floats(meta["joint_damping"])
        self.action_scale = csv_floats(meta["action_scale"])
        if not all(len(values) == 29 for values in (self.default_pos, self.stiffness, self.damping, self.action_scale)):
            raise ValueError("Policy metadata must contain 29 values for the G1 joints.")

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.005
        self.steps_per_control = round(self.control_dt / self.model.opt.timestep)
        self.joint_ids = np.asarray(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names]
        )
        if np.any(self.joint_ids < 0):
            missing = [name for name, joint_id in zip(self.joint_names, self.joint_ids) if joint_id < 0]
            raise ValueError(f"MuJoCo model is missing policy joints: {missing}")
        self.qpos_ids = self.model.jnt_qposadr[self.joint_ids]
        self.dof_ids = self.model.jnt_dofadr[self.joint_ids]
        self.force_limits = np.abs(self.model.jnt_actfrcrange[self.joint_ids, 1])
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.previous_action = np.zeros(29, dtype=np.float32)
        self.time_step = 0
        self.total_control_steps = 0
        self.reset_count = 0
        self.motion_length = motion_length or self._infer_motion_length()
        self.reset_on_fall = reset_on_fall
        self.fall_height = fall_height
        self.reset()

    def _infer_motion_length(self) -> int:
        # The exporter clamps out-of-range indices. Find the first index whose
        # reference equals the previous one, then use the preceding frame count.
        last = None
        for index in range(1, 100_000):
            ref = self.reference(index)[1]
            if last is not None and np.array_equal(ref, last):
                return index
            last = ref
        raise RuntimeError("Could not infer motion length from ONNX reference outputs.")

    def reference(self, time_step: int) -> list[np.ndarray]:
        outputs = self.session.run(
            None,
            {
                "obs": np.zeros((1, self.observation_size), dtype=np.float32),
                "time_step": np.array([[time_step]], dtype=np.float32),
            },
        )
        return outputs

    def reset(self) -> None:
        if self.total_control_steps:
            self.reset_count += 1
        mujoco.mj_resetData(self.model, self.data)
        outputs = self.reference(0)
        joint_pos, body_pos, body_quat = outputs[1][0], outputs[3][0], outputs[4][0]
        self.data.qpos[:3] = body_pos[0]
        self.data.qpos[3:7] = body_quat[0]
        self.data.qpos[self.qpos_ids] = joint_pos
        self.data.qvel[:] = 0.0
        self.previous_action[:] = 0.0
        self.time_step = 0
        mujoco.mj_forward(self.model, self.data)

    def observation(self, outputs: list[np.ndarray]) -> np.ndarray:
        joint_ref, joint_vel_ref = outputs[1][0], outputs[2][0]
        body_pos, body_quat = outputs[3][0], outputs[4][0]
        robot_anchor_pos = self.data.xpos[self.torso_id]
        robot_anchor_quat = self.data.xquat[self.torso_id]
        robot_rotation = quat_matrix(robot_anchor_quat)
        motion_anchor_pos_b = robot_rotation.T @ (body_pos[self.anchor_body_index] - robot_anchor_pos)
        relative_quat = quat_multiply(quat_conjugate(robot_anchor_quat), body_quat[self.anchor_body_index])
        motion_anchor_ori_b = quat_matrix(relative_quat)[:, :2].reshape(-1)

        root_velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.pelvis_id, root_velocity, 1
        )
        terms = {
            "command": np.concatenate([joint_ref, joint_vel_ref]),
            "motion_anchor_pos_b": motion_anchor_pos_b,
            "motion_anchor_ori_b": motion_anchor_ori_b,
            "base_lin_vel": root_velocity[3:],
            "base_ang_vel": root_velocity[:3],
            "joint_pos": self.data.qpos[self.qpos_ids] - self.default_pos,
            "joint_vel": self.data.qvel[self.dof_ids],
            "actions": self.previous_action,
        }
        unknown = [name for name in self.observation_names if name not in terms]
        if unknown:
            raise ValueError(f"Unsupported policy observation terms: {unknown}")
        obs = np.concatenate([terms[name] for name in self.observation_names]).astype(np.float32)
        if obs.shape != (self.observation_size,) or not np.all(np.isfinite(obs)):
            raise RuntimeError(f"Invalid policy observation: shape={obs.shape}, finite={np.all(np.isfinite(obs))}")
        return obs

    def step(self) -> None:
        reference = self.reference(self.time_step)
        obs = self.observation(reference)
        outputs = self.session.run(
            None,
            {"obs": obs[None, :], "time_step": np.array([[self.time_step]], dtype=np.float32)},
        )
        action = outputs[0][0]
        target = self.default_pos + self.action_scale * action
        for _ in range(self.steps_per_control):
            torque = self.stiffness * (target - self.data.qpos[self.qpos_ids])
            torque -= self.damping * self.data.qvel[self.dof_ids]
            self.data.qfrc_applied[self.dof_ids] = np.clip(torque, -self.force_limits, self.force_limits)
            mujoco.mj_step(self.model, self.data)
        self.previous_action = action.astype(np.float32)
        self.time_step += 1
        self.total_control_steps += 1
        pelvis_height = self.data.xpos[self.pelvis_id, 2]
        if self.reset_on_fall and pelvis_height < self.fall_height:
            self.reset()
        elif self.time_step >= self.motion_length:
            self.reset()


class VideoRecorder:
    """Render simulation frames and encode them as H.264 MP4."""

    def __init__(
        self,
        simulation: Sim2Sim,
        path: Path,
        width: int,
        height: int,
        fps: float,
        camera_distance: float,
        camera_azimuth: float,
        camera_elevation: float,
    ) -> None:
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError("Video width and height must be positive even numbers.")
        control_fps = 1.0 / simulation.control_dt
        if fps <= 0 or fps > control_fps:
            raise ValueError(f"Video FPS must be in the range (0, {control_fps:g}].")

        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise RuntimeError("Video recording requires imageio and imageio-ffmpeg.") from exc

        self.simulation = simulation
        self.fps = fps
        self.next_frame_time = 0.0
        self.frame_count = 0
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.unlink(missing_ok=True)

        # MuJoCo's default offscreen framebuffer can be smaller than an HD video.
        simulation.model.vis.global_.offwidth = max(simulation.model.vis.global_.offwidth, width)
        simulation.model.vis.global_.offheight = max(simulation.model.vis.global_.offheight, height)
        self.renderer = mujoco.Renderer(simulation.model, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = camera_distance
        self.camera.azimuth = camera_azimuth
        self.camera.elevation = camera_elevation
        self.writer = imageio.get_writer(
            str(self.path),
            format="FFMPEG",
            mode="I",
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=None,
        )

    def capture_if_due(self) -> None:
        simulation_time = self.simulation.total_control_steps * self.simulation.control_dt
        if simulation_time + 0.5 * self.simulation.control_dt < self.next_frame_time:
            return
        self.camera.lookat[:] = self.simulation.data.xpos[self.simulation.pelvis_id]
        self.renderer.update_scene(self.simulation.data, camera=self.camera)
        self.writer.append_data(self.renderer.render())
        self.frame_count += 1
        self.next_frame_time += 1.0 / self.fps

    def close(self) -> None:
        self.writer.close()
        self.renderer.close()


def main() -> None:
    args = parse_args()
    simulation = Sim2Sim(
        args.model.resolve(),
        args.xml.resolve(),
        args.motion_length,
        reset_on_fall=args.reset_on_fall,
        fall_height=args.fall_height,
    )
    video_fps = args.video_fps or 1.0 / simulation.control_dt
    recorder = None
    if args.record_video:
        recorder = VideoRecorder(
            simulation,
            args.video_path,
            args.video_width,
            args.video_height,
            video_fps,
            args.camera_distance,
            args.camera_azimuth,
            args.camera_elevation,
        )

    def should_continue() -> bool:
        simulated_time = simulation.total_control_steps * simulation.control_dt
        return (
            (args.duration is None or simulated_time < args.duration)
            and (args.max_steps is None or simulation.total_control_steps < args.max_steps)
            and (not args.record_one_motion or simulation.total_control_steps < simulation.motion_length)
        )

    try:
        if recorder is not None:
            recorder.capture_if_due()
        if args.headless:
            while should_continue():
                simulation.step()
                if recorder is not None:
                    recorder.capture_if_due()
        else:
            import mujoco.viewer

            with mujoco.viewer.launch_passive(simulation.model, simulation.data) as viewer:
                while viewer.is_running() and should_continue():
                    step_started = time.monotonic()
                    simulation.step()
                    if recorder is not None:
                        recorder.capture_if_due()
                    viewer.sync()
                    remaining = simulation.control_dt - (time.monotonic() - step_started)
                    if remaining > 0:
                        time.sleep(remaining)
    finally:
        if recorder is not None:
            recorder.close()

    print(
        f"Completed {simulation.total_control_steps} control steps with {simulation.reset_count} resets; "
        f"frame={simulation.time_step}, base_z={simulation.data.qpos[2]:.3f}, "
        f"finite={np.all(np.isfinite(simulation.data.qpos))}"
    )
    if recorder is not None:
        print(f"Saved {recorder.frame_count} frames to {recorder.path}")


if __name__ == "__main__":
    main()
