"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python replay_motion.py --motion_file source/whole_body_tracking/whole_body_tracking/assets/g1/motions/lafan_walk_short.npz
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import threading

import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--registry_name", type=str, required=False, help="The name of the wand registry.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to local motion npz (overrides registry)")
parser.add_argument("--output_video", type=str, default=None, help="Optional path for a one-cycle MP4 preview.")
parser.add_argument("--video_width", type=int, default=960, help="Per-camera preview width in pixels.")
parser.add_argument("--video_height", type=int, default=720, help="Preview video height in pixels.")
parser.add_argument(
    "--camera_layout",
    choices=("oblique", "front_rear"),
    default="oblique",
    help="Camera layout: legacy oblique view or synchronized front/rear split screen.",
)
parser.add_argument(
    "--camera_focal_length",
    type=float,
    choices=(18.0, 24.0, 35.0),
    default=18.0,
    help="Camera focal length in millimeters. Smaller values provide a wider field of view.",
)
parser.add_argument(
    "--shutdown_timeout",
    type=float,
    default=30.0,
    help="Force-exit after this many seconds if Isaac Sim hangs during shutdown. Set to 0 to disable.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher_kwargs = {}
if args_cli.device.startswith("cuda:"):
    selected_gpu_index = int(args_cli.device.removeprefix("cuda:"))
    app_launcher_kwargs = {
        "active_gpu": selected_gpu_index,
        "physics_gpu": selected_gpu_index,
        "multi_gpu": False,
    }
app_launcher = AppLauncher(args_cli, **app_launcher_kwargs)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.mdp import MotionLoader


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    camera: CameraCfg | None = None
    front_camera: CameraCfg | None = None
    rear_camera: CameraCfg | None = None


def camera_positions(root_state: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed front/rear camera poses from the motion's initial root state."""

    root_position = root_state[:3].cpu().numpy()
    root_quaternion = root_state[3:7]
    forward_local = torch.tensor([1.0, 0.0, 0.0], device=root_state.device, dtype=root_state.dtype)
    quaternion_vector = root_quaternion[1:]
    quaternion_scalar = root_quaternion[0]
    forward_world = (
        2.0 * torch.dot(quaternion_vector, forward_local) * quaternion_vector
        + (quaternion_scalar * quaternion_scalar - torch.dot(quaternion_vector, quaternion_vector)) * forward_local
        + 2.0 * quaternion_scalar * torch.linalg.cross(quaternion_vector, forward_local)
    )
    forward_xy = forward_world[:2].cpu().numpy()
    forward_xy /= max(float(np.linalg.norm(forward_xy)), 1e-6)
    camera_offset = np.array([forward_xy[0] * 3.2, forward_xy[1] * 3.2, 1.05])
    target = root_position + np.array([0.0, 0.0, 0.35])
    return root_position + camera_offset, root_position - camera_offset + np.array([0.0, 0.0, 2.1]), target


def label_split_screen(front_frame: np.ndarray, rear_frame: np.ndarray) -> np.ndarray:
    """Combine synchronized views and burn unambiguous camera labels into the frame."""

    from PIL import Image, ImageDraw, ImageFont

    combined = np.concatenate((front_frame, rear_frame), axis=1)
    image = Image.fromarray(combined)
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default(size=max(16, front_frame.shape[0] // 32))
    padding_x = 18
    padding_y = 10
    label_y = 20
    labels = (
        ("FIXED FRONT / CAMERA 01", 20, (216, 255, 69, 235)),
        ("FIXED REAR / CAMERA 02", front_frame.shape[1] + 20, (255, 106, 50, 235)),
    )
    for label, label_x, accent in labels:
        bounds = draw.textbbox((0, 0), label, font=font)
        box_width = bounds[2] - bounds[0] + padding_x * 2
        box_height = bounds[3] - bounds[1] + padding_y * 2
        draw.rectangle((label_x, label_y, label_x + box_width, label_y + box_height), fill=(8, 10, 9, 205))
        draw.rectangle((label_x, label_y, label_x + 5, label_y + box_height), fill=accent)
        draw.text((label_x + padding_x, label_y + padding_y), label, fill=(239, 240, 232, 255), font=font)
    divider_x = front_frame.shape[1]
    draw.rectangle((divider_x - 2, 0, divider_x + 2, combined.shape[0]), fill=(239, 240, 232, 150))
    return np.asarray(image)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # Determine motion file source: prefer local --motion_file, otherwise use WandB registry
    if args_cli.motion_file is not None:
        motion_file = args_cli.motion_file
    else:
        if args_cli.registry_name is None:
            raise ValueError("Please provide either --motion_file or --registry_name")
        registry_name = args_cli.registry_name
        if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
            registry_name += ":latest"
        import pathlib

        import wandb

        api = wandb.Api()
        artifact = api.artifact(registry_name)
        motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")

    motion = MotionLoader(
        motion_file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)
    camera: Camera | None = None
    front_camera: Camera | None = None
    rear_camera: Camera | None = None
    if args_cli.output_video:
        if args_cli.camera_layout == "front_rear":
            front_camera = scene["front_camera"]
            rear_camera = scene["rear_camera"]
        else:
            camera = scene["camera"]
    video_writer = None
    if args_cli.output_video:
        import imageio.v2 as imageio

        output_video_path = os.path.abspath(args_cli.output_video)
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        motion_fps = float(np.asarray(motion.fps).reshape(-1)[0])
        video_writer = imageio.get_writer(
            output_video_path,
            fps=motion_fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )

    # Simulation loop
    rendered_frame_count = 0
    fixed_dual_camera_pose: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    while simulation_app.is_running():

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)

        pos_lookat = root_states[0, :3].cpu().numpy()
        camera_eye = pos_lookat + np.array([2.4, 2.4, 1.0])
        camera_target = pos_lookat + np.array([0.0, 0.0, 0.15])

        if front_camera is not None and rear_camera is not None and video_writer is not None:
            if fixed_dual_camera_pose is None:
                fixed_dual_camera_pose = camera_positions(root_states[0])
                front_eye, rear_eye, dual_camera_target = fixed_dual_camera_pose
                sim.set_camera_view(front_eye, dual_camera_target)
                front_camera.set_world_poses_from_view(
                    torch.tensor(front_eye[None, :], device=sim.device, dtype=torch.float32),
                    torch.tensor(dual_camera_target[None, :], device=sim.device, dtype=torch.float32),
                )
                rear_camera.set_world_poses_from_view(
                    torch.tensor(rear_eye[None, :], device=sim.device, dtype=torch.float32),
                    torch.tensor(dual_camera_target[None, :], device=sim.device, dtype=torch.float32),
                )
            front_camera.update(sim_dt, force_recompute=True)
            rear_camera.update(sim_dt, force_recompute=True)
            front_frame = front_camera.data.output["rgb"][0, ..., :3].cpu().numpy()
            rear_frame = rear_camera.data.output["rgb"][0, ..., :3].cpu().numpy()
            video_writer.append_data(label_split_screen(front_frame, rear_frame))
        elif camera is not None and video_writer is not None:
            sim.set_camera_view(camera_eye, camera_target)
            camera.set_world_poses_from_view(
                torch.tensor(camera_eye[None, :], device=sim.device, dtype=torch.float32),
                torch.tensor(camera_target[None, :], device=sim.device, dtype=torch.float32),
            )
            camera.update(sim_dt, force_recompute=True)
            rgb_frame = camera.data.output["rgb"][0, ..., :3].cpu().numpy()
            video_writer.append_data(rgb_frame)

        rendered_frame_count += 1
        if args_cli.output_video and rendered_frame_count >= motion.time_step_total:
            video_writer.close()
            print(f"[INFO]: Preview video saved to: {args_cli.output_video}")
            return

        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        time_steps[reset_ids] = 0


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    if args_cli.output_video:
        camera_config = {
            "update_period": 0.0,
            "height": args_cli.video_height,
            "width": args_cli.video_width,
            "data_types": ["rgb"],
            "spawn": sim_utils.PinholeCameraCfg(
                focal_length=args_cli.camera_focal_length,
                focus_distance=4.0,
                horizontal_aperture=24.0,
                clipping_range=(0.1, 100.0),
            ),
        }
        if args_cli.camera_layout == "front_rear":
            scene_cfg.front_camera = CameraCfg(prim_path="/World/FrontPreviewCamera", **camera_config)
            scene_cfg.rear_camera = CameraCfg(prim_path="/World/RearPreviewCamera", **camera_config)
        else:
            scene_cfg.camera = CameraCfg(prim_path="/World/PreviewCamera", **camera_config)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    shutdown_complete = threading.Event()

    def force_exit_if_shutdown_hangs():
        if not shutdown_complete.wait(args_cli.shutdown_timeout):
            print(
                f"[WARN]: Isaac Sim shutdown exceeded {args_cli.shutdown_timeout:g}s; forcing a clean process exit.",
                file=sys.stderr,
                flush=True,
            )
            os._exit(0)

    if args_cli.shutdown_timeout > 0:
        threading.Thread(target=force_exit_if_shutdown_hangs, daemon=True).start()
    simulation_app.close()
    shutdown_complete.set()
