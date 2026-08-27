# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib.metadata as importlib_metadata
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--save_interval", type=int, default=None, help="Iterations between model checkpoints.")
parser.add_argument("--num_steps_per_env", type=int, default=None, help="Rollout steps collected per environment.")
parser.add_argument("--num_mini_batches", type=int, default=None, help="PPO mini-batches per learning epoch.")
parser.add_argument("--num_learning_epochs", type=int, default=None, help="PPO passes over each rollout batch.")
parser.add_argument("--learning_rate", type=float, default=None, help="Initial PPO learning rate.")
parser.add_argument("--desired_kl", type=float, default=None, help="Target KL divergence for adaptive scheduling.")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs.")
parser.add_argument(
    "--physical_gpu_ids",
    type=str,
    default=None,
    help="Comma-separated physical GPU IDs selected by an external distributed launcher.",
)
# Added: support local motion file and make registry optional
parser.add_argument("--motion_file", type=str, default=None, help="Path to local motion npz (overrides registry).")
parser.add_argument("--registry_name", type=str, default=None, help="The name of the wand registry (if not using --motion_file).")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

try:
    from isaaclab.utils.io import dump_pickle
except ImportError:
    # Isaac Lab 2.3 removed dump_pickle. YAML remains the portable config
    # snapshot; pickle output is optional and only retained for 2.1 users.
    dump_pickle = None
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

try:
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
except ImportError:
    handle_deprecated_rsl_rl_cfg = None
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    if args_cli.save_interval is not None:
        agent_cfg.save_interval = args_cli.save_interval
    if args_cli.num_steps_per_env is not None:
        agent_cfg.num_steps_per_env = args_cli.num_steps_per_env
    if args_cli.num_mini_batches is not None:
        agent_cfg.algorithm.num_mini_batches = args_cli.num_mini_batches
    if args_cli.num_learning_epochs is not None:
        agent_cfg.algorithm.num_learning_epochs = args_cli.num_learning_epochs
    if args_cli.learning_rate is not None:
        agent_cfg.algorithm.learning_rate = args_cli.learning_rate
    if args_cli.desired_kl is not None:
        agent_cfg.algorithm.desired_kl = args_cli.desired_kl

    # Isaac Lab 2.3 bundles RSL-RL 5 and provides the official migration
    # utility for legacy policy configs used by this Isaac Lab 2.1 project.
    if handle_deprecated_rsl_rl_cfg is not None:
        installed_rsl_rl_version = importlib_metadata.version("rsl-rl-lib")
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rsl_rl_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.device is not None:
        # Keep the simulator and RSL-RL networks on the same selected GPU.
        agent_cfg.device = args_cli.device

    # Match Isaac Lab's distributed RSL-RL launcher: each torchrun rank owns one
    # simulator/GPU, while RSL-RL synchronizes PPO gradients across ranks.
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError("Distributed training requires a CUDA device, not CPU.")
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # load the motion file from the wandb registry
    # Prefer local motion file if provided; otherwise, fall back to WandB registry
    if args_cli.motion_file is not None:
        # use the local npz directly
        registry_name = "none"
        env_cfg.commands.motion.motion_file = os.path.abspath(args_cli.motion_file)
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
        env_cfg.commands.motion.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    clip_actions = getattr(agent_cfg, "clip_actions", None)
    env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, registry_name=registry_name
    )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
        # Preserve an exact ONNX/PT snapshot before the first resumed PPO update.  The
        # learning loop now starts at N+1, so this cannot collide with model_N.
        loaded_iteration = runner.loaded_checkpoint_iteration
        preserved_checkpoint_path = os.path.join(log_dir, f"model_{loaded_iteration}.pt")
        runner.save(preserved_checkpoint_path)
        print(f"[INFO]: Preserved loaded checkpoint at: {preserved_checkpoint_path}")

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    if dump_pickle is not None:
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
