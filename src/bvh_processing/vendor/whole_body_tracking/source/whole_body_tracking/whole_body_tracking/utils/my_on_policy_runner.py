import copy
import os
import shutil
from types import SimpleNamespace

import torch
from rsl_rl.env import VecEnv
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


def _convert_split_policy_state(actor_state: dict, critic_state: dict) -> dict:
    """Convert split actor/critic checkpoints to the combined ActorCritic layout."""

    converted = {}
    for name, value in actor_state.items():
        if name.startswith("mlp."):
            converted[f"actor.{name.removeprefix('mlp.')}"] = value
        elif name.startswith("obs_normalizer."):
            converted[f"actor_obs_normalizer.{name.removeprefix('obs_normalizer.')}"] = value
        elif name == "distribution.std_param":
            converted["std"] = value
        else:
            raise KeyError(f"Unsupported actor checkpoint tensor: {name}")

    for name, value in critic_state.items():
        if name.startswith("mlp."):
            converted[f"critic.{name.removeprefix('mlp.')}"] = value
        elif name.startswith("obs_normalizer."):
            converted[f"critic_obs_normalizer.{name.removeprefix('obs_normalizer.')}"] = value
        else:
            raise KeyError(f"Unsupported critic checkpoint tensor: {name}")
    return converted


def _restore_combined_policy_normalizers(policy, actor_state: dict, critic_state: dict) -> None:
    """Enable checkpoint normalization when the current policy config disabled it."""

    policy_state_names = policy.state_dict().keys()
    normalizer_specs = (
        ("actor", actor_state, "actor_obs_normalization", "actor_obs_normalizer"),
        ("critic", critic_state, "critic_obs_normalization", "critic_obs_normalizer"),
    )
    device = next(policy.parameters()).device
    for label, split_state, enabled_attribute, module_attribute in normalizer_specs:
        mean = split_state.get("obs_normalizer._mean")
        if mean is None:
            continue
        state_prefix = f"{module_attribute}."
        if not any(name.startswith(state_prefix) for name in policy_state_names):
            setattr(policy, module_attribute, EmpiricalNormalization(mean.shape[-1]).to(device))
            setattr(policy, enabled_attribute, True)
            print(f"[INFO]: Restored {label} observation normalization from checkpoint.")


def _convert_split_optimizer_state(optimizer_state: dict) -> dict:
    """Validate and copy optimizer state shared by the two checkpoint layouts."""

    converted = copy.deepcopy(optimizer_state)
    if len(converted["param_groups"]) != 1:
        raise ValueError("Split checkpoint conversion only supports a single optimizer parameter group.")
    return converted


def _set_next_learning_iteration(runner: OnPolicyRunner, checkpoint_iteration: int) -> None:
    """Continue after a saved iteration instead of training and saving it again."""

    runner.loaded_checkpoint_iteration = int(checkpoint_iteration)
    runner.current_learning_iteration = runner.loaded_checkpoint_iteration + 1


def get_onnx_export_names(checkpoint_path: str) -> tuple[str, str, str]:
    """Return the export directory, versioned filename, and latest filename."""

    export_directory = os.path.dirname(checkpoint_path)
    checkpoint_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    run_directory_name = os.path.basename(os.path.normpath(export_directory))
    return export_directory, f"{checkpoint_stem}.onnx", f"{run_directory_name}.onnx"


def update_latest_onnx(export_directory: str, versioned_filename: str, latest_filename: str) -> None:
    """Update the legacy run-named ONNX file to the newest checkpoint export."""

    if versioned_filename == latest_filename:
        return
    shutil.copy2(
        os.path.join(export_directory, versioned_filename),
        os.path.join(export_directory, latest_filename),
    )


def get_export_policy(runner: OnPolicyRunner):
    """Return an ONNX-compatible policy for legacy and RSL-RL 5 runners."""

    if hasattr(runner.alg, "policy"):
        return runner.alg.policy, getattr(runner, "obs_normalizer", None)

    actor = runner.alg.actor
    exportable_actor = actor.as_onnx(verbose=False)
    export_policy = SimpleNamespace(
        actor=exportable_actor,
        is_recurrent=exportable_actor.is_recurrent,
    )
    # RSL-RL 5 embeds normalization inside actor.as_onnx().
    return export_policy, None


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        export_directory, versioned_filename, latest_filename = get_onnx_export_names(path)
        export_policy, export_normalizer = get_export_policy(self)
        export_policy_as_onnx(
            export_policy,
            normalizer=export_normalizer,
            path=export_directory,
            filename=versioned_filename,
        )
        # Attach minimal metadata (no wandb run path)
        try:
            attach_onnx_metadata(
                self.env.unwrapped,
                "none",
                path=export_directory,
                filename=versioned_filename,
            )
        except Exception:
            pass
        update_latest_onnx(export_directory, versioned_filename, latest_filename)


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None):
        """Load combined RSL-RL checkpoints and newer split actor/critic checkpoints."""

        checkpoint = torch.load(path, weights_only=False, map_location=map_location)
        if "model_state_dict" in checkpoint:
            infos = super().load(path, map_location=map_location)
            _set_next_learning_iteration(self, checkpoint["iter"])
            print(
                f"[INFO]: Loaded combined checkpoint at iteration {checkpoint['iter']}; "
                f"training will continue at iteration {self.current_learning_iteration}."
            )
            return infos
        if "actor_state_dict" not in checkpoint or "critic_state_dict" not in checkpoint:
            raise KeyError(
                "Unsupported checkpoint schema: expected 'model_state_dict' or both "
                "'actor_state_dict' and 'critic_state_dict'."
            )

        if hasattr(self.alg, "actor") and hasattr(self.alg, "critic"):
            self.alg.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.alg.critic.load_state_dict(checkpoint["critic_state_dict"])
            optimizer_state = checkpoint["optimizer_state_dict"]
        else:
            _restore_combined_policy_normalizers(
                self.alg.policy, checkpoint["actor_state_dict"], checkpoint["critic_state_dict"]
            )
            policy_state = _convert_split_policy_state(
                checkpoint["actor_state_dict"], checkpoint["critic_state_dict"]
            )
            self.alg.policy.load_state_dict(policy_state)
            optimizer_state = _convert_split_optimizer_state(checkpoint["optimizer_state_dict"])

        if load_optimizer:
            self.alg.optimizer.load_state_dict(optimizer_state)
        _set_next_learning_iteration(self, checkpoint["iter"])
        print(
            f"[INFO]: Loaded split actor/critic checkpoint at iteration {checkpoint['iter']}; "
            f"training will continue at iteration {self.current_learning_iteration}."
        )
        return checkpoint.get("infos")

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        export_directory, versioned_filename, latest_filename = get_onnx_export_names(path)
        export_policy, export_normalizer = get_export_policy(self)
        export_motion_policy_as_onnx(
            self.env.unwrapped,
            export_policy,
            normalizer=export_normalizer,
            path=export_directory,
            filename=versioned_filename,
        )
        # Attach minimal metadata (no wandb run path)
        try:
            attach_onnx_metadata(
                self.env.unwrapped,
                "none",
                path=export_directory,
                filename=versioned_filename,
            )
        except Exception:
            pass
        update_latest_onnx(export_directory, versioned_filename, latest_filename)
        # Do not link/use wandb artifacts
        self.registry_name = None
