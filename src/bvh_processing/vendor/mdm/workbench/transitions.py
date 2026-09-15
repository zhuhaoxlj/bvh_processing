"""MDM inpainting of a gap between two converted HumanML3D motions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Callable

import numpy as np
import torch

from bvh_processing.vendor.mdm.workbench.conversion import (
    DEFAULT_RESOURCE_ROOT,
    HumanMLAssets,
    TARGET_FPS,
    load_assets,
)
from bvh_processing.vendor.mdm.data.humanml.scripts.motion_process import recover_from_ric
from bvh_processing.vendor.mdm.model.cfg_sampler import ClassifierFreeSampleModel

MAX_FRAMES = 196
CONTEXT_FRAMES = 20
MAX_TRANSITION_SECONDS = (MAX_FRAMES - CONTEXT_FRAMES * 2) / TARGET_FPS
MAX_TEXT_LENGTH = 500
DEFAULT_GUIDANCE_PARAM = 2.5
MAX_GUIDANCE_PARAM = 5.0
CHECKPOINT = DEFAULT_RESOURCE_ROOT / "checkpoint/model000600000.pt"
CHECKPOINT_SIZE = 232_011_477
CHECKPOINT_SHA256 = "195664bed72143e071acef4c97ac1aa67f8aa00f57fdc691248e98d715356d92"
TEXT_ENCODER_SIZE = 267_954_768
TEXT_ENCODER_SHA256 = "5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063"
Progress = Callable[[str, int, int], None]


@dataclass(frozen=True)
class PreparedTransition:
    motion_a: np.ndarray
    motion_b: np.ndarray
    condition: np.ndarray
    keep_mask: np.ndarray
    left_frames: int
    right_frames: int
    gap_frames: int
    requested_seconds: float

    @property
    def window_frames(self) -> int:
        return self.left_frames + self.gap_frames + self.right_frames


@dataclass(frozen=True)
class TransitionResult:
    features: np.ndarray
    normalized: np.ndarray
    joints: np.ndarray
    condition: np.ndarray
    keep_mask: np.ndarray
    sampled_window: np.ndarray
    metadata: dict


def transition_frames(seconds: float) -> int:
    if not math.isfinite(seconds) or not 1 / TARGET_FPS <= seconds <= MAX_TRANSITION_SECONDS:
        raise ValueError(f"过渡时长必须为 0.05–{MAX_TRANSITION_SECONDS:g} 秒")
    return int(math.floor(seconds * TARGET_FPS + 0.5))


def prepare_transition(
    motion_a: np.ndarray, motion_b: np.ndarray, seconds: float, assets: HumanMLAssets
) -> PreparedTransition:
    gap = transition_frames(seconds)
    for label, motion in (("A", motion_a), ("B", motion_b)):
        if motion.ndim != 2 or motion.shape[1] != 263 or len(motion) < 2:
            raise ValueError(f"动作 {label} 需要至少 2 帧有效的 263 维 HumanML3D 特征")
        if not np.isfinite(motion).all():
            raise ValueError(f"动作 {label} 包含无效数值")
    left, right = min(CONTEXT_FRAMES, len(motion_a)), min(CONTEXT_FRAMES, len(motion_b))
    condition = np.zeros((1, 263, 1, MAX_FRAMES), dtype=np.float32)
    condition[0, :, 0, :left] = ((motion_a[-left:] - assets.mean) / assets.std).T
    condition[0, :, 0, left + gap:left + gap + right] = ((motion_b[:right] - assets.mean) / assets.std).T
    keep = np.ones_like(condition, dtype=bool)
    keep[..., left:left + gap] = False
    return PreparedTransition(motion_a, motion_b, condition, keep, left, right, gap, seconds)


def assemble_transition(
    prepared: PreparedTransition, sample: np.ndarray, assets: HumanMLAssets, model_info: dict, seed: int
) -> TransitionResult:
    if sample.shape != prepared.condition.shape or not np.isfinite(sample).all():
        raise ValueError("MDM 返回了无效的动作数据")
    context_error = float(np.max(np.abs(sample[prepared.keep_mask] - prepared.condition[prepared.keep_mask])))
    if context_error > 1e-5:
        raise ValueError("MDM 未保持指定的 A/B 上下文，已拒绝输出")
    left, gap = prepared.left_frames, prepared.gap_frames
    generated = sample[0, :, 0, left:left + gap].T * assets.std + assets.mean
    # HumanML3D stores root *velocities*, not absolute root poses. Integrate the
    # entire A + gap + B sequence once. B follows the generated endpoint with a
    # single rigid yaw/translation; resetting B's root would cause a teleport.
    features = np.concatenate((prepared.motion_a, generated, prepared.motion_b)).astype(np.float32)
    normalized = ((features - assets.mean) / assets.std).astype(np.float32)
    with torch.inference_mode():
        joints = recover_from_ric(torch.from_numpy(features), 22).numpy()
    if not np.isfinite(joints).all() or not np.isfinite(normalized).all():
        raise ValueError("MDM 结果恢复后包含无效坐标")
    start, end = len(prepared.motion_a), len(prepared.motion_a) + gap
    seam_steps = [float(np.linalg.norm(joints[index] - joints[index - 1], axis=-1).max()) for index in (start, end)]
    notes = ["A/B 保持转换后的动作特征；B 的整体位置与朝向随过渡末端衔接。"]
    if max(seam_steps) > 0.25:
        notes.append("接缝处存在较大的单帧位移，建议延长过渡或重新生成后比较。")
    floor_min = float(joints[start:end, [7, 8, 10, 11], 1].min())
    if floor_min < -0.05:
        notes.append("部分过渡帧的脚低于地面，请在预览中检查。")
    metadata = {
        "method": "MDM HumanML3D inpainting",
        "neural_model_used": True,
        "model": model_info,
        "seed": seed,
        "target_fps": TARGET_FPS,
        "requested_transition_seconds": prepared.requested_seconds,
        "transition_seconds": gap / TARGET_FPS,
        "transition_frames": gap,
        "transition_range": [start, end],
        "range_convention": "zero-based, end-exclusive",
        "motion_a_frames": start,
        "motion_b_frames": len(prepared.motion_b),
        "output_frames": len(features),
        "output_duration_seconds": len(features) / TARGET_FPS,
        "feature_shape": list(features.shape),
        "mdm_input_shape": list(prepared.condition.shape),
        "context_frames": {"a": left, "b": prepared.right_frames},
        "window_frames": prepared.window_frames,
        "preserved_feature_context_max_error": context_error,
        "seam_max_joint_step_m": {"a_to_transition": seam_steps[0], "transition_to_b": seam_steps[1]},
        "transition_min_foot_height_m": floor_min,
        "b_alignment": "single rigid yaw and XZ translation from integrated generated root velocities",
        "reference_files_sha256": assets.hashes,
        "all_finite": True,
        "notes": notes,
    }
    return TransitionResult(features, normalized, joints, prepared.condition, prepared.keep_mask, sample, metadata)


class MDMEngine:
    """Load the local checkpoint once; the job runner serializes GPU access."""

    def __init__(
        self,
        checkpoint: Path = CHECKPOINT,
        data_root: Path = DEFAULT_RESOURCE_ROOT / "humanml",
        text_encoder: Path = DEFAULT_RESOURCE_ROOT / "text_encoder",
    ):
        self.checkpoint = Path(checkpoint)
        self.data_root = Path(data_root)
        self.text_encoder = Path(text_encoder)
        self.model = None
        self.diffusion = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_info: dict = {}

    def status(self) -> dict:
        required = [
            self.checkpoint,
            self.checkpoint.with_name("args.json"),
            self.data_root / "Mean.npy",
            self.data_root / "Std.npy",
            self.data_root / "new_joints/000021.npy",
            self.text_encoder / "config.json",
            self.text_encoder / "model.safetensors",
            self.text_encoder / "tokenizer.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        expected_sizes = {
            self.checkpoint: CHECKPOINT_SIZE,
            self.text_encoder / "model.safetensors": TEXT_ENCODER_SIZE,
        }
        invalid = [
            str(path)
            for path, size in expected_sizes.items()
            if path.is_file() and path.stat().st_size != size
        ]
        return {"available": not missing and not invalid, "loaded": self.model is not None,
                "device": str(self.device), "missing": missing, "invalid": invalid,
                "fps": TARGET_FPS, "max_transition_seconds": MAX_TRANSITION_SECONDS,
                "context_frames": CONTEXT_FRAMES, "checkpoint": self.checkpoint.name}

    def _load(self) -> None:
        if self.model is not None:
            return
        status = self.status()
        if status["missing"]:
            raise ValueError("缺少 MDM 资源：" + "、".join(status["missing"]))
        if status["invalid"]:
            raise ValueError(
                "MDM 模型资源大小不正确，请执行 git lfs pull："
                + "、".join(status["invalid"])
            )
        hashes = {}
        for path, expected in (
            (self.checkpoint, CHECKPOINT_SHA256),
            (self.text_encoder / "model.safetensors", TEXT_ENCODER_SHA256),
        ):
            with path.open("rb") as file:
                digest = hashlib.file_digest(file, "sha256").hexdigest()
            if digest != expected:
                raise ValueError(f"MDM 模型资源 SHA-256 校验失败：{path}")
            hashes[path] = digest
        # All weights, including DistilBERT, were downloaded for this workspace.
        # A generation request must never silently start a network download.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from bvh_processing.vendor.mdm.utils.model_util import create_model_and_diffusion, load_saved_model

        settings = json.loads(self.checkpoint.with_name("args.json").read_text())
        if settings["dataset"] != "humanml" or settings.get("pred_len", 0) or settings.get("context_len", 0):
            raise ValueError("当前工作台需要 HumanML3D MDM 补间检查点")
        args = SimpleNamespace(**settings)
        args.text_encoder_path = str(self.text_encoder)
        torch.set_num_threads(min(4, torch.get_num_threads()))
        model, diffusion = create_model_and_diffusion(args, SimpleNamespace(dataset=SimpleNamespace(num_actions=1)))
        load_saved_model(model, str(self.checkpoint), use_avg=args.use_ema)
        model.eval()
        model.to(self.device)
        self.model_info = {"checkpoint": self.checkpoint.name,
                           "checkpoint_sha256": hashes[self.checkpoint], "diffusion_steps": diffusion.num_timesteps,
                           "use_ema": args.use_ema, "device": str(self.device)}
        self.model, self.diffusion = model, diffusion

    def generate(
        self, motion_a: np.ndarray, motion_b: np.ndarray, seconds: float, seed: int, progress: Progress,
        *, text_condition: str = "", guidance_param: float = DEFAULT_GUIDANCE_PARAM,
    ) -> TransitionResult:
        text_condition = text_condition.strip()
        if len(text_condition) > MAX_TEXT_LENGTH:
            raise ValueError(f"动作描述不能超过 {MAX_TEXT_LENGTH} 个字符")
        if not 0 <= guidance_param <= MAX_GUIDANCE_PARAM:
            raise ValueError(f"文本引导强度必须为 0–{MAX_GUIDANCE_PARAM:g}")
        guidance = guidance_param if text_condition else 0.0
        guided = guidance > 0
        assets = load_assets(self.data_root)
        prepared = prepare_transition(motion_a, motion_b, seconds, assets)
        progress("loading", 0, 0)
        self._load()
        condition = torch.from_numpy(prepared.condition).to(self.device)
        keep = torch.from_numpy(prepared.keep_mask).to(self.device)
        started = time.perf_counter()
        devices = [self.device.index] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.manual_seed(seed)
            # Empty text or zero strength uses the same unconditional sampling path.
            prompt = text_condition if guided else ""
            sampling_model = ClassifierFreeSampleModel(self.model) if guided else self.model
            kwargs = {"y": {
                "text": [prompt], "text_embed": self.model.encode_text([prompt]), "uncond": not guided,
                "lengths": torch.tensor([prepared.window_frames], device=self.device),
                "mask": (torch.arange(MAX_FRAMES, device=self.device) < prepared.window_frames)[None, None, None],
                "inpainted_motion": condition, "inpainting_mask": keep,
            }}
            if guided:
                kwargs["y"]["scale"] = torch.tensor([guidance], device=self.device)
            # The upstream inpainting clamp is applied at every denoising step.
            for step, output in enumerate(self.diffusion.p_sample_loop_progressive(
                sampling_model, condition.shape, clip_denoised=False, model_kwargs=kwargs, progress=False
            ), start=1):
                progress("sampling", step, self.diffusion.num_timesteps)
            sample = output["sample"].cpu().numpy()
        info = {**self.model_info, "sampling_seconds": round(time.perf_counter() - started, 3),
                "text_condition": text_condition, "guidance_param": guidance,
                "sampling": ("text-guided CFG" if guided else "unconditioned") + " p_sample_loop_progressive"}
        progress("saving", self.diffusion.num_timesteps, self.diffusion.num_timesteps)
        return assemble_transition(prepared, sample, assets, info, seed)
