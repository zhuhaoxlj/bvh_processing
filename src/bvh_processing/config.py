from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_PACKAGE_ROOT = Path(__file__).resolve().parent
_EMBEDDED_WBT_ROOT = _PACKAGE_ROOT / "vendor" / "whole_body_tracking"


class Settings(BaseSettings):
    download_timeout_seconds: float = 30.0
    max_file_size_mb: int = 100
    minio_allowed_hosts: str = ""
    callback_allowed_hosts: str = ""
    callback_token: str = "action-callback-token"
    train_timeout_seconds: float = 604800.0
    render_timeout_seconds: float = 1800.0
    train_max_concurrency: int = 1
    train_workspace_root: str = "/tmp/bvh-processing-training"
    wbt_project_root: str = str(_EMBEDDED_WBT_ROOT)
    wbt_python: str = (
        "/home/mark/Project/01-RL/IsaacLab/env_isaaclab/bin/python"
    )
    wbt_task: str = "Tracking-Flat-G1-Wo-State-Estimation-v0"
    wbt_mujoco_xml: str = (
        "source/whole_body_tracking/whole_body_tracking/assets/"
        "unitree_description/mjcf/g1.xml"
    )
    npz_max_uncompressed_mb: int = 2048
    linkseg_python: str = "/home/mark/Project/02-Python/LinkSeg/.venv/bin/python"
    audio_analysis_timeout_seconds: float = 1800.0

    model_config = SettingsConfigDict(
        env_prefix="BVH_",
        env_file=".env",
        extra="ignore",
    )

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def npz_max_uncompressed_bytes(self) -> int:
        return self.npz_max_uncompressed_mb * 1024 * 1024

    @property
    def allowed_hosts(self) -> frozenset[str]:
        return frozenset(
            host.strip().lower()
            for host in self.minio_allowed_hosts.split(",")
            if host.strip()
        )

    @property
    def allowed_callback_hosts(self) -> frozenset[str]:
        return frozenset(
            host.strip().lower()
            for host in self.callback_allowed_hosts.split(",")
            if host.strip()
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
