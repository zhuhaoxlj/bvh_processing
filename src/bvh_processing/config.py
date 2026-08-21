from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    download_timeout_seconds: float = 30.0
    max_file_size_mb: int = 100
    minio_allowed_hosts: str = ""
    callback_allowed_hosts: str = ""
    callback_token: str = "action-callback-token"

    model_config = SettingsConfigDict(
        env_prefix="BVH_",
        env_file=".env",
        extra="ignore",
    )

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

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
