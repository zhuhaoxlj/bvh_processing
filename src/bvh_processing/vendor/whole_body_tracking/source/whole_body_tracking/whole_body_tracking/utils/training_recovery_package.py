"""Versioned, portable training recovery package protocol."""

from __future__ import annotations

import hashlib
import json
import re
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

PACKAGE_FORMAT = "motion-inspector-training-recovery"
PACKAGE_VERSION = 1
MANIFEST_MEMBER = "manifest.json"
MOTION_MEMBER = "motion/motion.npz"
ENV_MEMBER = "run/params/env.yaml"
AGENT_MEMBER = "run/params/agent.yaml"
CHECKPOINT_PATTERN = re.compile(r"^run/model_(\d+)\.pt$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_PACKAGE_BYTES = 1024 * 1024 * 1024
MAX_MEMBER_BYTES = 768 * 1024 * 1024
MAX_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_MANIFEST_BYTES = 256 * 1024
MAX_MEMBERS = 5
_COPY_CHUNK_BYTES = 1024 * 1024


class RecoveryPackageError(ValueError):
    """Raised when a recovery package violates the protocol."""


@dataclass(frozen=True)
class ExtractedRecoveryPackage:
    manifest: dict[str, Any]
    motion_path: Path
    checkpoint_path: Path
    environment_config_path: Path
    agent_config_path: Path

    @property
    def checkpoint_iteration(self) -> int:
        return int(self.manifest["checkpoint"]["iteration"])

    @property
    def original_motion_name(self) -> str:
        return str(self.manifest["motion"]["original_name"])

    @property
    def source_run_directory(self) -> str:
        return str(self.manifest["source"]["run_directory"])

    @property
    def training_config(self) -> dict[str, Any]:
        return dict(self.manifest.get("training_config") or {})


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(_COPY_CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _file_record(path: Path) -> dict[str, int | str]:
    digest, size = _sha256_and_size(path)
    return {"size": size, "sha256": digest}


def build_recovery_package(
    destination: Path,
    *,
    motion_path: Path,
    checkpoint_path: Path,
    environment_config_path: Path,
    agent_config_path: Path,
    source_job_id: str,
    source_run_directory: str,
    original_motion_name: str,
    training_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a minimal recovery ZIP and return its manifest."""

    checkpoint_match = re.fullmatch(r"model_(\d+)\.pt", checkpoint_path.name)
    if not checkpoint_match:
        raise RecoveryPackageError("Checkpoint 文件名必须是 model_<iteration>.pt。")
    required_paths = (motion_path, checkpoint_path, environment_config_path, agent_config_path)
    if not all(path.is_file() and not path.is_symlink() for path in required_paths):
        raise RecoveryPackageError("训练恢复包所需文件不完整。")

    checkpoint_member = f"run/{checkpoint_path.name}"
    archive_files = {
        MOTION_MEMBER: motion_path,
        checkpoint_member: checkpoint_path,
        ENV_MEMBER: environment_config_path,
        AGENT_MEMBER: agent_config_path,
    }
    manifest: dict[str, Any] = {
        "format": PACKAGE_FORMAT,
        "version": PACKAGE_VERSION,
        "source": {
            "job_id": source_job_id,
            "run_directory": source_run_directory,
        },
        "motion": {"original_name": Path(original_motion_name).name},
        "checkpoint": {
            "iteration": int(checkpoint_match.group(1)),
            "member": checkpoint_member,
        },
        "training_config": training_config or {},
        "files": {member: _file_record(path) for member, path in archive_files.items()},
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        archive.writestr(
            MANIFEST_MEMBER,
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        )
        for member, path in archive_files.items():
            archive.write(path, member)
    return manifest


def _validate_member_name(name: str) -> None:
    path = PurePosixPath(name)
    if not name or "\\" in name or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RecoveryPackageError(f"ZIP 包含不安全路径：{name!r}。")


def _validate_zip_info(info: zipfile.ZipInfo) -> None:
    _validate_member_name(info.filename)
    if info.is_dir():
        raise RecoveryPackageError(f"ZIP 不允许目录条目：{info.filename}。")
    unix_mode = info.external_attr >> 16
    file_type = stat.S_IFMT(unix_mode)
    if file_type not in {0, stat.S_IFREG}:
        raise RecoveryPackageError(f"ZIP 不允许符号链接或特殊文件：{info.filename}。")
    if info.file_size < 0 or info.file_size > MAX_MEMBER_BYTES:
        raise RecoveryPackageError(f"ZIP 成员过大：{info.filename}。")


def _parse_manifest(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> dict[str, Any]:
    if info.file_size > MAX_MANIFEST_BYTES:
        raise RecoveryPackageError("manifest.json 过大。")
    try:
        payload = archive.read(info)
        manifest = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RuntimeError) as exc:
        raise RecoveryPackageError("manifest.json 无法读取或不是有效 JSON。") from exc
    if not isinstance(manifest, dict):
        raise RecoveryPackageError("manifest.json 顶层必须是对象。")
    return manifest


def _required_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RecoveryPackageError(f"manifest 的 {label} 必须是对象。")
    return value


def _validate_manifest(manifest: dict[str, Any]) -> tuple[str, dict[str, dict[str, Any]]]:
    if manifest.get("format") != PACKAGE_FORMAT:
        raise RecoveryPackageError("不是 Motion Inspector 训练恢复包。")
    if manifest.get("version") != PACKAGE_VERSION:
        raise RecoveryPackageError("训练恢复包版本不受支持。")

    source = _required_mapping(manifest.get("source"), "source")
    if not isinstance(source.get("job_id"), str) or not isinstance(source.get("run_directory"), str):
        raise RecoveryPackageError("manifest 的来源任务信息无效。")

    motion = _required_mapping(manifest.get("motion"), "motion")
    original_name = motion.get("original_name")
    if (
        not isinstance(original_name, str)
        or Path(original_name).name != original_name
        or Path(original_name).suffix.lower() != ".npz"
    ):
        raise RecoveryPackageError("manifest 的原动作文件名无效。")

    checkpoint = _required_mapping(manifest.get("checkpoint"), "checkpoint")
    checkpoint_member = checkpoint.get("member")
    checkpoint_match = CHECKPOINT_PATTERN.fullmatch(checkpoint_member or "") if isinstance(checkpoint_member, str) else None
    iteration = checkpoint.get("iteration")
    if (
        checkpoint_match is None
        or isinstance(iteration, bool)
        or not isinstance(iteration, int)
        or iteration < 0
        or int(checkpoint_match.group(1)) != iteration
    ):
        raise RecoveryPackageError("manifest 的 checkpoint 信息无效。")

    training_config = manifest.get("training_config")
    if not isinstance(training_config, dict):
        raise RecoveryPackageError("manifest 的 training_config 必须是对象。")

    files = _required_mapping(manifest.get("files"), "files")
    expected_members = {MOTION_MEMBER, checkpoint_member, ENV_MEMBER, AGENT_MEMBER}
    if set(files) != expected_members:
        raise RecoveryPackageError("manifest 文件清单与协议不一致。")
    records: dict[str, dict[str, Any]] = {}
    for member in expected_members:
        record = _required_mapping(files.get(member), f"files.{member}")
        size = record.get("size")
        digest = record.get("sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or size > MAX_MEMBER_BYTES
            or not isinstance(digest, str)
            or not SHA256_PATTERN.fullmatch(digest)
        ):
            raise RecoveryPackageError(f"manifest 的文件校验信息无效：{member}。")
        records[member] = record
    return checkpoint_member, records


def _copy_verified_member(
    source: BinaryIO,
    destination: Path,
    *,
    expected_size: int,
    expected_digest: str,
) -> None:
    digest = hashlib.sha256()
    size = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        while chunk := source.read(_COPY_CHUNK_BYTES):
            size += len(chunk)
            if size > expected_size or size > MAX_MEMBER_BYTES:
                raise RecoveryPackageError("ZIP 成员解压大小超过 manifest 声明。")
            digest.update(chunk)
            output.write(chunk)
    if size != expected_size or digest.hexdigest() != expected_digest:
        raise RecoveryPackageError("训练恢复包文件大小或 SHA-256 校验失败。")


def validate_and_extract_recovery_package(
    package_path: Path,
    destination: Path,
) -> ExtractedRecoveryPackage:
    """Validate every member and extract into an empty staging directory."""

    if not package_path.is_file() or package_path.is_symlink():
        raise RecoveryPackageError("训练恢复包不存在或不是普通文件。")
    if package_path.stat().st_size > MAX_PACKAGE_BYTES:
        raise RecoveryPackageError("训练恢复包超过大小限制。")
    destination.mkdir(parents=True, exist_ok=False)

    try:
        with zipfile.ZipFile(package_path, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise RecoveryPackageError("训练恢复包包含重复文件。")
            if len(infos) > MAX_MEMBERS:
                raise RecoveryPackageError("训练恢复包文件数量超过限制。")
            for info in infos:
                _validate_zip_info(info)
            if sum(info.file_size for info in infos) > MAX_TOTAL_BYTES:
                raise RecoveryPackageError("训练恢复包解压总大小超过限制。")

            info_by_name = {info.filename: info for info in infos}
            manifest_info = info_by_name.get(MANIFEST_MEMBER)
            if manifest_info is None:
                raise RecoveryPackageError("训练恢复包缺少 manifest.json。")
            manifest = _parse_manifest(archive, manifest_info)
            checkpoint_member, records = _validate_manifest(manifest)
            expected_members = {MANIFEST_MEMBER, *records}
            if set(info_by_name) != expected_members:
                raise RecoveryPackageError("训练恢复包包含缺失或额外文件。")

            extracted_paths: dict[str, Path] = {}
            for member, record in records.items():
                info = info_by_name[member]
                if info.file_size != record["size"]:
                    raise RecoveryPackageError(f"ZIP 文件大小与 manifest 不符：{member}。")
                output_path = destination.joinpath(*PurePosixPath(member).parts)
                with archive.open(info, "r") as source:
                    _copy_verified_member(
                        source,
                        output_path,
                        expected_size=record["size"],
                        expected_digest=record["sha256"],
                    )
                extracted_paths[member] = output_path
    except zipfile.BadZipFile as exc:
        raise RecoveryPackageError("上传文件不是有效 ZIP。") from exc

    return ExtractedRecoveryPackage(
        manifest=manifest,
        motion_path=extracted_paths[MOTION_MEMBER],
        checkpoint_path=extracted_paths[checkpoint_member],
        environment_config_path=extracted_paths[ENV_MEMBER],
        agent_config_path=extracted_paths[AGENT_MEMBER],
    )