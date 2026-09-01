"""GPU Training Control API 客户端。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from bvh_processing.config import Settings
from bvh_processing.errors import BvhServiceError
from bvh_processing.schemas import TrainBvhRequest
from bvh_processing.services.download import DownloadedBvh


@dataclass(frozen=True, slots=True)
class SubmittedTrainingJob:
    job_id: str
    gpu: int


def _api_url(settings: Settings, path: str) -> str:
    return f"{settings.gpu_control_api_url.rstrip('/')}{path}"


def _error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text.strip()[:500]
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, str):
            return detail
    return ""


def _remote_error(
    operation: str,
    response: httpx.Response,
    *,
    status_code: int = 502,
) -> BvhServiceError:
    detail = _error_detail(response)
    message = f"GPU 控制服务{operation}失败（HTTP {response.status_code}）"
    if detail:
        message = f"{message}：{detail}"
    return BvhServiceError(
        status_code=status_code,
        code="gpu_control_api_error",
        message=message,
    )


def _authorization_headers(settings: Settings) -> dict[str, str]:
    if not settings.gpu_control_api_token:
        raise BvhServiceError(
            status_code=503,
            code="gpu_control_not_configured",
            message="GPU 控制服务 Token 未配置",
        )
    return {"Authorization": f"Bearer {settings.gpu_control_api_token}"}


async def _upload_motion(
    client: httpx.AsyncClient,
    settings: Settings,
    source: DownloadedBvh,
) -> str:
    source.content.seek(0)
    try:
        response = await client.post(
            _api_url(settings, "/api/v1/artifacts/motions"),
            headers=_authorization_headers(settings),
            files={
                "file": (
                    source.source_filename,
                    source.content,
                    "application/octet-stream",
                )
            },
            timeout=settings.gpu_control_timeout_seconds,
        )
    except httpx.HTTPError as exc:
        raise BvhServiceError(
            status_code=502,
            code="gpu_control_unavailable",
            message="无法连接 GPU 控制服务上传 NPZ",
        ) from exc

    if not response.is_success:
        raise _remote_error("上传 NPZ", response)
    try:
        body: Any = response.json()
        artifact_id = body["id"]
    except (ValueError, KeyError, TypeError) as exc:
        raise BvhServiceError(
            status_code=502,
            code="invalid_gpu_control_response",
            message="GPU 控制服务上传 NPZ 的响应缺少动作文件 ID",
        ) from exc
    if not isinstance(artifact_id, str) or not artifact_id:
        raise BvhServiceError(
            status_code=502,
            code="invalid_gpu_control_response",
            message="GPU 控制服务返回了无效的动作文件 ID",
        )
    return artifact_id


async def _available_gpus(
    client: httpx.AsyncClient,
    settings: Settings,
) -> list[int]:
    try:
        response = await client.get(
            _api_url(settings, "/api/v1/gpus/simple"),
            timeout=settings.gpu_control_timeout_seconds,
        )
    except httpx.HTTPError as exc:
        raise BvhServiceError(
            status_code=502,
            code="gpu_control_unavailable",
            message="无法连接 GPU 控制服务查询显卡状态",
        ) from exc

    if not response.is_success:
        raise _remote_error("查询显卡", response)
    try:
        body: Any = response.json()
        gpus = body["gpus"]
    except (ValueError, KeyError, TypeError) as exc:
        raise BvhServiceError(
            status_code=502,
            code="invalid_gpu_control_response",
            message="GPU 控制服务返回了无效的显卡列表",
        ) from exc
    if not isinstance(gpus, list):
        raise BvhServiceError(
            status_code=502,
            code="invalid_gpu_control_response",
            message="GPU 控制服务返回了无效的显卡列表",
        )

    available: list[int] = []
    for gpu in gpus:
        if not isinstance(gpu, dict) or gpu.get("available") is not True:
            continue
        index = gpu.get("gpu")
        if isinstance(index, int) and not isinstance(index, bool) and index >= 0:
            available.append(index)
    return sorted(set(available))


def _job_payload(
    artifact_id: str,
    gpu: int,
    payload: TrainBvhRequest,
) -> dict[str, object]:
    return {
        "artifact_id": artifact_id,
        "devices": [f"cuda:{gpu}"],
        "num_envs": payload.num_envs,
        "max_iterations": payload.max_iterations,
        "seed": payload.seed,
    }


async def _create_job(
    client: httpx.AsyncClient,
    settings: Settings,
    artifact_id: str,
    gpu: int,
    payload: TrainBvhRequest,
) -> httpx.Response:
    try:
        return await client.post(
            _api_url(settings, "/api/v1/jobs"),
            headers=_authorization_headers(settings),
            json=_job_payload(artifact_id, gpu, payload),
            timeout=settings.gpu_control_timeout_seconds,
        )
    except httpx.HTTPError as exc:
        raise BvhServiceError(
            status_code=502,
            code="gpu_control_unavailable",
            message="无法连接 GPU 控制服务创建训练任务",
        ) from exc


async def submit_training_job(
    client: httpx.AsyncClient,
    settings: Settings,
    source: DownloadedBvh,
    payload: TrainBvhRequest,
) -> SubmittedTrainingJob:
    """上传 NPZ，按编号选择第一张空闲 GPU，并创建训练任务。"""

    artifact_id = await _upload_motion(client, settings, source)
    available_gpus = await _available_gpus(client, settings)
    if not available_gpus:
        raise BvhServiceError(
            status_code=409,
            code="gpu_capacity_full",
            message="当前显卡训练任务已满",
        )

    # GPU 状态查询与任务创建之间可能发生竞争。若某张卡刚被占用，继续尝试
    # 查询结果中的下一张空闲卡；全部冲突时再返回容量已满。
    for gpu in available_gpus:
        response = await _create_job(client, settings, artifact_id, gpu, payload)
        if response.status_code == 409:
            continue
        if not response.is_success:
            raise _remote_error("创建训练任务", response)
        try:
            body: Any = response.json()
            job_id = body["id"]
        except (ValueError, KeyError, TypeError) as exc:
            raise BvhServiceError(
                status_code=502,
                code="invalid_gpu_control_response",
                message="GPU 控制服务创建任务的响应缺少训练任务 ID",
            ) from exc
        if not isinstance(job_id, str) or not job_id:
            raise BvhServiceError(
                status_code=502,
                code="invalid_gpu_control_response",
                message="GPU 控制服务返回了无效的训练任务 ID",
            )
        return SubmittedTrainingJob(job_id=job_id, gpu=gpu)

    raise BvhServiceError(
        status_code=409,
        code="gpu_capacity_full",
        message="当前显卡训练任务已满",
    )
