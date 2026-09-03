"""训练期间轮询 GPU loss 并回调业务后端。"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from bvh_processing.config import Settings
from bvh_processing.schemas import TrainBvhRequest

logger = logging.getLogger(__name__)

LOSS_INTERVAL_SECONDS = 600.0
LOSS_MAX_POINTS = 500


def _authorization_headers(settings: Settings) -> dict[str, str]:
    if not settings.gpu_control_api_token:
        raise ValueError("GPU 控制服务 Token 未配置")
    return {"Authorization": f"Bearer {settings.gpu_control_api_token}"}


def _api_url(settings: Settings, job_id: str) -> str:
    return (
        f"{settings.gpu_control_api_url.rstrip('/')}/api/v1/jobs/"
        f"{job_id}/loss?max_points={LOSS_MAX_POINTS}"
    )


async def latest_training_job_id(
    client: httpx.AsyncClient,
    settings: Settings,
) -> str:
    """返回 GPU 控制服务最近创建的训练任务 ID。"""

    response = await client.get(
        f"{settings.gpu_control_api_url.rstrip('/')}/api/v1/jobs",
        headers=_authorization_headers(settings),
        timeout=settings.gpu_control_timeout_seconds,
    )
    response.raise_for_status()
    body: Any = response.json()
    jobs = body.get("jobs") if isinstance(body, dict) else None
    if not isinstance(jobs, list):
        raise TypeError("GPU 控制服务返回了无效的训练任务列表")
    for job in jobs:
        job_id = job.get("id") if isinstance(job, dict) else None
        if isinstance(job_id, str) and job_id:
            return job_id
    raise ValueError("GPU 控制服务暂无可用于 Mock loss 的训练任务")


async def _fetch_and_callback(
    client: httpx.AsyncClient,
    settings: Settings,
    job_id: str,
    payload: TrainBvhRequest,
) -> bool:
    """查询一次 GPU loss 并回调；返回是否成功完成本次回调。"""

    if payload.loss_callback_url is None:
        return False

    action_id = payload.action_id or f"training-{job_id}"
    response = await client.get(
        _api_url(settings, job_id),
        headers=_authorization_headers(settings),
        timeout=120.0,
    )
    response.raise_for_status()
    loss_data: Any = response.json()

    callback_headers: dict[str, str] = {}
    if settings.callback_token:
        callback_headers["X-Callback-Token"] = settings.callback_token
    callback = await client.post(
        str(payload.loss_callback_url),
        json={"actionId": action_id, "data": loss_data},
        headers=callback_headers,
        timeout=settings.gpu_control_timeout_seconds,
    )
    callback.raise_for_status()
    logger.info("已回调训练 loss：jobId=%s", job_id)
    return True


async def send_loss_callback(
    client: httpx.AsyncClient,
    settings: Settings,
    job_id: str,
    payload: TrainBvhRequest,
) -> None:
    """查询一次 GPU loss 并发送回调。"""

    await _fetch_and_callback(client, settings, job_id, payload)


async def send_initial_loss_callback(
    client: httpx.AsyncClient,
    settings: Settings,
    job_id: str,
    payload: TrainBvhRequest,
) -> None:
    """立即查询并回调一次；数据暂时不可用时不阻断训练。"""

    if payload.loss_callback_url is None:
        return

    try:
        await _fetch_and_callback(client, settings, job_id, payload)
    except asyncio.CancelledError:
        raise
    except (httpx.HTTPError, ValueError, TypeError) as error:
        logger.warning("首次训练 loss 查询或回调失败：jobId=%s error=%s", job_id, error)
    except Exception:
        logger.exception("首次训练 loss 查询或回调异常：jobId=%s", job_id)


async def run_metrics_polling(
    client: httpx.AsyncClient,
    settings: Settings,
    job_id: str,
    payload: TrainBvhRequest,
) -> None:
    """首次回调后每十分钟查询 GPU loss 并回调业务后端。"""

    if payload.loss_callback_url is None:
        return

    deadline = asyncio.get_running_loop().time() + settings.train_timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        try:
            await asyncio.sleep(LOSS_INTERVAL_SECONDS)
            if asyncio.get_running_loop().time() >= deadline:
                break
            await _fetch_and_callback(client, settings, job_id, payload)
        except asyncio.CancelledError:
            raise
        except (httpx.HTTPError, ValueError, TypeError) as error:
            logger.warning("训练 loss 轮询或回调失败：jobId=%s error=%s", job_id, error)
        except Exception:
            logger.exception("训练 loss 轮询异常：jobId=%s", job_id)
