from typing import Annotated
from uuid import uuid4

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, Request

from bvh_processing.config import Settings, get_settings
from bvh_processing.schemas import (
    HealthResponse,
    ProcessBvhRequest,
    ProcessBvhResponse,
)
from bvh_processing.services.callback import validate_callback_url
from bvh_processing.services.tasks import run_processing_task

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post(
    "/api/v1/bvh/process",
    response_model=ProcessBvhResponse,
    summary="提交 BVH 处理任务",
    description=(
        "异步接收 BVH 处理任务。任务执行期间，服务会对每个选中的处理选项 "
        "向 callbackUrl 发送一次 multipart/form-data 进度回调；全部完成后，"
        "再发送一次携带 file 的最终结果回调。"
    ),
    tags=["bvh"],
)
async def process(
    payload: ProcessBvhRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ProcessBvhResponse:
    validate_callback_url(str(payload.callback_url), settings.allowed_callback_hosts)

    task_id = str(uuid4())
    client: httpx.AsyncClient = request.app.state.http_client
    background_tasks.add_task(
        run_processing_task,
        client,
        settings,
        task_id,
        payload,
    )
    return ProcessBvhResponse(
        success=True,
        taskId=task_id,
        message="任务已接收",
    )
