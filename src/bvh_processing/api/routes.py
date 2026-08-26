from typing import Annotated
from uuid import uuid4

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, Request

from bvh_processing.config import Settings, get_settings
from bvh_processing.retargeting.task import run_retarget_task
from bvh_processing.schemas import (
    HealthResponse,
    MergeBvhRequest,
    ProcessBvhRequest,
    ProcessBvhResponse,
    RetargetBvhRequest,
    TrainBvhRequest,
)
from bvh_processing.services.callback import validate_callback_url
from bvh_processing.services.tasks import run_merge_task, run_processing_task
from bvh_processing.training.task import run_train_task

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


@router.post(
    "/api/v1/bvh/retarget",
    response_model=ProcessBvhResponse,
    summary="提交 BVH 重定向任务",
    description=(
        "异步下载 MinIO 中的 LAFAN1/Nokov BVH，使用 Robot Retargeter "
        "重定向为 Unitree G1 动作，并生成 Whole Body Tracking NPZ 和"
        "元数据 JSON。完成后通过 callbackUrl 一次上传两个文件。"
    ),
    tags=["bvh"],
)
async def retarget(
    payload: RetargetBvhRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ProcessBvhResponse:
    validate_callback_url(str(payload.callback_url), settings.allowed_callback_hosts)

    task_id = str(uuid4())
    client: httpx.AsyncClient = request.app.state.http_client
    background_tasks.add_task(
        run_retarget_task,
        client,
        settings,
        task_id,
        payload,
    )
    return ProcessBvhResponse(
        success=True,
        taskId=task_id,
        message="重定向任务已接收",
    )


@router.post(
    "/api/v1/bvh/train",
    response_model=ProcessBvhResponse,
    summary="提交机器人策略训练任务",
    description=(
        "异步下载 MinIO 中的重定向 NPZ，并按机器人、算法和域随机强度"
        "执行训练。returnType=1 时通过 callbackUrl 上传 MP4；"
        "returnType=2 时上传 ONNX。"
    ),
    tags=["bvh"],
)
async def train(
    payload: TrainBvhRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ProcessBvhResponse:
    validate_callback_url(str(payload.callback_url), settings.allowed_callback_hosts)

    task_id = str(uuid4())
    client: httpx.AsyncClient = request.app.state.http_client
    background_tasks.add_task(
        run_train_task,
        client,
        settings,
        task_id,
        payload,
    )
    return ProcessBvhResponse(
        success=True,
        taskId=task_id,
        message="训练任务已接收",
    )


@router.post(
    "/api/v1/bvh/merge",
    response_model=ProcessBvhResponse,
    summary="提交多个 BVH 合并任务",
    description=(
        "按 fileUrls 的顺序异步合并多个 BVH。intervalsSeconds 中的每个值"
        "表示对应两个文件之间的间隔秒数；间隔帧保持前一个文件的最后姿势。"
        "完成后通过 callbackUrl 上传合并文件。"
    ),
    tags=["bvh"],
)
async def merge(
    payload: MergeBvhRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ProcessBvhResponse:
    validate_callback_url(str(payload.callback_url), settings.allowed_callback_hosts)

    task_id = str(uuid4())
    client: httpx.AsyncClient = request.app.state.http_client
    background_tasks.add_task(
        run_merge_task,
        client,
        settings,
        task_id,
        payload,
    )
    return ProcessBvhResponse(
        success=True,
        taskId=task_id,
        message="合并任务已接收",
    )
