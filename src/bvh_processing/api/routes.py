import asyncio
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
from bvh_processing.services.download import download_npz
from bvh_processing.services.tasks import run_merge_task, run_processing_task
from bvh_processing.training.control import submit_training_job
from bvh_processing.training.metrics import (
    run_metrics_polling,
    send_initial_loss_callback,
)
from bvh_processing.training.mock import run_mock_train_task

# ============================================================================
# 接口总流程
#
# /process、/merge、/retarget：校验请求后注册后台任务并立即返回，由后台任务
# 下载源文件、执行处理、调用 callbackUrl 回传结果并释放资源。
#
# /train：Mock 模式按 returnType 回调内置演示文件，不下载 NPZ 或调用 GPU；
# 真实模式在请求内下载 NPZ、上传 GPU 控制服务、查询空闲 GPU 并创建远端
# 训练任务；创建成功后返回远端 job ID，GPU 不足或远端调用失败则直接返回错误。
# 远端训练进程本身异步运行，本服务不等待训练完成。
# ============================================================================


router = APIRouter()


# ----------------------------------------------------------------------------
# GET /health
#
# 入参：无请求体、无查询参数。
# 返回：HealthResponse，即 {"status": "ok"}，用于确认服务进程可用。
# ----------------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


# ----------------------------------------------------------------------------
# POST /api/v1/bvh/process
#
# 入参：ProcessBvhRequest（JSON）
# - actionId：业务动作记录 ID。
# - originalFileUrl：MinIO 中原始 BVH 的可下载地址。
# - handleOptions：处理选项列表；1 去噪、2 平滑、3 脚步锁定、4 循环优化。
# - callbackUrl：处理进度和最终结果的回调地址。
#
# 后台流程：
#   download_bvh() → process_bvh() → send_progress_callback()
#   → send_callback(file=处理结果)
#   当前 process_bvh() 是算法接入点，联调阶段原样返回 BVH 内容；
#   进度回调按 handleOptions 逐项发送，最终结果只发送一次。
#
# 返回：ProcessBvhResponse（立即返回）
# - success：是否成功接收任务。
# - taskId：后台任务 ID，可用于业务侧关联任务。
# - message：接收结果说明。
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# POST /api/v1/bvh/retarget
#
# 入参：RetargetBvhRequest（JSON）
# - originalFileUrl：LAFAN1/Nokov BVH 的 MinIO 下载地址。
# - robotType：机器人类型，目前 1 表示 Unitree G1。
# - callbackUrl：重定向结果回调地址。
#
# 后台流程：
#   run_retarget_task() → download_bvh() → 分类/解析 BVH
#   → Robot Retargeter 重定向 → 导出 Whole Body Tracking NPZ 和元数据 JSON
#   → send_callback(attachments=[NPZ, JSON])
#
# 返回：ProcessBvhResponse，字段含义与 /process 相同；最终文件通过回调上传。
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# POST /api/v1/bvh/train
#
# 入参：TrainBvhRequest（JSON）
# - actionId：可选的训练记录 ID。
# - robotType：机器人类型。
# - algorithmType：训练算法： 1 BeyondMimic（当前支持）、2 PHC、3 OmniH2O
# - npzFileUrl：重定向后 NPZ 文件的 MinIO 下载地址。
# - domainRandomization：当前仅支持 2（训练项目内置随机化配置）。
# - returnType：返回类型，1 MP4 仿真视频和 ONNX 模型 2 ONNX 模型。
# - gpu：可选，物理 GPU 编号，默认 0。（TODO 支持 GPU id 数组）
# - numEnvs：可选，Isaac Lab 并行环境数，默认 7168。
# - maxIterations：可选，最大训练迭代数，默认 100000。
# - seed：可选，训练随机种子，默认 42。
# - callbackUrl：训练结果回调地址。
#
# Mock 模式（默认）：
#   returnType=1 → 通过 callbackUrl 回调内置 MP4 和 ONNX
#   returnType=2 → 通过 callbackUrl 回调内置 ONNX
#   lossCallbackUrl → 立即获取云端最近一次训练任务的真实 loss 并回调，
#   此后每 10 分钟重新查询并回调一次；演示产物仍在 10 秒后回调。
#
# 真实模式（启动时指定 --mock 0）：（TODO 任务完成或者提前结束的按钮触发回调）
#   下载 NPZ → 上传 GPU Training Control → 查询空闲 GPU
#   → 按物理编号选择第一张空闲卡 → 创建远端训练任务
#   → 立即查询并回调 loss → 此后每 10 分钟重新查询并回调一次
#
# 返回：ProcessBvhResponse。Mock 模式的 taskId 为本地任务 ID；真实模式的
# taskId 为 GPU 控制服务返回的训练任务 ID。
# ----------------------------------------------------------------------------
@router.post(
    "/api/v1/bvh/train",
    response_model=ProcessBvhResponse,
    summary="提交机器人策略训练任务",
    description=(
        "默认使用 Mock 模式，returnType=1 回调内置 MP4 和 ONNX，"
        "returnType=2 仅回调内置 ONNX；提供 lossCallbackUrl 时使用 GPU "
        "Control 最近一次训练任务的真实 loss。使用 --mock 0 启动时，下载 MinIO "
        "中的 NPZ，上传到 GPU Training Control，自动选择空闲 GPU 并创建"
        "远端 BeyondMimic 训练任务。"
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
    if payload.loss_callback_url is not None:
        validate_callback_url(
            str(payload.loss_callback_url), settings.allowed_callback_hosts
        )

    client: httpx.AsyncClient = request.app.state.http_client
    if settings.mock:
        task_id = str(uuid4())
        background_tasks.add_task(
            run_mock_train_task,
            client,
            settings,
            task_id,
            payload,
        )
        return ProcessBvhResponse(
            success=True,
            taskId=task_id,
            message="Mock 训练任务已接收",
        )

    source = await download_npz(client, str(payload.npz_file_url), settings)
    try:
        submitted = await submit_training_job(client, settings, source, payload)
    finally:
        source.content.close()

    await send_initial_loss_callback(
        client,
        settings,
        submitted.job_id,
        payload,
    )
    asyncio.create_task(
        run_metrics_polling(
            client,
            settings,
            submitted.job_id,
            payload,
        )
    )

    return ProcessBvhResponse(
        success=True,
        taskId=submitted.job_id,
        message=f"训练任务已提交，使用 cuda:{submitted.gpu}",
    )


# ----------------------------------------------------------------------------
# POST /api/v1/bvh/merge
#
# 入参：MergeBvhRequest（JSON）
# - actionId/danceId：任务回调关联 ID 或舞蹈 ID，至少提供一个。
# - timelineOffsetSec：首动作时间轴偏移，仅用于编排，不写入 BVH。
# - segments：按合并顺序排列的片段，至少一个；每段包含：
#   segmentId、actionId、actionUrl、sourceInSec、sourceOutSec、
#   outputDurationSec、gapAfterSec。
# - callbackUrl：合并结果回调地址。
#
# 后台流程：
#   run_merge_task()
#     → download_bvh()：按 actionUrl 下载并复用相同源文件
#     → trim_bvh()：将 sourceInSec/sourceOutSec 映射到最近的已有帧并裁剪，
#       不在边界生成插值姿态
#     → normalize_bvh_frame_rates()：统一到最低帧率
#     → adjust_bvh_motion_durations()：按 outputDurationSec 重采样，
#       目标时长更短则加速，目标时长更长则放慢
#     → merge_bvh_files()：将非末段 gapAfterSec 换算为接缝过渡帧数，
#       对后一个动作执行根节点位置/朝向对齐，使用 Hermite 曲线插值根节点
#       位移、缓动旋转插值关节姿态，并锁定支撑脚
#     → send_callback(file=*_merged.bvh)：上传最终合并文件
#
# 返回：ProcessBvhResponse，表示合并任务是否已接收；最终 BVH 通过回调上传。
# ----------------------------------------------------------------------------
@router.post(
    "/api/v1/bvh/merge",
    response_model=ProcessBvhResponse,
    summary="提交多个 BVH 合并任务",
    description=(
        "按 segments 的顺序异步处理并合并 BVH 片段。每段先根据 "
        "sourceInSec/sourceOutSec 裁剪已有帧，再统一帧率并按 "
        "outputDurationSec 调整速度；非末段的 gapAfterSec 表示与下一动作的"
        "平滑过渡时长。过渡阶段会执行根节点对齐、旋转插值和支撑脚锁定。"
        "timelineOffsetSec 不写入输出 BVH。完成后通过 callbackUrl 上传文件。"
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
