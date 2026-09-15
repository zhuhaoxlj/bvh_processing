"""本地自测页面：上传 BVH → 调用真实 /api/v1/bvh/process → 取回结果给 three.js 预览。

仅在 ``BVH_DEV_UI=1`` 时由 :func:`bvh_processing.main.create_app` 挂载，供本地联调使用。
它把上传的文件暂存成可下载 URL，并充当 ``/api/v1/bvh/process`` 的进度与结果回调接收方，
因此走的是与业务侧完全相同的接口链路（下载、SHA-256 校验、格式识别、回调）。
单 worker 运行，请勿在生产环境开启。

启动时 :func:`announce` 会打印可点击的页面地址（``GET /`` 也会重定向过去），
配置 ``BVH_DEV_UI_OPEN_BROWSER=1`` 还能直接拉起系统浏览器。
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import quote
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse

from bvh_processing.config import Settings, get_settings
from bvh_processing.errors import BvhServiceError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["dev-ui"])

_DEV_UI_ROOT = Path(__file__).resolve().parent.parent / "dev_ui"
INDEX_HTML = _DEV_UI_ROOT / "index.html"
STATIC_ROOT = _DEV_UI_ROOT / "static"

_UPLOAD_CHUNK_SIZE = 1024 * 1024
_MAX_TASKS = 200
_MAX_TASK_AGE_SECONDS = 6 * 60 * 60
_ID_PATTERN = re.compile(r"^[0-9a-zA-Z_-]{8,64}$")
_DEFAULT_PORT = 9001
# 0.0.0.0 这类监听地址不能直接点开，换成回环地址更方便。
_UNCLICKABLE_HOSTS = frozenset({"", "0.0.0.0", "::", "[::]"})
_BROWSER_DELAY_SECONDS = 0.8

TaskStatus = Literal["pending", "succeeded", "failed"]


@dataclass(slots=True)
class _Upload:
    token: str
    filename: str
    path: Path
    sha256: str
    size: int


@dataclass(slots=True)
class _DevTask:
    task_id: str
    status: TaskStatus = "pending"
    message: str = ""
    progress: list[dict[str, Any]] = field(default_factory=list)
    result_path: Path | None = None
    result_filename: str | None = None
    result_size: int = 0
    updated_at: float = field(default_factory=time.time)


class _DevStore:
    """进程内的临时存储；目录懒创建，应用关闭时整体删除。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._root: Path | None = None
        self._uploads: dict[str, _Upload] = {}
        self._tasks: dict[str, _DevTask] = {}

    def _directory_locked(self, name: str) -> Path:
        if self._root is None:
            self._root = Path(tempfile.mkdtemp(prefix="bvh-dev-ui-"))
            logger.info("dev ui temporary root: %s", self._root)
        directory = self._root / name
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def new_upload_path(self, token: str) -> Path:
        with self._lock:
            return self._directory_locked("uploads") / f"{token}.bvh"

    def new_result_path(self, task_id: str) -> Path:
        with self._lock:
            return self._directory_locked("results") / f"{task_id}.bvh"

    def add_upload(self, upload: _Upload) -> None:
        with self._lock:
            self._uploads[upload.token] = upload

    def get_upload(self, token: str) -> _Upload | None:
        with self._lock:
            return self._uploads.get(token)

    def task(self, task_id: str) -> _DevTask:
        with self._lock:
            self._prune_locked()
            task = self._tasks.get(task_id)
            if task is None:
                task = _DevTask(task_id=task_id)
                self._tasks[task_id] = task
            return task

    def add_progress(self, task_id: str, payload: dict[str, Any]) -> None:
        task = self.task(task_id)
        with self._lock:
            task.progress.append(payload)
            task.updated_at = time.time()

    def finish(
        self,
        task_id: str,
        *,
        path: Path,
        filename: str,
        size: int,
        message: str,
    ) -> None:
        task = self.task(task_id)
        with self._lock:
            task.status = "succeeded"
            task.message = message
            task.result_path = path
            task.result_filename = filename
            task.result_size = size
            task.updated_at = time.time()

    def fail(self, task_id: str, message: str) -> None:
        task = self.task(task_id)
        with self._lock:
            task.status = "failed"
            task.message = message
            task.updated_at = time.time()

    def _prune_locked(self) -> None:
        if len(self._tasks) < _MAX_TASKS:
            return
        deadline = time.time() - _MAX_TASK_AGE_SECONDS
        for task_id in [
            key for key, task in self._tasks.items() if task.updated_at < deadline
        ]:
            del self._tasks[task_id]
        overflow = len(self._tasks) - _MAX_TASKS
        if overflow > 0:
            oldest = sorted(self._tasks, key=lambda key: self._tasks[key].updated_at)
            for task_id in oldest[: overflow + 1]:
                del self._tasks[task_id]

    def reset(self) -> None:
        with self._lock:
            root, self._root = self._root, None
            self._uploads.clear()
            self._tasks.clear()
        if root is not None:
            shutil.rmtree(root, ignore_errors=True)
            logger.info("dev ui temporary root removed: %s", root)


_store = _DevStore()


def reset_store() -> None:
    """删除自测页面的临时文件；未启用时是空操作。"""
    _store.reset()


def _command_line_option(name: str) -> str | None:
    """从 ``sys.argv`` 里取 ``--name value`` 或 ``--name=value``。"""
    argv = sys.argv[1:]
    for index, item in enumerate(argv):
        if item == name and index + 1 < len(argv):
            return argv[index + 1]
        if item.startswith(f"{name}="):
            return item.split("=", 1)[1]
    return None


def _serving_host_port(default_port: int = _DEFAULT_PORT) -> tuple[str, int]:
    """推断服务实际监听的地址，优先命令行参数，其次 uvicorn 的环境变量。"""
    host = _command_line_option("--host") or os.environ.get("UVICORN_HOST") or ""
    raw_port = _command_line_option("--port") or os.environ.get("UVICORN_PORT") or ""
    try:
        port = int(raw_port)
    except ValueError:
        port = default_port
    if host in _UNCLICKABLE_HOSTS:
        host = "127.0.0.1"
    elif ":" in host:
        host = f"[{host}]"
    return host, port


def dev_ui_url(default_port: int = _DEFAULT_PORT) -> str:
    """返回自测页面的完整地址，便于直接点开。"""
    host, port = _serving_host_port(default_port)
    return f"http://{host}:{port}/dev/bvh"


def _open_browser(url: str) -> None:
    # 立刻解析实现：延迟触发时不应受期间对 webbrowser.open 的替换影响。
    open_url = webbrowser.open

    def _open() -> None:
        try:
            if not open_url(url):
                logger.warning("未能自动打开浏览器，请手动访问：%s", url)
        except Exception as error:  # noqa: BLE001
            logger.warning("自动打开浏览器失败（%s），请手动访问：%s", error, url)

    # 等 uvicorn 绑定端口后再拉起浏览器，避免首屏连接被拒。
    timer = threading.Timer(_BROWSER_DELAY_SECONDS, _open)
    timer.daemon = True
    timer.start()


def announce(settings: Settings) -> None:
    """启动时打印自测页面地址，必要时拉起浏览器。"""
    url = dev_ui_url()
    logger.warning(
        "本地自测页面已启用：%s（仅供联调，请勿在生产环境开启，且需单 worker 运行）",
        url,
    )
    if settings.dev_ui_open_browser:
        _open_browser(url)


def _require_safe_id(task_id: str) -> str:
    if _ID_PATTERN.fullmatch(task_id) is None:
        raise BvhServiceError(
            status_code=400,
            code="invalid_dev_task_id",
            message="自测任务 ID 格式不正确",
        )
    return task_id


def _allow_dev_host(request: Request, settings: Settings) -> None:
    """自测页面与被测服务同源，把该主机并入下载/回调白名单，避免本地联调被拦截。"""
    host = (request.url.hostname or "").lower().rstrip(".")
    if not host:
        return
    for field_name in ("minio_allowed_hosts", "callback_allowed_hosts"):
        current = str(getattr(settings, field_name))
        hosts = [item.strip().lower() for item in current.split(",") if item.strip()]
        # 空列表表示放行全部主机，无需追加。
        if not hosts or host in hosts:
            continue
        setattr(settings, field_name, ",".join([*hosts, host]))
        logger.debug("dev ui added host %s to %s", host, field_name)


def _task_payload(task: _DevTask) -> dict[str, Any]:
    result = None
    if task.result_path is not None:
        result = {
            "filename": task.result_filename,
            "size": task.result_size,
            "url": f"/api/v1/dev/bvh/tasks/{task.task_id}/result",
        }
    return {
        "taskId": task.task_id,
        "status": task.status,
        "success": task.status == "succeeded",
        "message": task.message,
        "progress": task.progress,
        "result": result,
    }


@router.get("/", include_in_schema=False)
async def dev_ui_root() -> RedirectResponse:
    """开启自测页面时，直接访问根路径也能进页面。"""
    return RedirectResponse("/dev/bvh", status_code=302)


@router.get(
    "/dev/bvh",
    summary="[自测] BVH 处理测试页面",
    description="本地联调用的单页工具：上传 BVH、调用 /api/v1/bvh/process、预览处理结果。",
    include_in_schema=False,
)
async def dev_ui_page() -> FileResponse:
    return FileResponse(INDEX_HTML, media_type="text/html; charset=utf-8")


@router.post(
    "/api/v1/dev/bvh/uploads",
    summary="[自测] 暂存待处理的 BVH 文件",
    description="把浏览器上传的 BVH 存为临时文件，返回可直接作为 originalFileUrl 的地址与 SHA-256。",
)
async def create_dev_upload(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    file: Annotated[UploadFile, File(description="待处理的 BVH 文件")],
) -> dict[str, Any]:
    _allow_dev_host(request, settings)

    filename = Path(file.filename or "").name or "source.bvh"
    if not filename.lower().endswith(".bvh"):
        raise BvhServiceError(
            status_code=400,
            code="invalid_upload_filename",
            message="只允许上传 .bvh 文件",
        )

    token = uuid4().hex
    target = _store.new_upload_path(token)
    digest = hashlib.sha256()
    size = 0
    try:
        with target.open("wb") as sink:
            while chunk := await file.read(_UPLOAD_CHUNK_SIZE):
                size += len(chunk)
                if size > settings.max_file_size_bytes:
                    raise BvhServiceError(
                        status_code=413,
                        code="source_file_too_large",
                        message=f"BVH 文件不能超过 {settings.max_file_size_mb} MB",
                    )
                digest.update(chunk)
                sink.write(chunk)
    except BvhServiceError:
        target.unlink(missing_ok=True)
        raise

    if size == 0:
        target.unlink(missing_ok=True)
        raise BvhServiceError(
            status_code=422,
            code="empty_source_file",
            message="BVH 文件为空",
        )

    upload = _Upload(
        token=token,
        filename=filename,
        path=target,
        sha256=digest.hexdigest(),
        size=size,
    )
    _store.add_upload(upload)
    return {
        "token": token,
        "filename": filename,
        "size": size,
        "sha256": upload.sha256,
        "sourcePath": f"/api/v1/dev/bvh/uploads/{token}/{quote(filename)}",
    }


@router.get(
    "/api/v1/dev/bvh/uploads/{token}/{filename}",
    summary="[自测] 下载暂存的 BVH",
    description=(
        "作为 originalFileUrl 被 /api/v1/bvh/process 回源下载；"
        "路径末段保留原始文件名，好让处理结果命名为 <原名>_processed.bvh。"
    ),
)
async def download_dev_upload(token: str, filename: str) -> FileResponse:
    upload = _store.get_upload(_require_safe_id(token))
    if upload is None:
        raise BvhServiceError(
            status_code=404,
            code="dev_upload_not_found",
            message="暂存的 BVH 不存在或已被清理",
        )
    return FileResponse(
        upload.path,
        media_type="application/octet-stream",
        filename=upload.filename,
    )


@router.post(
    "/api/v1/dev/bvh/progress-callback/{task_id}",
    summary="[自测] 接收处理进度回调",
    description="/api/v1/bvh/process 会把 callbackUrl 中的 /callback 替换为 /progress-callback 后回调到这里。",
)
async def dev_progress_callback(task_id: str, request: Request) -> dict[str, bool]:
    payload = await request.json()
    if not isinstance(payload, dict):
        payload = {"raw": payload}
    _store.add_progress(_require_safe_id(task_id), payload)
    return {"received": True}


@router.post(
    "/api/v1/dev/bvh/callback/{task_id}",
    summary="[自测] 接收处理结果回调",
    description="接收 /api/v1/bvh/process 的最终 multipart 回调，把结果文件另存后供页面预览和下载。",
)
async def dev_result_callback(
    task_id: str,
    success: Annotated[bool, Form()] = False,
    message: Annotated[str, Form()] = "",
    action_id: Annotated[str | None, Form(alias="actionId")] = None,
    file: Annotated[UploadFile | None, File()] = None,
) -> dict[str, bool]:
    _store.task(_require_safe_id(task_id))

    if not success:
        _store.fail(task_id, message or "处理失败")
        return {"received": True}

    if file is None:
        _store.fail(task_id, "成功回调缺少 file 字段")
        return {"received": True}

    filename = Path(file.filename or "").name or f"{task_id}_processed.bvh"
    target = _store.new_result_path(task_id)
    size = 0
    with target.open("wb") as sink:
        while chunk := await file.read(_UPLOAD_CHUNK_SIZE):
            size += len(chunk)
            sink.write(chunk)

    _store.finish(
        task_id,
        path=target,
        filename=filename,
        size=size,
        message=message or "处理成功",
    )
    logger.info(
        "dev ui result stored: taskId=%s actionId=%s filename=%s size=%d",
        task_id,
        action_id,
        filename,
        size,
    )
    return {"received": True}


@router.get(
    "/api/v1/dev/bvh/tasks/{task_id}",
    summary="[自测] 查询自测任务状态",
    description="页面轮询该接口获取进度回调与最终结果，任务行按 task_id 懒创建。",
)
async def dev_task_status(task_id: str) -> dict[str, Any]:
    return _task_payload(_store.task(_require_safe_id(task_id)))


@router.get(
    "/api/v1/dev/bvh/tasks/{task_id}/result",
    summary="[自测] 下载处理结果 BVH",
)
async def dev_task_result(task_id: str) -> FileResponse:
    task = _store.task(_require_safe_id(task_id))
    if task.result_path is None:
        raise BvhServiceError(
            status_code=404,
            code="dev_result_not_ready",
            message="处理结果尚未就绪",
        )
    return FileResponse(
        task.result_path,
        media_type="application/octet-stream",
        filename=task.result_filename or f"{task_id}_processed.bvh",
    )
