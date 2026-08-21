from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from bvh_processing.api.routes import router
from bvh_processing.config import get_settings
from bvh_processing.errors import BvhServiceError


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    timeout = httpx.Timeout(settings.download_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        app.state.http_client = client
        yield


def create_app() -> FastAPI:
    application = FastAPI(
        title="BVH Processing API",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.include_router(router)

    @application.exception_handler(BvhServiceError)
    async def handle_service_error(
        _request: Request, error: BvhServiceError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content={
                "success": False,
                "taskId": None,
                "message": error.message,
            },
        )

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request, _error: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "taskId": None,
                "message": "请求参数不正确",
            },
        )

    return application


app = create_app()
