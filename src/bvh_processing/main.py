import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from bvh_processing.api.routes import router
from bvh_processing.config import get_settings
from bvh_processing.errors import BvhServiceError

logger = logging.getLogger(__name__)


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
        request: Request, error: BvhServiceError
    ) -> JSONResponse:
        logger.warning(
            "API request failed: method=%s path=%s status=%d code=%s message=%s",
            request.method,
            request.url.path,
            error.status_code,
            error.code,
            error.message,
        )
        return JSONResponse(
            status_code=error.status_code,
            content={
                "success": False,
                "taskId": None,
                "code": error.code,
                "message": error.message,
            },
        )

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        details = []
        for item in error.errors():
            location = item.get("loc", ())
            field_parts = [str(part) for part in location if part != "body"]
            details.append(
                {
                    "field": ".".join(field_parts) or "body",
                    "message": item.get("msg", "参数校验失败"),
                    "type": item.get("type", "validation_error"),
                }
            )

        logger.warning(
            "API request validation failed: method=%s path=%s errors=%s",
            request.method,
            request.url.path,
            details,
        )
        return JSONResponse(
            status_code=422,
            content=jsonable_encoder(
                {
                    "success": False,
                    "taskId": None,
                    "code": "request_validation_error",
                    "message": "请求参数校验失败",
                    "errors": details,
                }
            ),
        )

    @application.exception_handler(StarletteHTTPException)
    async def handle_http_error(
        request: Request, error: StarletteHTTPException
    ) -> JSONResponse:
        message = error.detail if isinstance(error.detail, str) else "HTTP 请求失败"
        logger.warning(
            "HTTP request failed: method=%s path=%s status=%d detail=%r",
            request.method,
            request.url.path,
            error.status_code,
            error.detail,
        )
        return JSONResponse(
            status_code=error.status_code,
            content=jsonable_encoder(
                {
                    "success": False,
                    "taskId": None,
                    "code": "http_error",
                    "message": message,
                    "detail": error.detail,
                }
            ),
            headers=error.headers,
        )

    @application.exception_handler(Exception)
    async def handle_unexpected_error(
        request: Request, error: Exception
    ) -> JSONResponse:
        error_id = str(uuid4())
        logger.exception(
            "Unexpected API error: id=%s method=%s path=%s error=%s",
            error_id,
            request.method,
            request.url.path,
            error,
        )
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "taskId": None,
                "code": "internal_server_error",
                "message": "服务器内部错误",
                "errorId": error_id,
            },
        )

    return application


app = create_app()
