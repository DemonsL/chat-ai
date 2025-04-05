import time
from typing import Callable

from fastapi import FastAPI, Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    请求日志中间件
    记录所有HTTP请求的信息、处理时间和响应状态
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)

            # 记录成功响应信息
            logger.info(
                f"请求完成 | {request.method} {request.url.path} | "
                f"{response.status_code} | {process_time:.4f}s"
            )
            return response
        except Exception as e:
            process_time = time.time() - start_time
            # 记录异常信息，使用异常类型名称
            logger.error(
                f"请求异常 | {request.method} {request.url.path} | "
                f"{type(e).__name__} | {process_time:.4f}s"
                # 可以选择性地包含异常详情: f" | Error: {e}"
            )
            # 必须重新抛出异常，以便上层（如 FastAPI 的异常处理器）能处理它
            raise
