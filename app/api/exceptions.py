from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import ValidationError

# from app.domain.exceptions import DomainException


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    处理HTTP异常
    """
    logger.warning(f"HTTP异常: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    处理请求验证异常
    """
    logger.warning(f"验证错误: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "请求参数验证失败",
            "errors": exc.errors(),
        },
    )


# async def domain_exception_handler(
#     request: Request, exc: DomainException
# ) -> JSONResponse:
#     """
#     处理领域异常
#     """
#     logger.warning(f"领域异常: {exc}")
#     return JSONResponse(
#         status_code=status.HTTP_400_BAD_REQUEST,
#         content={"detail": str(exc)},
#     )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    处理通用异常
    """
    logger.exception(f"未处理异常: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "服务器内部错误"},
    )


def setup_exception_handlers(app: FastAPI) -> None:
    """
    配置异常处理器
    """
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ValidationError, validation_exception_handler)
    # app.add_exception_handler(DomainException, domain_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
