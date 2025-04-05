import logging
import sys

from loguru import logger

from app.core.config import settings


class InterceptHandler(logging.Handler):
    """
    拦截标准库日志并重定向到Loguru
    """

    def emit(self, record):
        # 获取对应的Loguru级别
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # 获取调用者信息
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging():
    """
    配置应用日志
    """
    # 移除所有处理器
    logger.remove()

    # 添加控制台输出
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="INFO",
        colorize=True,
    )

    # 添加文件输出
    logger.add(
        "logs/app.log",
        rotation="10 MB",  # 每10MB切换日志文件
        retention="1 month",  # 保留1个月的日志
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="DEBUG",
    )

    # 配置Sentry集成（如果有DSN）
    if settings.SENTRY_DSN:
        import sentry_sdk
        from sentry_sdk.integrations.logging import LoggingIntegration

        sentry_logging = LoggingIntegration(
            level=logging.INFO,  # 捕获正常日志的最小级别
            event_level=logging.ERROR,  # 发送到Sentry的最小级别
        )

        sentry_sdk.init(
            dsn=settings.SENTRY_DSN,
            traces_sample_rate=0.1,  # 性能监控采样率
            integrations=[sentry_logging],
        )

    # 拦截Python日志
    logging.basicConfig(handlers=[InterceptHandler()], level=0)

    # 替换常见库的日志处理器
    for _log in ["uvicorn", "uvicorn.error", "fastapi"]:
        _logger = logging.getLogger(_log)
        _logger.handlers = [InterceptHandler()]

    logger.info("日志系统已初始化")
