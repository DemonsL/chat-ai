from typing import Callable

import prometheus_client
from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, Gauge, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Match

# 定义指标
REQUEST_COUNT = Counter(
    "app_request_count", "应用请求总数", ["method", "endpoint", "status_code"]
)

REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "请求处理时间（秒）", ["method", "endpoint"]
)

ACTIVE_REQUESTS = Gauge("app_active_requests", "当前活跃请求数", ["method", "endpoint"])

ERROR_COUNT = Counter(
    "app_error_count", "应用错误总数", ["method", "endpoint", "error_type"]
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    Prometheus监控中间件
    收集HTTP请求指标
    """

    def get_path_template(self, request: Request) -> str:
        """辅助函数，获取用于Prometheus的路径模板"""
        for route in request.app.routes:
            # 使用 route.matches 检查请求是否匹配路由
            match, _ = route.matches(request.scope)
            if match == Match.FULL:
                # 如果完全匹配，返回路由的路径模板
                return route.path
        # 如果没有找到匹配的路由模板，则回退到原始路径 (可能包含路径参数)
        return request.url.path

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        method = request.method
        # 使用路由模板作为 endpoint 标签
        endpoint = self.get_path_template(request)

        ACTIVE_REQUESTS.labels(method=method, endpoint=endpoint).inc()
        response = None  # 初始化 response 变量
        status_code = 500  # 默认为500，以防异常阻止设置状态码

        try:
            # 记录请求处理时间
            with REQUEST_LATENCY.labels(method=method, endpoint=endpoint).time():
                response = await call_next(request)
            status_code = response.status_code  # 从成功响应中获取状态码
        except Exception as e:
            # 记录错误计数器
            ERROR_COUNT.labels(
                method=method, endpoint=endpoint, error_type=type(e).__name__
            ).inc()
            # status_code 保持 500
            raise  # 重新抛出异常，以便 FastAPI 的异常处理器处理
        finally:
            # 减少活跃请求数
            ACTIVE_REQUESTS.labels(method=method, endpoint=endpoint).dec()
            # 仅在成功生成响应或已知异常状态码时记录请求计数
            # (注意：如果在异常处理中设置了 response，这里的 status_code 需相应调整)
            if response is not None:
                REQUEST_COUNT.labels(
                    method=method, endpoint=endpoint, status_code=status_code
                ).inc()
            # 如果需要，可以在捕获特定 FastAPI HTTPExceptions 时也记录 REQUEST_COUNT

        return response


def setup_metrics(app: FastAPI) -> None:
    """
    配置Prometheus监控
    """
    # 添加Prometheus中间件
    app.add_middleware(PrometheusMiddleware)

    # 创建指标端点
    @app.get("/metrics")
    async def metrics():
        return Response(prometheus_client.generate_latest(), media_type="text/plain")
