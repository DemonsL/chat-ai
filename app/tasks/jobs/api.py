import json
from typing import Dict, Optional

import httpx
from loguru import logger

from app.tasks.base import async_task


@async_task(
    name="tasks.api.call_external_api",
    queue="api_calls",
    max_retries=3,
    retry_backoff=True,
)
async def call_external_api_task(
    self,
    url: str,
    method: str = "GET",
    data: Optional[Dict] = None,
    headers: Optional[Dict] = None,
    timeout: int = 30,
    callback_task: Optional[str] = None,
):
    """
    调用外部API的异步任务

    参数:
        url: API地址
        method: HTTP方法
        data: 请求数据
        headers: 请求头
        timeout: 超时时间(秒)
        callback_task: 回调任务名称
    """
    logger.info(f"调用外部API: {method} {url}")

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            if method.upper() == "GET":
                response = await client.get(url, params=data, headers=headers)
            elif method.upper() == "POST":
                response = await client.post(url, json=data, headers=headers)
            elif method.upper() == "PUT":
                response = await client.put(url, json=data, headers=headers)
            elif method.upper() == "DELETE":
                response = await client.delete(url, json=data, headers=headers)
            else:
                raise ValueError(f"不支持的HTTP方法: {method}")

            # 确保请求成功
            response.raise_for_status()

            try:
                result = response.json()
            except json.JSONDecodeError:
                result = {"text": response.text}

            logger.info(f"API调用成功: {response.status_code}")

            # 如果有回调任务，触发它
            if callback_task:
                from app.tasks.celery import celery_app

                # 触发回调任务
                callback_data = {
                    "success": True,
                    "status_code": response.status_code,
                    "result": result,
                }
                celery_app.send_task(
                    callback_task, kwargs={"response_data": callback_data}
                )

            return {
                "success": True,
                "status_code": response.status_code,
                "result": result,
            }

        except httpx.HTTPStatusError as e:
            logger.error(
                f"API调用HTTP错误: {e.response.status_code} - {e.response.text}"
            )

            # 如果有回调任务，也要触发它，但标记为失败
            if callback_task:
                from app.tasks.celery import celery_app

                error_data = {
                    "success": False,
                    "status_code": e.response.status_code,
                    "error": e.response.text,
                }
                celery_app.send_task(
                    callback_task, kwargs={"response_data": error_data}
                )

            raise e

        except httpx.RequestError as e:
            logger.error(f"API调用请求错误: {str(e)}")

            # 触发错误回调
            if callback_task:
                from app.tasks.celery import celery_app

                error_data = {
                    "success": False,
                    "error": str(e),
                }
                celery_app.send_task(
                    callback_task, kwargs={"response_data": error_data}
                )

            raise e
