from uuid import UUID

from loguru import logger

from app.tasks.base import async_task, get_async_db_session


@async_task(name="tasks.credits.reduce_credits", queue="credits", max_retries=2)
async def reduce_credits_task(self, user_id: str, amount: int, description: str = None):
    """
    减少用户积分

    参数:
        user_id: 用户ID
        amount: 积分数量
        description: 操作描述
    """
    async for session in get_async_db_session():
        from app.db.repositories.user_repository import UserRepository

        user_repo = UserRepository(session)
        user_id_uuid = UUID(user_id)

        # 获取用户
        user = await user_repo.get_by_id(user_id_uuid)
        if not user:
            raise ValueError(f"用户不存在: {user_id}")

        if user.credits < amount:
            raise ValueError(f"用户积分不足: {user.credits} < {amount}")

        # 更新积分
        await user_repo.update_credits(user_id_uuid, -amount)

        # 添加积分记录
        await user_repo.add_credit_record(
            user_id=user_id_uuid,
            amount=-amount,
            description=description or "积分消费",
            operation_type="consume",
        )

        return {
            "success": True,
            "user_id": user_id,
            "remaining_credits": user.credits - amount,
        }


@async_task(name="tasks.credits.award_credits", queue="credits", max_retries=2)
async def award_credits_task(self, user_id: str, amount: int, reason: str = None):
    """
    奖励用户积分

    参数:
        user_id: 用户ID
        amount: 积分数量
        reason: 奖励原因
    """
    async for session in get_async_db_session():
        from app.db.repositories.user_repository import UserRepository

        user_repo = UserRepository(session)
        user_id_uuid = UUID(user_id)

        # 更新积分
        await user_repo.update_credits(user_id_uuid, amount)

        # 添加积分记录
        await user_repo.add_credit_record(
            user_id=user_id_uuid,
            amount=amount,
            description=reason or "积分奖励",
            operation_type="award",
        )

        # 获取更新后的用户
        user = await user_repo.get_by_id(user_id_uuid)

        return {"success": True, "user_id": user_id, "current_credits": user.credits}
