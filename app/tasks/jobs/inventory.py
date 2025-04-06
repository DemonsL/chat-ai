from uuid import UUID

from app.tasks.base import async_task, get_async_db_session


@async_task(name="tasks.inventory.reduce_stock", queue="inventory")
async def reduce_stock_task(self, product_id: str, quantity: int, order_id: str = None):
    """
    减少商品库存

    参数:
        product_id: 商品ID
        quantity: 数量
        order_id: 订单ID(可选)
    """
    async for session in get_async_db_session():
        from app.db.repositories.product_repository import ProductRepository

        product_repo = ProductRepository(session)
        product_id_uuid = UUID(product_id)

        # 更新库存
        product = await product_repo.get_by_id(product_id_uuid)
        if not product:
            raise ValueError(f"商品不存在: {product_id}")

        if product.stock < quantity:
            raise ValueError(f"商品库存不足: {product.stock} < {quantity}")

        # 更新库存
        await product_repo.update_stock(product_id_uuid, -quantity)

        # 记录库存变更
        await product_repo.add_stock_record(
            product_id=product_id_uuid,
            quantity=-quantity,
            order_id=UUID(order_id) if order_id else None,
            operation_type="order",
        )

        return {
            "success": True,
            "product_id": product_id,
            "remaining_stock": product.stock - quantity,
        }


@async_task(name="tasks.inventory.check_low_stock", queue="scheduled")
async def check_low_stock_task(self, threshold: int = 10):
    """
    检查低库存商品并发送通知

    参数:
        threshold: 库存阈值
    """
    async for session in get_async_db_session():
        from app.db.repositories.product_repository import ProductRepository
        from app.tasks.jobs.email import send_notification_task

        product_repo = ProductRepository(session)

        # 获取低库存商品
        low_stock_products = await product_repo.get_low_stock_products(threshold)

        if low_stock_products:
            # 发送低库存通知
            product_list = "\n".join(
                [f"- {p.name}: 剩余库存 {p.stock}" for p in low_stock_products]
            )

            await send_notification_task.delay(
                subject="低库存提醒",
                content=f"以下商品库存不足，请及时补货：\n{product_list}",
                recipients=["inventory@example.com"],
            )

        return {
            "checked_count": len(low_stock_products),
            "low_stock_count": len(low_stock_products),
        }
