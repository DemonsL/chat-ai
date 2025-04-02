from typing import List, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundException, PermissionDeniedException
from app.db.models.conversation import Conversation
from app.db.repositories.conversation_repository import ConversationRepository
from app.db.repositories.model_config_repository import ModelConfigRepository
from app.db.repositories.user_file_repository import UserFileRepository
from app.schemas.conversation import ConversationCreate, ConversationUpdate


class ConversationService:
    """
    会话服务，处理会话的创建、获取和更新
    """
    def __init__(self, db_session: AsyncSession):
        self.conversation_repo = ConversationRepository(db_session)
        self.model_config_repo = ModelConfigRepository(db_session)
        self.user_file_repo = UserFileRepository(db_session)
        self.db_session = db_session

    async def create(self, user_id: UUID, conv_create: ConversationCreate) -> Conversation:
        """
        创建新会话
        """
        # 检查模型是否存在
        model = await self.model_config_repo.get_by_model_id(conv_create.model_id)
        if not model or not model.is_active:
            raise NotFoundException(detail="指定的模型不存在或已停用")

        # 如果指定了文件，检查文件是否存在且属于该用户
        if conv_create.file_ids:
            files = await self.user_file_repo.get_by_ids_for_user(conv_create.file_ids, user_id)
            if len(files) != len(conv_create.file_ids):
                raise NotFoundException(detail="一个或多个指定的文件不存在")

        # 创建会话
        conv_data = conv_create.model_dump(exclude={"file_ids"})
        conversation = await self.conversation_repo.create(
            obj_in=ConversationCreate(
                **conv_data,
                user_id=user_id
            )
        )

        # 如果指定了文件，更新会话与文件的关联
        if conv_create.file_ids:
            await self.conversation_repo.update_files(conversation, conv_create.file_ids)
            # 重新获取会话以包含更新后的关联
            conversation = await self.conversation_repo.get_by_id(conversation.id)

        return conversation

    async def get_by_id(self, conversation_id: UUID, user_id: UUID) -> Optional[Conversation]:
        """
        获取会话详情
        """
        # 获取会话并检查权限
        conversation = await self.conversation_repo.get_by_id_for_user(conversation_id, user_id)
        if not conversation:
            raise NotFoundException(detail="会话不存在")
        return conversation

    async def get_by_user_id(self, user_id: UUID, skip: int = 0, limit: int = 100) -> List[Conversation]:
        """
        获取用户的所有会话
        """
        return await self.conversation_repo.get_by_user_id(user_id, skip=skip, limit=limit)

    async def update(self, conversation_id: UUID, user_id: UUID, conv_update: ConversationUpdate) -> Conversation:
        """
        更新会话设置
        """
        # 获取会话并检查权限
        conversation = await self.conversation_repo.get_by_id_for_user(conversation_id, user_id)
        if not conversation:
            raise NotFoundException(detail="会话不存在")

        # 如果更新了模型，检查模型是否存在
        if conv_update.model_id:
            model = await self.model_config_repo.get_by_model_id(conv_update.model_id)
            if not model or not model.is_active:
                raise NotFoundException(detail="指定的模型不存在或已停用")

        # 提取文件 ID
        file_ids = None
        if hasattr(conv_update, "file_ids"):
            file_ids = conv_update.file_ids
            conv_update_dict = conv_update.model_dump(exclude={"file_ids"})
        else:
            conv_update_dict = conv_update.model_dump()

        # 更新会话设置
        updated_conversation = await self.conversation_repo.update(
            db_obj=conversation,
            obj_in=conv_update_dict
        )

        # 如果更新了文件关联
        if file_ids is not None:
            # 验证文件所有权
            if file_ids:
                files = await self.user_file_repo.get_by_ids_for_user(file_ids, user_id)
                if len(files) != len(file_ids):
                    raise NotFoundException(detail="一个或多个指定的文件不存在")
            
            # 更新会话与文件的关联
            await self.conversation_repo.update_files(updated_conversation, file_ids)
            # 重新获取会话以包含更新后的关联
            updated_conversation = await self.conversation_repo.get_by_id(updated_conversation.id)

        return updated_conversation

    async def delete(self, conversation_id: UUID, user_id: UUID) -> bool:
        """
        删除会话
        """
        # 获取会话并检查权限
        conversation = await self.conversation_repo.get_by_id_for_user(conversation_id, user_id)
        if not conversation:
            raise NotFoundException(detail="会话不存在")

        # 删除会话
        await self.conversation_repo.delete(id=conversation_id)
        return True