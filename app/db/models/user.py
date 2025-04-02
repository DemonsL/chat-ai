from sqlalchemy import Boolean, Column, String
from app.db.models.base import Base


class User(Base):
    """
    用户模型
    """
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    
    # 用户偏好设置
    default_model_id = Column(String, nullable=True)  # 默认使用的模型ID
    ui_theme = Column(String, default="light")  # 用户界面主题偏好 