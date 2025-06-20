"""add_google_to_model_provider_enum

Revision ID: 565ce19416ff
Revises: 19d70f4103d1
Create Date: 2025-06-19 18:23:12.395226

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '565ce19416ff'
down_revision: Union[str, None] = '19d70f4103d1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # 添加 'google_genai' 值到 model_provider 枚举类型
    op.execute("ALTER TYPE model_provider ADD VALUE 'google_genai'")


def downgrade() -> None:
    """Downgrade schema."""
    # 注意：PostgreSQL 不支持直接删除枚举值，需要重建枚举类型
    # 这个降级操作比较复杂，通常在生产环境中需要谨慎处理
    
    # 临时重命名旧枚举
    op.execute("ALTER TYPE model_provider RENAME TO model_provider_old")
    
    # 创建新的枚举类型（不包含 google_genai）
    op.execute("CREATE TYPE model_provider AS ENUM ('openai', 'anthropic', 'deepseek', 'other')")
    
    # 更新表列类型（如果表中没有使用 'google' 值的记录）
    op.execute("""
        ALTER TABLE modelconfig 
        ALTER COLUMN provider TYPE model_provider 
        USING provider::text::model_provider
    """)
    
    # 删除旧枚举类型
    op.execute("DROP TYPE model_provider_old")
