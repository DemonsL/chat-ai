"""add_search_mode_to_conversation

Revision ID: 19d70f4103d1
Revises: 7bd61077962b
Create Date: 2025-06-16 22:06:47.285212

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '19d70f4103d1'
down_revision: Union[str, None] = '7bd61077962b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add 'search' to the conversation_mode enum
    op.execute("ALTER TYPE conversation_mode ADD VALUE 'search'")


def downgrade() -> None:
    """Downgrade schema."""
    # Note: PostgreSQL doesn't support removing enum values directly
    # This would require recreating the enum type and updating all references
    # For simplicity, we'll leave the enum as is during downgrade
    # In production, you might want to implement a more complex downgrade process
    pass
