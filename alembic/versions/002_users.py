"""Add users table + user_id columns

Revision ID: 002
Revises: 001
Create Date: 2026-06-21

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("email", sa.String(255), nullable=False, unique=True, index=True),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("name", sa.String(128), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.add_column("conversations", sa.Column("user_id", sa.Integer(), nullable=True, server_default="0"))
    op.add_column("feedback", sa.Column("user_id", sa.Integer(), nullable=True, server_default="0"))
    op.add_column("documents", sa.Column("user_id", sa.Integer(), nullable=True, server_default="0"))


def downgrade() -> None:
    op.drop_column("documents", "user_id")
    op.drop_column("feedback", "user_id")
    op.drop_column("conversations", "user_id")
    op.drop_table("users")
