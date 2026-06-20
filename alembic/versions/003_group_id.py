"""Add group_id to documents table

Revision ID: 003
Revises: 002
Create Date: 2026-06-21

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("group_id", sa.String(36), nullable=True))


def downgrade() -> None:
    op.drop_column("documents", "group_id")
