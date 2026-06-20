from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Text, DateTime, Integer, JSON, select
from src.config.settings import settings
import json
import logging

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class DocumentRecord(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(256), nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    num_texts: Mapped[int] = mapped_column(Integer, default=0)
    num_tables: Mapped[int] = mapped_column(Integer, default=0)
    num_images: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="processed")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class ChatHistoryManager:
    def __init__(self):
        self.engine = create_async_engine(settings.postgres_url, pool_size=settings.postgres_pool_size)
        self.async_session = async_sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)

    async def init_db(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("PostgreSQL tables created")

    async def add_message(self, session_id: str, role: str, content: str, metadata: Optional[dict] = None):
        async with self.async_session() as session:
            msg = Conversation(
                session_id=session_id,
                role=role,
                content=content,
                metadata=metadata,
            )
            session.add(msg)
            await session.commit()

    async def get_history(self, session_id: str, limit: int = 20) -> list[dict]:
        async with self.async_session() as session:
            stmt = (
                select(Conversation)
                .where(Conversation.session_id == session_id)
                .order_by(Conversation.created_at.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            messages = result.scalars().all()
            return [
                {"role": m.role, "content": m.content, "metadata": m.metadata}
                for m in reversed(messages)
            ]

    async def record_document(self, filename: str, file_path: str, stats: dict):
        async with self.async_session() as session:
            doc = DocumentRecord(
                filename=filename,
                file_path=file_path,
                num_texts=stats.get("texts", 0),
                num_tables=stats.get("tables", 0),
                num_images=stats.get("images", 0),
            )
            session.add(doc)
            await session.commit()

    async def close(self):
        await self.engine.dispose()
