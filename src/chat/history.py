from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Text, DateTime, Integer, Boolean, JSON, select, delete as sa_delete, func
from src.config.settings import settings
from src.models.schemas import DocumentInfo
import logging

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=True, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    message_id: Mapped[int] = mapped_column(Integer, nullable=False)
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=True, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class DocumentRecord(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(256), nullable=False)
    file_format: Mapped[str] = mapped_column(String(32), default="pdf")
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    num_texts: Mapped[int] = mapped_column(Integer, default=0)
    num_tables: Mapped[int] = mapped_column(Integer, default=0)
    num_images: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="processed")
    user_id: Mapped[int] = mapped_column(Integer, nullable=True, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class ChatHistoryManager:
    def __init__(self):
        self.engine = create_async_engine(settings.postgres_url, pool_size=settings.postgres_pool_size)
        self.async_session = async_sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)

    async def init_db(self):
        try:
            from alembic.config import Config
            from alembic.command import upgrade
            alembic_cfg = Config("alembic.ini")
            async with self.engine.begin() as conn:
                await conn.run_sync(lambda sync_conn: upgrade(alembic_cfg, "head"))
            logger.info("PostgreSQL migrations applied")
        except Exception:
            logger.warning("Alembic migration failed, falling back to create_all")
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("PostgreSQL tables created")

    async def add_message(self, session_id: str, role: str, content: str, metadata: Optional[dict] = None, user_id: int = 0):
        async with self.async_session() as session:
            msg = Conversation(
                session_id=session_id,
                role=role,
                content=content,
                metadata=metadata,
                user_id=user_id,
            )
            session.add(msg)
            await session.commit()

    async def get_history(self, session_id: str, limit: int = 20, user_id: int = 0) -> list[dict]:
        async with self.async_session() as session:
            stmt = (
                select(Conversation)
                .where(Conversation.session_id == session_id, Conversation.user_id == user_id)
                .order_by(Conversation.created_at.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            messages = result.scalars().all()
            return [
                {"role": m.role, "content": m.content, "metadata": m.metadata}
                for m in reversed(messages)
            ]

    async def record_document(self, filename: str, file_path: str, file_format: str, stats: dict, user_id: int = 0) -> int:
        async with self.async_session() as session:
            doc = DocumentRecord(
                filename=filename,
                file_path=file_path,
                file_format=file_format,
                num_texts=stats.get("texts", 0),
                num_tables=stats.get("tables", 0),
                num_images=stats.get("images", 0),
                user_id=user_id,
            )
            session.add(doc)
            await session.commit()
            await session.refresh(doc)
            return doc.id

    async def list_documents(self, skip: int = 0, limit: int = 50, user_id: int = 0) -> tuple[list[DocumentInfo], int]:
        async with self.async_session() as session:
            count_stmt = select(func.count()).select_from(DocumentRecord).where(DocumentRecord.user_id == user_id)
            total = (await session.execute(count_stmt)).scalar() or 0

            stmt = (
                select(DocumentRecord)
                .where(DocumentRecord.user_id == user_id)
                .order_by(DocumentRecord.created_at.desc())
                .offset(skip).limit(limit)
            )
            result = await session.execute(stmt)
            docs = result.scalars().all()
            return [
                DocumentInfo(
                    id=d.id,
                    filename=d.filename,
                    file_format=d.file_format,
                    file_path=d.file_path,
                    num_texts=d.num_texts,
                    num_tables=d.num_tables,
                    num_images=d.num_images,
                    status=d.status,
                    created_at=d.created_at,
                )
                for d in docs
            ], total

    async def get_user_document(self, doc_id: int, user_id: int) -> Optional[DocumentInfo]:
        async with self.async_session() as session:
            stmt = select(DocumentRecord).where(DocumentRecord.id == doc_id, DocumentRecord.user_id == user_id)
            result = await session.execute(stmt)
            d = result.scalar_one_or_none()
            if not d:
                return None
            return DocumentInfo(
                id=d.id,
                filename=d.filename,
                file_format=d.file_format,
                file_path=d.file_path,
                num_texts=d.num_texts,
                num_tables=d.num_tables,
                num_images=d.num_images,
                status=d.status,
                created_at=d.created_at,
            )

    async def delete_user_document_record(self, doc_id: int, user_id: int) -> bool:
        async with self.async_session() as session:
            stmt = select(DocumentRecord).where(DocumentRecord.id == doc_id, DocumentRecord.user_id == user_id)
            result = await session.execute(stmt)
            d = result.scalar_one_or_none()
            if not d:
                return False
            await session.delete(d)
            await session.commit()
            return True

    async def get_document(self, doc_id: int) -> Optional[DocumentInfo]:
        async with self.async_session() as session:
            stmt = select(DocumentRecord).where(DocumentRecord.id == doc_id)
            result = await session.execute(stmt)
            d = result.scalar_one_or_none()
            if not d:
                return None
            return DocumentInfo(
                id=d.id,
                filename=d.filename,
                file_format=d.file_format,
                file_path=d.file_path,
                num_texts=d.num_texts,
                num_tables=d.num_tables,
                num_images=d.num_images,
                status=d.status,
                created_at=d.created_at,
            )

    async def delete_document_record(self, doc_id: int) -> bool:
        async with self.async_session() as session:
            stmt = select(DocumentRecord).where(DocumentRecord.id == doc_id)
            result = await session.execute(stmt)
            d = result.scalar_one_or_none()
            if not d:
                return False
            await session.delete(d)
            await session.commit()
            return True

    async def add_feedback(self, session_id: str, message_id: int, rating: int, comment: Optional[str] = None, user_id: int = 0):
        async with self.async_session() as session:
            fb = Feedback(
                session_id=session_id,
                message_id=message_id,
                rating=rating,
                comment=comment,
                user_id=user_id,
            )
            session.add(fb)
            await session.commit()

    async def get_feedback(self, session_id: str, limit: int = 50, user_id: int = 0) -> list[dict]:
        async with self.async_session() as session:
            stmt = (
                select(Feedback)
                .where(Feedback.session_id == session_id, Feedback.user_id == user_id)
                .order_by(Feedback.created_at.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            return [
                {"id": f.id, "message_id": f.message_id, "rating": f.rating, "comment": f.comment}
                for f in result.scalars().all()
            ]

    async def close(self):
        await self.engine.dispose()
