import os
import logging
from typing import Optional
from src.ingestion.document_processor import DocumentProcessor
from src.summarization.orchestrator import SummarizationOrchestrator
from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.retriever import MultiModalRetriever
from src.generation.rag_chain import RAGChain
from src.chat.history import ChatHistoryManager
from src.core.cache import QueryCache
from src.core.metrics import track_request, get_tracker, rag_cache_hits, rag_cache_misses, rag_document_count
from src.models.schemas import QueryResponse, DocumentInfo, FileFormat

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self):
        logger.info("Initializing RAG Pipeline")
        self.doc_processor = DocumentProcessor()
        self.summarizer = SummarizationOrchestrator()
        self.vector_store = VectorStoreManager()
        self.retriever = MultiModalRetriever(self.vector_store.get_retriever())
        self.rag_chain = RAGChain(self.retriever)
        self.chat_history = ChatHistoryManager()
        self.cache = QueryCache()

    async def cleanup(self):
        await self.chat_history.cleanup_expired()

    async def ingest(self, source: str, file_format: Optional[FileFormat] = None, user_id: int = 0, tags: list[str] | None = None) -> dict:
        logger.info(f"Ingesting document: {source}")
        document = await self.doc_processor.extract(source, file_format=file_format)
        summaries = self.summarizer.summarize(document)
        group_id = self.vector_store.index_document(document, summaries)

        stats = {
            "texts": len(document.texts),
            "tables": len(document.tables),
            "images": len(document.images),
        }
        filename = os.path.basename(source) if not source.startswith("http") else source
        fmt = (file_format or self.doc_processor.detect_format(source)).value
        doc_id = await self.chat_history.record_document(filename, source, fmt, stats, user_id=user_id, group_id=group_id, tags=tags)

        logger.info(f"Document ingested: id={doc_id}, group={group_id}, {source}")
        return {"id": doc_id, "group_id": group_id, "filename": filename, **stats}

    async def ingest_file(self, filename: str, content: bytes, user_id: int = 0, tags: list[str] | None = None) -> dict:
        logger.info(f"Ingesting file: {filename}")
        fmt = self.doc_processor.detect_format(filename)
        if fmt == FileFormat.UNKNOWN:
            raise ValueError(f"Unsupported file format: {filename}")

        document = await self.doc_processor.extract_from_bytes(filename, content, file_format=fmt)
        summaries = self.summarizer.summarize(document)
        group_id = self.vector_store.index_document(document, summaries)

        stats = {
            "texts": len(document.texts),
            "tables": len(document.tables),
            "images": len(document.images),
        }
        temp_path = f"/tmp/{filename}"
        doc_id = await self.chat_history.record_document(filename, temp_path, fmt.value, stats, user_id=user_id, group_id=group_id, tags=tags)
        return {"id": doc_id, "group_id": group_id, "filename": filename, **stats}

    async def search_documents(self, query: str, user_id: int = 0, skip: int = 0, limit: int = 50) -> tuple[list[DocumentInfo], int]:
        return await self.chat_history.search_documents(query, user_id=user_id, skip=skip, limit=limit)

    async def update_document_tags(self, doc_id: int, tags: list[str], user_id: int = 0) -> bool:
        doc = await self.chat_history.get_user_document(doc_id, user_id=user_id)
        if not doc:
            return False
        return await self.chat_history.update_document_tags(doc_id, tags)

    async def list_documents(self, skip: int = 0, limit: int = 50, user_id: int = 0) -> tuple[list[DocumentInfo], int]:
        return await self.chat_history.list_documents(skip=skip, limit=limit, user_id=user_id)

    async def get_document(self, doc_id: int, user_id: int = 0) -> Optional[DocumentInfo]:
        return await self.chat_history.get_user_document(doc_id, user_id=user_id)

    async def delete_document(self, doc_id: int, user_id: int = 0) -> bool:
        doc = await self.chat_history.get_user_document(doc_id, user_id=user_id)
        if not doc:
            return False

        if doc.group_id:
            self.vector_store.delete_document(doc.group_id)

        self.cache.invalidate_all()
        await self.chat_history.delete_user_document_record(doc_id, user_id=user_id)
        logger.info(f"Document {doc_id} fully deleted (group={doc.group_id})")
        return True

    async def reindex_all(self) -> int:
        docs, _ = await self.chat_history.list_documents(limit=10000)
        self.vector_store.clear_collection()
        count = 0
        for doc in docs:
            if os.path.exists(doc.file_path):
                try:
                    await self.ingest(doc.file_path)
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to reindex {doc.filename}: {e}")
        return count

    async def aquery(self, question: str, k: int = 5, session_id: str | None = None, user_id: int = 0, metadata_filter: dict | None = None) -> QueryResponse:
        with track_request(question=question, user_id=user_id, session_id=session_id or ""):
            tracker = get_tracker()
            tracker.checkpoint("start")

            cached = self.cache.get(question, k)
            if cached:
                tracker.cache_hit = True
                tracker.checkpoint("cache_hit")
                tracker.finalize()
                rag_cache_hits.inc()
                return QueryResponse(**cached)
            tracker.checkpoint("cache_miss")
            rag_cache_misses.inc()

            chat_history = []
            if session_id:
                chat_history = await self.chat_history.get_history(session_id, user_id=user_id)
            tracker.checkpoint("history_loaded")

            result = self.rag_chain.invoke_with_sources(question, chat_history=chat_history, metadata_filter=metadata_filter)
            tracker.checkpoint("response_generated")

            guardrail_blocked = result.guardrail_input and not result.guardrail_input.get("passed", True)
            tracker.guardrail_blocked = guardrail_blocked

            if not guardrail_blocked and session_id:
                await self.chat_history.add_message(session_id, "user", question, user_id=user_id)
                await self.chat_history.add_message(session_id, "assistant", result.answer, user_id=user_id)

            if not guardrail_blocked:
                self.cache.set(question, k, result.model_dump())

            tracker.checkpoint("done")

        return result

    async def astream(self, question: str, session_id: str | None = None, user_id: int = 0, metadata_filter: dict | None = None):
        with track_request(question=question, user_id=user_id, session_id=session_id or ""):
            tracker = get_tracker()
            tracker.checkpoint("start")
        chat_history = []
        if session_id:
            chat_history = await self.chat_history.get_history(session_id, user_id=user_id)
            await self.chat_history.add_message(session_id, "user", question, user_id=user_id)

        full = ""
        async for chunk in self.rag_chain.astream(question, chat_history=chat_history, session_id=session_id, metadata_filter=metadata_filter):
            full += chunk
            yield chunk

        if session_id:
            await self.chat_history.add_message(session_id, "assistant", full, user_id=user_id)

        tracker.checkpoint("done")
