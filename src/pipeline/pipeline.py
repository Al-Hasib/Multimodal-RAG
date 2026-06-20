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

    async def ingest(self, source: str, file_format: Optional[FileFormat] = None) -> dict:
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
        doc_id = await self.chat_history.record_document(filename, source, fmt, stats)

        logger.info(f"Document ingested: id={doc_id}, group={group_id}, {source}")
        return {"id": doc_id, "group_id": group_id, "filename": filename, **stats}

    async def ingest_file(self, filename: str, content: bytes) -> dict:
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
        doc_id = await self.chat_history.record_document(filename, temp_path, fmt.value, stats)
        return {"id": doc_id, "group_id": group_id, "filename": filename, **stats}

    async def list_documents(self, skip: int = 0, limit: int = 50) -> tuple[list[DocumentInfo], int]:
        return await self.chat_history.list_documents(skip=skip, limit=limit)

    async def get_document(self, doc_id: int) -> Optional[DocumentInfo]:
        return await self.chat_history.get_document(doc_id)

    async def delete_document(self, doc_id: int) -> bool:
        doc = await self.chat_history.get_document(doc_id)
        if not doc:
            return False

        # Delete from Qdrant + Redis docstore
        self.cache.invalidate_all()
        await self.chat_history.delete_document_record(doc_id)
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

    async def aquery(self, question: str, k: int = 5, session_id: str | None = None) -> QueryResponse:
        cached = self.cache.get(question, k)
        if cached:
            return QueryResponse(**cached)

        chat_history = []
        if session_id:
            chat_history = await self.chat_history.get_history(session_id)

        result = self.rag_chain.invoke_with_sources(question, chat_history=chat_history)

        if session_id:
            await self.chat_history.add_message(session_id, "user", question)
            await self.chat_history.add_message(session_id, "assistant", result.answer)

        self.cache.set(question, k, result.model_dump())
        return result

    async def astream(self, question: str, session_id: str | None = None):
        chat_history = []
        if session_id:
            chat_history = await self.chat_history.get_history(session_id)
            await self.chat_history.add_message(session_id, "user", question)

        full = ""
        async for chunk in self.rag_chain.astream(question, chat_history=chat_history, session_id=session_id):
            full += chunk
            yield chunk

        if session_id:
            await self.chat_history.add_message(session_id, "assistant", full)
