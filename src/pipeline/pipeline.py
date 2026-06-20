import os
import logging
from src.ingestion.pdf_processor import PDFProcessor
from src.summarization.orchestrator import SummarizationOrchestrator
from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.retriever import MultiModalRetriever
from src.generation.rag_chain import RAGChain
from src.chat.history import ChatHistoryManager
from src.core.cache import QueryCache
from src.models.schemas import QueryResponse

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self):
        logger.info("Initializing RAG Pipeline")
        self.pdf_processor = PDFProcessor()
        self.summarizer = SummarizationOrchestrator()
        self.vector_store = VectorStoreManager()
        self.retriever = MultiModalRetriever(self.vector_store.get_retriever())
        self.rag_chain = RAGChain(self.retriever)
        self.chat_history = ChatHistoryManager()
        self.cache = QueryCache()

    async def ingest(self, pdf_path: str) -> dict:
        logger.info(f"Ingesting document: {pdf_path}")
        document = await self.pdf_processor.extract(pdf_path)
        summaries = self.summarizer.summarize(document)
        self.vector_store.index_document(document, summaries)

        stats = {
            "texts": len(document.texts),
            "tables": len(document.tables),
            "images": len(document.images),
        }
        filename = os.path.basename(pdf_path)
        await self.chat_history.record_document(filename, pdf_path, stats)

        logger.info(f"Document ingested successfully: {pdf_path}")
        return {"filename": filename, **stats}

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
