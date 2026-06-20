from src.ingestion.pdf_processor import PDFProcessor
from src.summarization.orchestrator import SummarizationOrchestrator
from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.retriever import MultiModalRetriever
from src.generation.rag_chain import RAGChain
from src.models.schemas import QueryResponse
import logging

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self):
        logger.info("Initializing RAG Pipeline")
        self.pdf_processor = PDFProcessor()
        self.summarizer = SummarizationOrchestrator()
        self.vector_store = VectorStoreManager()
        self.retriever = MultiModalRetriever(self.vector_store.get_retriever())
        self.rag_chain = RAGChain(self.retriever)

    def ingest(self, pdf_path: str) -> None:
        logger.info(f"Ingesting document: {pdf_path}")
        document = self.pdf_processor.extract(pdf_path)
        summaries = self.summarizer.summarize(document)
        self.vector_store.index_document(document, summaries)
        logger.info(f"Document ingested successfully: {pdf_path}")

    def query(self, question: str, k: int = 5) -> QueryResponse:
        return self.rag_chain.invoke_with_sources(question, k=k)
