import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def mock_pdf_path(tmp_path):
    path = tmp_path / "test.pdf"
    path.write_text("%PDF-1.4 mock content")
    return str(path)


@pytest.fixture
def mock_settings():
    with patch("src.config.settings.settings") as mock:
        mock.qdrant_url = "http://localhost:6333"
        mock.redis_url = "redis://localhost:6379/0"
        mock.postgres_url = "postgresql+asyncpg://postgres:postgres@localhost:5432/multimodal_rag"
        mock.openai_api_key = "sk-test"
        mock.groq_api_key = "gsk-test"
        mock.retrieval_k = 5
        mock.rerank_enabled = False
        mock.query_transformer_enabled = False
        mock.redis_cache_enabled = False
        mock.langfuse_enabled = False
        yield mock


@pytest.fixture
def mock_pipeline(mock_settings):
    with patch("src.pipeline.pipeline.PDFProcessor") as mock_proc, \
         patch("src.pipeline.pipeline.SummarizationOrchestrator") as mock_sum, \
         patch("src.pipeline.pipeline.VectorStoreManager") as mock_vs, \
         patch("src.pipeline.pipeline.MultiModalRetriever") as mock_ret, \
         patch("src.pipeline.pipeline.RAGChain") as mock_rag, \
         patch("src.pipeline.pipeline.ChatHistoryManager") as mock_chat:

        mock_proc.return_value.extract = AsyncMock()
        mock_chat.return_value.init_db = AsyncMock()
        mock_chat.return_value.get_history = AsyncMock(return_value=[])
        mock_chat.return_value.add_message = AsyncMock()
        mock_chat.return_value.record_document = AsyncMock()
        mock_chat.return_value.close = AsyncMock()

        from src.pipeline.pipeline import RAGPipeline
        pipe = RAGPipeline()
        yield pipe
