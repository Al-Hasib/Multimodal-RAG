import pytest
from src.retrieval.query_transformer import QueryTransformer
from src.retrieval.reranker import Reranker


@pytest.mark.asyncio
async def test_query_transformer_initialization():
    qt = QueryTransformer()
    assert qt.method in ("hyde", "multi_query", "step_back")


@pytest.mark.asyncio
async def test_reranker_initialization():
    reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    assert reranker is not None


def test_parse_docs_empty():
    from src.retrieval.retriever import MultiModalRetriever
    from unittest.mock import MagicMock
    mock_inner = MagicMock()
    retriever = MultiModalRetriever(mock_inner)
    result = retriever._parse_docs([])
    assert len(result.texts) == 0
    assert len(result.images) == 0


def test_parse_docs_mixed():
    from src.retrieval.retriever import MultiModalRetriever
    from unittest.mock import MagicMock
    import base64

    mock_inner = MagicMock()
    retriever = MultiModalRetriever(mock_inner)
    sample_b64 = base64.b64encode(b"fake-image-data").decode()

    docs = ["plain text here", sample_b64]
    result = retriever._parse_docs(docs)
    assert len(result.texts) == 1
    assert len(result.images) == 1
    assert result.texts[0] == "plain text here"
    assert result.images[0] == sample_b64
