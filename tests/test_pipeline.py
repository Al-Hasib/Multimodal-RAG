import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_pipeline_health(mock_pipeline):
    assert mock_pipeline is not None
    assert hasattr(mock_pipeline, "ingest")
    assert hasattr(mock_pipeline, "aquery")
    assert hasattr(mock_pipeline, "astream")


@pytest.mark.asyncio
async def test_pipeline_ingest(mock_pipeline, mock_pdf_path):
    mock_pipeline.pdf_processor.extract.return_value = MagicMock(
        texts=[MagicMock()],
        tables=[],
        images=[],
    )
    mock_pipeline.summarizer.summarize.return_value = MagicMock(
        text_summaries=["summary"],
        table_summaries=[],
        image_summaries=[],
    )

    result = await mock_pipeline.ingest(mock_pdf_path)
    assert "filename" in result
    assert result["texts"] == 1


@pytest.mark.asyncio
async def test_pipeline_query(mock_pipeline):
    mock_pipeline.rag_chain.invoke_with_sources.return_value = MagicMock(
        answer="test answer",
        context_texts=["context"],
        context_images=[],
    )

    from src.models.schemas import QueryResponse
    result = await mock_pipeline.aquery("test question")
    assert isinstance(result, MagicMock)
    assert result.answer == "test answer"
