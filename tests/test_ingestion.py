import pytest
from src.ingestion.pdf_processor import PDFProcessor
from src.models.schemas import ExtractedDocument


@pytest.mark.asyncio
async def test_pdf_processor_initialization():
    processor = PDFProcessor()
    assert processor.extraction_strategy == "hi_res"
    assert processor.infer_table_structure is True


@pytest.mark.asyncio
async def test_extract_nonexistent_pdf():
    processor = PDFProcessor()
    with pytest.raises(Exception):
        await processor.extract("/nonexistent/file.pdf")
