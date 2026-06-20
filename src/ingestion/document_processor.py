import os
import tempfile
from typing import Optional
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from src.config.settings import settings
from src.models.schemas import ExtractedDocument, ExtractedElement, DocumentType, FileFormat
import logging

logger = logging.getLogger(__name__)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
DOCX_EXTENSIONS = {".docx", ".doc"}
HTML_EXTENSIONS = {".html", ".htm", ".xhtml"}
URL_SCHEMES = {"http://", "https://"}


class DocumentProcessor:
    def __init__(self):
        self.pdf_processor = PDFProcessor()

    def detect_format(self, source: str) -> FileFormat:
        lower = source.lower()
        if any(lower.startswith(s) for s in URL_SCHEMES):
            return FileFormat.URL
        ext = os.path.splitext(lower)[1]
        if ext == ".pdf":
            return FileFormat.PDF
        if ext in IMAGE_EXTENSIONS:
            return FileFormat.IMAGE
        if ext in DOCX_EXTENSIONS:
            return FileFormat.DOCX
        if ext in HTML_EXTENSIONS:
            return FileFormat.HTML
        return FileFormat.UNKNOWN

    async def extract(self, source: str, file_format: Optional[FileFormat] = None) -> ExtractedDocument:
        fmt = file_format or self.detect_format(source)
        logger.info(f"Extracting from {source} (format: {fmt.value})")

        if fmt == FileFormat.PDF:
            return await self.pdf_processor.extract(source)

        kwargs = self._get_unstructured_kwargs(fmt)
        elements = partition(filename=source, **kwargs)
        return self._elements_to_document(source, fmt, elements)

    async def extract_from_bytes(
        self, filename: str, content: bytes, file_format: Optional[FileFormat] = None
    ) -> ExtractedDocument:
        fmt = file_format or self.detect_format(filename)

        if fmt == FileFormat.PDF:
            temp_path = f"/tmp/{filename}"
            async with __import__("aiofiles").open(temp_path, "wb") as f:
                await f.write(content)
            return await self.pdf_processor.extract(temp_path)

        suffix = os.path.splitext(filename)[1] or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            kwargs = self._get_unstructured_kwargs(fmt)
            elements = partition(filename=tmp_path, **kwargs)
            return self._elements_to_document(filename, fmt, elements)
        finally:
            os.unlink(tmp_path)

    def _get_unstructured_kwargs(self, fmt: FileFormat) -> dict:
        kwargs: dict = {}
        if fmt == FileFormat.IMAGE:
            kwargs["strategy"] = "auto"
        elif fmt == FileFormat.URL:
            kwargs["ssl_verify"] = False
        elif fmt == FileFormat.DOCX:
            kwargs["include_metadata"] = True
        elif fmt == FileFormat.HTML:
            kwargs["include_metadata"] = True
        return kwargs

    def _elements_to_document(self, source: str, fmt: FileFormat, elements: list) -> ExtractedDocument:
        filename = os.path.basename(source) if not source.startswith("http") else source
        doc = ExtractedDocument(filename=filename, file_format=fmt)

        for el in elements:
            el_type = str(type(el))
            if "Table" in el_type:
                doc.tables.append(ExtractedElement(
                    type=DocumentType.TABLE,
                    content=el,
                    metadata={
                        "text_as_html": getattr(el.metadata, "text_as_html", None),
                        **getattr(el, "metadata", {}),
                    },
                ))
            elif "Image" in el_type:
                b64 = getattr(el.metadata, "image_base64", None)
                doc.images.append(ExtractedElement(
                    type=DocumentType.IMAGE,
                    content=el,
                    image_base64=b64,
                ))
            else:
                doc.texts.append(ExtractedElement(
                    type=DocumentType.TEXT,
                    content=el,
                    metadata=getattr(el, "metadata", {}),
                ))

        logger.info(f"Extracted {len(doc.texts)} texts, {len(doc.tables)} tables, {len(doc.images)} images")
        return doc
