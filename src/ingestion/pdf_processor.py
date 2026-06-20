from unstructured.partition.pdf import partition_pdf
from src.config.settings import settings
from src.models.schemas import ExtractedDocument, ExtractedElement, DocumentType
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PDFProcessor:
    def __init__(self):
        self.extraction_strategy = settings.pdf_extraction_strategy
        self.infer_table_structure = settings.pdf_infer_table_structure
        self.extract_image_block_types = settings.pdf_extract_image_block_types
        self.extract_image_block_to_payload = settings.pdf_extract_image_block_to_payload
        self.chunking_strategy = settings.pdf_chunking_strategy
        self.max_characters = settings.pdf_max_characters
        self.combine_text_under_n_chars = settings.pdf_combine_text_under_n_chars
        self.new_after_n_chars = settings.pdf_new_after_n_chars

    def extract(self, file_path: str, image_output_dir: Optional[str] = None) -> ExtractedDocument:
        logger.info(f"Extracting elements from: {file_path}")

        chunks = partition_pdf(
            filename=file_path,
            infer_table_structure=self.infer_table_structure,
            strategy=self.extraction_strategy,
            extract_image_block_types=self.extract_image_block_types,
            image_output_dir_path=image_output_dir,
            extract_image_block_to_payload=self.extract_image_block_to_payload,
            chunking_strategy=self.chunking_strategy,
            max_characters=self.max_characters,
            combine_text_under_n_chars=self.combine_text_under_n_chars,
            new_after_n_chars=self.new_after_n_chars,
        )

        return self._separate_elements(chunks, file_path)

    def _separate_elements(self, chunks: list, filename: str) -> ExtractedDocument:
        tables = []
        texts = []
        images_b64 = []

        for chunk in chunks:
            chunk_type = str(type(chunk))
            if "Table" in chunk_type:
                tables.append(chunk)
            elif "CompositeElement" in chunk_type:
                texts.append(chunk)
                images_b64.extend(self._extract_images_from_chunk(chunk))

        extracted = ExtractedDocument(filename=filename)

        for t in texts:
            extracted.texts.append(ExtractedElement(
                type=DocumentType.TEXT,
                content=t,
                metadata=getattr(t, "metadata", {}),
            ))

        for t in tables:
            extracted.tables.append(ExtractedElement(
                type=DocumentType.TABLE,
                content=t,
                metadata={
                    "text_as_html": getattr(t.metadata, "text_as_html", None),
                    **getattr(t, "metadata", {}),
                },
            ))

        for img in images_b64:
            extracted.images.append(ExtractedElement(
                type=DocumentType.IMAGE,
                content=img,
                image_base64=img,
            ))

        logger.info(f"Extracted {len(extracted.texts)} texts, {len(extracted.tables)} tables, {len(extracted.images)} images")
        return extracted

    def _extract_images_from_chunk(self, chunk) -> list[str]:
        images = []
        orig_elements = getattr(chunk.metadata, "orig_elements", [])
        for el in orig_elements:
            if "Image" in str(type(el)):
                b64 = getattr(el.metadata, "image_base64", None)
                if b64:
                    images.append(b64)
        return images
