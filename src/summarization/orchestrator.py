from src.models.schemas import ExtractedDocument, SummaryResult
from src.summarization.text_summarizer import TextSummarizer
from src.summarization.image_summarizer import ImageSummarizer
import logging

logger = logging.getLogger(__name__)


class SummarizationOrchestrator:
    def __init__(self):
        self.text_summarizer = TextSummarizer()
        self.image_summarizer = ImageSummarizer()

    def summarize(self, document: ExtractedDocument) -> SummaryResult:
        logger.info("Starting summarization of extracted document")

        text_summaries = []
        if document.texts:
            text_contents = [t.content for t in document.texts]
            text_summaries = self.text_summarizer.summarize_batch(text_contents)
            logger.info(f"Summarized {len(text_summaries)} text chunks")

        table_summaries = []
        if document.tables:
            tables_html = []
            for t in document.tables:
                html = t.metadata.get("text_as_html", str(t.content))
                tables_html.append(html)
            table_summaries = self.text_summarizer.summarize_batch(tables_html)
            logger.info(f"Summarized {len(table_summaries)} tables")

        image_summaries = []
        if document.images:
            image_b64_list = [img.image_base64 for img in document.images if img.image_base64]
            image_summaries = self.image_summarizer.summarize_batch(image_b64_list)
            logger.info(f"Summarized {len(image_summaries)} images")

        return SummaryResult(
            text_summaries=text_summaries,
            table_summaries=table_summaries,
            image_summaries=image_summaries,
        )
