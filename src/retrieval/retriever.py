from langchain.retrievers.multi_vector import MultiVectorRetriever
from base64 import b64decode
from src.config.settings import settings
from src.models.schemas import RetrievalResult
import logging

logger = logging.getLogger(__name__)


class MultiModalRetriever:
    def __init__(self, retriever: MultiVectorRetriever):
        self.retriever = retriever
        self.k = settings.retrieval_k

    def retrieve(self, query: str, k: int | None = None) -> RetrievalResult:
        top_k = k or self.k
        logger.info(f"Retrieving for query with k={top_k}: {query}")

        docs = self.retriever.invoke(query)
        return self._parse_docs(docs)

    def _parse_docs(self, docs) -> RetrievalResult:
        b64_images = []
        texts = []

        for doc in docs:
            if isinstance(doc, str):
                try:
                    b64decode(doc)
                    b64_images.append(doc)
                except Exception:
                    texts.append(doc)
            else:
                texts.append(doc)

        return RetrievalResult(texts=texts, images=b64_images)
