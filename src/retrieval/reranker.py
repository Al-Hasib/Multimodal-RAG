from sentence_transformers import CrossEncoder
from typing import Any
import logging

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: list[Any], top_k: int = 5) -> list[Any]:
        if not documents:
            return documents

        pairs = []
        for doc in documents:
            text = doc.text if hasattr(doc, "text") else str(doc)
            pairs.append((query, text))

        scores = self.model.predict(pairs)
        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        reranked = [doc for doc, _ in scored[:top_k]]

        logger.info(f"Reranked {len(documents)} docs -> kept top {len(reranked)}")
        return reranked
