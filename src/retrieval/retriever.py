from langchain.retrievers.multi_vector import MultiVectorRetriever
from base64 import b64decode
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from src.config.settings import settings
from src.models.schemas import RetrievalResult
from src.retrieval.reranker import Reranker
from src.retrieval.query_transformer import QueryTransformer
import logging

logger = logging.getLogger(__name__)


class MultiModalRetriever:
    def __init__(self, retriever: MultiVectorRetriever):
        self.retriever = retriever
        self.k = settings.retrieval_k
        self.hybrid = settings.qdrant_hybrid
        self.hybrid_alpha = settings.retrieval_hybrid_alpha
        self.reranker = Reranker(model_name=settings.rerank_model) if settings.rerank_enabled else None
        self.query_transformer = QueryTransformer()

        if self.hybrid:
            self.qdrant_client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                prefer_grpc=settings.qdrant_prefer_grpc,
            )

    def retrieve(self, query: str, k: int | None = None) -> RetrievalResult:
        top_k = k or self.k
        logger.info(f"Retrieving for query with k={top_k}: {query}")

        queries = self.query_transformer.transform(query)
        all_docs: list = []

        for q in queries:
            if self.hybrid:
                docs = self._hybrid_search(q, top_k)
            else:
                docs = self.retriever.invoke(q)
            all_docs.extend(docs)

        if self.reranker:
            all_docs = self.reranker.rerank(query, all_docs, top_k=settings.rerank_top_k)

        result = self._parse_docs(all_docs[:top_k])
        logger.info(f"Retrieved {len(result.texts)} texts, {len(result.images)} images")
        return result

    def _hybrid_search(self, query: str, k: int) -> list:
        embedding = OpenAIEmbeddings(model=settings.openai_embedding_model)
        dense_vector = embedding.embed_query(query)

        search_result = self.qdrant_client.search(
            collection_name=settings.qdrant_collection_name,
            query_vector=dense_vector,
            limit=k * 2,
        )
        point_ids = [hit.id for hit in search_result]
        if not point_ids:
            return []

        stored_docs = self.retriever.docstore.mget([str(pid) for pid in point_ids])
        docs = [sd for sd in stored_docs if sd is not None]
        return docs

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
            elif hasattr(doc, "page_content") and doc.page_content:
                try:
                    b64decode(doc.page_content)
                    b64_images.append(doc.page_content)
                except Exception:
                    texts.append(doc)
            else:
                texts.append(doc)

        return RetrievalResult(texts=texts, images=b64_images)
