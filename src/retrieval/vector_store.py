import uuid
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document as LCDocument
from src.config.settings import settings
from src.models.schemas import ExtractedDocument
from src.core.retry import default_retry, SummaryResult
from src.storage.redis_docstore import RedisDocStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, SparseIndexParams,
    HnswConfigDiff, Filter, FieldCondition, MatchValue, PointIdsList,
)
import logging

logger = logging.getLogger(__name__)

DOC_POINTS_KEY_PREFIX = "doc_points:"


class VectorStoreManager:
    def __init__(self):
        self.id_key = "doc_id"
        self.collection_name = settings.qdrant_collection_name

        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            prefer_grpc=settings.qdrant_prefer_grpc,
        )

        self._ensure_collection()

        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=OpenAIEmbeddings(model=settings.openai_embedding_model),
        )

        self.docstore = RedisDocStore()

        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            id_key=self.id_key,
        )

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        existing = [c.name for c in collections]

        if self.collection_name in existing:
            logger.info(f"Using existing Qdrant collection: {self.collection_name}")
            return

        logger.info(f"Creating Qdrant collection: {self.collection_name}")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=1536,
                distance=Distance.COSINE,
            ),
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            } if settings.qdrant_hybrid else None,
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
        )

    @default_retry
    def index_document(self, document: ExtractedDocument, summaries: SummaryResult) -> str:
        group_id = str(uuid.uuid4())
        logger.info(f"Indexing document (group={group_id}) into Qdrant")

        all_point_ids = []

        if document.texts and summaries.text_summaries:
            doc_ids = [str(uuid.uuid4()) for _ in document.texts]
            summary_docs = [
                LCDocument(
                    page_content=summary,
                    metadata={self.id_key: doc_ids[i], "group_id": group_id},
                )
                for i, summary in enumerate(summaries.text_summaries)
            ]
            self.vectorstore.add_documents(summary_docs)
            self.docstore.mset(list(zip(doc_ids, [t.content for t in document.texts])))
            all_point_ids.extend(doc_ids)
            logger.info(f"Indexed {len(document.texts)} text chunks")

        if document.tables and summaries.table_summaries:
            table_ids = [str(uuid.uuid4()) for _ in document.tables]
            table_summary_docs = [
                LCDocument(
                    page_content=summary,
                    metadata={self.id_key: table_ids[i], "group_id": group_id},
                )
                for i, summary in enumerate(summaries.table_summaries)
            ]
            self.vectorstore.add_documents(table_summary_docs)
            self.docstore.mset(list(zip(table_ids, [t.content for t in document.tables])))
            all_point_ids.extend(table_ids)
            logger.info(f"Indexed {len(document.tables)} tables")

        if document.images and summaries.image_summaries:
            img_ids = [str(uuid.uuid4()) for _ in document.images]
            img_summary_docs = [
                LCDocument(
                    page_content=summary,
                    metadata={self.id_key: img_ids[i], "group_id": group_id},
                )
                for i, summary in enumerate(summaries.image_summaries)
            ]
            self.vectorstore.add_documents(img_summary_docs)
            self.docstore.mset(list(zip(img_ids, [img.image_base64 for img in document.images])))
            all_point_ids.extend(img_ids)
            logger.info(f"Indexed {len(document.images)} images")

        if all_point_ids:
            key = f"{DOC_POINTS_KEY_PREFIX}{group_id}"
            self.docstore.client.sadd(key, *all_point_ids)
            self.docstore.client.expire(key, 86400 * 7)

        return group_id

    @default_retry
    def delete_document(self, group_id: str) -> bool:
        key = f"{DOC_POINTS_KEY_PREFIX}{group_id}"
        point_ids = self.docstore.client.smembers(key)
        if not point_ids:
            logger.warning(f"No points found for group: {group_id}")
            return False

        pids = [pid.decode() if isinstance(pid, bytes) else pid for pid in point_ids]
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=pids),
        )

        self.docstore.mdelete(pids)
        self.docstore.client.delete(key)
        logger.info(f"Deleted {len(pids)} points for group {group_id}")
        return True

    @default_retry
    def clear_collection(self):
        self.client.delete_collection(collection_name=self.collection_name)
        self._ensure_collection()
        logger.info(f"Cleared collection: {self.collection_name}")

    def scroll_all_points(self, limit: int = 100) -> list[dict]:
        from qdrant_client.models import ScoredPoint
        results = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                offset=offset,
                with_payload=True,
            )
            for p in points:
                results.append(p.payload or {})
            if offset is None:
                break
        return results

    def get_retriever(self) -> MultiVectorRetriever:
        return self.retriever
