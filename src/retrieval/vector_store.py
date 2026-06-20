import uuid
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.document import Document
from src.config.settings import settings
from src.models.schemas import ExtractedDocument, SummaryResult
from qdrant_client import QdrantClient
import logging

logger = logging.getLogger(__name__)


class VectorStoreManager:
    def __init__(self):
        self.id_key = "doc_id"
        self.collection_name = settings.qdrant_collection_name

        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            prefer_grpc=settings.qdrant_prefer_grpc,
        )

        self.vectorstore = QdrantVectorStore(
            client=client,
            collection_name=self.collection_name,
            embedding=OpenAIEmbeddings(model=settings.openai_embedding_model),
        )

        self.docstore = InMemoryStore()

        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            id_key=self.id_key,
        )

    def index_document(self, document: ExtractedDocument, summaries: SummaryResult) -> None:
        logger.info("Indexing document into Qdrant")

        if document.texts and summaries.text_summaries:
            doc_ids = [str(uuid.uuid4()) for _ in document.texts]
            summary_docs = [
                Document(page_content=summary, metadata={self.id_key: doc_ids[i]})
                for i, summary in enumerate(summaries.text_summaries)
            ]
            self.vectorstore.add_documents(summary_docs)
            self.docstore.mset(list(zip(doc_ids, [t.content for t in document.texts])))
            logger.info(f"Indexed {len(document.texts)} text chunks")

        if document.tables and summaries.table_summaries:
            table_ids = [str(uuid.uuid4()) for _ in document.tables]
            table_summary_docs = [
                Document(page_content=summary, metadata={self.id_key: table_ids[i]})
                for i, summary in enumerate(summaries.table_summaries)
            ]
            self.vectorstore.add_documents(table_summary_docs)
            self.docstore.mset(list(zip(table_ids, [t.content for t in document.tables])))
            logger.info(f"Indexed {len(document.tables)} tables")

        if document.images and summaries.image_summaries:
            img_ids = [str(uuid.uuid4()) for _ in document.images]
            img_summary_docs = [
                Document(page_content=summary, metadata={self.id_key: img_ids[i]})
                for i, summary in enumerate(summaries.image_summaries)
            ]
            self.vectorstore.add_documents(img_summary_docs)
            self.docstore.mset(list(zip(img_ids, [img.image_base64 for img in document.images])))
            logger.info(f"Indexed {len(document.images)} images")

    def get_retriever(self) -> MultiVectorRetriever:
        return self.retriever
