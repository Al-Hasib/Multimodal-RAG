from abc import ABC, abstractmethod
from typing import List
from src.config.settings import settings
import logging

logger = logging.getLogger(__name__)


class BaseEmbeddings(ABC):
    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        ...

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ...


class OpenAIEmbeddingsAdapter(BaseEmbeddings):
    def __init__(self, model: str = "text-embedding-3-small"):
        from langchain_openai import OpenAIEmbeddings as LangChainOpenAIEmbeddings
        self._inner = LangChainOpenAIEmbeddings(model=model)

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._inner.embed_documents(texts)


class HuggingFaceEmbeddingsAdapter(BaseEmbeddings):
    def __init__(self, model: str = "BAAI/bge-m3"):
        from langchain_huggingface import HuggingFaceEmbeddings
        self._inner = HuggingFaceEmbeddings(model_name=model)

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._inner.embed_documents(texts)


class EmbeddingFactory:
    _instances: dict[str, BaseEmbeddings] = {}

    @classmethod
    def get_embeddings(cls, provider: str | None = None, model: str | None = None) -> BaseEmbeddings:
        provider = provider or settings.embedding_provider
        model = model or settings.embedding_model
        key = f"{provider}:{model}"

        if key not in cls._instances:
            logger.info(f"Creating embeddings: provider={provider}, model={model}")
            if provider == "openai":
                cls._instances[key] = OpenAIEmbeddingsAdapter(model=model)
            elif provider == "huggingface":
                cls._instances[key] = HuggingFaceEmbeddingsAdapter(model=model)
            else:
                raise ValueError(f"Unknown embedding provider: {provider}")

        return cls._instances[key]

    @classmethod
    def get_langchain_embeddings(cls):
        provider = settings.embedding_provider
        model = settings.embedding_model
        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model=model)
        elif provider == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=model)
        raise ValueError(f"Unknown embedding provider: {provider}")
