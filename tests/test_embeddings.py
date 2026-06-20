import pytest
from src.core.embeddings import EmbeddingFactory, OpenAIEmbeddingsAdapter, HuggingFaceEmbeddingsAdapter, BaseEmbeddings


def test_embedding_factory_openai():
    embeddings = EmbeddingFactory.get_embeddings(provider="openai", model="text-embedding-3-small")
    assert isinstance(embeddings, BaseEmbeddings)
    assert isinstance(embeddings, OpenAIEmbeddingsAdapter)


def test_embedding_factory_huggingface():
    embeddings = EmbeddingFactory.get_embeddings(provider="huggingface", model="BAAI/bge-m3")
    assert isinstance(embeddings, BaseEmbeddings)
    assert isinstance(embeddings, HuggingFaceEmbeddingsAdapter)


def test_embedding_factory_invalid_provider():
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        EmbeddingFactory.get_embeddings(provider="invalid_provider")


def test_embedding_factory_caching():
    e1 = EmbeddingFactory.get_embeddings(provider="openai", model="text-embedding-3-small")
    e2 = EmbeddingFactory.get_embeddings(provider="openai", model="text-embedding-3-small")
    assert e1 is e2
