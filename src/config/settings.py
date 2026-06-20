from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    openai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    langchain_api_key: Optional[str] = None
    langchain_tracing_v2: bool = False
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_project: str = "multimodal-rag"

    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_enabled: bool = False

    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    groq_chat_model: str = "llama-3.1-8b-instant"

    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_collection_name: str = "multi_modal_rag"
    qdrant_prefer_grpc: bool = False
    qdrant_hybrid: bool = True
    qdrant_sparse_model: str = "Qdrant/bm25"

    redis_url: str = "redis://localhost:6379/0"
    redis_docstore_ttl: int = 86400
    redis_cache_ttl: int = 3600
    redis_cache_enabled: bool = True

    postgres_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/multimodal_rag"
    postgres_pool_size: int = 10

    pdf_extraction_strategy: str = "hi_res"
    pdf_infer_table_structure: bool = True
    pdf_extract_image_block_types: list[str] = ["Image"]
    pdf_extract_image_block_to_payload: bool = True
    pdf_chunking_strategy: str = "by_title"
    pdf_max_characters: int = 10000
    pdf_combine_text_under_n_chars: int = 2000
    pdf_new_after_n_chars: int = 6000

    summarization_max_concurrency: int = 5
    retrieval_k: int = 10
    retrieval_hybrid_alpha: float = 0.7
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 5
    query_transformer_enabled: bool = True
    query_transformer_method: str = "hyde"

    log_level: str = "INFO"
    log_format: str = "json"

    prompt_use_langfuse: bool = False
    prompt_default_version: int = 1

    guardrail_enabled: bool = True
    guardrail_input_check: bool = True
    guardrail_output_check: bool = True
    guardrail_block_on_input_violation: bool = True
    guardrail_block_on_output_violation: bool = False
    guardrail_log_to_langfuse: bool = True
    guardrail_prompt_injection_model: str = "llama-3.1-8b-instant"

    audio_transcription_provider: str = "openai"
    audio_transcription_model: str = "whisper-1"
    audio_transcription_language: Optional[str] = None
    audio_chunk_seconds: int = 300
    audio_max_file_size_mb: int = 200

    api_keys: list[str] = []
    rate_limit: str = "10/minute"
    cors_origins: str = "*"
    request_timeout_seconds: int = 120

    arq_redis_url: str = "redis://localhost:6379/1"
    arq_queue_name: str = "ingestion"

    model_config = {"env_prefix": "", "case_sensitive": False, "env_file": ".env", "extra": "ignore"}


settings = Settings()
