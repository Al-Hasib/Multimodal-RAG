from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    openai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    langchain_api_key: Optional[str] = None
    langchain_tracing_v2: bool = False

    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    groq_chat_model: str = "llama-3.1-8b-instant"

    chroma_collection_name: str = "multi_modal_rag"
    chroma_persist_directory: str = "./chroma_db"

    pdf_extraction_strategy: str = "hi_res"
    pdf_infer_table_structure: bool = True
    pdf_extract_image_block_types: list[str] = ["Image"]
    pdf_extract_image_block_to_payload: bool = True
    pdf_chunking_strategy: str = "by_title"
    pdf_max_characters: int = 10000
    pdf_combine_text_under_n_chars: int = 2000
    pdf_new_after_n_chars: int = 6000

    summarization_max_concurrency: int = 5
    retrieval_k: int = 5

    model_config = {"env_prefix": "", "case_sensitive": False, "env_file": ".env", "extra": "ignore"}


settings = Settings()
