import json
import logging
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from src.config.settings import settings

logger = logging.getLogger(__name__)

ALLOWED_ATTRIBUTES = {
    "file_format": {"type": "string", "values": ["pdf", "image", "docx", "html", "url", "audio"]},
    "filename": {"type": "string"},
    "group_id": {"type": "string"},
}


class StructuredQuery(BaseModel):
    query: str = Field(description="The refined search query text")
    filter_expr: Optional[dict] = Field(default=None, description="Qdrant filter condition as dict")


SELF_QUERY_SYSTEM = """You are a query analyzer for a multimodal RAG system.
Given a user question, extract:
1. A refined search query (remove filter-like language)
2. Optional metadata filters based on these allowed fields:

{allowed_fields}

Return a JSON object with keys "query" (str) and "filter_expr" (dict or null).

Examples:
- "Find PDFs about machine learning" → {{"query": "machine learning", "filter_expr": {{"must": [{{"key": "file_format", "match": {{"value": "pdf"}}}}]}}}}
- "Show audio transcriptions of meetings" → {{"query": "meetings", "filter_expr": {{"must": [{{"key": "file_format", "match": {{"value": "audio"}}}}]}}}}
- "What is the transformer architecture?" → {{"query": "transformer architecture", "filter_expr": null}}"""


class SelfQueryRetriever:
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(
            model=settings.openai_chat_model,
            temperature=0,
            api_key=settings.openai_api_key,
        )

    def parse_query(self, question: str) -> StructuredQuery:
        allowed = "\n".join(f"- {k}: {v['type']}" for k, v in ALLOWED_ATTRIBUTES.items())
        try:
            response = self.llm.invoke(
                SELF_QUERY_SYSTEM.format(allowed_fields=allowed)
                + f"\n\nUser question: {question}"
            )
            parsed = json.loads(response.content.strip().removeprefix("```json").removesuffix("```").strip())
            logger.info(f"Self-query parsed: {parsed}")
            return StructuredQuery(**parsed)
        except Exception as e:
            logger.warning(f"Self-query parsing failed, using raw query: {e}")
            return StructuredQuery(query=question, filter_expr=None)

    def build_qdrant_filter(self, filter_expr: Optional[dict]) -> Optional[dict]:
        if not filter_expr:
            return None
        return filter_expr
