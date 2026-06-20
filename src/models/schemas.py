from enum import Enum
from pydantic import BaseModel, Field
from typing import Any


class DocumentType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


class ExtractedElement(BaseModel):
    type: DocumentType
    content: Any
    metadata: dict = Field(default_factory=dict)
    image_base64: str | None = None


class ExtractedDocument(BaseModel):
    filename: str
    texts: list[ExtractedElement] = Field(default_factory=list)
    tables: list[ExtractedElement] = Field(default_factory=list)
    images: list[ExtractedElement] = Field(default_factory=list)


class SummaryResult(BaseModel):
    text_summaries: list[str] = Field(default_factory=list)
    table_summaries: list[str] = Field(default_factory=list)
    image_summaries: list[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    texts: list[Any] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    context_texts: list[str] = Field(default_factory=list)
    context_images: list[str] = Field(default_factory=list)
