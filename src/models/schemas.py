from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from typing import Any


class DocumentType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


class FileFormat(str, Enum):
    PDF = "pdf"
    IMAGE = "image"
    DOCX = "docx"
    HTML = "html"
    URL = "url"
    AUDIO = "audio"
    UNKNOWN = "unknown"


class ExtractedElement(BaseModel):
    type: DocumentType
    content: Any
    metadata: dict = Field(default_factory=dict)
    image_base64: str | None = None


class ExtractedDocument(BaseModel):
    filename: str
    file_format: FileFormat = FileFormat.PDF
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
    session_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    context_texts: list[str] = Field(default_factory=list)
    context_images: list[str] = Field(default_factory=list)
    guardrail_input: dict | None = None
    guardrail_output: dict | None = None


class DocumentInfo(BaseModel):
    id: int
    filename: str
    file_format: str
    file_path: str
    num_texts: int = 0
    num_tables: int = 0
    num_images: int = 0
    status: str = "processed"
    created_at: datetime | None = None


class DocumentList(BaseModel):
    documents: list[DocumentInfo]
    total: int
