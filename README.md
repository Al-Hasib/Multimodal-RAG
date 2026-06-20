# Multimodal-RAG

Production-ready multimodal RAG application with hybrid search, re-ranking, query transformation, streaming, chat history, and a web UI.

## Architecture

```
                        ┌─────────────┐
                        │    PDF      │
                        └──────┬──────┘
                               │
                        ┌──────▼──────┐
                        │  Ingestion  │  Unstructured partition_pdf
                        └──────┬──────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │  Text    │   │  Table   │   │  Image   │
        │Summary   │   │Summary   │   │Summary   │
        │(Groq)    │   │(Groq)    │   │(GPT-4o)  │
        └────┬─────┘   └────┬─────┘   └────┬─────┘
             │              │              │
             └──────────────┼──────────────┘
                            ▼
              ┌─────────────────────────┐
              │   MultiVector Retriever │
              │   ├─ Qdrant (hybrid)    │  Dense + Sparse vectors
              │   └─ Redis (docstore)   │  Persistent document store
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   Query Transformer     │  HyDE / Multi-query / Step-back
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   Cross-encoder         │  Re-ranking retrieved docs
              │   Reranker              │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   LangGraph RAG Chain   │
              │   (text + images)       │  GPT-4o-mini
              └────────────┬────────────┘
                           │
                    ┌──────▼──────┐
                    │  Response   │  Streaming SSE
                    └─────────────┘
```

## Quick Start

### Prerequisites
- Docker & Docker Compose

### Docker Compose (recommended)

```bash
docker compose up --build
```

This starts all services:

| Service | Port | Description |
|---------|------|-------------|
| `multimodal-rag` | 8000 | FastAPI app (hot-reload enabled) |
| `qdrant` | 6333/6334 | Vector database (hybrid search) |
| `redis` | 6379 | Document store + caching |
| `postgres` | 5432 | Chat history + document metadata |

### Usage

**Web UI:** [http://localhost:8501](http://localhost:8501) (streamlit)

**API:**
```bash
# Ingest a PDF
curl -X POST -F "file=@paper.pdf" http://localhost:8000/ingest

# Query
curl -X POST -H "Content-Type: application/json" \
  -d '{"question": "What is the Transformer architecture?", "session_id": "abc123"}' \
  http://localhost:8000/query

# Stream a response
curl -N "http://localhost:8000/query/stream?question=Explain+attention&session_id=abc123"

# Get chat history
curl "http://localhost:8000/history?session_id=abc123"
```

### Local (no Docker)

```bash
# Start infra
docker run -d -p 6333:6333 qdrant/qdrant
docker run -d -p 6379:6379 redis:7-alpine
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=multimodal_rag postgres:16-alpine

# Install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run
cp .env.example .env  # edit with your API keys
python -m src.main api
```

## Features

| Feature | Description |
|---------|-------------|
| **Hybrid Search** | Dense (OpenAI) + sparse (BM25) vectors in Qdrant for better retrieval |
| **Query Transformation** | HyDE, multi-query, or step-back prompting before retrieval |
| **Re-ranking** | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-ranks retrieved documents |
| **MultiVector Retriever** | Stores summaries in vector DB, links to full content in Redis docstore |
| **Multimodal** | Answers grounded in text, tables, and images from PDFs |
| **Chat History** | Persistent conversation storage in PostgreSQL per session |
| **Streaming** | Server-sent events for token-by-token responses |
| **Web UI** | Streamlit interface for PDF upload + chat |
| **Hot Reload** | Source code mounted live in Docker — no rebuild on edits |

## Project Structure

```
src/
├── config/          # Settings via pydantic-settings
├── models/          # Pydantic schemas
├── ingestion/       # PDF parsing with unstructured
├── summarization/   # Text (Groq/Llama) + Image (GPT-4o-mini) summarization
├── retrieval/       # Qdrant + hybrid search + query transformer + reranker
│   ├── vector_store.py     # Qdrant hybrid collection setup
│   ├── retriever.py        # MultiModalRetriever
│   ├── query_transformer.py # HyDE / multi-query / step-back
│   └── reranker.py         # Cross-encoder re-ranking
├── generation/      # LangGraph-based RAG chain with streaming
├── storage/         # Redis-backed docstore
├── chat/            # PostgreSQL chat history
├── pipeline/        # Orchestration
├── api/             # FastAPI endpoints (REST + SSE streaming)
├── ui/              # Streamlit web app
└── main.py          # CLI entry point
```

## Configuration

All settings via `.env` or environment variables:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `GROQ_API_KEY` | — | Groq API key |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_COLLECTION_NAME` | `multi_modal_rag` | Qdrant collection name |
| `QDRANT_HYBRID` | `true` | Enable hybrid (dense + sparse) search |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis URL for docstore |
| `POSTGRES_URL` | `postgresql+asyncpg://...` | PostgreSQL URL for chat history |
| `RERANK_ENABLED` | `true` | Enable cross-encoder re-ranking |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Re-ranking model |
| `QUERY_TRANSFORMER_ENABLED` | `true` | Enable query transformation |
| `QUERY_TRANSFORMER_METHOD` | `hyde` | `hyde`, `multi_query`, or `step_back` |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Chat model for generation |
| `GROQ_CHAT_MODEL` | `llama-3.1-8b-instant` | Text/table summarization model |

## References

- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [LangChain Multi-vector Retriever](https://python.langchain.com/docs/how_to/multi_vector/)
- [Qdrant Hybrid Search](https://qdrant.tech/articles/hybrid-search/)
- [Unstructured PDF Partitioning](https://docs.unstructured.io/open-source/core-functionality/chunking)
