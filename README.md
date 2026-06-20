# Multimodal-RAG

Production-ready multimodal RAG application with hybrid search, re-ranking, query transformation, LangGraph orchestration,
prompt versioning (LangFuse), streaming, chat history (PostgreSQL), Redis caching, and a Streamlit web UI.

## Architecture

```
                        ┌─────────────┐
                        │    PDF      │
                        └──────┬──────┘
                               │
                        ┌──────▼──────┐
                        │  Ingestion  │  Unstructured (async)
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
              │   └─ Redis (docstore)   │  Persistent TTL-backed store
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   Query Transformer     │  HyDE / Multi-query / Step-back
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   Cross-encoder         │  Re-ranking (ms-marco)
              │   Reranker              │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   LangGraph State Graph │
              │   transform → retrieve  │
              │   → build → generate    │  GPT-4o-mini
              └────────────┬────────────┘
                           │
                    ┌──────▼──────┐
                    │  Response   │  Streaming SSE + Cache
                    └─────────────┘
```

## Features

| Feature | Description |
|---------|-------------|
| **Hybrid Search** | Dense (OpenAI) + sparse (BM25) vectors in Qdrant |
| **Query Transformation** | HyDE, multi-query, step-back prompting |
| **Re-ranking** | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-ranks results |
| **LangGraph** | State graph with 4 nodes: transform → retrieve → build → generate |
| **Prompt Versioning** | Prompts managed via LangFuse dashboard with local fallback |
| **LangFuse Tracing** | Full observability of LLM calls, retrievals, and generations |
| **Async Ingestion** | Non-blocking PDF processing with `aiofiles` |
| **Redis Cache** | Query result caching with configurable TTL |
| **Chat History** | Persistent PostgreSQL storage per session |
| **Streaming** | Server-sent events for token-by-token responses |
| **Web UI** | Streamlit interface for PDF upload + chat |
| **Embedding Abstraction** | Swap OpenAI ↔ HuggingFace via config |
| **Health Checks** | Redis, Qdrant, PostgreSQL status endpoint |
| **Production Build** | Multi-stage Dockerfile + gunicorn workers |
| **Evaluation** | `scripts/evaluate.py` using `ragas` metrics |

## Quick Start

### Docker Compose (dev)

```bash
docker compose up --build
```

| Service | Port | Description |
|---------|------|-------------|
| `multimodal-rag` | 8000 | FastAPI (hot-reload) |
| `qdrant` | 6333 | Vector database |
| `redis` | 6379 | Docstore + cache |
| `postgres` | 5432 | Chat history |

### Docker Compose (prod)

```bash
docker compose --profile prod up --build
```

### Local

```bash
# Start infra
docker compose up -d qdrant redis postgres

# Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env  # edit with API keys

# Run
python -m src.main api        # dev
python -m src.main ingest ./paper.pdf
python -m src.main query "What is the Transformer?" --session-id abc
python -m src.main ui          # Streamlit
python -m src.main evaluate scripts/test_set_example.json
```

### API Usage

```bash
# Ingest
curl -X POST -F "file=@paper.pdf" http://localhost:8000/ingest

# Query
curl -X POST -H "Content-Type: application/json" \
  -d '{"question": "What is attention?", "session_id": "abc"}' \
  http://localhost:8000/query

# Stream
curl -N "http://localhost:8000/query/stream?question=Explain+attention&session_id=abc"

# Health
curl http://localhost:8000/health

# History
curl "http://localhost:8000/history?session_id=abc"
```

## Configuration

All settings via `.env`:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `GROQ_API_KEY` | — | Groq API key |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis URL |
| `POSTGRES_URL` | `postgresql+asyncpg://...` | PostgreSQL URL |
| `LANGFUSE_PUBLIC_KEY` | — | LangFuse public key |
| `LANGFUSE_SECRET_KEY` | — | LangFuse secret key |
| `LANGFUSE_ENABLED` | `false` | Enable LangFuse tracing |
| `EMBEDDING_PROVIDER` | `openai` | `openai` or `huggingface` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model name |
| `REDIS_CACHE_ENABLED` | `true` | Enable query result caching |
| `RERANK_ENABLED` | `true` | Enable cross-encoder re-ranking |
| `QUERY_TRANSFORMER_ENABLED` | `true` | Enable query transformation |
| `QUERY_TRANSFORMER_METHOD` | `hyde` | `hyde`, `multi_query`, `step_back` |
| `PROMPT_USE_LANGFUSE` | `false` | Fetch prompts from LangFuse |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FORMAT` | `json` | `json` or `text` |

## Project Structure

```
src/
├── config/          # pydantic-settings
├── core/            # EmbeddingFactory, QueryCache, logging
├── models/          # Pydantic schemas
├── ingestion/       # Async PDF parsing (unstructured)
├── summarization/   # Groq + GPT-4o-mini summarization
├── retrieval/       # Qdrant hybrid, reranker, query transformer
├── generation/      # LangGraph graph + RAGChain + PromptManager
├── storage/         # Redis-backed docstore
├── chat/            # PostgreSQL chat history
├── pipeline/        # Pipeline orchestrator
├── api/             # FastAPI (REST + SSE streaming)
├── ui/              # Streamlit web app
└── main.py          # CLI (api, prod, ingest, query, ui, evaluate)

tests/               # pytest tests (unit + mock)
scripts/             # evaluate.py, test_set_example.json
```

## Development

```bash
pip install pytest pytest-asyncio ruff mypy pre-commit coverage
pre-commit install

pytest                          # run tests
coverage run -m pytest && coverage report
ruff check src/ && ruff format src/
mypy src/
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## References

- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [LangChain Multi-vector Retriever](https://python.langchain.com/docs/how_to/multi_vector/)
- [Qdrant Hybrid Search](https://qdrant.tech/articles/hybrid-search/)
- [LangFuse Prompt Management](https://langfuse.com/docs/prompts)
- [Ragas Evaluation](https://docs.ragas.io/)
