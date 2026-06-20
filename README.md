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
| **Document Management** | List, get, delete documents via REST API |
| **Multi-format Ingestion** | PDF, images (JPG/PNG), DOCX, HTML, URLs, audio (MP3, WAV, M4A, FLAC, OGG, AAC) |
| **Re-index** | Wipe and re-index all documents from scratch |
| **URL Ingestion** | `POST /ingest/url` to fetch and index web pages |
| **Input Guardrails** | Prompt injection detection (regex + LLM), toxicity check, PII scan, topic relevance |
| **Output Guardrails** | Hallucination detection (factual grounding), answer relevance, output toxicity check |
| **LangFuse Integration** | Guardrail violations logged as LangFuse traces for audit |
| **User Management** | Register/login/session via JWT tokens + bcrypt password hashing |
| **API Key Auth** | Fallback static API key auth when `API_KEYS` is set |
| **Rate Limiting** | Configurable per-endpoint rate limits via `slowapi` |
| **User Isolation** | Every user sees only their own documents, chats, and feedback |
| **Feedback Loop** | Thumbs up/down per answer, stored in PostgreSQL |
| **Background Jobs** | Async ingestion via `arq` Redis task queue for files > 10 MB |
| **Prometheus Metrics** | `/metrics` endpoint for QPS, latency, error rate |
| **Cost Tracking** | Per-request token usage + cost estimation (model-specific pricing) |
| **Latency Breakdown** | Per-component timing (retrieval, generation, guardrails) via Prometheus histograms |
| **Usage Stats** | `/metrics/stats` endpoint for aggregated cost, latency, cache hit ratio |
| **Per-request Logging** | JSON structured logs with latency, tokens, cost, guardrail status |
| **Structured Errors** | Consistent `{code, message}` error responses |
| **Request ID Tracing** | `X-Request-ID` header on every response |
| **CORS** | Configurable cross-origin support |
| **Database Migrations** | Alembic-managed PostgreSQL schema evolution |
| **CI Pipeline** | GitHub Actions (lint, test, docker build) |
| **Per-format Chunking** | Different chunk strategies for PDF, DOCX, HTML, audio, images |
| **Retry with Backoff** | Tenacity retry on Qdrant operations |

## Quick Start

### Docker Compose (dev)

```bash
docker compose up --build
```

| Service | Port | Description |
|---------|------|-------------|
| `multimodal-rag` | 8000 | FastAPI (hot-reload) |
| `multimodal-rag-worker` | — | arq background job worker (prod only) |
| `qdrant` | 6333 | Vector database |
| `redis` | 6379 | Docstore + cache |
| `postgres` | 5432 | Chat history + feedback |

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
# Register a new user
curl -X POST -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "secret123", "name": "Alice"}' \
  http://localhost:8000/auth/register

# Login
TOKEN=$(curl -s -X POST -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "secret123"}' \
  http://localhost:8000/auth/login | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# Use token for authenticated requests
AUTH="-H 'Authorization: Bearer $TOKEN'"

# Get current user
curl $AUTH http://localhost:8000/auth/me

# Ingest (PDF, image, DOCX, HTML, audio)
curl -X POST $AUTH -F "file=@paper.pdf" http://localhost:8000/ingest

# Ingest from URL
curl -X POST $AUTH "http://localhost:8000/ingest/url?url=https://example.com/article.html"

# List documents (user-scoped)
curl $AUTH http://localhost:8000/documents

# Get document details
curl $AUTH http://localhost:8000/documents/1

# Delete a document
curl -X DELETE $AUTH http://localhost:8000/documents/1

# Re-index all
curl -X POST $AUTH http://localhost:8000/reindex

# Query
curl -X POST $AUTH -H "Content-Type: application/json" \
  -d '{"question": "What is attention?", "session_id": "abc"}' \
  http://localhost:8000/query

# Stream
curl -N "http://localhost:8000/query/stream?question=Explain+attention&session_id=abc"

# Health
curl http://localhost:8000/health

# History
curl "http://localhost:8000/history?session_id=abc"

# Feedback (thumbs up)
curl -X POST $AUTH -H "Content-Type: application/json" \
  -d '{"session_id": "abc", "message_id": 1, "rating": 1}' \
  http://localhost:8000/feedback

# Metrics (Prometheus)
curl http://localhost:8000/metrics

# Usage stats (cost, latency, cache ratio)
curl http://localhost:8000/metrics/stats
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
| `GUARDRAIL_ENABLED` | `true` | Enable all guardrails |
| `GUARDRAIL_INPUT_CHECK` | `true` | Check user questions for injection/toxicity/PII |
| `GUARDRAIL_OUTPUT_CHECK` | `true` | Check LLM answers for hallucination/relevance/toxicity |
| `GUARDRAIL_BLOCK_ON_INPUT_VIOLATION` | `true` | Block & return safe message on input violation |
| `GUARDRAIL_BLOCK_ON_OUTPUT_VIOLATION` | `false` | Re-generate on output violation (false = flag only) |
| `GUARDRAIL_LOG_TO_LANGFUSE` | `true` | Log violations as LangFuse traces |
| `AUDIO_TRANSCRIPTION_PROVIDER` | `openai` | `openai` (Whisper API) or `local` (openai-whisper) |
| `AUDIO_TRANSCRIPTION_MODEL` | `whisper-1` | Model name for provider |
| `AUDIO_TRANSCRIPTION_LANGUAGE` | — | Optional ISO language code (e.g. `en`) |
| `AUDIO_CHUNK_SECONDS` | `300` | Chunk size for large files (seconds) |
| `AUDIO_MAX_FILE_SIZE_MB` | `200` | Max file size before forced chunking |
| `API_KEYS` | `[]` | Comma-separated list of valid API keys |
| `RATE_LIMIT` | `10/minute` | Default rate limit per endpoint |
| `CORS_ORIGINS` | `*` | Allowed CORS origins (comma-separated) |
| `REQUEST_TIMEOUT_SECONDS` | `120` | Per-request timeout |
| `ARQ_REDIS_URL` | `redis://localhost:6379/1` | Redis URL for task queue |
| `JWT_SECRET` | `change-me-in-production` | Secret key for JWT signing |
| `JWT_ALGORITHM` | `HS256` | JWT signing algorithm |
| `JWT_ACCESS_TOKEN_EXPIRE_SECONDS` | `3600` | Access token TTL (1h) |
| `JWT_REFRESH_TOKEN_EXPIRE_SECONDS` | `2592000` | Refresh token TTL (30d) |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FORMAT` | `json` | `json` or `text` |

## Project Structure

```
src/
├── config/          # pydantic-settings
├── core/            # EmbeddingFactory, QueryCache, retry, logging
├── models/          # Pydantic schemas
├── ingestion/       # PDF, audio (Whisper), unstructured multi-format
├── summarization/   # Groq + GPT-4o-mini summarization
├── retrieval/       # Qdrant hybrid, reranker, query transformer
├── generation/      # LangGraph graph + RAGChain + PromptManager
├── storage/         # Redis-backed docstore
├── chat/            # PostgreSQL chat history + feedback
├── pipeline/        # Pipeline orchestrator
├── auth/            # User model, JWT helpers, auth dependency
├── api/             # FastAPI, arq worker
├── ui/              # Streamlit web app
└── main.py          # CLI entry point

alembic/             # Database migrations
tests/               # pytest tests (unit + mock)
scripts/             # evaluate.py, test_set_example.json
.github/workflows/   # CI pipeline
```

## Development

```bash
pip install -r requirements.txt
pip install pytest pytest-asyncio ruff mypy pre-commit coverage alembic
pre-commit install

alembic upgrade head            # apply DB migrations
pytest                          # run tests
coverage run -m pytest && coverage report
ruff check src/ && ruff format src/
mypy src/

# Run background worker (for async ingestion)
arq src.api.worker.WorkerSettings
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## References

- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [LangChain Multi-vector Retriever](https://python.langchain.com/docs/how_to/multi_vector/)
- [Qdrant Hybrid Search](https://qdrant.tech/articles/hybrid-search/)
- [LangFuse Prompt Management](https://langfuse.com/docs/prompts)
- [Ragas Evaluation](https://docs.ragas.io/)
