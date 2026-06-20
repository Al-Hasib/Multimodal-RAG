# Contributing

## Development Setup

```bash
# Clone and enter
git clone <repo> && cd Multimodal-RAG

# Python environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Install dev extras
pip install pytest pytest-asyncio ruff mypy pre-commit coverage
pre-commit install

# Start infra
docker compose up -d qdrant redis postgres

# Copy env and edit with API keys
cp .env.example .env
```

## Running Tests

```bash
# All tests
pytest

# With coverage
coverage run -m pytest && coverage report

# Specific file
pytest tests/test_retrieval.py -v
```

## Code Quality

```bash
# Lint
ruff check src/

# Format
ruff format src/

# Type check
mypy src/
```

## Project Structure

```
src/
├── config/          # Settings (pydantic-settings)
├── core/            # Embedding factory, cache, logging
├── models/          # Pydantic schemas
├── ingestion/       # PDF parsing (async)
├── summarization/   # Summarization with Groq + GPT-4o
├── retrieval/       # Qdrant hybrid search, reranker, query transformer
├── generation/      # LangGraph RAG chain, prompt management
├── storage/         # Redis docstore
├── chat/            # PostgreSQL chat history
├── pipeline/        # Orchestrator
├── api/             # FastAPI (REST + SSE)
├── ui/              # Streamlit
└── main.py          # CLI
```

## Adding a new embedding provider

1. Implement `BaseEmbeddings` in `src/core/embeddings.py`
2. Register it in `EmbeddingFactory`
3. Add config to `Settings`

## Prompt versioning

Prompts are managed via LangFuse. To add a new prompt:
1. Create it in the LangFuse dashboard
2. Add a fallback in `src/generation/prompts.py`'s `_local_prompt` registry
3. Set `PROMPT_USE_LANGFUSE=true` in `.env`

## Evaluation

```bash
# Run evaluation against a test set
python scripts/evaluate.py scripts/test_set_example.json --output results.json
```
