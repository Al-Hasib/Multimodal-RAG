# Multimodal-RAG

Production-ready multimodal RAG application that extracts text, tables, and images from PDFs, generates summaries via LLMs, and answers questions using retrieved multimodal context.

## Architecture

```
                    ┌─────────────┐
                    │    PDF      │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Ingestion  │  Unstructured partition_pdf
                    │  (text,     │  → CompositeElement, Table, Image
                    │   tables,   │
                    │   images)   │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │  Text    │ │  Table   │ │  Image   │
        │Summary   │ │Summary   │ │Summary   │
        │(Groq)    │ │(Groq)    │ │(GPT-4o)  │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │            │            │
             └────────────┼────────────┘
                          ▼
              ┌─────────────────────┐
              │   MultiVector       │
              │   Retriever         │  Qdrant (embeddings)
              │   docstore (data)   │  InMemoryStore (raw content)
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │   RAG Chain         │  GPT-4o-mini
              │   (text + images)   │  Multimodal prompt
              └──────────┬──────────┘
                         ▼
                   ┌──────────┐
                   │ Response │
                   └──────────┘
```

## Quick Start

### Prerequisites
- Python 3.10+
- System deps: `poppler-utils`, `tesseract-ocr`, `libmagic-dev`

```bash
# Ubuntu
sudo apt-get install poppler-utils tesseract-ocr libmagic-dev

# macOS
brew install poppler tesseract libmagic
```

### Installation

```bash
git clone <repo>
cd Multimodal-RAG

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your API keys
```

### Local (no Docker)

First start Qdrant:
```bash
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Then run the app:
```bash
# Ingest a PDF
python -m src.main ingest ./paper.pdf

# Ask a question
python -m src.main query "What is multi-head attention?" --k 5

# Run API server
python -m src.main api
```

### Docker Compose (recommended)

One command — starts Qdrant + the app with hot-reload:

```bash
docker compose up --build
```

The `src/` directory is mounted live, so any code change is reflected immediately without rebuilding.

Usage:
```bash
# Ingest a PDF
curl -X POST -F "file=@paper.pdf" http://localhost:8000/ingest

# Query
curl -X POST -H "Content-Type: application/json" \
  -d '{"question": "What is the Transformer architecture?"}' \
  http://localhost:8000/query
```

## Project Structure

```
src/
├── config/          # Settings via pydantic-settings
├── models/          # Pydantic schemas
├── ingestion/       # PDF parsing with unstructured
├── summarization/   # Text (Groq/Llama) + Image (GPT-4o-mini) summarization
├── retrieval/       # Qdrant + MultiVectorRetriever
├── generation/      # RAG chain with multimodal context
├── pipeline/        # Orchestration
├── api/             # FastAPI endpoints
└── main.py          # CLI entry point
```

## Configuration

All settings via `.env` or environment variables:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `GROQ_API_KEY` | — | Groq API key |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | — | Qdrant API key (if auth enabled) |
| `QDRANT_COLLECTION_NAME` | `multi_modal_rag` | Qdrant collection name |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Chat model for image summaries & RAG |
| `GROQ_CHAT_MODEL` | `llama-3.1-8b-instant` | Text/table summarization model |

## References

- [LangChain Multi-vector Retriever](https://python.langchain.com/docs/how_to/multi_vector/)
- [Unstructured PDF Partitioning](https://docs.unstructured.io/open-source/core-functionality/chunking)
- [Qdrant](https://qdrant.tech/documentation/)
