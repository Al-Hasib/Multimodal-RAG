import json
import logging
import aiofiles
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
from src.models.schemas import QueryRequest, QueryResponse, DocumentList, DocumentInfo
from src.pipeline.pipeline import RAGPipeline
from src.config.settings import settings
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

_pipeline: RAGPipeline | None = None
SUPPORTED_EXTENSIONS = {
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp",
    ".docx", ".doc", ".html", ".htm", ".xhtml",
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".opus",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    logger.info("Starting Multimodal RAG API v3")
    _pipeline = RAGPipeline()
    await _pipeline.chat_history.init_db()

    if settings.langfuse_enabled:
        try:
            from langfuse import Langfuse
            langfuse = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            langfuse.auth_check()
            logger.info("LangFuse connected")
        except Exception as e:
            logger.warning(f"LangFuse connection failed: {e}")

    yield
    await _pipeline.chat_history.close()


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


app = FastAPI(title="Multimodal RAG API", version="3.1.0", lifespan=lifespan)


def create_app() -> FastAPI:
    return app


# ── Health ──────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    checks = {"status": "ok", "version": "3.1.0"}
    try:
        from redis import Redis as RedisClient
        r = RedisClient.from_url(settings.redis_url)
        r.ping()
        checks["redis"] = "ok"
        r.close()
    except Exception as e:
        checks["redis"] = f"error: {e}"
    try:
        from qdrant_client import QdrantClient
        q = QdrantClient(url=settings.qdrant_url)
        q.get_collections()
        checks["qdrant"] = "ok"
    except Exception as e:
        checks["qdrant"] = f"error: {e}"
    try:
        from asyncpg import connect
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"error: {e}"
    checks["langfuse"] = "configured" if settings.langfuse_enabled else "disabled"
    checks["guardrails"] = {
        "enabled": settings.guardrail_enabled,
        "input_check": settings.guardrail_input_check,
        "output_check": settings.guardrail_output_check,
    }
    return checks


# ── Ingestion (multi-format) ────────────────────────────────────────


@app.post("/ingest", summary="Ingest a document (PDF, image, DOCX, HTML)")
async def ingest(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    content = await file.read()
    try:
        pipe = get_pipeline()
        result = await pipe.ingest_file(file.filename, content)
        return {"message": f"Ingested {file.filename}", "status": "success", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/url", summary="Ingest a document from URL")
async def ingest_url(url: str = Query(..., description="URL to ingest")):
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid URL")
    try:
        pipe = get_pipeline()
        result = await pipe.ingest(url, file_format=None)
        return {"message": f"Ingested {url}", "status": "success", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Document Management ─────────────────────────────────────────────


@app.get("/documents", response_model=DocumentList, summary="List all ingested documents")
async def list_documents(skip: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
    pipe = get_pipeline()
    docs, total = await pipe.list_documents(skip=skip, limit=limit)
    return DocumentList(documents=docs, total=total)


@app.get("/documents/{doc_id}", response_model=DocumentInfo, summary="Get document details")
async def get_document(doc_id: int):
    pipe = get_pipeline()
    doc = await pipe.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.delete("/documents/{doc_id}", summary="Delete a document")
async def delete_document(doc_id: int):
    pipe = get_pipeline()
    ok = await pipe.delete_document(doc_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": f"Document {doc_id} deleted", "status": "success"}


@app.post("/reindex", summary="Re-index all documents from scratch")
async def reindex():
    pipe = get_pipeline()
    count = await pipe.reindex_all()
    return {"message": f"Re-indexed {count} documents", "status": "success"}


# ── Query ───────────────────────────────────────────────────────────


@app.post("/query", response_model=QueryResponse, summary="Ask a question")
async def query(request: QueryRequest):
    try:
        pipe = get_pipeline()
        return await pipe.aquery(
            question=request.question,
            k=request.k,
            session_id=request.session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query/stream", summary="Stream a response")
async def query_stream(question: str = Query(..., min_length=1), session_id: str = Query(None)):
    async def event_stream():
        pipe = get_pipeline()
        async for chunk in pipe.astream(question=question, session_id=session_id):
            yield f"data: {json.dumps({'token': chunk})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/history", summary="Get chat history")
async def get_history(session_id: str = Query(...)):
    pipe = get_pipeline()
    history = await pipe.chat_history.get_history(session_id)
    return {"session_id": session_id, "messages": history}
