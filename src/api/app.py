import json
import logging
import aiofiles
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
from src.models.schemas import QueryRequest, QueryResponse
from src.pipeline.pipeline import RAGPipeline
from src.config.settings import settings
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

_pipeline: RAGPipeline | None = None


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


app = FastAPI(title="Multimodal RAG API", version="3.0.0", lifespan=lifespan)


def create_app() -> FastAPI:
    return app


@app.get("/health")
async def health():
    checks = {"status": "ok", "version": "3.0.0"}
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
    checks["langfuse"] = "configured" if settings.langfuse_enabled else "disabled"
    return checks


@app.post("/ingest", summary="Ingest a PDF document")
async def ingest(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    content = await file.read()
    temp_path = f"/tmp/{file.filename}"
    async with aiofiles.open(temp_path, "wb") as f:
        await f.write(content)
    try:
        pipe = get_pipeline()
        result = await pipe.ingest(temp_path)
        return {"message": f"Ingested {file.filename}", "status": "success", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
