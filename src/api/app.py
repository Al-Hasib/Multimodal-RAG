import json
import uuid
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_fastapi_instrumentator import Instrumentator
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.schemas import (
    QueryRequest, QueryResponse, DocumentList, DocumentInfo,
    FeedbackRequest, JobInfo, JobStatus,
)
from src.auth.schemas import RegisterRequest, LoginRequest, TokenResponse, UserResponse
from src.auth.deps import get_current_user, create_access_token, create_refresh_token
from src.auth.models import get_user_by_email, create_user, User
from src.chat.history import ChatHistoryManager
from src.pipeline.pipeline import RAGPipeline
from src.config.settings import settings

logger = logging.getLogger(__name__)

_pipeline: RAGPipeline | None = None
SUPPORTED_EXTENSIONS = {
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp",
    ".docx", ".doc", ".html", ".htm", ".xhtml",
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".opus",
}


# ── Request ID middleware ──────────────────────────────────────────────

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        with logger.bind(request_id=request_id):
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response


# ── Error responses ────────────────────────────────────────────────────

class AppError(HTTPException):
    def __init__(self, code: str, message: str, status_code: int = 400):
        self.code = code
        super().__init__(status_code=status_code, detail={"code": code, "message": message})


ERROR_RESPONSES = {
    400: {"description": "Bad request"},
    401: {"description": "Unauthorized"},
    404: {"description": "Not found"},
    409: {"description": "Conflict"},
    422: {"description": "Validation error"},
    429: {"description": "Rate limit exceeded"},
    500: {"description": "Internal server error"},
}


# ── Rate limiter ───────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)


# ── Lifespan ───────────────────────────────────────────────────────────

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

    Instrumentator().instrument(app).expose(app, include_in_schema=False)

    yield
    await _pipeline.chat_history.close()


app = FastAPI(
    title="Multimodal RAG API",
    version="3.3.0",
    lifespan=lifespan,
    responses=ERROR_RESPONSES,
)


# ── Middleware ──────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(",") if settings.cors_origins != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestIDMiddleware)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def create_app() -> FastAPI:
    return app


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


# ── Global exception handlers ────────────────────────────────────────

@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    return JSONResponse(status_code=exc.status_code, content={"code": exc.code, "message": exc.detail["message"]})

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(status_code=500, content={"code": "internal_error", "message": "An unexpected error occurred"})


# ── Auth ──────────────────────────────────────────────────────────────

@app.post("/auth/register", response_model=TokenResponse, summary="Register a new user")
async def register(body: RegisterRequest):
    pipe = get_pipeline()
    async with pipe.chat_history.async_session() as session:
        existing = await get_user_by_email(session, body.email)
        if existing:
            raise AppError(code="email_exists", message="Email already registered", status_code=409)
        user = await create_user(session, body.email, body.password, body.name)
    access = create_access_token(user.id, user.email)
    refresh = create_refresh_token(user.id, user.email)
    return TokenResponse(access_token=access, refresh_token=refresh)


@app.post("/auth/login", response_model=TokenResponse, summary="Login with email and password")
async def login(body: LoginRequest):
    pipe = get_pipeline()
    async with pipe.chat_history.async_session() as session:
        user = await get_user_by_email(session, body.email)
        if not user or not user.verify_password(body.password):
            raise AppError(code="invalid_credentials", message="Invalid email or password", status_code=401)
    access = create_access_token(user.id, user.email)
    refresh = create_refresh_token(user.id, user.email)
    return TokenResponse(access_token=access, refresh_token=refresh)


@app.get("/auth/me", response_model=UserResponse, summary="Get current user info")
async def get_me(current_user: dict = Depends(get_current_user)):
    if current_user["user_id"] == 0:
        return UserResponse(id=0, email="api_key", name="API Key")
    pipe = get_pipeline()
    async with pipe.chat_history.async_session() as session:
        stmt = __import__("sqlalchemy").select(User).where(User.id == current_user["user_id"])
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()
        if not user:
            raise AppError(code="not_found", message="User not found", status_code=404)
        return UserResponse(id=user.id, email=user.email, name=user.name, is_active=user.is_active, created_at=user.created_at)


# ── Health ──────────────────────────────────────────────────────────

@app.get("/health", summary="Health check with dependency status")
async def health():
    checks = {"status": "ok", "version": "3.3.0"}
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


# ── Ingestion ───────────────────────────────────────────────────────

@app.post("/ingest", summary="Ingest a document", dependencies=[Depends(get_current_user)])
@limiter.limit(settings.rate_limit)
async def ingest(request: Request, file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    if not file.filename:
        raise AppError(code="missing_filename", message="No filename provided", status_code=400)

    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise AppError(
            code="unsupported_format",
            message=f"Unsupported format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            status_code=400,
        )

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)

    if size_mb > 10:
        try:
            from arq import create_pool
            pool = await create_pool(settings.arq_redis_url)
            from src.api.worker import ingest_job
            job = await pool.enqueue_job("ingest_job", file.filename, content, current_user["user_id"])
            await pool.close()
            return {"status": "queued", "job_id": job.job_id, "filename": file.filename, "size_mb": round(size_mb, 1)}
        except Exception as e:
            logger.warning(f"Task queue unavailable, processing inline: {e}")

    try:
        pipe = get_pipeline()
        result = await pipe.ingest_file(file.filename, content, user_id=current_user["user_id"])
        return {"status": "completed", **result}
    except Exception as e:
        raise AppError(code="ingestion_failed", message=str(e), status_code=500)


@app.post("/ingest/url", summary="Ingest a document from URL", dependencies=[Depends(get_current_user)])
@limiter.limit(settings.rate_limit)
async def ingest_url(request: Request, url: str = Query(...), current_user: dict = Depends(get_current_user)):
    if not url.startswith(("http://", "https://")):
        raise AppError(code="invalid_url", message="URL must start with http:// or https://", status_code=400)
    try:
        pipe = get_pipeline()
        result = await pipe.ingest(url, file_format=None, user_id=current_user["user_id"])
        return {"status": "completed", **result}
    except Exception as e:
        raise AppError(code="ingestion_failed", message=str(e), status_code=500)


# ── Job Status ─────────────────────────────────────────────────────

@app.get("/jobs/{job_id}", response_model=JobInfo, summary="Check background job status")
async def get_job_status(job_id: str):
    try:
        from arq import create_pool
        pool = await create_pool(settings.arq_redis_url)
        info = await pool.get_job_result(job_id)
        await pool.close()
        if info is None:
            return JobInfo(job_id=job_id, status=JobStatus.RUNNING)
        success, result, _ = info
        if success:
            return JobInfo(job_id=job_id, status=JobStatus.COMPLETED, result=result)
        else:
            return JobInfo(job_id=job_id, status=JobStatus.FAILED, error=str(result))
    except Exception as e:
        return JobInfo(job_id=job_id, status=JobStatus.FAILED, error=str(e))


# ── Document Management ─────────────────────────────────────────────

@app.get("/documents", response_model=DocumentList, summary="List documents", dependencies=[Depends(get_current_user)])
async def list_documents(skip: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200), current_user: dict = Depends(get_current_user)):
    pipe = get_pipeline()
    docs, total = await pipe.list_documents(skip=skip, limit=limit, user_id=current_user["user_id"])
    return DocumentList(documents=docs, total=total)


@app.get("/documents/{doc_id}", response_model=DocumentInfo, summary="Get document details", dependencies=[Depends(get_current_user)])
async def get_document(doc_id: int, current_user: dict = Depends(get_current_user)):
    pipe = get_pipeline()
    doc = await pipe.get_document(doc_id, user_id=current_user["user_id"])
    if not doc:
        raise AppError(code="not_found", message="Document not found", status_code=404)
    return doc


@app.delete("/documents/{doc_id}", summary="Delete a document", dependencies=[Depends(get_current_user)])
async def delete_document(doc_id: int, current_user: dict = Depends(get_current_user)):
    pipe = get_pipeline()
    ok = await pipe.delete_document(doc_id, user_id=current_user["user_id"])
    if not ok:
        raise AppError(code="not_found", message="Document not found", status_code=404)
    return {"message": f"Document {doc_id} deleted", "status": "success"}


@app.post("/reindex", summary="Re-index all documents", dependencies=[Depends(get_current_user)])
async def reindex(current_user: dict = Depends(get_current_user)):
    pipe = get_pipeline()
    count = await pipe.reindex_all()
    return {"message": f"Re-indexed {count} documents", "status": "success"}


# ── Query ───────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, summary="Ask a question", dependencies=[Depends(get_current_user)])
@limiter.limit(settings.rate_limit)
async def query(request: Request, body: QueryRequest, current_user: dict = Depends(get_current_user)):
    try:
        pipe = get_pipeline()
        return await pipe.aquery(
            question=body.question,
            k=body.k,
            session_id=body.session_id,
            user_id=current_user["user_id"],
            metadata_filter=body.metadata_filter,
        )
    except Exception as e:
        raise AppError(code="query_failed", message=str(e), status_code=500)


@app.get("/query/stream", summary="Stream a response", dependencies=[Depends(get_current_user)])
@limiter.limit(settings.rate_limit)
async def query_stream(
    request: Request,
    question: str = Query(..., min_length=1),
    session_id: str = Query(None),
    metadata_filter: str = Query(None, description="JSON-encoded Qdrant filter dict"),
    current_user: dict = Depends(get_current_user),
):
    async def event_stream():
        pipe = get_pipeline()
        parsed_filter = json.loads(metadata_filter) if metadata_filter else None
        async for chunk in pipe.astream(question=question, session_id=session_id, user_id=current_user["user_id"], metadata_filter=parsed_filter):
            yield f"data: {json.dumps({'token': chunk})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Chat History ────────────────────────────────────────────────────

@app.get("/history", summary="Get chat history", dependencies=[Depends(get_current_user)])
async def get_history(session_id: str = Query(...), current_user: dict = Depends(get_current_user)):
    pipe = get_pipeline()
    history = await pipe.chat_history.get_history(session_id, user_id=current_user["user_id"])
    return {"session_id": session_id, "messages": history}


# ── Feedback ────────────────────────────────────────────────────────

@app.post("/feedback", summary="Submit feedback on an answer", dependencies=[Depends(get_current_user)])
async def submit_feedback(fb: FeedbackRequest, current_user: dict = Depends(get_current_user)):
    pipe = get_pipeline()
    await pipe.chat_history.add_feedback(
        session_id=fb.session_id,
        message_id=fb.message_id,
        rating=fb.rating,
        comment=fb.comment,
        user_id=current_user["user_id"],
    )
    return {"status": "ok", "message": "Feedback recorded"}


@app.get("/feedback/{session_id}", summary="Get feedback for a session", dependencies=[Depends(get_current_user)])
async def get_feedback(session_id: str, current_user: dict = Depends(get_current_user)):
    pipe = get_pipeline()
    items = await pipe.chat_history.get_feedback(session_id, user_id=current_user["user_id"])
    return {"session_id": session_id, "feedback": items}
