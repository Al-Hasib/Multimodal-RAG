import logging
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
from src.models.schemas import QueryRequest, QueryResponse
from src.pipeline.pipeline import RAGPipeline
from src.config.settings import settings

logger = logging.getLogger(__name__)


def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    app = FastAPI(title="Multimodal RAG API", version="2.0.0")

    if pipeline is None:
        pipeline = RAGPipeline()

    @app.on_event("startup")
    async def startup():
        logger.info("Starting Multimodal RAG API v2")
        await pipeline.chat_history.init_db()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/ingest", summary="Ingest a PDF document")
    async def ingest(file: UploadFile = File(...)):
        if not file.filename or not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        content = await file.read()
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(content)

        try:
            result = pipeline.ingest(temp_path)
            return {"message": f"Ingested {file.filename}", "status": "success", **result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/query", response_model=QueryResponse, summary="Ask a question")
    async def query(request: QueryRequest):
        try:
            result = await pipeline.aquery(
                question=request.question,
                k=request.k,
                session_id=request.session_id,
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/query/stream", summary="Stream a response")
    async def query_stream(
        question: str = Query(..., min_length=1),
        session_id: str = Query(None),
    ):
        async def event_stream():
            async for chunk in pipeline.astream(question=question, session_id=session_id):
                yield f"data: {json.dumps({'token': chunk})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/history", summary="Get chat history")
    async def get_history(session_id: str = Query(...)):
        history = await pipeline.chat_history.get_history(session_id)
        return {"session_id": session_id, "messages": history}

    return app
