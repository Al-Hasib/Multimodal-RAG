import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from src.models.schemas import QueryRequest, QueryResponse
from src.pipeline.pipeline import RAGPipeline
from src.config.settings import settings

logger = logging.getLogger(__name__)


def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    app = FastAPI(title="Multimodal RAG API", version="1.0.0")

    if pipeline is None:
        pipeline = RAGPipeline()

    @app.on_event("startup")
    async def startup():
        logger.info("Starting Multimodal RAG API")

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
            pipeline.ingest(temp_path)
            return {"message": f"Ingested {file.filename}", "status": "success"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/query", response_model=QueryResponse, summary="Ask a question")
    async def query(request: QueryRequest):
        try:
            result = pipeline.query(request.question, k=request.k)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app
