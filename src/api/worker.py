import logging
from src.pipeline.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


async def ingest_job(ctx, filename: str, content: bytes) -> dict:
    logger.info(f"Background ingestion: {filename}")
    pipeline = RAGPipeline()
    await pipeline.chat_history.init_db()
    result = await pipeline.ingest_file(filename, content)
    return result


class WorkerSettings:
    redis_settings = None
    functions = [ingest_job]
    poll_delay = 1.0
    max_jobs = 10
