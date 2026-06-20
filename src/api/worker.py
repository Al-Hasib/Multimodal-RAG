import logging
from src.pipeline.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


async def ingest_job(ctx, filename: str, content: bytes, user_id: int = 0) -> dict:
    logger.info(f"Background ingestion: {filename} (user_id={user_id})")
    pipeline = RAGPipeline()
    await pipeline.chat_history.init_db()
    result = await pipeline.ingest_file(filename, content, user_id=user_id)
    return result


class WorkerSettings:
    redis_settings = None
    functions = [ingest_job]
    poll_delay = 1.0
    max_jobs = 10
