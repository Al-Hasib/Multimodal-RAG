import hashlib
import json
import logging
from typing import Optional
from redis import Redis as RedisClient
from src.config.settings import settings

logger = logging.getLogger(__name__)


class QueryCache:
    def __init__(self):
        self.enabled = settings.redis_cache_enabled
        self.ttl = settings.redis_cache_ttl
        self.client: Optional[RedisClient] = None
        if self.enabled:
            try:
                self.client = RedisClient.from_url(settings.redis_url, decode_responses=True)
                self.client.ping()
                logger.info("Query cache connected to Redis")
            except Exception as e:
                logger.warning(f"Cache unavailable, running without: {e}")
                self.enabled = False

    def _make_key(self, question: str, k: int) -> str:
        raw = f"{question}:{k}"
        return f"cache:query:{hashlib.sha256(raw.encode()).hexdigest()}"

    def get(self, question: str, k: int) -> Optional[dict]:
        if not self.enabled or not self.client:
            return None
        key = self._make_key(question, k)
        data = self.client.get(key)
        if data:
            logger.debug(f"Cache hit for: {question[:50]}...")
            return json.loads(data)
        return None

    def set(self, question: str, k: int, result: dict) -> None:
        if not self.enabled or not self.client:
            return
        key = self._make_key(question, k)
        self.client.setex(key, self.ttl, json.dumps(result, default=str))
        logger.debug(f"Cached result for: {question[:50]}...")

    def invalidate(self, question: str, k: int) -> None:
        if not self.enabled or not self.client:
            return
        key = self._make_key(question, k)
        self.client.delete(key)
