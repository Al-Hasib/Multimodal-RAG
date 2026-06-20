import json
import pickle
from typing import Any, Iterable, Iterator, List, Optional, Tuple
from langchain.storage import BaseStore
from redis import Redis as RedisClient
from src.config.settings import settings
import logging

logger = logging.getLogger(__name__)


class RedisDocStore(BaseStore[str, Any]):
    def __init__(self, redis_url: Optional[str] = None, ttl: Optional[int] = None):
        url = redis_url or settings.redis_url
        self.ttl = ttl or settings.redis_docstore_ttl
        self.client = RedisClient.from_url(url, decode_responses=False)
        logger.info(f"Connected to Redis docstore at {url}")

    def mget(self, keys: List[str]) -> List[Optional[Any]]:
        results = self.client.mget(keys)
        return [pickle.loads(r) if r else None for r in results]

    def mset(self, key_value_pairs: List[Tuple[str, Any]]) -> None:
        pipe = self.client.pipeline()
        for key, value in key_value_pairs:
            data = pickle.dumps(value)
            pipe.setex(key, self.ttl, data)
        pipe.execute()

    def mdelete(self, keys: List[str]) -> None:
        self.client.delete(*keys)

    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
        pattern = f"{prefix or ''}*"
        cursor = 0
        while True:
            cursor, keys = self.client.scan(cursor=cursor, match=pattern, count=100)
            for key in keys:
                yield key.decode() if isinstance(key, bytes) else key
            if cursor == 0:
                break
