import time
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from prometheus_client import Counter, Histogram, Gauge
from src.config.settings import settings

logger = logging.getLogger(__name__)

# ── Prometheus metrics ──────────────────────────────────────────────

rag_requests_total = Counter("rag_requests_total", "Total query requests", ["status"])
rag_requests_duration = Histogram(
    "rag_requests_duration_seconds", "Query latency (total)",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)
rag_retrieval_duration = Histogram(
    "rag_retrieval_duration_seconds", "Retrieval latency",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)
rag_generation_duration = Histogram(
    "rag_generation_duration_seconds", "Generation latency",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)
rag_guardrail_duration = Histogram(
    "rag_guardrail_duration_seconds", "Guardrail latency",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
)
rag_total_tokens = Counter("rag_total_tokens", "Total tokens consumed", ["type"])
rag_cost_total = Counter("rag_cost_total", "Total cost (USD)")
rag_cache_hits = Counter("rag_cache_hits", "Cache hit count")
rag_cache_misses = Counter("rag_cache_misses", "Cache miss count")
rag_document_count = Gauge("rag_document_count", "Number of ingested documents")

# Token → cost mapping (per 1K tokens)
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
    "whisper-1": {"input": 0.006, "output": 0.0},  # per minute of audio
}


def estimate_cost(model: str, input_tokens: int = 0, output_tokens: int = 0) -> float:
    pricing = MODEL_PRICING.get(model, {"input": 0.001, "output": 0.002})
    return (input_tokens / 1000) * pricing["input"] + (output_tokens / 1000) * pricing["output"]


# ── Per-request tracker ─────────────────────────────────────────────

_tracker_local = threading.local()


@dataclass
class RequestTracker:
    question: str = ""
    user_id: int = 0
    session_id: str = ""
    cache_hit: bool = False
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    guardrail_blocked: bool = False
    timings: dict = field(default_factory=dict)
    _start: float = field(default_factory=time.time)

    def checkpoint(self, name: str):
        self.timings[name] = time.time()

    def duration(self, start: str, end: str) -> float:
        return self.timings.get(end, time.time()) - self.timings.get(start, self._start)

    def finalize(self, total_tokens: int = 0, input_tokens: int = 0, output_tokens: int = 0, model: str = ""):
        elapsed = time.time() - self._start
        self.total_tokens = total_tokens or (input_tokens + output_tokens)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        if model:
            self.cost = estimate_cost(model, input_tokens, output_tokens)

        # Prometheus
        rag_requests_duration.observe(elapsed)
        rag_requests_total.labels(status="blocked" if self.guardrail_blocked else "ok").inc()
        if self.total_tokens:
            rag_total_tokens.labels(type="input").inc(self.input_tokens)
            rag_total_tokens.labels(type="output").inc(self.output_tokens)
        if self.cost:
            rag_cost_total.inc(self.cost)

        # Log
        logger.info(
            "Request completed",
            extra={
                "question": self.question[:100],
                "user_id": self.user_id,
                "cache_hit": self.cache_hit,
                "latency_seconds": round(elapsed, 3),
                "total_tokens": self.total_tokens,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cost": round(self.cost, 6),
                "guardrail_blocked": self.guardrail_blocked,
                "timings": {k: round(v - self._start, 3) for k, v in self.timings.items()},
            },
        )


def get_tracker() -> RequestTracker:
    if not hasattr(_tracker_local, "tracker"):
        _tracker_local.tracker = RequestTracker()
    return _tracker_local.tracker


def set_tracker(tracker: RequestTracker):
    _tracker_local.tracker = tracker


@contextmanager
def track_request(question: str = "", user_id: int = 0, session_id: str = ""):
    tracker = RequestTracker(question=question, user_id=user_id, session_id=session_id)
    set_tracker(tracker)
    try:
        yield tracker
    finally:
        tracker.finalize()
        if hasattr(_tracker_local, "tracker"):
            del _tracker_local.tracker
