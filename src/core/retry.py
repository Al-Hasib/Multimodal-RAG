import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


def default_retry(func=None, attempts: int = 3, min_wait: float = 1.0, max_wait: float = 10.0):
    decorator = retry(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            f"Retry {retry_state.attempt_number}/{attempts} after {retry_state.outcome.exception()}"
        ),
    )
    if func:
        return decorator(func)
    return decorator
