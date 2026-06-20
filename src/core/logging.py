import logging
import sys
from src.config.settings import settings


def setup_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    if settings.log_format == "json":
        try:
            from pythonjsonlogger import jsonlogger
            handler = logging.StreamHandler(sys.stdout)
            formatter = jsonlogger.JsonFormatter(
                fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
            handler.setFormatter(formatter)
            logging.basicConfig(level=level, handlers=[handler])
        except ImportError:
            _setup_text_logging(level)
    else:
        _setup_text_logging(level)


def _setup_text_logging(level: int) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
