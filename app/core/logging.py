"""Structured logging configuration.

Configures Python's standard logging with JSON-style structured output.
Sensitive data (journal text, tokens) is never logged by default.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone


class StructuredFormatter(logging.Formatter):
    """Formats log records as structured key=value lines.

    Produces human-readable structured logs suitable for local development
    and easy to parse in production log aggregators.
    """

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        level = record.levelname
        logger = record.name
        message = record.getMessage()

        # Base structured line
        line = f"ts={timestamp} level={level} logger={logger} msg={message!r}"

        # Include exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            exc_text = self.formatException(record.exc_info)
            line += f" exception={exc_text!r}"

        return line


def setup_logging(log_level: str = "INFO") -> None:
    """Configure application logging.

    Args:
        log_level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create structured formatter
    formatter = StructuredFormatter()

    # Configure root handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing non-test handlers to avoid duplicate output while preserving pytest caplog
    root_logger.handlers = [
        h
        for h in root_logger.handlers
        if "LogCapture" in type(h).__name__ or "_pytest" in getattr(type(h), "__module__", "")
    ]
    root_logger.addHandler(handler)

    # Quiet noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging initialized at level=%s",
        log_level,
    )
