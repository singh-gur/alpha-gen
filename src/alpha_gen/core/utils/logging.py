"""Logging utilities for Alpha Gen application."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from langfuse.langchain import CallbackHandler


def setup_logging(
    level: str = "INFO",
    json_logs: bool = False,
) -> None:
    """Configure application logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_logs: Whether to output logs in JSON format
    """
    # Configure structlog
    if json_logs:
        structlog.configure(
            processors=[
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, level.upper())
            ),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=False,
        )
    else:
        structlog.configure(
            processors=[
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, level.upper())
            ),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=False,
        )

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        stream=sys.stdout,
        force=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    return structlog.get_logger(name)


class LogContext:
    """Context manager for adding temporary context to logs."""

    def __init__(self, **context: Any) -> None:
        self.context = context
        self._previous_context: dict[str, Any] = {}

    def __enter__(self) -> LogContext:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


def get_langfuse_handler() -> CallbackHandler | None:
    """Get Langfuse callback handler if configured.

    Returns:
        CallbackHandler instance if Langfuse is enabled and configured, None otherwise
    """
    import os

    from alpha_gen.core.config.settings import get_config

    config = get_config()

    if not config.observability.enabled or not config.observability.is_configured:
        return None

    try:
        # Set environment variables for Langfuse
        os.environ["LANGFUSE_PUBLIC_KEY"] = config.observability.public_key or ""
        os.environ["LANGFUSE_SECRET_KEY"] = config.observability.secret_key or ""
        os.environ["LANGFUSE_HOST"] = config.observability.host

        handler = CallbackHandler()
        return handler
    except Exception as e:
        get_logger(__name__).warning(
            "Failed to initialize Langfuse handler", error=str(e)
        )
        return None

    try:
        handler = CallbackHandler(
            public_key=config.observability.public_key,
            secret_key=config.observability.secret_key,
            host=config.observability.host,
        )
        return handler
    except Exception as e:
        get_logger(__name__).warning(
            "Failed to initialize Langfuse handler", error=str(e)
        )
        return None


def log_execution(
    func_name: str,
    args: dict[str, Any] | None = None,
    result: Any | None = None,
    error: Exception | None = None,
    duration_ms: float | None = None,
) -> None:
    """Log function execution details.

    Args:
        func_name: Name of the function being executed
        args: Function arguments (sanitized)
        result: Function result
        error: Any error that occurred
        duration_ms: Execution duration in milliseconds
    """
    log_data: dict[str, Any] = {"function": func_name}

    if args:
        log_data["args"] = args
    if result is not None:
        log_data["result"] = str(result)[:1000]  # Truncate long results
    if error:
        log_data["error"] = str(error)
        get_logger(__name__).error(**log_data)
    elif duration_ms is not None:
        log_data["duration_ms"] = duration_ms
        get_logger(__name__).info(**log_data)
    else:
        get_logger(__name__).info(**log_data)
