"""Decorators and utilities for CLI commands."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, ParamSpec

import structlog
import typer
from rich import print as rprint

logger = structlog.get_logger(__name__)

P = ParamSpec("P")


def async_command[**P](
    func: Callable[P, Awaitable[dict[str, Any]]],
) -> Callable[P, None]:
    """Decorator to handle async command execution with error handling.

    Wraps an async function that returns a result dict and handles:
    - Running the async function
    - Error handling and logging
    - Success/failure status checking
    - Exit codes

    Args:
        func: Async function that returns a dict with 'status' key

    Returns:
        Synchronous wrapper function suitable for Typer commands
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
        try:
            result = asyncio.run(func(*args, **kwargs))

            if result.get("status") != "success":
                error_msg = result.get("error", "Unknown error")
                rprint(f"[red]Error: {error_msg}[/red]")
                raise typer.Exit(1)

        except typer.Exit:
            raise
        except Exception as e:
            logger.error(f"{func.__name__} failed", error=str(e))
            rprint(f"[red]Error: {e!s}[/red]")
            raise typer.Exit(1)

    return wrapper
