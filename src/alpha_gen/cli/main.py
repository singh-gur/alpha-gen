"""CLI module for Alpha Gen - Main entry point."""

from __future__ import annotations

import structlog
import typer

from alpha_gen.cli.commands import (
    analyze_app,
    news_app,
    opportunities_app,
    research_app,
)
from alpha_gen.config.settings import get_config
from alpha_gen.utils.logging import setup_logging

# Create main Typer app
app = typer.Typer(
    name="alpha-gen",
    help="AI-powered investment research assistant",
    add_completion=False,
    no_args_is_help=True,
)

# Register subcommands
app.add_typer(research_app, name="research", invoke_without_command=True)
app.add_typer(opportunities_app, name="opportunities", invoke_without_command=True)
app.add_typer(news_app, name="news", invoke_without_command=True)
app.add_typer(analyze_app, name="analyze", invoke_without_command=True)


logger = structlog.get_logger(__name__)


def version_callback(value: bool) -> None:
    """Show version information."""
    if value:
        config = get_config()
        from rich import print as rprint

        rprint(f"Alpha Gen v{config.app_version}")
        raise typer.Exit()


def setup_app(debug: bool = False, log_level: str = "INFO") -> None:
    """Setup application configuration."""
    _ = get_config()  # Load config to ensure it's available
    setup_logging(level=log_level, json_logs=False)


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version information",
        callback=version_callback,
        is_eager=True,
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug mode",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Set logging level",
    ),
) -> None:
    """Alpha Gen - AI-powered investment research assistant."""
    setup_app(debug=debug, log_level=log_level)


def create_app() -> typer.Typer:
    """Create and configure the Typer application."""
    return app


def entrypoint() -> None:
    """Main entry point for the CLI application."""
    app()


if __name__ == "__main__":
    entrypoint()
