"""CLI module for Alpha Gen - Main entry point."""

from __future__ import annotations

import structlog
import typer

from alpha_gen.cli.commands import (
    analyze_app,
    news_app,
    opportunities_app,
)
from alpha_gen.cli.commands.research import research_command
from alpha_gen.core.config.settings import get_config
from alpha_gen.core.utils.logging import setup_logging

# Create main Typer app
app = typer.Typer(
    name="alpha-gen",
    help="""
    🤖 Alpha Gen - AI-Powered Investment Research Assistant

    Leverage multi-agentic AI to analyze stocks, discover opportunities, and stay informed
    about market trends using real-time data from Alpha Vantage API.

    Features:
      • Deep-dive company research with comprehensive analysis
      • Market opportunity discovery from underperforming stocks
      • News sentiment analysis for investment insights
      • Quick stock analysis with optional news integration

    Examples:
      alpha-gen research AAPL              # Research Apple Inc.
      alpha-gen opportunities --limit 50   # Find top 50 opportunities
      alpha-gen news                       # Analyze market news
      alpha-gen analyze NVDA --news        # Quick analysis with news
    """,
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Register commands
app.command(name="research", help="📊 Deep-Dive Company Research")(research_command)
app.add_typer(opportunities_app, name="opportunities")
app.add_typer(news_app, name="news")
app.add_typer(analyze_app, name="analyze")


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
        help="Show version and exit",
        callback=version_callback,
        is_eager=True,
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug mode with verbose logging",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Set logging level (DEBUG, INFO, WARNING, ERROR)",
    ),
) -> None:
    """
    🤖 Alpha Gen - AI-Powered Investment Research Assistant

    Multi-agentic AI system for comprehensive investment research and analysis.
    """
    setup_app(debug=debug, log_level=log_level)


def create_app() -> typer.Typer:
    """Create and configure the Typer application."""
    return app


def entrypoint() -> None:
    """Main entry point for the CLI application."""
    app()


if __name__ == "__main__":
    entrypoint()
