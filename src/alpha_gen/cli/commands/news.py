"""News command for analyzing news-based opportunities."""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
import typer
from rich import print as rprint

from alpha_gen.agents import analyze_news
from alpha_gen.cli.helpers import output_result

logger = structlog.get_logger(__name__)

news_app = typer.Typer(
    name="news",
    help="Analyze recent news for investment opportunities",
    no_args_is_help=True,
)


@news_app.command("news")
def news_command(
    output: str = typer.Option(
        "text",
        "--output",
        "-o",
        help="Output format (text, json, markdown)",
    ),
) -> None:
    """Analyze recent news for investment opportunities."""
    rprint("[bold]Analyzing market news for investment opportunities...[/bold]")

    async def run_news() -> dict[str, Any]:
        return await analyze_news()

    try:
        result = asyncio.run(run_news())

        if result.get("status") == "success":
            analysis = result.get("analysis", "No analysis available")
            output_result(
                output_format=output,
                title="News-Based Investment Opportunities",
                content=analysis,
            )
        else:
            rprint(f"[red]Error: {result.get('error', 'Unknown error')}[/red]")
            raise typer.Exit(1)

    except Exception as e:
        logger.error("News command failed", error=str(e))
        rprint(f"[red]Error: {e!s}[/red]")
        raise typer.Exit(1)
