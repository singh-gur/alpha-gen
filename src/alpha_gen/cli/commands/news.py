"""News command for analyzing news-based opportunities."""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
import typer
from rich import print as rprint

from alpha_gen.cli.helpers import output_result
from alpha_gen.core.agents import analyze_news

logger = structlog.get_logger(__name__)

news_app = typer.Typer(
    name="news",
    help="📰 Market News Analysis - Analyze recent market news to identify investment opportunities and trends",
    no_args_is_help=False,
    rich_markup_mode="rich",
)


@news_app.callback(invoke_without_command=True)
def news_command(
    ctx: typer.Context,
    output: str = typer.Option(
        "text",
        "--output",
        "-o",
        help="Output format: 'text' (rich console), 'json' (structured data), 'markdown' (formatted report)",
    ),
) -> None:
    """
    📰 Analyze recent market news for investment opportunities

    Performs AI-powered sentiment analysis on recent market news to identify
    emerging trends, opportunities, and potential risks in the market.

    Example: alpha-gen news
    """
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
