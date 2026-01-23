"""News command for analyzing news-based opportunities."""

from __future__ import annotations

from typing import Any

import typer
from rich import print as rprint

from alpha_gen.cli.base import OutputOption, SaveOption
from alpha_gen.cli.decorators import async_command
from alpha_gen.core.agents import analyze_news

news_app = typer.Typer(
    name="news",
    help="📰 Market News Analysis - Analyze recent market news to identify investment opportunities and trends",
    invoke_without_command=True,
    no_args_is_help=False,
    rich_markup_mode="rich",
)


@news_app.callback()
@async_command
async def news_command(
    output: OutputOption = "text",
    save: SaveOption = False,
) -> dict[str, Any]:
    """
    📰 Analyze recent market news for investment opportunities

    Performs AI-powered sentiment analysis on recent market news to identify
    emerging trends, opportunities, and potential risks in the market.

    Example: alpha-gen news --save
    """
    rprint("[bold]Analyzing market news for investment opportunities...[/bold]")

    # Run news analysis
    result = await analyze_news()

    # Handle output if successful
    if result.get("status") == "success":
        analysis = result.get("analysis", "No analysis available")

        from alpha_gen.cli.helpers import output_result

        output_result(
            output_format=output,
            title="News-Based Investment Opportunities",
            content=analysis,
            save=save,
            filename_prefix="news",
        )

    return result
