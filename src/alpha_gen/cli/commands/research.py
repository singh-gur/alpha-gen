"""Research command for deep-dive company analysis."""

from __future__ import annotations

from typing import Annotated, Any

import typer
from rich import print as rprint

from alpha_gen.cli.base import OutputOption, SaveOption
from alpha_gen.cli.decorators import async_command
from alpha_gen.core.agents import research_company


@async_command
async def research_command(
    ticker: Annotated[
        str,
        typer.Argument(
            ...,
            help="Stock ticker symbol (e.g., AAPL, MSFT, TSLA)",
        ),
    ],
    skip_gather: Annotated[
        bool,
        typer.Option(
            "--skip-gather",
            "-g",
            help="Use pre-gathered data from vector store instead of fetching from API",
        ),
    ] = False,
    output: OutputOption = "text",
    save: SaveOption = False,
) -> dict[str, Any]:
    """
    📊 Conduct comprehensive research on a company

    Performs deep-dive analysis including fundamentals, market trends, and AI-generated insights.
    Uses real-time data from Alpha Vantage API for accurate market information.

    Use --skip-gather to use pre-gathered data for faster analysis (run 'gather' first).

    Examples:
        alpha-gen research AAPL              # Research with live data
        alpha-gen research AAPL --skip-gather # Use pre-gathered data
        alpha-gen research AAPL --save       # Save report to file
    """
    ticker = ticker.upper()

    if skip_gather:
        rprint(f"[bold]Researching {ticker} using pre-gathered data...[/bold]")
    else:
        rprint(f"[bold]Researching {ticker}...[/bold]")

    # Run research
    result = await research_company(ticker, skip_gather=skip_gather)

    # Handle output if successful
    if result.get("status") == "success":
        analysis = result.get("analysis", "No analysis available")
        duration = result.get("duration_ms", 0)
        latest_quarter = result.get("latest_quarter", "N/A")
        latest_news = result.get("latest_news_time", "N/A")

        from alpha_gen.cli.helpers import output_result

        output_result(
            output_format=output,
            title=f"Research Report: {ticker}",
            content=analysis,
            metadata={
                "ticker": ticker,
                "duration_ms": f"{duration:.0f}",
                "financial_data_as_of": latest_quarter,
                "latest_news_as_of": latest_news,
            },
            save=save,
            filename_prefix=f"research_{ticker}",
        )

    return result
