"""Research command for deep-dive company analysis."""

from __future__ import annotations

from typing import Any

import typer
from rich import print as rprint

from alpha_gen.cli.base import OutputOption, SaveOption
from alpha_gen.cli.decorators import async_command
from alpha_gen.core.agents import research_company

research_app = typer.Typer(
    name="research",
    help="📊 Deep-Dive Company Research - Conduct comprehensive AI-powered research on any publicly traded company",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@research_app.callback(invoke_without_command=True)
@async_command
async def research_command(
    ctx: typer.Context,
    ticker: str = typer.Argument(
        None,
        help="Stock ticker symbol (e.g., AAPL, MSFT, TSLA)",
    ),
    output: OutputOption = "text",
    save: SaveOption = False,
) -> dict[str, Any]:
    """
    📊 Conduct comprehensive research on a company

    Performs deep-dive analysis including fundamentals, market trends, and AI-generated insights.
    Uses real-time data from Alpha Vantage API for accurate market information.

    Example: alpha-gen research AAPL --save
    """
    # If no ticker provided, show help
    if ticker is None:
        rprint(ctx.get_help())
        raise typer.Exit()

    ticker = ticker.upper()
    rprint(f"[bold]Researching {ticker}...[/bold]")

    # Run research
    result = await research_company(ticker)

    # Handle output if successful
    if result.get("status") == "success":
        analysis = result.get("analysis", "No analysis available")
        duration = result.get("duration_ms", 0)

        from alpha_gen.cli.helpers import output_result

        output_result(
            output_format=output,
            title=f"Research Report: {ticker}",
            content=analysis,
            metadata={
                "ticker": ticker,
                "duration_ms": f"{duration:.0f}",
            },
            save=save,
            filename_prefix=f"research_{ticker}",
        )

    return result
