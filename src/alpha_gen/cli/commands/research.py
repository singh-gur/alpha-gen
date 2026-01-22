"""Research command for deep-dive company analysis."""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
import typer
from rich import print as rprint

from alpha_gen.cli.helpers import output_result
from alpha_gen.core.agents import research_company

logger = structlog.get_logger(__name__)

research_app = typer.Typer(
    name="research",
    help="📊 Deep-Dive Company Research - Conduct comprehensive AI-powered research on any publicly traded company",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@research_app.callback(invoke_without_command=True)
def research_command(
    ctx: typer.Context,
    ticker: str = typer.Argument(
        None,
        help="Stock ticker symbol (e.g., AAPL, MSFT, TSLA)",
    ),
    output: str = typer.Option(
        "text",
        "--output",
        "-o",
        help="Output format: 'text' (rich console), 'json' (structured data), 'markdown' (formatted report)",
    ),
) -> None:
    """
    📊 Conduct comprehensive research on a company

    Performs deep-dive analysis including fundamentals, market trends, and AI-generated insights.
    Uses real-time data from Alpha Vantage API for accurate market information.

    Example: alpha-gen research AAPL
    """
    # If no ticker provided, show help
    if ticker is None:
        rprint(ctx.get_help())
        raise typer.Exit()
    rprint(f"[bold]Researching {ticker}...[/bold]")

    async def run_research() -> dict[str, Any]:
        return await research_company(ticker.upper())

    try:
        result = asyncio.run(run_research())

        if result.get("status") == "success":
            analysis = result.get("analysis", "No analysis available")
            duration = result.get("duration_ms", 0)
            output_result(
                output_format=output,
                title=f"Research Report: {ticker.upper()}",
                content=analysis,
                metadata={
                    "ticker": ticker.upper(),
                    "duration_ms": f"{duration:.0f}",
                },
            )
        else:
            rprint(f"[red]Error: {result.get('error', 'Unknown error')}[/red]")
            raise typer.Exit(1)

    except Exception as e:
        logger.error("Research command failed", error=str(e))
        rprint(f"[red]Error: {e!s}[/red]")
        raise typer.Exit(1)
