"""Research command for deep-dive company analysis."""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
import typer
from rich import print as rprint

from ...agents import research_company
from ..helpers import output_result

logger = structlog.get_logger(__name__)

research_app = typer.Typer(
    name="research",
    help="Conduct deep-dive research on a company",
    no_args_is_help=True,
)


@research_app.command("research")
def research_command(
    ticker: str = typer.Argument(
        ...,
        help="Stock ticker symbol (e.g., AAPL, MSFT)",
    ),
    output: str = typer.Option(
        "text",
        "--output",
        "-o",
        help="Output format (text, json, markdown)",
    ),
) -> None:
    """Conduct deep-dive research on a company."""
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
