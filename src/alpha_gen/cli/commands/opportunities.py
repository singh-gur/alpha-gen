"""Opportunities command for finding investment opportunities."""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
import typer
from rich import print as rprint

from alpha_gen.cli.helpers import output_result
from alpha_gen.core.agents import find_opportunities

logger = structlog.get_logger(__name__)

opportunities_app = typer.Typer(
    name="opportunities",
    help="💎 Investment Opportunity Discovery - Discover potential investment opportunities by analyzing underperforming stocks",
    no_args_is_help=False,
    rich_markup_mode="rich",
)


@opportunities_app.callback(invoke_without_command=True)
def opportunities_command(
    ctx: typer.Context,
    limit: int = typer.Option(
        25,
        "--limit",
        "-l",
        help="Number of top losers to analyze (default: 25, max recommended: 100)",
        min=1,
        max=100,
    ),
    output: str = typer.Option(
        "text",
        "--output",
        "-o",
        help="Output format: 'text' (rich console), 'json' (structured data), 'markdown' (formatted report)",
    ),
) -> None:
    """
    💎 Discover investment opportunities from market losers

    Analyzes underperforming stocks to identify potential recovery candidates.
    AI evaluates fundamentals, market conditions, and sentiment to find hidden gems.

    Examples:
      alpha-gen opportunities              # Analyze top 25 losers
      alpha-gen opportunities --limit 50   # Analyze top 50 losers
    """
    rprint(
        f"[bold]Finding investment opportunities from losers list (limit: {limit})...[/bold]"
    )

    async def run_opportunities() -> dict[str, Any]:
        return await find_opportunities(limit=limit)

    try:
        result = asyncio.run(run_opportunities())

        if result.get("status") == "success":
            losers_data = result.get("losers_data", {}).get("losers", [])
            analysis = result.get("analysis", "No analysis available")

            output_result(
                output_format=output,
                title="Investment Opportunities",
                content=analysis,
                metadata={"limit": limit},
                losers_data=losers_data,
            )
        else:
            rprint(f"[red]Error: {result.get('error', 'Unknown error')}[/red]")
            raise typer.Exit(1)

    except Exception as e:
        logger.error("Opportunities command failed", error=str(e))
        rprint(f"[red]Error: {e!s}[/red]")
        raise typer.Exit(1)
