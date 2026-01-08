"""Opportunities command for finding investment opportunities."""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
import typer
from rich import print as rprint

from alpha_gen.agents import find_opportunities
from alpha_gen.cli.helpers import output_result

logger = structlog.get_logger(__name__)

opportunities_app = typer.Typer(
    name="opportunities",
    help="Find investment opportunities from underperforming stocks",
    no_args_is_help=True,
)


@opportunities_app.command("opportunities")
def opportunities_command(
    limit: int = typer.Option(
        25,
        "--limit",
        "-l",
        help="Number of losers to analyze",
    ),
    output: str = typer.Option(
        "text",
        "--output",
        "-o",
        help="Output format (text, json, markdown)",
    ),
) -> None:
    """Find investment opportunities from underperforming stocks."""
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
