"""Opportunities command for finding investment opportunities."""

from __future__ import annotations

from typing import Any

import typer
from rich import print as rprint

from alpha_gen.cli.base import OutputOption, SaveOption
from alpha_gen.cli.decorators import async_command
from alpha_gen.core.agents import find_opportunities

opportunities_app = typer.Typer(
    name="opportunities",
    help="💎 Find investment opportunities from underperforming stocks",
    invoke_without_command=True,
    no_args_is_help=False,
    rich_markup_mode=None,
)


@opportunities_app.callback()
@async_command
async def opportunities_command(
    limit: int = typer.Option(
        25,
        "--limit",
        "-l",
        help="Number of top losers to analyze (default: 25, max recommended: 100)",
        min=1,
        max=100,
    ),
    output: OutputOption = "text",
    save: SaveOption = False,
) -> dict[str, Any]:
    """
    💎 Discover investment opportunities from market losers

    Analyzes underperforming stocks to identify potential recovery candidates.
    AI evaluates fundamentals, market conditions, and sentiment to find hidden gems.

    Examples:
      alpha-gen opportunities --save              # Analyze top 25 losers and save
      alpha-gen opportunities --limit 50 --save   # Analyze top 50 losers and save
    """
    rprint(
        f"[bold]Finding investment opportunities from losers list (limit: {limit})...[/bold]"
    )

    # Run opportunities analysis
    result = await find_opportunities(limit=limit)

    # Handle output if successful
    if result.get("status") == "success":
        losers_data = result.get("losers_data", {}).get("losers", [])
        analysis = result.get("analysis", "No analysis available")

        from alpha_gen.cli.helpers import output_result

        output_result(
            output_format=output,
            title="Investment Opportunities",
            content=analysis,
            metadata={"limit": limit},
            losers_data=losers_data,
            save=save,
            filename_prefix="opportunities",
        )

    return result
