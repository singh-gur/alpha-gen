"""Analyze command for quick stock analysis."""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
import typer
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table

from alpha_gen.cli.helpers import output_result
from alpha_gen.core.agents import analyze_news, research_company

logger = structlog.get_logger(__name__)

analyze_app = typer.Typer(
    name="analyze",
    help="⚡ Quick Stock Analysis - Fast, focused analysis of any stock ticker with optional news integration",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@analyze_app.callback(invoke_without_command=True)
def analyze_command(
    ctx: typer.Context,
    ticker: str = typer.Argument(
        None,
        help="Stock ticker symbol to analyze (e.g., NVDA, GOOGL, AMZN)",
    ),
    news: bool = typer.Option(
        False,
        "--news",
        "-n",
        help="Include recent news sentiment analysis in the report",
    ),
    output: str = typer.Option(
        "text",
        "--output",
        "-o",
        help="Output format: 'text' (rich console), 'json' (structured data), 'markdown' (formatted report)",
    ),
) -> None:
    """
    ⚡ Quick analysis of a stock ticker

    Provides rapid insights with key metrics and AI-generated analysis.
    Add --news flag to include sentiment analysis from recent market news.

    Examples:
      alpha-gen analyze NVDA           # Quick analysis
      alpha-gen analyze TSLA --news    # Include news sentiment
    """
    # If no ticker provided, show help
    if ticker is None:
        rprint(ctx.get_help())
        raise typer.Exit()
    ticker = ticker.upper()
    rprint(f"[bold]Quick analysis for {ticker}...[/bold]")

    async def run_analysis() -> dict[str, Any]:
        result = await research_company(ticker)

        if news:
            news_result = await analyze_news()
            result["news_analysis"] = news_result.get("analysis")

        return result

    try:
        result = asyncio.run(run_analysis())

        if result.get("status") == "success":
            duration = result.get("duration_ms", 0)
            analysis = result.get("analysis", "")

            if output == "markdown":
                # Full markdown output
                content = f"## Quick Analysis\n\n{analysis}"
                if news and result.get("news_analysis"):
                    content += f"\n\n## News Analysis\n\n{result['news_analysis']}"

                output_result(
                    output_format=output,
                    title=f"Quick Analysis: {ticker}",
                    content=content,
                    metadata={
                        "ticker": ticker,
                        "duration_ms": f"{duration:.0f}",
                    },
                )
            else:
                # Text/Rich output
                table = Table(title=f"Quick Analysis: {ticker}")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="magenta")

                table.add_row("Ticker", ticker)
                table.add_row("Analysis Status", "Complete")
                table.add_row("Duration", f"{duration:.0f}ms")

                rprint(table)
                rprint("")

                if analysis:
                    abbreviated = (
                        analysis[:2000] + "..." if len(analysis) > 2000 else analysis
                    )
                    rprint(
                        Panel(
                            abbreviated,
                            title=f"Analysis Preview: {ticker}",
                            expand=False,
                        )
                    )

                if news and result.get("news_analysis"):
                    rprint(
                        Panel(
                            result["news_analysis"][:2000] + "..."
                            if len(result["news_analysis"]) > 2000
                            else result["news_analysis"],
                            title="News Analysis",
                            expand=False,
                        )
                    )
        else:
            rprint(f"[red]Error: {result.get('error', 'Unknown error')}[/red]")
            raise typer.Exit(1)

    except Exception as e:
        logger.error("Analyze command failed", error=str(e))
        rprint(f"[red]Error: {e!s}[/red]")
        raise typer.Exit(1)
