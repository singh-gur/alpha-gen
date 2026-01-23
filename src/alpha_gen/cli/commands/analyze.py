"""Analyze command for quick stock analysis."""

from __future__ import annotations

from typing import Any

import typer
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table

from alpha_gen.cli.base import OutputOption, SaveOption
from alpha_gen.cli.decorators import async_command
from alpha_gen.core.agents import analyze_news, research_company

analyze_app = typer.Typer(
    name="analyze",
    help="⚡ Quick Stock Analysis - Fast, focused analysis of any stock ticker with optional news integration",
    invoke_without_command=True,
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@analyze_app.callback()
@async_command
async def analyze_command(
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
    output: OutputOption = "text",
    save: SaveOption = False,
) -> dict[str, Any]:
    """
    ⚡ Quick analysis of a stock ticker

    Provides rapid insights with key metrics and AI-generated analysis.
    Add --news flag to include sentiment analysis from recent market news.

    Examples:
      alpha-gen analyze NVDA --save           # Quick analysis and save
      alpha-gen analyze TSLA --news --save    # Include news sentiment and save
    """
    # If no ticker provided, show help
    if ticker is None:
        raise typer.Exit()

    ticker = ticker.upper()
    rprint(f"[bold]Quick analysis for {ticker}...[/bold]")

    # Run analysis
    result = await research_company(ticker)

    if news:
        news_result = await analyze_news()
        result["news_analysis"] = news_result.get("analysis")

    # Handle output if successful
    if result.get("status") == "success":
        duration = result.get("duration_ms", 0)
        analysis = result.get("analysis", "")
        latest_quarter = result.get("latest_quarter", "N/A")
        latest_news = result.get("latest_news_time", "N/A")

        if output == "markdown":
            # Full markdown output
            content = f"## Quick Analysis\n\n{analysis}"
            if news and result.get("news_analysis"):
                content += f"\n\n## News Analysis\n\n{result['news_analysis']}"

            from alpha_gen.cli.helpers import output_result

            output_result(
                output_format=output,
                title=f"Quick Analysis: {ticker}",
                content=content,
                metadata={
                    "ticker": ticker,
                    "duration_ms": f"{duration:.0f}",
                    "financial_data_as_of": latest_quarter,
                    "latest_news_as_of": latest_news,
                },
                save=save,
                filename_prefix=f"analyze_{ticker}",
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

    return result
