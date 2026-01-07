"""CLI module for Alpha Gen."""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
import typer
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table

from ..agents import analyze_news, find_opportunities, research_company
from ..config.settings import get_config
from ..utils.logging import setup_logging

# Create Typer app
app = typer.Typer(
    name="alpha-gen",
    help="AI-powered investment research assistant",
    add_completion=False,
)


logger = structlog.get_logger(__name__)


def version_callback(value: bool) -> None:
    """Show version information."""
    if value:
        config = get_config()
        rprint(f"Alpha Gen v{config.app_version}")
        raise typer.Exit()


def setup_app(debug: bool = False, log_level: str = "INFO") -> None:
    """Setup application configuration."""
    _ = get_config()  # Load config to ensure it's available
    setup_logging(level=log_level, json_logs=False)


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version information",
        callback=version_callback,
        is_eager=True,
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug mode",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Set logging level",
    ),
) -> None:
    """Alpha Gen - AI-powered investment research assistant."""
    setup_app(debug=debug, log_level=log_level)


@app.command("research")
def research_command(
    ticker: str = typer.Argument(
        ...,
        help="Stock ticker symbol (e.g., AAPL, MSFT)",
    ),
    output: str = typer.Option(
        "text",
        "--output",
        "-o",
        help="Output format (text, json)",
    ),
) -> None:
    """Conduct deep-dive research on a company."""
    rprint(f"[bold]Researching {ticker}...[/bold]")

    async def run_research() -> dict[str, Any]:
        return await research_company(ticker.upper())

    try:
        result = asyncio.run(run_research())

        if result.get("status") == "success":
            rprint(Panel(
                result.get("analysis", "No analysis available"),
                title=f"Research Report: {ticker.upper()}",
                expand=False,
            ))

            if output == "json":
                import json
                rprint(json.dumps(result, indent=2, default=str))
        else:
            rprint(f"[red]Error: {result.get('error', 'Unknown error')}[/red]")
            raise typer.Exit(1)

    except Exception as e:
        logger.error("Research command failed", error=str(e))
        rprint(f"[red]Error: {e!s}[/red]")
        raise typer.Exit(1)


@app.command("opportunities")
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
        help="Output format (text, json)",
    ),
) -> None:
    """Find investment opportunities from underperforming stocks."""
    rprint(f"[bold]Finding investment opportunities from losers list (limit: {limit})...[/bold]")

    async def run_opportunities() -> dict[str, Any]:
        return await find_opportunities(limit=limit)

    try:
        result = asyncio.run(run_opportunities())

        if result.get("status") == "success":
            # Show losers table
            losers_data = result.get("losers_data", {}).get("losers", [])
            if losers_data:
                table = Table(title="Top Losers Analyzed")
                table.add_column("Ticker")
                table.add_column("Name")
                table.add_column("Price")
                table.add_column("Change")
                table.add_column("Volume")

                for loser in losers_data[:10]:
                    table.add_row(
                        loser.get("ticker", "N/A"),
                        loser.get("name", "N/A")[:30],
                        loser.get("price", "N/A"),
                        loser.get("change", "N/A"),
                        loser.get("volume", "N/A"),
                    )

                rprint(table)
                rprint("")

            rprint(Panel(
                result.get("analysis", "No analysis available"),
                title="Investment Opportunities",
                expand=False,
            ))

            if output == "json":
                import json
                rprint(json.dumps(result, indent=2, default=str))
        else:
            rprint(f"[red]Error: {result.get('error', 'Unknown error')}[/red]")
            raise typer.Exit(1)

    except Exception as e:
        logger.error("Opportunities command failed", error=str(e))
        rprint(f"[red]Error: {e!s}[/red]")
        raise typer.Exit(1)


@app.command("news")
def news_command(
    output: str = typer.Option(
        "text",
        "--output",
        "-o",
        help="Output format (text, json)",
    ),
) -> None:
    """Analyze recent news for investment opportunities."""
    rprint("[bold]Analyzing market news for investment opportunities...[/bold]")

    async def run_news() -> dict[str, Any]:
        return await analyze_news()

    try:
        result = asyncio.run(run_news())

        if result.get("status") == "success":
            rprint(Panel(
                result.get("analysis", "No analysis available"),
                title="News-Based Investment Opportunities",
                expand=False,
            ))

            if output == "json":
                import json
                rprint(json.dumps(result, indent=2, default=str))
        else:
            rprint(f"[red]Error: {result.get('error', 'Unknown error')}[/red]")
            raise typer.Exit(1)

    except Exception as e:
        logger.error("News command failed", error=str(e))
        rprint(f"[red]Error: {e!s}[/red]")
        raise typer.Exit(1)


@app.command("analyze")
def analyze_command(
    ticker: str = typer.Argument(
        ...,
        help="Stock ticker symbol to analyze",
    ),
    news: bool = typer.Option(
        False,
        "--news",
        "-n",
        help="Include news analysis",
    ),
    fundamentals: bool = typer.Option(
        False,
        "--fundamentals",
        "-f",
        help="Include fundamental analysis",
    ),
) -> None:
    """Quick analysis of a stock ticker."""
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
            # Create summary table
            table = Table(title=f"Quick Analysis: {ticker}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")

            table.add_row("Ticker", ticker)
            table.add_row("Analysis Status", "Complete")
            table.add_row("Duration", f"{result.get('duration_ms', 0):.0f}ms")

            rprint(table)
            rprint("")

            # Show abbreviated analysis
            analysis = result.get("analysis", "")
            if analysis:
                # Show first 2000 chars
                abbreviated = analysis[:2000] + "..." if len(analysis) > 2000 else analysis
                rprint(Panel(
                    abbreviated,
                    title=f"Analysis Preview: {ticker}",
                    expand=False,
                ))

            if news and result.get("news_analysis"):
                rprint(Panel(
                    result["news_analysis"][:2000] + "..." if len(result["news_analysis"]) > 2000 else result["news_analysis"],
                    title="News Analysis",
                    expand=False,
                ))

        else:
            rprint(f"[red]Error: {result.get('error', 'Unknown error')}[/red]")
            raise typer.Exit(1)

    except Exception as e:
        logger.error("Analyze command failed", error=str(e))
        rprint(f"[red]Error: {e!s}[/red]")
        raise typer.Exit(1)


def create_app() -> typer.Typer:
    """Create and configure the Typer application."""
    return app


def entrypoint() -> None:
    """Main entry point for the CLI application."""
    app()


if __name__ == "__main__":
    entrypoint()


if __name__ == "__main__":
    main()
