"""Gather command for collecting and storing financial data."""

from __future__ import annotations

from typing import Annotated, Any

import typer
from rich import print as rprint

from alpha_gen.cli.base import OutputOption, SaveOption
from alpha_gen.cli.decorators import async_command
from alpha_gen.core.agents import gather_multiple_tickers


def parse_tickers(tickers_str: str) -> list[str]:
    """Parse comma-separated tickers into a list.

    Args:
        tickers_str: Comma-separated ticker symbols

    Returns:
        List of uppercase ticker symbols
    """
    return [t.strip().upper() for t in tickers_str.split(",") if t.strip()]


@async_command
async def gather_command(
    tickers: Annotated[
        str,
        typer.Argument(
            ...,
            help="Stock ticker symbol(s), comma-separated (e.g., AAPL, MSFT, TSLA)",
        ),
    ],
    output: OutputOption = "text",
    save: SaveOption = False,
) -> dict[str, Any]:
    """
    📥 Gather and store financial data for tickers

    Fetches company fundamentals, news, and sentiment data from Alpha Vantage
    and stores it in the vector database for later use in research.

    This pre-gathers data so you can run 'research --skip-gather' for faster analysis.

    Examples:
        alpha-gen gather AAPL              # Gather data for Apple
        alpha-gen gather AAPL,MSFT,TSLA    # Gather data for multiple tickers
        alpha-gen gather AAPL --save       # Gather and save report
    """
    ticker_list = parse_tickers(tickers)

    if not ticker_list:
        rprint("[red]Error: No valid tickers provided[/red]")
        return {"status": "error", "error": "No valid tickers provided"}

    if len(ticker_list) == 1:
        rprint(f"[bold]Gathering data for {ticker_list[0]}...[/bold]")
    else:
        rprint(f"[bold]Gathering data for {len(ticker_list)} tickers...[/bold]")
        rprint(f"Tickers: {', '.join(ticker_list)}")

    # Run gather
    result = await gather_multiple_tickers(ticker_list)

    # Handle output if successful
    if result.get("status") == "success":
        successful = result.get("successful", 0)
        failed = result.get("failed", 0)
        total = result.get("total_tickers", 0)

        # Build summary content
        lines: list[str] = []
        lines.append("Gather Summary")
        lines.append("")
        lines.append(f"Total tickers: {total}")
        lines.append(f"Successful: {successful}")
        lines.append(f"Failed: {failed}")
        lines.append("")

        # Add details for each ticker
        for ticker_result in result.get("results", []):
            ticker = ticker_result.get("ticker", "Unknown")
            docs = ticker_result.get("docs_added", 0)
            news = ticker_result.get("news_articles_stored", 0)
            quarter = ticker_result.get("latest_quarter", "N/A")
            lines.append(f"{ticker}:")
            lines.append(f"  - Documents stored: {docs}")
            lines.append(f"  - News articles: {news}")
            lines.append(f"  - Latest quarter: {quarter}")
            lines.append("")

        # Add errors if any
        errors = result.get("errors")
        if errors:
            lines.append("Errors:")
            for error_info in errors:
                ticker = error_info.get("ticker", "Unknown")
                error_msg = error_info.get("error", "Unknown error")
                lines.append(f"  - {ticker}: {error_msg}")
            lines.append("")

        content = "\n".join(lines)

        from alpha_gen.cli.helpers import output_result

        output_result(
            output_format=output,
            title=f"Gather Report: {', '.join(ticker_list)}",
            content=content,
            metadata={
                "total_tickers": str(total),
                "successful": str(successful),
                "failed": str(failed),
            },
            save=save,
            filename_prefix=f"gather_{ticker_list[0]}",
        )

    return result
