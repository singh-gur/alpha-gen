"""Alpha Vantage API client for financial data."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

from alpha_gen.core.data_sources.base import BaseDataSource, SourceData

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class NewsArticle:
    """News article with sentiment data."""

    title: str
    url: str
    time_published: str
    authors: list[str]
    summary: str
    source: str
    category_within_source: str
    topics: list[dict[str, str]]
    overall_sentiment_score: float
    overall_sentiment_label: str
    ticker_sentiment: list[dict[str, Any]]


@dataclass(frozen=True)
class TopGainerLoser:
    """Top gainer or loser stock data."""

    ticker: str
    price: str
    change_amount: str
    change_percentage: str
    volume: str


@dataclass(frozen=True)
class CompanyOverview:
    """Company overview data."""

    symbol: str
    name: str
    description: str
    exchange: str
    currency: str
    country: str
    sector: str
    industry: str
    market_capitalization: str | None
    ebitda: str | None
    pe_ratio: str | None
    peg_ratio: str | None
    book_value: str | None
    dividend_per_share: str | None
    dividend_yield: str | None
    eps: str | None
    revenue_per_share_ttm: str | None
    profit_margin: str | None
    operating_margin_ttm: str | None
    return_on_assets_ttm: str | None
    return_on_equity_ttm: str | None
    revenue_ttm: str | None
    gross_profit_ttm: str | None
    diluted_eps_ttm: str | None
    quarterly_earnings_growth_yoy: str | None
    quarterly_revenue_growth_yoy: str | None
    analyst_target_price: str | None
    trailing_pe: str | None
    forward_pe: str | None
    price_to_sales_ratio_ttm: str | None
    price_to_book_ratio: str | None
    ev_to_revenue: str | None
    ev_to_ebitda: str | None
    beta: str | None
    week_52_high: str | None
    week_52_low: str | None
    day_50_moving_average: str | None
    day_200_moving_average: str | None
    shares_outstanding: str | None
    dividend_date: str | None
    ex_dividend_date: str | None


class AlphaVantageClient(BaseDataSource):
    """Client for Alpha Vantage API."""

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str, timeout: int = 30) -> None:
        super().__init__(timeout=timeout)
        self.api_key = api_key
        self._client = httpx.AsyncClient(timeout=timeout)

    async def fetch(self, **kwargs: Any) -> SourceData:
        """Generic fetch method (required by BaseDataSource)."""
        function = kwargs.get("function")
        if not function:
            raise ValueError("function parameter is required")

        params = {"apikey": self.api_key, **kwargs}
        return await self._make_request(params)

    async def _make_request(self, params: dict[str, Any]) -> SourceData:
        """Make HTTP request to Alpha Vantage API."""
        try:
            # Add API key to params if not already present
            if "apikey" not in params:
                params["apikey"] = self.api_key

            self._logger.info(
                "Making Alpha Vantage API request", function=params.get("function")
            )
            response = await self._client.get(self.BASE_URL, params=params)
            response.raise_for_status()

            data = response.json()

            # Check for API errors
            if "Error Message" in data:
                raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")
            if "Note" in data:
                self._logger.warning("Alpha Vantage API rate limit", note=data["Note"])
                raise ValueError(f"Alpha Vantage API rate limit: {data['Note']}")

            return SourceData(
                source="alpha_vantage",
                url=str(response.url),
                content=data,
                timestamp=time.time(),
            )
        except httpx.HTTPError as e:
            self._logger.error("Alpha Vantage API request failed", error=str(e))
            raise

    async def get_news_sentiment(
        self,
        tickers: str | None = None,
        topics: str | None = None,
        time_from: str | None = None,
        time_to: str | None = None,
        sort: str = "LATEST",
        limit: int = 50,
    ) -> SourceData:
        """Get news and sentiment data.

        Args:
            tickers: Comma-separated list of stock tickers (e.g., "AAPL,TSLA")
            topics: Comma-separated list of topics (e.g., "technology,earnings")
            time_from: Start time in YYYYMMDDTHHMM format
            time_to: End time in YYYYMMDDTHHMM format
            sort: Sort order (LATEST, EARLIEST, RELEVANCE)
            limit: Number of results (max 1000)

        Returns:
            SourceData containing news articles with sentiment
        """
        params: dict[str, Any] = {
            "function": "NEWS_SENTIMENT",
            "sort": sort,
            "limit": str(limit),
        }

        if tickers:
            params["tickers"] = tickers
        if topics:
            params["topics"] = topics
        if time_from:
            params["time_from"] = time_from
        if time_to:
            params["time_to"] = time_to

        return await self._make_request(params)

    async def get_top_gainers_losers(self) -> SourceData:
        """Get top gainers, losers, and most actively traded stocks.

        Returns:
            SourceData containing top gainers, losers, and most active stocks
        """
        params = {"function": "TOP_GAINERS_LOSERS"}
        return await self._make_request(params)

    async def get_company_overview(self, symbol: str) -> SourceData:
        """Get company overview and fundamental data.

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL")

        Returns:
            SourceData containing company overview and fundamentals
        """
        params = {
            "function": "OVERVIEW",
            "symbol": symbol,
        }
        return await self._make_request(params)

    async def get_quote(self, symbol: str) -> SourceData:
        """Get real-time quote data for a symbol.

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL")

        Returns:
            SourceData containing real-time quote data
        """
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
        }
        return await self._make_request(params)

    async def search_symbol(self, keywords: str) -> SourceData:
        """Search for stock symbols by keywords.

        Args:
            keywords: Search keywords (e.g., "Apple")

        Returns:
            SourceData containing matching symbols
        """
        params = {
            "function": "SYMBOL_SEARCH",
            "keywords": keywords,
        }
        return await self._make_request(params)

    async def get_market_status(self) -> SourceData:
        """Get global market status (open/closed).

        Returns:
            SourceData containing market status for global exchanges
        """
        params = {"function": "MARKET_STATUS"}
        return await self._make_request(params)

    async def close(self) -> None:
        """Clean up HTTP client resources."""
        await self._client.aclose()


# Convenience functions for common operations


async def fetch_news_sentiment(
    api_key: str,
    tickers: str | None = None,
    topics: str | None = None,
    limit: int = 50,
    timeout: int = 30,
) -> SourceData:
    """Fetch news sentiment data.

    Args:
        api_key: Alpha Vantage API key
        tickers: Comma-separated list of stock tickers
        topics: Comma-separated list of topics
        limit: Number of results
        timeout: Request timeout in seconds

    Returns:
        SourceData containing news articles with sentiment
    """
    client = AlphaVantageClient(api_key=api_key, timeout=timeout)
    try:
        return await client.get_news_sentiment(
            tickers=tickers, topics=topics, limit=limit
        )
    finally:
        await client.close()


async def fetch_top_gainers_losers(
    api_key: str,
    timeout: int = 30,
) -> SourceData:
    """Fetch top gainers and losers.

    Args:
        api_key: Alpha Vantage API key
        timeout: Request timeout in seconds

    Returns:
        SourceData containing top gainers, losers, and most active stocks
    """
    client = AlphaVantageClient(api_key=api_key, timeout=timeout)
    try:
        return await client.get_top_gainers_losers()
    finally:
        await client.close()


async def fetch_company_overview(
    api_key: str,
    symbol: str,
    timeout: int = 30,
) -> SourceData:
    """Fetch company overview data.

    Args:
        api_key: Alpha Vantage API key
        symbol: Stock ticker symbol
        timeout: Request timeout in seconds

    Returns:
        SourceData containing company overview and fundamentals
    """
    client = AlphaVantageClient(api_key=api_key, timeout=timeout)
    try:
        return await client.get_company_overview(symbol=symbol)
    finally:
        await client.close()
