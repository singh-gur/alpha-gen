"""Data sources module for Alpha Gen."""

from alpha_gen.core.data_sources.alpha_vantage import (
    AlphaVantageClient,
    CompanyOverview,
    NewsArticle,
    TopGainerLoser,
    fetch_company_overview,
    fetch_news_sentiment,
    fetch_top_gainers_losers,
)
from alpha_gen.core.data_sources.base import BaseDataSource, SourceData
from alpha_gen.core.data_sources.yahoo_finance import (
    YahooFinanceScraper,
    YahooNewsArticle,
    fetch_yahoo_general_news,
    fetch_yahoo_ticker_news,
)

__all__ = [
    "AlphaVantageClient",
    "BaseDataSource",
    "CompanyOverview",
    "NewsArticle",
    "SourceData",
    "TopGainerLoser",
    "YahooFinanceScraper",
    "YahooNewsArticle",
    "fetch_company_overview",
    "fetch_news_sentiment",
    "fetch_top_gainers_losers",
    "fetch_yahoo_general_news",
    "fetch_yahoo_ticker_news",
]
