"""Data sources module for Alpha Gen."""

from alpha_gen.data_sources.alpha_vantage import (
    AlphaVantageClient,
    CompanyOverview,
    NewsArticle,
    TopGainerLoser,
    fetch_company_overview,
    fetch_news_sentiment,
    fetch_top_gainers_losers,
)
from alpha_gen.data_sources.base import BaseDataSource, SourceData

__all__ = [
    "AlphaVantageClient",
    "BaseDataSource",
    "CompanyOverview",
    "NewsArticle",
    "SourceData",
    "TopGainerLoser",
    "fetch_company_overview",
    "fetch_news_sentiment",
    "fetch_top_gainers_losers",
]
