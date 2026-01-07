"""Scrapers module for Alpha Gen."""

from .base import BaseScraper, ScrapedData
from .yahoo_finance import (
    YahooFinanceScraper,
    scrape_company_data,
    scrape_losers,
)

__all__ = [
    "BaseScraper",
    "ScrapedData",
    "YahooFinanceScraper",
    "scrape_company_data",
    "scrape_losers",
]
