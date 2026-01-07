"""Scrapers module for Alpha Gen."""

from alpha_gen.scrapers.base import BaseScraper, ScrapedData
from alpha_gen.scrapers.yahoo_finance import (
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
