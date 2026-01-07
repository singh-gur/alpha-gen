"""Yahoo Finance scraper module using Playwright."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from bs4 import BeautifulSoup, Tag
from playwright.async_api import async_playwright

from ..utils.logging import get_logger
from .base import BaseScraper, ScrapedData

logger = get_logger(__name__)


@dataclass(frozen=True)
class FinancialData:
    """Financial data for a company."""

    ticker: str
    company_name: str
    current_price: float
    price_change: float
    percent_change: float
    market_cap: float | None
    pe_ratio: float | None
    eps: float | None
    dividend_yield: float | None
    high_52_week: float | None
    low_52_week: float | None
    avg_volume: int | None
    revenue: float | None
    net_income: float | None
    eps_growth: float | None
    roe: float | None
    debt_to_equity: float | None


@dataclass(frozen=True)
class CompetitorData:
    """Competitor information."""

    ticker: str
    company_name: str
    market_cap: float | None
    pe_ratio: float | None


@dataclass(frozen=True)
class NewsArticle:
    """News article data."""

    title: str
    url: str
    publisher: str
    publish_date: datetime
    summary: str
    sentiment: str  # "positive", "negative", "neutral"


class YahooFinanceScraper(BaseScraper):
    """Scraper for Yahoo Finance data."""

    BASE_URL = "https://finance.yahoo.com"

    def __init__(
        self,
        timeout: int = 30,
        headless: bool = True,
    ) -> None:
        super().__init__(timeout=timeout, headless=headless)
        self._playwright: Any | None = None
        self._browser: Any | None = None
        self._context: Any | None = None

    async def _ensure_browser(self) -> None:
        """Ensure browser is initialized."""
        if self._browser is None:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=self.headless)
            assert self._browser is not None, "Browser should be initialized"
            self._context = await self._browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1920, "height": 1080},
            )

    async def _fetch_page(self, url: str) -> str:
        """Fetch page content using Playwright."""
        await self._ensure_browser()
        assert self._context is not None, "Browser context not initialized"

        page = await self._context.new_page()  # type: ignore[union-attr]
        try:
            self._logger.info("Fetching page", url=url)
            await page.goto(url, timeout=self.timeout * 1000, wait_until="domcontentloaded")

            # Wait for dynamic content to load
            await page.wait_for_load_state("networkidle")

            # Get page content
            content = await page.content()
            return content
        except Exception as e:
            self._logger.error("Failed to fetch page", url=url, error=str(e))
            raise
        finally:
            await page.close()

    async def scrape(self, **kwargs: Any) -> ScrapedData:  # type: ignore[override]
        """Scrape data from Yahoo Finance (required by BaseScraper)."""
        ticker = kwargs.get("ticker")
        if ticker:
            return await self.get_company_info(ticker)
        raise ValueError("No ticker provided for scraping")

    async def get_company_info(self, ticker: str) -> ScrapedData:
        """Get company information for a ticker."""
        url = f"{self.BASE_URL}/quote/{ticker}"
        content = await self._fetch_page(url)

        data = self._parse_company_info(content, ticker, url)
        return ScrapedData(
            source="yahoo_finance",
            url=url,
            content=data,
            timestamp=time.time(),
        )

    def _parse_company_info(self, html: str, ticker: str, url: str) -> dict[str, Any]:
        """Parse company information from HTML."""
        soup = BeautifulSoup(html, "lxml")

        data: dict[str, Any] = {
            "ticker": ticker,
            "url": url,
            "raw_html": str(soup)[:5000],
        }

        try:
            # Extract company name
            company_name_elem = soup.select_one("h1")
            if company_name_elem:
                data["company_name"] = company_name_elem.get_text(strip=True)

            # Extract price data
            price_elem = soup.select_one('[data-test="qsp-price"]')
            if price_elem:
                data["current_price"] = price_elem.get_text(strip=True)

            # Extract key statistics
            stats_table = soup.select('[data-test="qsp-statistics"]')
            if stats_table:
                data["statistics_html"] = str(stats_table)

            # Extract financial table
            financial_table = soup.select('[data-test="qsp-financials"]')
            if financial_table:
                data["financials_html"] = str(financial_table)

        except Exception as e:
            self._logger.warning("Failed to parse company info", ticker=ticker, error=str(e))

        return data

    async def get_financials(self, ticker: str) -> ScrapedData:
        """Get financial statements for a ticker."""
        url = f"{self.BASE_URL}/quote/{ticker}/financials"
        content = await self._fetch_page(url)

        data = self._parse_financials(content, ticker, url)
        return ScrapedData(
            source="yahoo_finance_financials",
            url=url,
            content=data,
            timestamp=time.time(),
        )

    def _parse_financials(self, html: str, ticker: str, url: str) -> dict[str, Any]:
        """Parse financial statements from HTML."""
        soup = BeautifulSoup(html, "lxml")

        data: dict[str, Any] = {
            "ticker": ticker,
            "url": url,
        }

        try:
            # Look for income statement, balance sheet, cash flow sections
            for section in ["income-statement", "balance-sheet", "cash-flow-statement"]:
                section_elem = soup.select_one(f'[data-test="{section}"]')
                if section_elem:
                    data[section] = str(section_elem)

        except Exception as e:
            self._logger.warning("Failed to parse financials", ticker=ticker, error=str(e))

        return data

    async def get_competitors(self, ticker: str) -> ScrapedData:
        """Get competitors for a ticker."""
        url = f"{self.BASE_URL}/quote/{ticker}/competitors"
        content = await self._fetch_page(url)

        data = self._parse_competitors(content, ticker, url)
        return ScrapedData(
            source="yahoo_finance_competitors",
            url=url,
            content=data,
            timestamp=time.time(),
        )

    def _parse_competitors(self, html: str, ticker: str, url: str) -> dict[str, Any]:
        """Parse competitor data from HTML."""
        soup = BeautifulSoup(html, "lxml")

        data: dict[str, Any] = {
            "ticker": ticker,
            "url": url,
        }

        try:
            # Look for competitor table
            table = soup.select_one("table")
            if table:
                data["competitors_html"] = str(table)

        except Exception as e:
            self._logger.warning("Failed to parse competitors", ticker=ticker, error=str(e))

        return data

    async def get_news(self, ticker: str, limit: int = 10) -> ScrapedData:
        """Get news articles for a ticker."""
        url = f"{self.BASE_URL}/quote/{ticker}/news"
        content = await self._fetch_page(url)

        data = self._parse_news(content, ticker, url, limit)
        return ScrapedData(
            source="yahoo_finance_news",
            url=url,
            content=data,
            timestamp=time.time(),
        )

    def _parse_news(self, html: str, ticker: str, url: str, limit: int) -> dict[str, Any]:
        """Parse news articles from HTML."""
        soup = BeautifulSoup(html, "lxml")

        data: dict[str, Any] = {
            "ticker": ticker,
            "url": url,
            "articles": [],
        }

        try:
            # Look for news items
            news_items = soup.select("li")[:limit]
            for item in news_items:
                article = self._parse_news_item(item)
                if article:
                    data["articles"].append(article)

        except Exception as e:
            self._logger.warning("Failed to parse news", ticker=ticker, error=str(e))

        return data

    def _parse_news_item(self, item: Tag) -> dict[str, Any] | None:
        """Parse a single news item."""
        try:
            link = item.select_one("a")
            if not link:
                return None

            title = link.get_text(strip=True)
            href_raw = link.get("href", "")
            href = str(href_raw) if href_raw else ""
            if href and not href.startswith("http"):
                href = f"{self.BASE_URL}{href}"

            # Extract publisher and date
            meta = item.select_one("span")
            publisher_date = meta.get_text(strip=True) if meta else ""

            return {
                "title": title,
                "url": href,
                "publisher_date": publisher_date,
            }
        except Exception:
            return None

    async def get_losers(self, limit: int = 25) -> ScrapedData:
        """Get the losers list from Yahoo Finance."""
        url = f"{self.BASE_URL}/losers"
        content = await self._fetch_page(url)

        data = self._parse_losers(content, url, limit)
        return ScrapedData(
            source="yahoo_finance_losers",
            url=url,
            content=data,
            timestamp=time.time(),
        )

    def _parse_losers(self, html: str, url: str, limit: int) -> dict[str, Any]:
        """Parse losers list from HTML."""
        soup = BeautifulSoup(html, "lxml")

        data: dict[str, Any] = {
            "url": url,
            "losers": [],
        }

        try:
            # Look for table rows in losers table
            table = soup.select_one("table")
            if table:
                rows = table.select("tr")[:limit]
                for row in rows:
                    loser = self._parse_loser_row(row)
                    if loser:
                        data["losers"].append(loser)

        except Exception as e:
            logger.warning("Failed to parse losers", error=str(e))

        return data

    def _parse_loser_row(self, row: Tag) -> dict[str, Any] | None:
        """Parse a single loser row."""
        try:
            cells = row.select("td")
            if len(cells) < 6:
                return None

            ticker_link = cells[0].select_one("a")
            if not ticker_link:
                return None

            ticker = ticker_link.get_text(strip=True)
            name = cells[1].get_text(strip=True) if len(cells) > 1 else ""
            price = cells[2].get_text(strip=True) if len(cells) > 2 else ""
            change = cells[3].get_text(strip=True) if len(cells) > 3 else ""
            pct_change = cells[4].get_text(strip=True) if len(cells) > 4 else ""
            volume = cells[5].get_text(strip=True) if len(cells) > 5 else ""

            return {
                "ticker": ticker,
                "name": name,
                "price": price,
                "change": change,
                "percent_change": pct_change,
                "volume": volume,
            }
        except Exception:
            return None

    async def get_most_active(self, limit: int = 25) -> ScrapedData:
        """Get the most active stocks list from Yahoo Finance."""
        url = f"{self.BASE_URL}/most-active"
        content = await self._fetch_page(url)

        data = self._parse_most_active(content, url, limit)
        return ScrapedData(
            source="yahoo_finance_most_active",
            url=url,
            content=data,
            timestamp=time.time(),
        )

    def _parse_most_active(self, html: str, url: str, limit: int) -> dict[str, Any]:
        """Parse most active list from HTML."""
        soup = BeautifulSoup(html, "lxml")

        data: dict[str, Any] = {
            "url": url,
            "stocks": [],
        }

        try:
            table = soup.select_one("table")
            if table:
                rows = table.select("tr")[:limit]
                for row in rows:
                    stock = self._parse_active_row(row)
                    if stock:
                        data["stocks"].append(stock)

        except Exception as e:
            logger.warning("Failed to parse most active", error=str(e))

        return data

    def _parse_active_row(self, row: Tag) -> dict[str, Any] | None:
        """Parse a single most active row."""
        try:
            cells = row.select("td")
            if len(cells) < 6:
                return None

            ticker_link = cells[0].select_one("a")
            if not ticker_link:
                return None

            ticker = ticker_link.get_text(strip=True)
            name = cells[1].get_text(strip=True) if len(cells) > 1 else ""
            price = cells[2].get_text(strip=True) if len(cells) > 2 else ""
            change = cells[3].get_text(strip=True) if len(cells) > 3 else ""
            pct_change = cells[4].get_text(strip=True) if len(cells) > 4 else ""
            volume = cells[5].get_text(strip=True) if len(cells) > 5 else ""

            return {
                "ticker": ticker,
                "name": name,
                "price": price,
                "change": change,
                "percent_change": pct_change,
                "volume": volume,
            }
        except Exception:
            return None

    async def close(self) -> None:
        """Clean up browser resources."""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None


async def scrape_company_data(
    ticker: str,
    timeout: int = 30,
    headless: bool = True,
) -> dict[str, ScrapedData]:
    """Scrape all company data in parallel.

    Args:
        ticker: Stock ticker symbol
        timeout: Request timeout in seconds
        headless: Whether to run browser in headless mode

    Returns:
        Dictionary of scraped data by category
    """
    scraper = YahooFinanceScraper(timeout=timeout, headless=headless)

    try:
        # Run all scrapes concurrently
        tasks = [
            scraper.get_company_info(ticker),
            scraper.get_financials(ticker),
            scraper.get_competitors(ticker),
            scraper.get_news(ticker),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        data: dict[str, ScrapedData] = {}
        sources = ["company_info", "financials", "competitors", "news"]

        for source, result in zip(sources, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to scrape {source}", error=str(result))
            elif isinstance(result, ScrapedData):
                data[source] = result

        return data

    finally:
        await scraper.close()


async def scrape_losers(
    limit: int = 25,
    timeout: int = 30,
    headless: bool = True,
) -> ScrapedData:
    """Scrape losers list.

    Args:
        limit: Number of losers to return
        timeout: Request timeout in seconds
        headless: Whether to run browser in headless mode

    Returns:
        ScrapedData containing losers list
    """
    scraper = YahooFinanceScraper(timeout=timeout, headless=headless)

    try:
        return await scraper.get_losers(limit=limit)
    finally:
        await scraper.close()
