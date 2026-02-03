"""Yahoo Finance news scraper using Playwright for browser-based scraping."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import structlog
from playwright.async_api import Browser, async_playwright

from alpha_gen.core.data_sources.base import BaseDataSource, SourceData

logger = structlog.get_logger(__name__)

# Yahoo Finance base URL
YAHOO_FINANCE_BASE_URL = "https://finance.yahoo.com"


@dataclass(frozen=True)
class YahooNewsArticle:
    """A news article scraped from Yahoo Finance."""

    title: str
    url: str
    source: str
    thumbnail_url: str | None = None
    related_tickers: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class YahooStockData:
    """Stock data scraped from Yahoo Finance market movers pages."""

    symbol: str
    name: str
    price: float
    change: float
    percent_change: float
    volume: int
    avg_volume_3m: int
    market_cap: str
    pe_ratio: str | None = None
    ytd_change: str | None = None


class YahooFinanceScraper(BaseDataSource):
    """Scraper for Yahoo Finance news using Playwright.

    Uses a headless Chromium browser to scrape news articles from Yahoo Finance,
    which requires JavaScript rendering to load content.

    Supports two modes:
    - Ticker-specific news: ``/quote/{ticker}/news/``
    - General financial news: ``/news/``
    """

    DEFAULT_BASE_URL = YAHOO_FINANCE_BASE_URL

    def __init__(
        self,
        timeout: int = 30,
        headless: bool = True,
        base_url: str | None = None,
    ) -> None:
        """Initialize Yahoo Finance scraper.

        Args:
            timeout: Page load timeout in seconds.
            headless: Whether to run the browser in headless mode.
            base_url: Custom base URL for Yahoo Finance (useful for testing).
        """
        super().__init__(timeout=timeout)
        self.headless = headless
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self._browser: Browser | None = None

    async def _ensure_browser(self) -> Browser:
        """Ensure a browser instance is available, launching one if needed.

        Returns:
            An active Playwright Browser instance.
        """
        if self._browser is None or not self._browser.is_connected():
            self._logger.info("Launching Playwright browser", headless=self.headless)
            pw = await async_playwright().start()
            self._browser = await pw.chromium.launch(headless=self.headless)
        return self._browser

    async def fetch(self, **kwargs: Any) -> SourceData:
        """Fetch data from Yahoo Finance.

        Dispatches to appropriate method based on keyword arguments:
        - ``ticker``: Get ticker-specific news
        - ``mode="losers"``: Get top losers
        - ``mode="gainers"``: Get top gainers
        - No args: Get general news

        Args:
            **kwargs: Keyword arguments for the specific fetch method.
                - ``ticker`` (str): For ticker-specific news
                - ``mode`` (str): "losers" or "gainers" for market movers
                - ``limit`` (int): Maximum number of items to return

        Returns:
            SourceData containing scraped data.

        Raises:
            ValueError: If invalid arguments are provided.
        """
        ticker: str | None = kwargs.get("ticker")
        mode: str | None = kwargs.get("mode")
        limit: int = kwargs.get("limit", 25)

        if ticker:
            return await self.get_ticker_news(ticker=ticker, limit=limit)
        if mode == "losers":
            return await self.get_losers(limit=limit)
        if mode == "gainers":
            return await self.get_gainers(limit=limit)
        if mode:
            raise ValueError(f"Invalid mode: {mode!r}. Must be 'losers' or 'gainers'.")
        return await self.get_general_news(limit=limit)

    async def get_ticker_news(
        self,
        ticker: str,
        limit: int = 25,
    ) -> SourceData:
        """Scrape news articles for a specific stock ticker.

        Args:
            ticker: Stock ticker symbol (e.g., ``"INTC"``, ``"AAPL"``).
            limit: Maximum number of articles to return.

        Returns:
            SourceData with ``content`` containing a list of article dicts
            under the ``"articles"`` key, plus ``"ticker"`` and ``"count"``.

        Raises:
            ValueError: If ticker is empty or contains invalid characters.
        """
        ticker = self._validate_ticker(ticker)
        url = f"{self.base_url}/quote/{ticker}/news/"

        self._logger.info("Scraping Yahoo Finance ticker news", ticker=ticker, url=url)

        articles = await self._scrape_news_page(url=url, limit=limit)

        return SourceData(
            source="yahoo_finance",
            url=url,
            content={
                "ticker": ticker,
                "articles": [self._article_to_dict(a) for a in articles],
                "count": len(articles),
            },
            timestamp=time.time(),
        )

    async def get_general_news(
        self,
        limit: int = 25,
    ) -> SourceData:
        """Scrape general financial news from Yahoo Finance.

        Args:
            limit: Maximum number of articles to return.

        Returns:
            SourceData with ``content`` containing a list of article dicts
            under the ``"articles"`` key, plus ``"count"``.
        """
        url = f"{self.base_url}/news/"

        self._logger.info("Scraping Yahoo Finance general news", url=url)

        articles = await self._scrape_news_page(url=url, limit=limit)

        return SourceData(
            source="yahoo_finance",
            url=url,
            content={
                "articles": [self._article_to_dict(a) for a in articles],
                "count": len(articles),
            },
            timestamp=time.time(),
        )

    async def get_losers(
        self,
        limit: int = 20,
    ) -> SourceData:
        """Scrape top losing stocks from Yahoo Finance.

        Args:
            limit: Maximum number of stocks to return (max 100).

        Returns:
            SourceData with ``content`` containing a list of stock dicts
            under the ``"stocks"`` key, plus ``"count"``.
        """
        # Yahoo Finance uses pagination with start and count params
        url = f"{self.base_url}/markets/stocks/losers/?start=0&count={limit}"

        self._logger.info("Scraping Yahoo Finance losers", url=url, limit=limit)

        stocks = await self._scrape_market_movers_page(url=url, limit=limit)

        return SourceData(
            source="yahoo_finance",
            url=url,
            content={
                "stocks": [self._stock_to_dict(s) for s in stocks],
                "count": len(stocks),
                "type": "losers",
            },
            timestamp=time.time(),
        )

    async def get_gainers(
        self,
        limit: int = 20,
    ) -> SourceData:
        """Scrape top gaining stocks from Yahoo Finance.

        Args:
            limit: Maximum number of stocks to return (max 100).

        Returns:
            SourceData with ``content`` containing a list of stock dicts
            under the ``"stocks"`` key, plus ``"count"``.
        """
        # Yahoo Finance uses pagination with start and count params
        url = f"{self.base_url}/markets/stocks/gainers/?start=0&count={limit}"

        self._logger.info("Scraping Yahoo Finance gainers", url=url, limit=limit)

        stocks = await self._scrape_market_movers_page(url=url, limit=limit)

        return SourceData(
            source="yahoo_finance",
            url=url,
            content={
                "stocks": [self._stock_to_dict(s) for s in stocks],
                "count": len(stocks),
                "type": "gainers",
            },
            timestamp=time.time(),
        )

    async def _scrape_news_page(
        self,
        url: str,
        limit: int = 25,
    ) -> list[YahooNewsArticle]:
        """Scrape news articles from a Yahoo Finance page.

        Opens the URL in a Playwright browser, waits for content to render,
        handles the consent dialog if present, and extracts article data
        from the DOM.

        Args:
            url: Full URL of the Yahoo Finance news page.
            limit: Maximum number of articles to extract.

        Returns:
            List of scraped YahooNewsArticle instances.
        """
        browser = await self._ensure_browser()
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()

        try:
            await page.goto(url, timeout=self.timeout * 1000)

            # Handle Yahoo consent dialog if it appears
            await self._handle_consent_dialog(page)

            # Wait for news content to load
            await page.wait_for_selector(
                "[data-testid='storyitem'], .stream-items li, .news-stream li",
                timeout=self.timeout * 1000,
            )

            # Extract articles from the page
            raw_articles: list[dict[str, Any]] = await page.evaluate(
                """(limit) => {
                    const articles = [];
                    const seen = new Set();

                    // Strategy 1: data-testid storyitem elements
                    const storyItems = document.querySelectorAll(
                        '[data-testid="storyitem"]'
                    );
                    for (const item of storyItems) {
                        if (articles.length >= limit) break;
                        const link = item.querySelector('a[href*="/news/"]')
                            || item.querySelector('a[href*="finance.yahoo.com"]')
                            || item.querySelector('a');
                        if (!link) continue;

                        const href = link.href;
                        if (seen.has(href)) continue;
                        seen.add(href);

                        const titleEl = item.querySelector('h3')
                            || item.querySelector('[class*="title"]')
                            || link;
                        const sourceEl = item.querySelector(
                            '[class*="publishing"], [class*="source"],'
                            + ' [data-testid="author"]'
                        );
                        const imgEl = item.querySelector('img');

                        articles.push({
                            title: (titleEl.textContent || '').trim(),
                            url: href,
                            source: sourceEl
                                ? (sourceEl.textContent || '').trim()
                                : '',
                            thumbnail_url: imgEl ? imgEl.src : null,
                        });
                    }

                    // Strategy 2: list items in news streams
                    if (articles.length === 0) {
                        const listItems = document.querySelectorAll(
                            '.stream-items li, .news-stream li,'
                            + ' [class*="newsStream"] li,'
                            + ' [class*="news"] li'
                        );
                        for (const item of listItems) {
                            if (articles.length >= limit) break;
                            const link = item.querySelector('a');
                            if (!link || !link.href) continue;

                            const href = link.href;
                            if (seen.has(href)) continue;
                            seen.add(href);

                            const titleEl = item.querySelector('h3')
                                || item.querySelector('h2')
                                || item.querySelector('[class*="title"]')
                                || link;
                            const sourceEl = item.querySelector(
                                '[class*="publishing"], [class*="source"]'
                            );
                            const imgEl = item.querySelector('img');

                            const title = (titleEl.textContent || '').trim();
                            if (!title) continue;

                            articles.push({
                                title: title,
                                url: href,
                                source: sourceEl
                                    ? (sourceEl.textContent || '').trim()
                                    : '',
                                thumbnail_url: imgEl ? imgEl.src : null,
                            });
                        }
                    }

                    // Strategy 3: any anchor tags with news-like hrefs
                    if (articles.length === 0) {
                        const allLinks = document.querySelectorAll(
                            'a[href*="/news/"]'
                        );
                        for (const link of allLinks) {
                            if (articles.length >= limit) break;
                            const href = link.href;
                            if (seen.has(href)) continue;
                            seen.add(href);

                            const title = (link.textContent || '').trim();
                            if (!title || title.length < 10) continue;

                            articles.push({
                                title: title,
                                url: href,
                                source: '',
                                thumbnail_url: null,
                            });
                        }
                    }

                    return articles.slice(0, limit);
                }""",
                limit,
            )

            articles = [self._parse_raw_article(raw) for raw in raw_articles]

            self._logger.info(
                "Scraped Yahoo Finance news",
                url=url,
                article_count=len(articles),
            )

            return articles

        except Exception as e:
            self._logger.error(
                "Failed to scrape Yahoo Finance news",
                url=url,
                error=str(e),
            )
            raise
        finally:
            await context.close()

    async def _scrape_market_movers_page(
        self,
        url: str,
        limit: int = 20,
    ) -> list[YahooStockData]:
        """Scrape market movers (gainers/losers) from a Yahoo Finance page.

        Opens the URL in a Playwright browser, waits for the table to render,
        handles the consent dialog if present, and extracts stock data from
        the table rows.

        Args:
            url: Full URL of the Yahoo Finance market movers page.
            limit: Maximum number of stocks to extract.

        Returns:
            List of scraped YahooStockData instances.
        """
        browser = await self._ensure_browser()
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()

        try:
            await page.goto(
                url, wait_until="domcontentloaded", timeout=self.timeout * 1000
            )

            # Handle Yahoo consent dialog if it appears
            await self._handle_consent_dialog(page)

            # Wait for table content to load
            await page.wait_for_selector(
                "table tbody tr",
                timeout=self.timeout * 1000,
            )

            # Give the page a moment to fully render
            await page.wait_for_timeout(2000)

            # Extract stock data from the table
            raw_stocks: list[dict[str, Any]] = await page.evaluate(
                """(limit) => {
                    const stocks = [];
                    const rows = document.querySelectorAll('table tbody tr');

                    for (const row of rows) {
                        if (stocks.length >= limit) break;

                        const cells = row.querySelectorAll('td');
                        if (cells.length < 12) continue;

                        // Helper to parse numeric values
                        const parseNum = (text) => {
                            if (!text || text === '--') return null;
                            // Remove commas and convert to number
                            const cleaned = text.replace(/,/g, '');
                            const num = parseFloat(cleaned);
                            return isNaN(num) ? null : num;
                        };

                        // Helper to parse volume (handles M, B suffixes)
                        const parseVolume = (text) => {
                            if (!text || text === '--') return 0;
                            const cleaned = text.replace(/,/g, '');
                            if (cleaned.endsWith('M')) {
                                return parseFloat(cleaned) * 1000000;
                            } else if (cleaned.endsWith('B')) {
                                return parseFloat(cleaned) * 1000000000;
                            }
                            return parseFloat(cleaned) || 0;
                        };

                        const symbol = cells[0].innerText.trim();
                        const name = cells[1].innerText.trim();
                        const price = parseNum(cells[3].innerText.trim());
                        const change = parseNum(cells[4].innerText.trim());
                        const percentChange = parseNum(
                            cells[5].innerText.trim().replace('%', '')
                        );
                        const volume = parseVolume(cells[6].innerText.trim());
                        const avgVolume = parseVolume(cells[7].innerText.trim());
                        const marketCap = cells[8].innerText.trim();
                        const peRatio = cells[9].innerText.trim();
                        const ytdChange = cells[10].innerText.trim();

                        if (symbol && name && price !== null) {
                            stocks.push({
                                symbol: symbol,
                                name: name,
                                price: price,
                                change: change || 0,
                                percent_change: percentChange || 0,
                                volume: volume,
                                avg_volume_3m: avgVolume,
                                market_cap: marketCap,
                                pe_ratio: peRatio === '--' ? null : peRatio,
                                ytd_change: ytdChange === '--' ? null : ytdChange,
                            });
                        }
                    }

                    return stocks;
                }""",
                limit,
            )

            stocks = [self._parse_raw_stock(raw) for raw in raw_stocks]

            self._logger.info(
                "Scraped Yahoo Finance market movers",
                url=url,
                stock_count=len(stocks),
            )

            return stocks

        except Exception as e:
            self._logger.error(
                "Failed to scrape Yahoo Finance market movers",
                url=url,
                error=str(e),
            )
            raise
        finally:
            await context.close()

    async def _handle_consent_dialog(self, page: Any) -> None:
        """Handle Yahoo's GDPR/privacy consent dialog if present.

        Yahoo Finance shows a consent page for EU users and sometimes
        for other regions. This method detects and accepts the consent
        to proceed to the actual content.

        Args:
            page: Playwright Page instance.
        """
        try:
            consent_button = page.locator(
                'button[name="agree"], '
                'button:has-text("Accept all"), '
                'button:has-text("Accept"), '
                'button:has-text("Scroll to continue"), '
                '[class*="consent"] button'
            ).first
            if await consent_button.is_visible(timeout=3000):
                self._logger.debug("Consent dialog detected, accepting")
                await consent_button.click()
                await page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:
            # Consent dialog not present or already handled
            pass

    def _validate_ticker(self, ticker: str) -> str:
        """Validate and normalize a stock ticker symbol.

        Args:
            ticker: Raw ticker string.

        Returns:
            Uppercased, stripped ticker string.

        Raises:
            ValueError: If ticker is empty or contains invalid characters.
        """
        ticker = ticker.strip().upper()
        if not ticker:
            raise ValueError("Ticker symbol cannot be empty")
        if not all(c.isalnum() or c in (".", "-", "^") for c in ticker):
            raise ValueError(
                f"Invalid ticker symbol: {ticker!r}. "
                "Only alphanumeric characters, '.', '-', and '^' are allowed."
            )
        return ticker

    def _parse_raw_article(self, raw: dict[str, Any]) -> YahooNewsArticle:
        """Parse a raw article dict from JavaScript evaluation into a dataclass.

        Args:
            raw: Dictionary with keys ``title``, ``url``, ``source``,
                and optionally ``thumbnail_url``.

        Returns:
            A YahooNewsArticle instance.
        """
        return YahooNewsArticle(
            title=str(raw.get("title", "")).strip(),
            url=str(raw.get("url", "")),
            source=str(raw.get("source", "")).strip(),
            thumbnail_url=raw.get("thumbnail_url"),
        )

    def _article_to_dict(self, article: YahooNewsArticle) -> dict[str, Any]:
        """Convert a YahooNewsArticle to a plain dictionary.

        Args:
            article: The article dataclass to convert.

        Returns:
            Dictionary representation of the article.
        """
        return {
            "title": article.title,
            "url": article.url,
            "source": article.source,
            "thumbnail_url": article.thumbnail_url,
            "related_tickers": article.related_tickers,
        }

    def _parse_raw_stock(self, raw: dict[str, Any]) -> YahooStockData:
        """Parse a raw stock dict from JavaScript evaluation into a dataclass.

        Args:
            raw: Dictionary with stock data fields.

        Returns:
            A YahooStockData instance.
        """
        return YahooStockData(
            symbol=str(raw.get("symbol", "")).strip(),
            name=str(raw.get("name", "")).strip(),
            price=float(raw.get("price", 0.0)),
            change=float(raw.get("change", 0.0)),
            percent_change=float(raw.get("percent_change", 0.0)),
            volume=int(raw.get("volume", 0)),
            avg_volume_3m=int(raw.get("avg_volume_3m", 0)),
            market_cap=str(raw.get("market_cap", "")).strip(),
            pe_ratio=raw.get("pe_ratio"),
            ytd_change=raw.get("ytd_change"),
        )

    def _stock_to_dict(self, stock: YahooStockData) -> dict[str, Any]:
        """Convert a YahooStockData to a plain dictionary.

        Args:
            stock: The stock dataclass to convert.

        Returns:
            Dictionary representation of the stock.
        """
        return {
            "symbol": stock.symbol,
            "name": stock.name,
            "price": stock.price,
            "change": stock.change,
            "percent_change": stock.percent_change,
            "volume": stock.volume,
            "avg_volume_3m": stock.avg_volume_3m,
            "market_cap": stock.market_cap,
            "pe_ratio": stock.pe_ratio,
            "ytd_change": stock.ytd_change,
        }

    async def close(self) -> None:
        """Close the Playwright browser and release resources."""
        if self._browser and self._browser.is_connected():
            self._logger.info("Closing Playwright browser")
            await self._browser.close()
            self._browser = None


# Convenience functions for common operations


async def fetch_yahoo_ticker_news(
    ticker: str,
    limit: int = 25,
    timeout: int = 30,
    headless: bool = True,
    base_url: str | None = None,
) -> SourceData:
    """Fetch news articles for a specific ticker from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (e.g., ``"INTC"``).
        limit: Maximum number of articles to return.
        timeout: Page load timeout in seconds.
        headless: Whether to run the browser in headless mode.
        base_url: Custom base URL (useful for testing).

    Returns:
        SourceData containing scraped news articles for the ticker.
    """
    scraper = YahooFinanceScraper(
        timeout=timeout,
        headless=headless,
        base_url=base_url,
    )
    try:
        return await scraper.get_ticker_news(ticker=ticker, limit=limit)
    finally:
        await scraper.close()


async def fetch_yahoo_general_news(
    limit: int = 25,
    timeout: int = 30,
    headless: bool = True,
    base_url: str | None = None,
) -> SourceData:
    """Fetch general financial news from Yahoo Finance.

    Args:
        limit: Maximum number of articles to return.
        timeout: Page load timeout in seconds.
        headless: Whether to run the browser in headless mode.
        base_url: Custom base URL (useful for testing).

    Returns:
        SourceData containing scraped general news articles.
    """
    scraper = YahooFinanceScraper(
        timeout=timeout,
        headless=headless,
        base_url=base_url,
    )
    try:
        return await scraper.get_general_news(limit=limit)
    finally:
        await scraper.close()


async def fetch_yahoo_losers(
    limit: int = 20,
    timeout: int = 30,
    headless: bool = True,
    base_url: str | None = None,
) -> SourceData:
    """Fetch top losing stocks from Yahoo Finance.

    Args:
        limit: Maximum number of stocks to return (max 100).
        timeout: Page load timeout in seconds.
        headless: Whether to run the browser in headless mode.
        base_url: Custom base URL (useful for testing).

    Returns:
        SourceData containing scraped loser stocks.
    """
    scraper = YahooFinanceScraper(
        timeout=timeout,
        headless=headless,
        base_url=base_url,
    )
    try:
        return await scraper.get_losers(limit=limit)
    finally:
        await scraper.close()


async def fetch_yahoo_gainers(
    limit: int = 20,
    timeout: int = 30,
    headless: bool = True,
    base_url: str | None = None,
) -> SourceData:
    """Fetch top gaining stocks from Yahoo Finance.

    Args:
        limit: Maximum number of stocks to return (max 100).
        timeout: Page load timeout in seconds.
        headless: Whether to run the browser in headless mode.
        base_url: Custom base URL (useful for testing).

    Returns:
        SourceData containing scraped gainer stocks.
    """
    scraper = YahooFinanceScraper(
        timeout=timeout,
        headless=headless,
        base_url=base_url,
    )
    try:
        return await scraper.get_gainers(limit=limit)
    finally:
        await scraper.close()
