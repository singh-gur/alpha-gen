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
        """Fetch news articles from Yahoo Finance.

        Dispatches to ``get_ticker_news`` or ``get_general_news`` based on
        whether a ``ticker`` keyword argument is provided.

        Args:
            **kwargs: Must include either ``ticker`` (str) for ticker-specific
                news, or no ticker for general news. Optional ``limit`` (int)
                controls the maximum number of articles returned.

        Returns:
            SourceData containing scraped news articles.

        Raises:
            ValueError: If an invalid ticker format is provided.
        """
        ticker: str | None = kwargs.get("ticker")
        limit: int = kwargs.get("limit", 25)

        if ticker:
            return await self.get_ticker_news(ticker=ticker, limit=limit)
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
