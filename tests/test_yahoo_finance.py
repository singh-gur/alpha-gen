"""Tests for Yahoo Finance scraper data source."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alpha_gen.core.data_sources.base import SourceData
from alpha_gen.core.data_sources.yahoo_finance import (
    YahooFinanceScraper,
    YahooNewsArticle,
    YahooStockData,
    fetch_yahoo_gainers,
    fetch_yahoo_general_news,
    fetch_yahoo_losers,
    fetch_yahoo_ticker_news,
)


class TestYahooNewsArticle:
    """Tests for YahooNewsArticle dataclass."""

    def test_create_article(self) -> None:
        """Test creating a YahooNewsArticle instance with all fields."""
        article = YahooNewsArticle(
            title="Intel Reports Q4 Earnings",
            url="https://finance.yahoo.com/news/intel-q4-earnings",
            source="Reuters",
            thumbnail_url="https://example.com/thumb.jpg",
            related_tickers=["INTC"],
        )

        assert article.title == "Intel Reports Q4 Earnings"
        assert article.url == "https://finance.yahoo.com/news/intel-q4-earnings"
        assert article.source == "Reuters"
        assert article.thumbnail_url == "https://example.com/thumb.jpg"
        assert article.related_tickers == ["INTC"]

    def test_create_article_defaults(self) -> None:
        """Test creating a YahooNewsArticle with default optional fields."""
        article = YahooNewsArticle(
            title="Market Update",
            url="https://finance.yahoo.com/news/market-update",
            source="Yahoo Finance",
        )

        assert article.title == "Market Update"
        assert article.thumbnail_url is None
        assert article.related_tickers == []

    def test_article_is_frozen(self) -> None:
        """Test that YahooNewsArticle is immutable."""
        article = YahooNewsArticle(
            title="Test",
            url="https://example.com",
            source="Test Source",
        )

        with pytest.raises(AttributeError):
            article.title = "Modified"  # type: ignore[misc]


class TestYahooStockData:
    """Tests for YahooStockData dataclass."""

    def test_create_stock_data(self) -> None:
        """Test creating a YahooStockData instance with all fields."""
        stock = YahooStockData(
            symbol="AAPL",
            name="Apple Inc.",
            price=150.25,
            change=-5.50,
            percent_change=-3.53,
            volume=50000000,
            avg_volume_3m=45000000,
            market_cap="2.5T",
            pe_ratio="25.5",
            ytd_change="+15.2%",
        )

        assert stock.symbol == "AAPL"
        assert stock.name == "Apple Inc."
        assert stock.price == 150.25
        assert stock.change == -5.50
        assert stock.percent_change == -3.53
        assert stock.volume == 50000000
        assert stock.avg_volume_3m == 45000000
        assert stock.market_cap == "2.5T"
        assert stock.pe_ratio == "25.5"
        assert stock.ytd_change == "+15.2%"

    def test_create_stock_data_defaults(self) -> None:
        """Test creating a YahooStockData with default optional fields."""
        stock = YahooStockData(
            symbol="TSLA",
            name="Tesla, Inc.",
            price=200.0,
            change=10.0,
            percent_change=5.26,
            volume=100000000,
            avg_volume_3m=90000000,
            market_cap="600B",
        )

        assert stock.symbol == "TSLA"
        assert stock.pe_ratio is None
        assert stock.ytd_change is None

    def test_stock_data_is_frozen(self) -> None:
        """Test that YahooStockData is immutable."""
        stock = YahooStockData(
            symbol="MSFT",
            name="Microsoft Corporation",
            price=300.0,
            change=5.0,
            percent_change=1.69,
            volume=25000000,
            avg_volume_3m=24000000,
            market_cap="2.2T",
        )

        with pytest.raises(AttributeError):
            stock.price = 305.0  # type: ignore[misc]


class TestYahooFinanceScraper:
    """Tests for YahooFinanceScraper."""

    def test_init_defaults(self) -> None:
        """Test default initialization values."""
        scraper = YahooFinanceScraper()

        assert scraper.timeout == 30
        assert scraper.headless is True
        assert scraper.base_url == "https://finance.yahoo.com"

    def test_init_custom_values(self) -> None:
        """Test initialization with custom values."""
        scraper = YahooFinanceScraper(
            timeout=60,
            headless=False,
            base_url="https://custom.yahoo.com",
        )

        assert scraper.timeout == 60
        assert scraper.headless is False
        assert scraper.base_url == "https://custom.yahoo.com"

    def test_validate_ticker_valid(self) -> None:
        """Test ticker validation with valid tickers."""
        scraper = YahooFinanceScraper()

        assert scraper._validate_ticker("AAPL") == "AAPL"
        assert scraper._validate_ticker("aapl") == "AAPL"
        assert scraper._validate_ticker("  intc  ") == "INTC"
        assert scraper._validate_ticker("BRK.B") == "BRK.B"
        assert scraper._validate_ticker("BF-B") == "BF-B"
        assert scraper._validate_ticker("^GSPC") == "^GSPC"

    def test_validate_ticker_empty(self) -> None:
        """Test ticker validation rejects empty strings."""
        scraper = YahooFinanceScraper()

        with pytest.raises(ValueError, match="cannot be empty"):
            scraper._validate_ticker("")

        with pytest.raises(ValueError, match="cannot be empty"):
            scraper._validate_ticker("   ")

    def test_validate_ticker_invalid_chars(self) -> None:
        """Test ticker validation rejects invalid characters."""
        scraper = YahooFinanceScraper()

        with pytest.raises(ValueError, match="Invalid ticker symbol"):
            scraper._validate_ticker("AAPL!")

        with pytest.raises(ValueError, match="Invalid ticker symbol"):
            scraper._validate_ticker("AA PL")

        with pytest.raises(ValueError, match="Invalid ticker symbol"):
            scraper._validate_ticker("AAPL/X")

    def test_parse_raw_article(self) -> None:
        """Test parsing raw article dict into YahooNewsArticle."""
        scraper = YahooFinanceScraper()

        raw = {
            "title": "  Test Article  ",
            "url": "https://finance.yahoo.com/news/test",
            "source": "  Reuters  ",
            "thumbnail_url": "https://example.com/img.jpg",
        }

        article = scraper._parse_raw_article(raw)

        assert article.title == "Test Article"
        assert article.url == "https://finance.yahoo.com/news/test"
        assert article.source == "Reuters"
        assert article.thumbnail_url == "https://example.com/img.jpg"

    def test_parse_raw_article_missing_fields(self) -> None:
        """Test parsing raw article with missing optional fields."""
        scraper = YahooFinanceScraper()

        raw: dict[str, str | None] = {
            "title": "Minimal Article",
            "url": "https://example.com",
            "source": "",
            "thumbnail_url": None,
        }

        article = scraper._parse_raw_article(raw)

        assert article.title == "Minimal Article"
        assert article.source == ""
        assert article.thumbnail_url is None

    def test_parse_raw_article_empty_dict(self) -> None:
        """Test parsing an empty raw article dict."""
        scraper = YahooFinanceScraper()

        article = scraper._parse_raw_article({})

        assert article.title == ""
        assert article.url == ""
        assert article.source == ""
        assert article.thumbnail_url is None

    def test_article_to_dict(self) -> None:
        """Test converting YahooNewsArticle to dict."""
        scraper = YahooFinanceScraper()

        article = YahooNewsArticle(
            title="Test",
            url="https://example.com",
            source="Reuters",
            thumbnail_url="https://example.com/img.jpg",
            related_tickers=["AAPL", "MSFT"],
        )

        result = scraper._article_to_dict(article)

        assert result == {
            "title": "Test",
            "url": "https://example.com",
            "source": "Reuters",
            "thumbnail_url": "https://example.com/img.jpg",
            "related_tickers": ["AAPL", "MSFT"],
        }

    def test_article_to_dict_defaults(self) -> None:
        """Test converting YahooNewsArticle with defaults to dict."""
        scraper = YahooFinanceScraper()

        article = YahooNewsArticle(
            title="Test",
            url="https://example.com",
            source="Source",
        )

        result = scraper._article_to_dict(article)

        assert result["thumbnail_url"] is None
        assert result["related_tickers"] == []

    def test_parse_raw_stock(self) -> None:
        """Test parsing raw stock dict into YahooStockData."""
        scraper = YahooFinanceScraper()

        raw = {
            "symbol": "  AAPL  ",
            "name": "  Apple Inc.  ",
            "price": 150.25,
            "change": -5.50,
            "percent_change": -3.53,
            "volume": 50000000,
            "avg_volume_3m": 45000000,
            "market_cap": "  2.5T  ",
            "pe_ratio": "25.5",
            "ytd_change": "+15.2%",
        }

        stock = scraper._parse_raw_stock(raw)

        assert stock.symbol == "AAPL"
        assert stock.name == "Apple Inc."
        assert stock.price == 150.25
        assert stock.change == -5.50
        assert stock.percent_change == -3.53
        assert stock.volume == 50000000
        assert stock.avg_volume_3m == 45000000
        assert stock.market_cap == "2.5T"
        assert stock.pe_ratio == "25.5"
        assert stock.ytd_change == "+15.2%"

    def test_parse_raw_stock_missing_fields(self) -> None:
        """Test parsing raw stock with missing optional fields."""
        scraper = YahooFinanceScraper()

        raw: dict[str, str | float | int | None] = {
            "symbol": "TSLA",
            "name": "Tesla, Inc.",
            "price": 200.0,
            "change": 10.0,
            "percent_change": 5.26,
            "volume": 100000000,
            "avg_volume_3m": 90000000,
            "market_cap": "600B",
            "pe_ratio": None,
            "ytd_change": None,
        }

        stock = scraper._parse_raw_stock(raw)

        assert stock.symbol == "TSLA"
        assert stock.pe_ratio is None
        assert stock.ytd_change is None

    def test_stock_to_dict(self) -> None:
        """Test converting YahooStockData to dict."""
        scraper = YahooFinanceScraper()

        stock = YahooStockData(
            symbol="MSFT",
            name="Microsoft Corporation",
            price=300.0,
            change=5.0,
            percent_change=1.69,
            volume=25000000,
            avg_volume_3m=24000000,
            market_cap="2.2T",
            pe_ratio="28.5",
            ytd_change="+20.5%",
        )

        result = scraper._stock_to_dict(stock)

        assert result == {
            "symbol": "MSFT",
            "name": "Microsoft Corporation",
            "price": 300.0,
            "change": 5.0,
            "percent_change": 1.69,
            "volume": 25000000,
            "avg_volume_3m": 24000000,
            "market_cap": "2.2T",
            "pe_ratio": "28.5",
            "ytd_change": "+20.5%",
        }

    @pytest.mark.asyncio
    async def test_close_no_browser(self) -> None:
        """Test closing scraper when no browser was launched."""
        scraper = YahooFinanceScraper()
        # Should not raise
        await scraper.close()

    @pytest.mark.asyncio
    async def test_close_with_browser(self) -> None:
        """Test closing scraper properly closes the browser."""
        scraper = YahooFinanceScraper()

        mock_browser = MagicMock()
        mock_browser.is_connected.return_value = True
        mock_browser.close = AsyncMock()
        scraper._browser = mock_browser

        await scraper.close()

        mock_browser.close.assert_awaited_once()
        assert scraper._browser is None

    @pytest.mark.asyncio
    async def test_fetch_dispatches_to_ticker_news(self) -> None:
        """Test that fetch() with ticker kwarg dispatches to get_ticker_news."""
        scraper = YahooFinanceScraper()
        scraper.get_ticker_news = AsyncMock()  # type: ignore[method-assign]
        scraper.get_ticker_news.return_value = MagicMock()

        await scraper.fetch(ticker="AAPL", limit=10)

        scraper.get_ticker_news.assert_awaited_once_with(ticker="AAPL", limit=10)

    @pytest.mark.asyncio
    async def test_fetch_dispatches_to_general_news(self) -> None:
        """Test that fetch() without ticker kwarg dispatches to get_general_news."""
        scraper = YahooFinanceScraper()
        scraper.get_general_news = AsyncMock()  # type: ignore[method-assign]
        scraper.get_general_news.return_value = MagicMock()

        await scraper.fetch(limit=5)

        scraper.get_general_news.assert_awaited_once_with(limit=5)

    @pytest.mark.asyncio
    async def test_fetch_dispatches_to_losers(self) -> None:
        """Test that fetch() with mode='losers' dispatches to get_losers."""
        scraper = YahooFinanceScraper()
        scraper.get_losers = AsyncMock()  # type: ignore[method-assign]
        scraper.get_losers.return_value = MagicMock()

        await scraper.fetch(mode="losers", limit=10)

        scraper.get_losers.assert_awaited_once_with(limit=10)

    @pytest.mark.asyncio
    async def test_fetch_dispatches_to_gainers(self) -> None:
        """Test that fetch() with mode='gainers' dispatches to get_gainers."""
        scraper = YahooFinanceScraper()
        scraper.get_gainers = AsyncMock()  # type: ignore[method-assign]
        scraper.get_gainers.return_value = MagicMock()

        await scraper.fetch(mode="gainers", limit=10)

        scraper.get_gainers.assert_awaited_once_with(limit=10)

    @pytest.mark.asyncio
    async def test_fetch_invalid_mode_raises_error(self) -> None:
        """Test that fetch() with invalid mode raises ValueError."""
        scraper = YahooFinanceScraper()

        with pytest.raises(ValueError, match="Invalid mode"):
            await scraper.fetch(mode="invalid")

    @pytest.mark.asyncio
    async def test_get_ticker_news_returns_source_data(self) -> None:
        """Test get_ticker_news returns properly structured SourceData."""
        scraper = YahooFinanceScraper()

        mock_articles = [
            YahooNewsArticle(
                title="INTC News",
                url="https://finance.yahoo.com/news/intc",
                source="Reuters",
            ),
        ]
        scraper._scrape_news_page = AsyncMock(return_value=mock_articles)  # type: ignore[method-assign]

        result = await scraper.get_ticker_news(ticker="INTC", limit=10)

        assert result.source == "yahoo_finance"
        assert result.url == "https://finance.yahoo.com/quote/INTC/news/"
        assert result.content["ticker"] == "INTC"
        assert result.content["count"] == 1
        assert len(result.content["articles"]) == 1
        assert result.content["articles"][0]["title"] == "INTC News"
        assert result.timestamp > 0

    @pytest.mark.asyncio
    async def test_get_ticker_news_uses_custom_base_url(self) -> None:
        """Test get_ticker_news uses custom base URL."""
        scraper = YahooFinanceScraper(base_url="https://custom.example.com")
        scraper._scrape_news_page = AsyncMock(return_value=[])  # type: ignore[method-assign]

        result = await scraper.get_ticker_news(ticker="AAPL")

        assert result.url == "https://custom.example.com/quote/AAPL/news/"

    @pytest.mark.asyncio
    async def test_get_general_news_returns_source_data(self) -> None:
        """Test get_general_news returns properly structured SourceData."""
        scraper = YahooFinanceScraper()

        mock_articles = [
            YahooNewsArticle(
                title="Market Update",
                url="https://finance.yahoo.com/news/market",
                source="AP",
            ),
            YahooNewsArticle(
                title="Fed Decision",
                url="https://finance.yahoo.com/news/fed",
                source="Bloomberg",
            ),
        ]
        scraper._scrape_news_page = AsyncMock(return_value=mock_articles)  # type: ignore[method-assign]

        result = await scraper.get_general_news(limit=25)

        assert result.source == "yahoo_finance"
        assert result.url == "https://finance.yahoo.com/news/"
        assert "ticker" not in result.content
        assert result.content["count"] == 2
        assert len(result.content["articles"]) == 2
        assert result.timestamp > 0

    @pytest.mark.asyncio
    async def test_get_ticker_news_validates_ticker(self) -> None:
        """Test that get_ticker_news validates the ticker symbol."""
        scraper = YahooFinanceScraper()

        with pytest.raises(ValueError, match="cannot be empty"):
            await scraper.get_ticker_news(ticker="")

        with pytest.raises(ValueError, match="Invalid ticker symbol"):
            await scraper.get_ticker_news(ticker="AAPL!")

    @pytest.mark.asyncio
    async def test_get_losers_returns_source_data(self) -> None:
        """Test get_losers returns properly structured SourceData."""
        scraper = YahooFinanceScraper()

        mock_stocks = [
            YahooStockData(
                symbol="AAPL",
                name="Apple Inc.",
                price=150.0,
                change=-10.0,
                percent_change=-6.25,
                volume=50000000,
                avg_volume_3m=45000000,
                market_cap="2.5T",
            ),
        ]
        scraper._scrape_market_movers_page = AsyncMock(return_value=mock_stocks)  # type: ignore[method-assign]

        result = await scraper.get_losers(limit=10)

        assert result.source == "yahoo_finance"
        assert result.url is not None
        assert "/markets/stocks/losers" in result.url
        assert result.content["type"] == "losers"
        assert result.content["count"] == 1
        assert len(result.content["stocks"]) == 1
        assert result.content["stocks"][0]["symbol"] == "AAPL"
        assert result.timestamp > 0

    @pytest.mark.asyncio
    async def test_get_gainers_returns_source_data(self) -> None:
        """Test get_gainers returns properly structured SourceData."""
        scraper = YahooFinanceScraper()

        mock_stocks = [
            YahooStockData(
                symbol="TSLA",
                name="Tesla, Inc.",
                price=200.0,
                change=15.0,
                percent_change=8.11,
                volume=100000000,
                avg_volume_3m=90000000,
                market_cap="600B",
            ),
            YahooStockData(
                symbol="NVDA",
                name="NVIDIA Corporation",
                price=500.0,
                change=25.0,
                percent_change=5.26,
                volume=75000000,
                avg_volume_3m=70000000,
                market_cap="1.2T",
            ),
        ]
        scraper._scrape_market_movers_page = AsyncMock(return_value=mock_stocks)  # type: ignore[method-assign]

        result = await scraper.get_gainers(limit=20)

        assert result.source == "yahoo_finance"
        assert result.url is not None
        assert "/markets/stocks/gainers" in result.url
        assert result.content["type"] == "gainers"
        assert result.content["count"] == 2
        assert len(result.content["stocks"]) == 2
        assert result.timestamp > 0


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.mark.asyncio
    async def test_fetch_yahoo_ticker_news(self) -> None:
        """Test fetch_yahoo_ticker_news convenience function."""
        with (
            patch.object(
                YahooFinanceScraper,
                "get_ticker_news",
                new_callable=AsyncMock,
            ) as mock_method,
            patch.object(
                YahooFinanceScraper,
                "close",
                new_callable=AsyncMock,
            ) as mock_close,
        ):
            from alpha_gen.core.data_sources.base import SourceData

            mock_method.return_value = SourceData(
                source="yahoo_finance",
                url="https://finance.yahoo.com/quote/INTC/news/",
                content={
                    "ticker": "INTC",
                    "articles": [{"title": "Test"}],
                    "count": 1,
                },
                timestamp=1234567890.0,
            )

            result = await fetch_yahoo_ticker_news(
                ticker="INTC",
                limit=10,
                timeout=60,
                headless=True,
            )

            assert result.source == "yahoo_finance"
            assert result.content["ticker"] == "INTC"
            mock_method.assert_awaited_once_with(ticker="INTC", limit=10)
            mock_close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetch_yahoo_general_news(self) -> None:
        """Test fetch_yahoo_general_news convenience function."""
        with (
            patch.object(
                YahooFinanceScraper,
                "get_general_news",
                new_callable=AsyncMock,
            ) as mock_method,
            patch.object(
                YahooFinanceScraper,
                "close",
                new_callable=AsyncMock,
            ) as mock_close,
        ):
            from alpha_gen.core.data_sources.base import SourceData

            mock_method.return_value = SourceData(
                source="yahoo_finance",
                url="https://finance.yahoo.com/news/",
                content={
                    "articles": [{"title": "General News"}],
                    "count": 1,
                },
                timestamp=1234567890.0,
            )

            result = await fetch_yahoo_general_news(
                limit=5,
                timeout=45,
                headless=False,
            )

            assert result.source == "yahoo_finance"
            assert result.content["count"] == 1
            mock_method.assert_awaited_once_with(limit=5)
            mock_close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetch_yahoo_ticker_news_closes_on_error(self) -> None:
        """Test that convenience function closes scraper even on error."""
        with (
            patch.object(
                YahooFinanceScraper,
                "get_ticker_news",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Scrape failed"),
            ),
            patch.object(
                YahooFinanceScraper,
                "close",
                new_callable=AsyncMock,
            ) as mock_close,
        ):
            with pytest.raises(RuntimeError, match="Scrape failed"):
                await fetch_yahoo_ticker_news(ticker="FAIL")

            mock_close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetch_yahoo_general_news_closes_on_error(self) -> None:
        """Test that convenience function closes scraper even on error."""
        with (
            patch.object(
                YahooFinanceScraper,
                "get_general_news",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Scrape failed"),
            ),
            patch.object(
                YahooFinanceScraper,
                "close",
                new_callable=AsyncMock,
            ) as mock_close,
        ):
            with pytest.raises(RuntimeError, match="Scrape failed"):
                await fetch_yahoo_general_news()

            mock_close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetch_yahoo_losers(self) -> None:
        """Test fetch_yahoo_losers convenience function."""
        with (
            patch.object(
                YahooFinanceScraper,
                "get_losers",
                new_callable=AsyncMock,
            ) as mock_method,
            patch.object(
                YahooFinanceScraper,
                "close",
                new_callable=AsyncMock,
            ) as mock_close,
        ):
            from alpha_gen.core.data_sources.base import SourceData

            mock_method.return_value = SourceData(
                source="yahoo_finance",
                url="https://finance.yahoo.com/markets/stocks/losers/",
                content={
                    "stocks": [{"symbol": "AAPL", "name": "Apple Inc."}],
                    "count": 1,
                    "type": "losers",
                },
                timestamp=1234567890.0,
            )

            result = await fetch_yahoo_losers(
                limit=10,
                timeout=60,
                headless=True,
            )

            assert result.source == "yahoo_finance"
            assert result.content["type"] == "losers"
            mock_method.assert_awaited_once_with(limit=10)
            mock_close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetch_yahoo_gainers(self) -> None:
        """Test fetch_yahoo_gainers convenience function."""
        with (
            patch.object(
                YahooFinanceScraper,
                "get_gainers",
                new_callable=AsyncMock,
            ) as mock_method,
            patch.object(
                YahooFinanceScraper,
                "close",
                new_callable=AsyncMock,
            ) as mock_close,
        ):
            from alpha_gen.core.data_sources.base import SourceData

            mock_method.return_value = SourceData(
                source="yahoo_finance",
                url="https://finance.yahoo.com/markets/stocks/gainers/",
                content={
                    "stocks": [{"symbol": "TSLA", "name": "Tesla, Inc."}],
                    "count": 1,
                    "type": "gainers",
                },
                timestamp=1234567890.0,
            )

            result = await fetch_yahoo_gainers(
                limit=10,
                timeout=60,
                headless=True,
            )

            assert result.source == "yahoo_finance"
            assert result.content["type"] == "gainers"
            mock_method.assert_awaited_once_with(limit=10)
            mock_close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetch_yahoo_losers_closes_on_error(self) -> None:
        """Test that convenience function closes scraper even on error."""
        with (
            patch.object(
                YahooFinanceScraper,
                "get_losers",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Scrape failed"),
            ),
            patch.object(
                YahooFinanceScraper,
                "close",
                new_callable=AsyncMock,
            ) as mock_close,
        ):
            with pytest.raises(RuntimeError, match="Scrape failed"):
                await fetch_yahoo_losers()

            mock_close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetch_yahoo_gainers_closes_on_error(self) -> None:
        """Test that convenience function closes scraper even on error."""
        with (
            patch.object(
                YahooFinanceScraper,
                "get_gainers",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Scrape failed"),
            ),
            patch.object(
                YahooFinanceScraper,
                "close",
                new_callable=AsyncMock,
            ) as mock_close,
        ):
            with pytest.raises(RuntimeError, match="Scrape failed"):
                await fetch_yahoo_gainers()

            mock_close.assert_awaited_once()


# ---------------------------------------------------------------------------
# Integration tests - actually scrape Yahoo Finance
# ---------------------------------------------------------------------------
# Run with: uv run pytest tests/test_yahoo_finance.py -m integration -v
# These are excluded from the default test run (use -m "not integration").


def _assert_valid_source_data(data: SourceData, expected_source: str) -> None:
    """Shared assertions for SourceData returned by the scraper."""
    assert isinstance(data, SourceData)
    assert data.source == expected_source
    assert data.url is not None
    assert data.url.startswith("https://")
    assert data.timestamp > 0


def _assert_valid_article(article: dict[str, object]) -> None:
    """Shared assertions for a single article dict."""
    assert isinstance(article, dict)

    title = article.get("title")
    assert isinstance(title, str), f"title should be str, got {type(title)}"
    assert len(title) > 0, "article title must not be empty"

    url = article.get("url")
    assert isinstance(url, str), f"url should be str, got {type(url)}"
    assert url.startswith("http"), f"url should start with http, got {url!r}"

    # source may be empty string but must be a string
    source = article.get("source")
    assert isinstance(source, str), f"source should be str, got {type(source)}"


def _assert_valid_stock(stock: dict[str, object]) -> None:
    """Shared assertions for a single stock dict."""
    assert isinstance(stock, dict)

    symbol = stock.get("symbol")
    assert isinstance(symbol, str), f"symbol should be str, got {type(symbol)}"
    assert len(symbol) > 0, "stock symbol must not be empty"

    name = stock.get("name")
    assert isinstance(name, str), f"name should be str, got {type(name)}"
    assert len(name) > 0, "stock name must not be empty"

    price = stock.get("price")
    assert isinstance(price, (int, float)), (
        f"price should be numeric, got {type(price)}"
    )
    assert price > 0, f"price should be positive, got {price}"

    change = stock.get("change")
    assert isinstance(change, (int, float)), (
        f"change should be numeric, got {type(change)}"
    )

    percent_change = stock.get("percent_change")
    assert isinstance(percent_change, (int, float)), (
        f"percent_change should be numeric, got {type(percent_change)}"
    )

    volume = stock.get("volume")
    assert isinstance(volume, int), f"volume should be int, got {type(volume)}"
    assert volume >= 0, f"volume should be non-negative, got {volume}"

    market_cap = stock.get("market_cap")
    assert isinstance(market_cap, str), (
        f"market_cap should be str, got {type(market_cap)}"
    )


@pytest.mark.integration
class TestYahooFinanceScraperIntegration:
    """Integration tests that perform real scraping against Yahoo Finance.

    These tests launch a headless Chromium browser, navigate to Yahoo Finance,
    and verify that articles are actually extracted.
    """

    @pytest.mark.asyncio
    async def test_scrape_ticker_news_intc(self) -> None:
        """Scrape INTC ticker news and verify articles are returned."""
        scraper = YahooFinanceScraper(timeout=30, headless=True)
        try:
            result = await scraper.get_ticker_news(ticker="INTC", limit=5)

            _assert_valid_source_data(result, "yahoo_finance")
            assert result.url is not None
            assert "/quote/INTC/news" in result.url

            assert result.content["ticker"] == "INTC"
            assert isinstance(result.content["count"], int)
            assert result.content["count"] > 0, "Expected at least 1 article for INTC"
            assert result.content["count"] == len(result.content["articles"])

            for article in result.content["articles"]:
                _assert_valid_article(article)
        finally:
            await scraper.close()

    @pytest.mark.asyncio
    async def test_scrape_ticker_news_aapl(self) -> None:
        """Scrape AAPL ticker news and verify articles are returned."""
        scraper = YahooFinanceScraper(timeout=30, headless=True)
        try:
            result = await scraper.get_ticker_news(ticker="AAPL", limit=5)

            _assert_valid_source_data(result, "yahoo_finance")
            assert result.content["ticker"] == "AAPL"
            assert result.content["count"] > 0, "Expected at least 1 article for AAPL"

            for article in result.content["articles"]:
                _assert_valid_article(article)
        finally:
            await scraper.close()

    @pytest.mark.asyncio
    async def test_scrape_general_news(self) -> None:
        """Scrape general Yahoo Finance news and verify articles are returned."""
        scraper = YahooFinanceScraper(timeout=30, headless=True)
        try:
            result = await scraper.get_general_news(limit=5)

            _assert_valid_source_data(result, "yahoo_finance")
            assert result.url is not None
            assert result.url.endswith("/news/")

            assert "ticker" not in result.content
            assert isinstance(result.content["count"], int)
            assert result.content["count"] > 0, (
                "Expected at least 1 general news article"
            )
            assert result.content["count"] == len(result.content["articles"])

            for article in result.content["articles"]:
                _assert_valid_article(article)
        finally:
            await scraper.close()

    @pytest.mark.asyncio
    async def test_scrape_respects_limit(self) -> None:
        """Verify the limit parameter caps the number of returned articles."""
        scraper = YahooFinanceScraper(timeout=30, headless=True)
        try:
            result = await scraper.get_general_news(limit=3)

            assert result.content["count"] <= 3
            assert len(result.content["articles"]) <= 3
        finally:
            await scraper.close()

    @pytest.mark.asyncio
    async def test_scrape_articles_have_no_duplicates(self) -> None:
        """Verify scraped articles have unique URLs (no duplicates)."""
        scraper = YahooFinanceScraper(timeout=30, headless=True)
        try:
            result = await scraper.get_ticker_news(ticker="NVDA", limit=10)

            urls = [a["url"] for a in result.content["articles"]]
            assert len(urls) == len(set(urls)), (
                f"Duplicate URLs found: {[u for u in urls if urls.count(u) > 1]}"
            )
        finally:
            await scraper.close()

    @pytest.mark.asyncio
    async def test_scrape_browser_reuse(self) -> None:
        """Verify the scraper reuses the browser across multiple calls."""
        scraper = YahooFinanceScraper(timeout=30, headless=True)
        try:
            result1 = await scraper.get_ticker_news(ticker="MSFT", limit=3)
            result2 = await scraper.get_general_news(limit=3)

            assert result1.content["count"] > 0
            assert result2.content["count"] > 0

            # Browser should still be connected after multiple calls
            assert scraper._browser is not None
            assert scraper._browser.is_connected()
        finally:
            await scraper.close()

    @pytest.mark.asyncio
    async def test_fetch_dispatches_correctly_integration(self) -> None:
        """Verify the generic fetch() method dispatches correctly."""
        scraper = YahooFinanceScraper(timeout=30, headless=True)
        try:
            ticker_result = await scraper.fetch(ticker="GOOG", limit=3)
            assert ticker_result.content["ticker"] == "GOOG"
            assert ticker_result.content["count"] > 0

            general_result = await scraper.fetch(limit=3)
            assert "ticker" not in general_result.content
            assert general_result.content["count"] > 0
        finally:
            await scraper.close()

    @pytest.mark.asyncio
    async def test_convenience_fetch_yahoo_ticker_news(self) -> None:
        """Verify the fetch_yahoo_ticker_news convenience function end-to-end."""
        result = await fetch_yahoo_ticker_news(ticker="TSLA", limit=3, timeout=30)

        _assert_valid_source_data(result, "yahoo_finance")
        assert result.content["ticker"] == "TSLA"
        assert result.content["count"] > 0

        for article in result.content["articles"]:
            _assert_valid_article(article)

    @pytest.mark.asyncio
    async def test_convenience_fetch_yahoo_general_news(self) -> None:
        """Verify the fetch_yahoo_general_news convenience function end-to-end."""
        result = await fetch_yahoo_general_news(limit=3, timeout=30)

        _assert_valid_source_data(result, "yahoo_finance")
        assert result.content["count"] > 0

        for article in result.content["articles"]:
            _assert_valid_article(article)

    @pytest.mark.asyncio
    async def test_scrape_losers(self) -> None:
        """Scrape top losers and verify stocks are returned."""
        scraper = YahooFinanceScraper(timeout=60, headless=True)
        try:
            result = await scraper.get_losers(limit=5)

            _assert_valid_source_data(result, "yahoo_finance")
            assert result.url is not None
            assert "/markets/stocks/losers" in result.url

            assert result.content["type"] == "losers"
            assert isinstance(result.content["count"], int)
            assert result.content["count"] > 0, "Expected at least 1 loser stock"
            assert result.content["count"] == len(result.content["stocks"])

            for stock in result.content["stocks"]:
                _assert_valid_stock(stock)
        finally:
            await scraper.close()

    @pytest.mark.asyncio
    async def test_scrape_gainers(self) -> None:
        """Scrape top gainers and verify stocks are returned."""
        scraper = YahooFinanceScraper(timeout=60, headless=True)
        try:
            result = await scraper.get_gainers(limit=5)

            _assert_valid_source_data(result, "yahoo_finance")
            assert result.url is not None
            assert "/markets/stocks/gainers" in result.url

            assert result.content["type"] == "gainers"
            assert isinstance(result.content["count"], int)
            assert result.content["count"] > 0, "Expected at least 1 gainer stock"
            assert result.content["count"] == len(result.content["stocks"])

            for stock in result.content["stocks"]:
                _assert_valid_stock(stock)
        finally:
            await scraper.close()

    @pytest.mark.asyncio
    async def test_convenience_fetch_yahoo_losers(self) -> None:
        """Verify the fetch_yahoo_losers convenience function end-to-end."""
        result = await fetch_yahoo_losers(limit=3, timeout=60)

        _assert_valid_source_data(result, "yahoo_finance")
        assert result.content["type"] == "losers"
        assert result.content["count"] > 0

        for stock in result.content["stocks"]:
            _assert_valid_stock(stock)

    @pytest.mark.asyncio
    async def test_convenience_fetch_yahoo_gainers(self) -> None:
        """Verify the fetch_yahoo_gainers convenience function end-to-end."""
        result = await fetch_yahoo_gainers(limit=3, timeout=60)

        _assert_valid_source_data(result, "yahoo_finance")
        assert result.content["type"] == "gainers"
        assert result.content["count"] > 0

        for stock in result.content["stocks"]:
            _assert_valid_stock(stock)
