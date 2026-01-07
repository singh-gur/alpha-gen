"""Tests for scrapers module."""

from __future__ import annotations

import pytest

from alpha_gen.scrapers.base import BaseScraper, ScrapedData


class TestScrapedData:
    """Tests for ScrapedData dataclass."""

    def test_create_scraped_data(self) -> None:
        """Test creating ScrapedData instance."""
        data = ScrapedData(
            source="test_source",
            url="https://example.com",
            content={"key": "value"},
            timestamp=1234567890.0,
        )

        assert data.source == "test_source"
        assert data.url == "https://example.com"
        assert data.content == {"key": "value"}
        assert data.timestamp == 1234567890.0


class TestBaseScraper:
    """Tests for BaseScraper."""

    def test_abstract_methods(self) -> None:
        """Test that BaseScraper has abstract methods."""
        assert hasattr(BaseScraper, "scrape")
        assert hasattr(BaseScraper, "close")

    def test_init(self) -> None:
        """Test BaseScraper initialization."""
        # Can't instantiate directly due to abstract methods
        # Just verify the init method exists and works
        class TestScraper(BaseScraper):
            async def scrape(self, **kwargs):
                return ScrapedData(
                    source="test",
                    url="",
                    content={},
                    timestamp=0.0,
                )

        scraper = TestScraper(timeout=45, headless=False)
        assert scraper.timeout == 45
        assert scraper.headless is False
