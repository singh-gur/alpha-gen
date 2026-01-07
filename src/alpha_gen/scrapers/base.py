"""Base scraper module for Alpha Gen."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ScrapedData:
    """Base class for scraped data."""

    source: str
    url: str
    content: dict[str, Any]
    timestamp: float


class BaseScraper(ABC):
    """Base class for web scrapers."""

    def __init__(self, timeout: int = 30, headless: bool = True) -> None:
        self.timeout = timeout
        self.headless = headless
        self._logger = logger.bind(scraper=self.__class__.__name__)

    @abstractmethod
    async def scrape(self, **kwargs: Any) -> ScrapedData:
        """Scrape data from the source."""
        ...

    async def close(self) -> None:
        """Clean up resources.

        Override this method in subclasses if needed.
        """
