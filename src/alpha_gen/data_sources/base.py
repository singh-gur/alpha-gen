"""Base data source module for Alpha Gen."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class SourceData:
    """Base class for data from any source (API, scraper, etc.)."""

    source: str
    url: str | None
    content: dict[str, Any]
    timestamp: float


class BaseDataSource(ABC):
    """Base class for all data sources (APIs, scrapers, etc.)."""

    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout
        self._logger = logger.bind(data_source=self.__class__.__name__)

    @abstractmethod
    async def fetch(self, **kwargs: Any) -> SourceData:
        """Fetch data from the source."""
        ...

    async def close(self) -> None:
        """Clean up resources.

        Override this method in subclasses if needed.
        """
