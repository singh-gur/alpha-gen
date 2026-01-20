"""Tests for data sources module."""

from __future__ import annotations

import pytest

from alpha_gen.core.data_sources.base import BaseDataSource, SourceData


class TestSourceData:
    """Tests for SourceData dataclass."""

    def test_create_source_data(self) -> None:
        """Test creating SourceData instance."""
        data = SourceData(
            source="test_source",
            url="https://example.com",
            content={"key": "value"},
            timestamp=1234567890.0,
        )

        assert data.source == "test_source"
        assert data.url == "https://example.com"
        assert data.content == {"key": "value"}
        assert data.timestamp == 1234567890.0


class TestBaseDataSource:
    """Tests for BaseDataSource."""

    def test_abstract_methods(self) -> None:
        """Test that BaseDataSource has abstract methods."""
        assert hasattr(BaseDataSource, "fetch")
        assert hasattr(BaseDataSource, "close")

    def test_init(self) -> None:
        """Test BaseDataSource initialization."""

        # Can't instantiate directly due to abstract methods
        # Just verify the init method exists and works
        class TestDataSource(BaseDataSource):
            async def fetch(self, **kwargs):
                return SourceData(
                    source="test",
                    url="",
                    content={},
                    timestamp=0.0,
                )

        data_source = TestDataSource(timeout=45)
        assert data_source.timeout == 45
