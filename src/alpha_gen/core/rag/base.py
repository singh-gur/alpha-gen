"""Base classes for vector store implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from langchain_core.vectorstores import VectorStoreRetriever


@dataclass(frozen=True)
class RetrievalResult:
    """Result from document retrieval."""

    content: str
    source: str
    score: float
    metadata: dict[str, Any]


class BaseVectorStore(ABC):
    """Abstract base class for vector store implementations."""

    @abstractmethod
    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add documents to the vector store.

        Args:
            texts: List of text documents to add
            metadatas: Optional list of metadata dicts
            ids: Optional list of document IDs

        Returns:
            List of document IDs
        """
        pass

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Search for similar documents.

        Args:
            query: Search query
            k: Number of results to return
            metadata_filter: Optional metadata filter

        Returns:
            List of RetrievalResult objects
        """
        pass

    @abstractmethod
    def get_retriever(
        self, search_kwargs: dict[str, Any] | None = None
    ) -> VectorStoreRetriever:
        """Get a retriever for the vector store.

        Args:
            search_kwargs: Optional search parameters

        Returns:
            VectorStoreRetriever instance
        """
        pass

    @abstractmethod
    def delete_collection(self) -> None:
        """Delete the current collection."""
        pass

    @abstractmethod
    def clear_all(self) -> None:
        """Clear all data from the vector store."""
        pass
