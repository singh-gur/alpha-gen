"""Chroma vector store implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from alpha_gen.core.rag.base import BaseVectorStore, RetrievalResult
from alpha_gen.core.utils.logging import get_logger

logger = get_logger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """Chroma vector store implementation."""

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str,
        embeddings: Embeddings,
    ) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embeddings = embeddings
        self._vector_store: Chroma | None = None

    @property
    def vector_store(self) -> Chroma:
        """Get or create vector store."""
        if self._vector_store is None:
            self._vector_store = Chroma(
                persist_directory=str(self.persist_directory),
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
            )
        return self._vector_store

    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add documents to the vector store."""
        # Ensure texts are strings, not Document objects
        processed_texts = []
        for text in texts:
            if isinstance(text, Document):
                logger.warning(
                    "Received Document object instead of string, extracting page_content"
                )
                processed_texts.append(text.page_content)
            else:
                processed_texts.append(text)

        documents = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(
                processed_texts, metadatas or [{}] * len(processed_texts)
            )
        ]

        ids = self.vector_store.add_documents(documents, ids=ids)
        logger.info("Added documents to Chroma", count=len(documents))
        return ids  # type: ignore[return-value]

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Search for similar documents."""
        # Only pass filter if it's not None
        if metadata_filter is not None:
            docs = self.vector_store.similarity_search(
                query, k=k, filter=metadata_filter
            )
        else:
            docs = self.vector_store.similarity_search(query, k=k)

        return [
            RetrievalResult(
                content=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                score=doc.metadata.get("score", 0.0),
                metadata=doc.metadata,
            )
            for doc in docs
        ]

    def get_retriever(
        self, search_kwargs: dict[str, Any] | None = None
    ) -> VectorStoreRetriever:
        """Get a retriever for the vector store."""
        return self.vector_store.as_retriever(search_kwargs=search_kwargs or {"k": 5})

    def delete_collection(self) -> None:
        """Delete the current collection."""
        if self._vector_store:
            self._vector_store.delete_collection()
            self._vector_store = None
            logger.info("Deleted Chroma collection", collection=self.collection_name)

    def clear_all(self) -> None:
        """Clear all data from the vector store."""
        import shutil

        if self.persist_directory.exists():
            shutil.rmtree(self.persist_directory)
            self._vector_store = None
            logger.info("Cleared Chroma data", directory=str(self.persist_directory))
