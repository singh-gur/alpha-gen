"""RAG (Retrieval-Augmented Generation) module for Alpha Gen."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from sentence_transformers import SentenceTransformer

from alpha_gen.core.config.settings import get_config
from alpha_gen.core.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RetrievalResult:
    """Result from document retrieval."""

    content: str
    source: str
    score: float
    metadata: dict[str, Any]


class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper for SentenceTransformer models."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents."""
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        return self.model.encode([text], show_progress_bar=False)[0].tolist()


class VectorStoreManager:
    """Manager for vector store operations."""

    def __init__(
        self,
        persist_directory: Path | None = None,
        collection_name: str = "alpha_gen_docs",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.persist_directory = persist_directory or Path("./data/vector_store")
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        self._embeddings: SentenceTransformerEmbeddings | None = None
        self._vector_store: Chroma | None = None

    @property
    def embeddings(self) -> SentenceTransformerEmbeddings:
        """Get or create embeddings model."""
        if self._embeddings is None:
            self._embeddings = SentenceTransformerEmbeddings(self.embedding_model)
        return self._embeddings

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
        """Add documents to the vector store.

        Args:
            texts: List of text documents to add
            metadatas: Optional list of metadata dicts
            ids: Optional list of document IDs

        Returns:
            List of document IDs
        """
        documents = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(texts, metadatas or [{}] * len(texts))
        ]

        ids = self.vector_store.add_documents(documents, ids=ids)
        logger.info("Added documents to vector store", count=len(documents))
        return ids  # type: ignore[return-value]

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
        docs = self.vector_store.similarity_search(query, k=k, filter=metadata_filter)

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
        """Get a retriever for the vector store.

        Args:
            search_kwargs: Optional search parameters

        Returns:
            VectorStoreRetriever instance
        """
        return self.vector_store.as_retriever(search_kwargs=search_kwargs or {"k": 5})

    def delete_collection(self) -> None:
        """Delete the current collection."""
        if self._vector_store:
            self._vector_store.delete_collection()
            self._vector_store = None
            logger.info(
                "Deleted vector store collection", collection=self.collection_name
            )

    def clear_all(self) -> None:
        """Clear all data from the vector store."""
        import shutil

        if self.persist_directory.exists():
            shutil.rmtree(self.persist_directory)
            self._vector_store = None
            logger.info("Cleared vector store", directory=str(self.persist_directory))


class DocumentProcessor:
    """Processor for documents before indexing."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process(
        self,
        content: str,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Process content into chunks.

        Args:
            content: Raw content to process
            source: Source identifier
            metadata: Optional metadata

        Returns:
            List of Document chunks
        """
        metadata = metadata or {}
        metadata["source"] = source

        # Simple chunking by sentences/paragraphs
        chunks = self._chunk_content(content, self.chunk_size, self.chunk_overlap)

        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = {
                **metadata,
                "chunk_id": i,
                "total_chunks": len(chunks),
            }
            documents.append(Document(page_content=chunk, metadata=doc_metadata))

        logger.debug(
            "Processed content into chunks", source=source, chunks=len(documents)
        )
        return documents

    def _chunk_content(
        self,
        content: str,
        chunk_size: int,
        overlap: int,
    ) -> list[str]:
        """Split content into chunks with overlap."""
        import re

        # Split by sentences
        sentences = re.split(r"(?<=[.!?])\s+", content)

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_length = len(sentence)

            if current_length + sentence_length > chunk_size and current_chunk:
                # Add current chunk
                chunks.append(" ".join(current_chunk))

                # Keep overlap
                if overlap > 0 and len(current_chunk) > 0:
                    overlap_text = " ".join(current_chunk[-overlap:])
                    current_chunk = [overlap_text]
                    current_length = len(overlap_text)
                else:
                    current_chunk = []
                    current_length = 0

            current_chunk.append(sentence)
            current_length += sentence_length

        # Add remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


def get_vector_store_manager() -> VectorStoreManager:
    """Get a VectorStoreManager instance with configuration."""
    config = get_config()

    return VectorStoreManager(
        persist_directory=config.vector_store.persist_directory,
        collection_name=config.vector_store.collection_name,
        embedding_model=config.vector_store.embedding_model,
    )
