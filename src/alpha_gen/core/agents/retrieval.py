"""Data retrieval utilities for accessing gathered data from vector store."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from alpha_gen.core.agents.gather import is_data_fresh
from alpha_gen.core.rag import get_vector_store_manager
from alpha_gen.core.utils.logging import get_logger

logger = get_logger(__name__)


def retrieve_gathered_data(
    ticker: str,
    max_age_hours: int = 24,
) -> dict[str, Any] | None:
    """Retrieve pre-gathered data from vector store.

    Args:
        ticker: Stock ticker symbol
        max_age_hours: Maximum age in hours for data to be considered fresh

    Returns:
        Dictionary with company data or None if not found/stale
    """
    try:
        vector_store = get_vector_store_manager()

        # Retrieve company overview
        overview_docs = vector_store.similarity_search(
            query=f"{ticker} company overview description",
            k=10,
            metadata_filter={"ticker": ticker, "type": "company_overview"},
        )

        # Retrieve financial metrics
        metrics_docs = vector_store.similarity_search(
            query=f"{ticker} financial metrics",
            k=20,
            metadata_filter={"ticker": ticker, "type": "financial_metric"},
        )

        # Retrieve news articles
        news_docs = vector_store.similarity_search(
            query=f"{ticker} news sentiment",
            k=20,
            metadata_filter={"ticker": ticker, "type": "news"},
        )

        if not overview_docs and not metrics_docs and not news_docs:
            logger.warning("No pre-gathered data found in vector store", ticker=ticker)
            return None

        # Check data freshness
        fetched_at = "N/A"
        if overview_docs:
            fetched_at = overview_docs[0].metadata.get("fetched_at", "N/A")
        elif metrics_docs:
            fetched_at = metrics_docs[0].metadata.get("fetched_at", "N/A")

        if fetched_at != "N/A" and not is_data_fresh(fetched_at, max_age_hours):
            logger.warning(
                "Pre-gathered data is stale",
                ticker=ticker,
                fetched_at=fetched_at,
                max_age_hours=max_age_hours,
            )
            return None

        # Reconstruct company overview
        overview = _reconstruct_overview(ticker, overview_docs, metrics_docs)

        # Reconstruct news sentiment
        news_sentiment = _reconstruct_news_sentiment(news_docs)

        # Extract metadata
        latest_quarter = "N/A"
        overview_fetched_at = "N/A"
        news_fetched_at = "N/A"

        if overview_docs:
            latest_quarter = overview_docs[0].metadata.get("latest_quarter", "N/A")
            overview_fetched_at = overview_docs[0].metadata.get("fetched_at", "N/A")

        if news_docs:
            news_fetched_at = news_docs[0].metadata.get("fetched_at", "N/A")

        # Get latest news time
        latest_news_time = "N/A"
        if news_sentiment.get("feed"):
            latest_news_time = news_sentiment["feed"][0].get("time_published", "N/A")
            if latest_news_time != "N/A" and len(latest_news_time) >= 8:
                try:
                    dt = datetime.strptime(latest_news_time[:15], "%Y%m%dT%H%M%S")
                    latest_news_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    pass

        logger.info(
            "Retrieved pre-gathered data from vector store",
            ticker=ticker,
            overview_docs=len(overview_docs),
            metrics_docs=len(metrics_docs),
            news_docs=len(news_docs),
            fetched_at=overview_fetched_at,
        )

        return {
            "ticker": ticker,
            "company_overview": overview,
            "news_sentiment": news_sentiment,
            "latest_quarter": latest_quarter,
            "latest_news_time": latest_news_time,
            "overview_fetched_at": overview_fetched_at,
            "news_fetched_at": news_fetched_at,
            "data_source": "vector_store",
        }

    except Exception as e:
        logger.error(
            "Failed to retrieve data from vector store", ticker=ticker, error=str(e)
        )
        return None


def _reconstruct_overview(
    ticker: str,
    overview_docs: list[Any],
    metrics_docs: list[Any],
) -> dict[str, Any]:
    """Reconstruct company overview from vector store documents.

    Args:
        ticker: Stock ticker symbol
        overview_docs: Overview description documents
        metrics_docs: Financial metrics documents

    Returns:
        Reconstructed company overview dictionary
    """
    overview: dict[str, Any] = {"Symbol": ticker}

    if overview_docs:
        # Combine description chunks (sorted by chunk_id)
        sorted_docs = sorted(
            overview_docs,
            key=lambda d: d.metadata.get("chunk_id", 0),
        )
        description = " ".join([doc.content for doc in sorted_docs])
        overview["Description"] = description

        # Extract metadata from first doc
        first_doc = sorted_docs[0]
        overview["Sector"] = first_doc.metadata.get("sector", "N/A")
        overview["Industry"] = first_doc.metadata.get("industry", "N/A")
        overview["Name"] = ticker  # Use ticker as name

    # Add financial metrics from structured metadata
    if metrics_docs:
        for doc in metrics_docs:
            metric_name = doc.metadata.get("metric_name")
            metric_value = doc.metadata.get("metric_value")
            if metric_name and metric_value:
                overview[metric_name] = metric_value

    return overview


def _reconstruct_news_sentiment(news_docs: list[Any]) -> dict[str, Any]:
    """Reconstruct news sentiment data from vector store documents.

    Args:
        news_docs: News article documents

    Returns:
        Reconstructed news sentiment dictionary
    """
    # Group news by article (using metadata)
    articles_map: dict[str, dict[str, Any]] = {}

    for doc in news_docs:
        metadata = doc.metadata
        title = metadata.get("title", "N/A")
        time_published = metadata.get("time_published", "N/A")

        # Use title + time as unique key
        key = f"{title}_{time_published}"

        if key not in articles_map:
            articles_map[key] = {
                "title": title,
                "summary": doc.content,
                "source": metadata.get("source", "N/A"),
                "overall_sentiment_label": metadata.get("sentiment", "N/A"),
                "overall_sentiment_score": metadata.get("sentiment_score", 0.0),
                "time_published": time_published,
                "chunk_id": metadata.get("chunk_id", 0),
            }
        else:
            # Append chunk to summary
            chunk_id = metadata.get("chunk_id", 0)
            if chunk_id > articles_map[key]["chunk_id"]:
                articles_map[key]["summary"] += " " + doc.content
                articles_map[key]["chunk_id"] = chunk_id

    # Convert to list and sort by time (newest first)
    feed = list(articles_map.values())
    feed.sort(
        key=lambda x: x.get("time_published", ""),
        reverse=True,
    )

    # Remove chunk_id from final output
    for article in feed:
        article.pop("chunk_id", None)

    return {"feed": feed}
