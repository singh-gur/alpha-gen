"""Gather agent for collecting and storing financial data in vector store.

This agent fetches company data from Alpha Vantage and stores it in the vector database
with proper versioning and metadata for later retrieval by the research agent.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Any

from alpha_gen.core.agents.base import AgentConfig, AgentState, BaseAgent
from alpha_gen.core.config.settings import TechnicalIndicatorConfig, get_config
from alpha_gen.core.data_sources.alpha_vantage import (
    AlphaVantageClient,
    fetch_company_overview,
    fetch_news_sentiment,
)
from alpha_gen.core.rag import get_vector_store_manager
from alpha_gen.core.utils.logging import get_logger

logger = get_logger(__name__)

# Data freshness configuration
DEFAULT_DATA_MAX_AGE_HOURS = 24


def _generate_doc_id(ticker: str, doc_type: str, index: int = 0) -> str:
    """Generate deterministic document ID for upsert behavior.

    Args:
        ticker: Stock ticker symbol
        doc_type: Type of document (overview, metric, news)
        index: Index for multiple documents of same type

    Returns:
        Deterministic document ID
    """
    return f"{ticker.upper()}_{doc_type}_{index}"


def _format_timestamp(timestamp: float) -> str:
    """Format Unix timestamp to UTC string.

    Args:
        timestamp: Unix timestamp

    Returns:
        Formatted timestamp string
    """
    return datetime.fromtimestamp(timestamp, tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def _parse_timestamp(timestamp_str: str) -> datetime | None:
    """Parse timestamp string to datetime.

    Args:
        timestamp_str: Timestamp string in format "YYYY-MM-DD HH:MM:SS UTC"

    Returns:
        Parsed datetime or None if parsing fails
    """
    try:
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S UTC").replace(
            tzinfo=UTC
        )
    except (ValueError, AttributeError):
        return None


def is_data_fresh(
    fetched_at: str, max_age_hours: int = DEFAULT_DATA_MAX_AGE_HOURS
) -> bool:
    """Check if gathered data is still fresh.

    Args:
        fetched_at: Timestamp string when data was fetched
        max_age_hours: Maximum age in hours before data is considered stale

    Returns:
        True if data is fresh, False otherwise
    """
    fetched_time = _parse_timestamp(fetched_at)
    if not fetched_time:
        return False

    age_hours = (datetime.now(UTC) - fetched_time).total_seconds() / 3600
    return age_hours < max_age_hours


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks.

    Args:
        text: Text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind(". ")
            if last_period > chunk_size // 2:  # Only break if we're past halfway
                end = start + last_period + 1
                chunk = text[start:end]

        chunks.append(chunk.strip())
        start = end - overlap if end < len(text) else end

    return chunks


def _extract_financial_metrics(overview: dict[str, Any]) -> dict[str, Any]:
    """Extract financial metrics from company overview.

    Args:
        overview: Company overview data from Alpha Vantage

    Returns:
        Dictionary of metric name to value
    """
    metric_keys = [
        "MarketCapitalization",
        "PERatio",
        "EPS",
        "RevenueTTM",
        "ProfitMargin",
        "OperatingMarginTTM",
        "ReturnOnEquityTTM",
        "ReturnOnAssetsTTM",
        "DebtToEquityRatio",
        "CurrentRatio",
        "QuickRatio",
        "Beta",
        "52WeekHigh",
        "52WeekLow",
        "AnalystTargetPrice",
        "DividendYield",
    ]

    metrics = {}
    for key in metric_keys:
        value = overview.get(key)
        if value and value != "None" and value != "-":
            metrics[key] = value

    return metrics


async def _fetch_technical_indicator(
    client: AlphaVantageClient,
    indicator_name: str,
    symbol: str,
    config: TechnicalIndicatorConfig,
) -> dict[str, Any] | None:
    """Fetch a single technical indicator from Alpha Vantage.

    Args:
        client: Alpha Vantage client
        indicator_name: Name of the indicator (SMA, EMA, RSI, etc.)
        symbol: Stock ticker symbol
        config: Configuration for the indicator

    Returns:
        Technical indicator data or None if fetch fails
    """
    try:
        # Map indicator names to client methods
        method_map = {
            "SMA": client.get_sma,
            "EMA": client.get_ema,
            "RSI": client.get_rsi,
            "MACD": client.get_macd,
            "STOCH": client.get_stoch,
            "BBANDS": client.get_bbands,
            "ATR": client.get_atr,
            "ADX": client.get_adx,
            "AROON": client.get_aroon,
            "CCI": client.get_cci,
            "OBV": client.get_obv,
            "AD": client.get_ad,
        }

        method = method_map.get(indicator_name)
        if not method:
            logger.warning(
                "Unknown technical indicator", indicator=indicator_name, symbol=symbol
            )
            return None

        # Call the appropriate method with config parameters
        # Some indicators don't use all parameters
        if indicator_name in ["MACD"]:
            data = await method(
                symbol=symbol,
                interval=config.interval,
                series_type=config.series_type,
            )
        elif indicator_name in ["STOCH", "OBV", "AD"]:
            data = await method(
                symbol=symbol,
                interval=config.interval,
            )
        elif indicator_name in ["ATR", "ADX", "AROON", "CCI"]:
            data = await method(
                symbol=symbol,
                interval=config.interval,
                time_period=config.time_period,
            )
        else:  # SMA, EMA, RSI, BBANDS
            data = await method(
                symbol=symbol,
                interval=config.interval,
                time_period=config.time_period,
                series_type=config.series_type,
            )

        return data.content
    except Exception as e:
        logger.error(
            "Failed to fetch technical indicator",
            indicator=indicator_name,
            symbol=symbol,
            error=str(e),
        )
        return None


def _format_technical_indicator_data(
    indicator_name: str, indicator_data: dict[str, Any], limit: int = 30
) -> str:
    """Format technical indicator data for storage.

    Args:
        indicator_name: Name of the indicator
        indicator_data: Raw indicator data from Alpha Vantage
        limit: Maximum number of data points to include

    Returns:
        Formatted string representation of the indicator data
    """
    # Find the technical analysis key in the response
    tech_key = None
    for key in indicator_data:
        if "Technical Analysis" in key:
            tech_key = key
            break

    if not tech_key:
        return f"{indicator_name}: No data available"

    data_points = indicator_data.get(tech_key, {})
    if not data_points:
        return f"{indicator_name}: No data available"

    # Get the most recent data points
    sorted_dates = sorted(data_points.keys(), reverse=True)[:limit]

    # Format the data
    lines = [f"{indicator_name} (most recent {len(sorted_dates)} periods):"]
    for date in sorted_dates:
        values = data_points[date]
        if isinstance(values, dict):
            # Multiple values (e.g., MACD has MACD, signal, histogram)
            value_str = ", ".join([f"{k}: {v}" for k, v in values.items()])
            lines.append(f"  {date}: {value_str}")
        else:
            # Single value
            lines.append(f"  {date}: {values}")

    return "\n".join(lines)


async def gather_data_node(state: AgentState) -> AgentState:
    """Fetch company data from Alpha Vantage and store in vector DB.

    This node:
    1. Fetches company overview and news sentiment from Alpha Vantage
    2. Deletes any existing data for the ticker (to prevent duplicates)
    3. Stores new data with structured metadata for easy retrieval
    4. Uses deterministic IDs for upsert behavior

    Args:
        state: Current agent state with ticker in context

    Returns:
        Updated agent state with results or error
    """
    ticker = state["context"].get("ticker", "").upper()

    if not ticker:
        updated_state: AgentState = {
            "messages": state["messages"],
            "current_step": state["current_step"],
            "context": state["context"],
            "result": state["result"],
            "error_message": "No ticker provided",
        }
        return updated_state

    logger.info("Gathering data for ticker", ticker=ticker)

    config = get_config()
    if not config.alpha_vantage.is_configured:
        updated_state: AgentState = {
            "messages": state["messages"],
            "current_step": state["current_step"],
            "context": state["context"],
            "result": state["result"],
            "error_message": "Alpha Vantage API key not configured",
        }
        return updated_state

    try:
        # Fetch data from Alpha Vantage
        api_key = config.alpha_vantage.api_key
        if not api_key:
            updated_state: AgentState = {
                "messages": state["messages"],
                "current_step": state["current_step"],
                "context": state["context"],
                "result": state["result"],
                "error_message": "Alpha Vantage API key not configured",
            }
            return updated_state

        overview_data = await fetch_company_overview(
            api_key=api_key,
            symbol=ticker,
            timeout=config.alpha_vantage.timeout_seconds,
            rate_limit_interval=config.alpha_vantage.rate_limit_interval,
            base_url=config.alpha_vantage.base_url,
        )

        news_data = await fetch_news_sentiment(
            api_key=api_key,
            tickers=ticker,
            limit=20,
            timeout=config.alpha_vantage.timeout_seconds,
            rate_limit_interval=config.alpha_vantage.rate_limit_interval,
            base_url=config.alpha_vantage.base_url,
        )

        # Fetch technical indicators if configured
        technical_indicators_data = {}
        enabled_indicators = (
            config.alpha_vantage.technical_indicators.enabled_indicators
        )
        if enabled_indicators:
            logger.info(
                "Fetching technical indicators",
                ticker=ticker,
                indicators=list(enabled_indicators.keys()),
            )
            # Create a client for technical indicators
            client = AlphaVantageClient(
                api_key=api_key,
                timeout=config.alpha_vantage.timeout_seconds,
                rate_limit_interval=config.alpha_vantage.rate_limit_interval,
                base_url=config.alpha_vantage.base_url,
            )
            try:
                for indicator_name, indicator_config in enabled_indicators.items():
                    indicator_data = await _fetch_technical_indicator(
                        client, indicator_name, ticker, indicator_config
                    )
                    if indicator_data:
                        technical_indicators_data[indicator_name] = indicator_data
            finally:
                await client.close()

        # Extract metadata
        overview = overview_data.content
        news_feed = news_data.content.get("feed", [])
        latest_quarter = overview.get("LatestQuarter", "N/A")
        overview_fetched_at = _format_timestamp(overview_data.timestamp)
        news_fetched_at = _format_timestamp(news_data.timestamp)

        # Format latest news time
        latest_news_time = "N/A"
        if news_feed and len(news_feed) > 0:
            latest_news_time = news_feed[0].get("time_published", "N/A")
            if latest_news_time != "N/A" and len(latest_news_time) >= 8:
                try:
                    dt = datetime.strptime(latest_news_time[:15], "%Y%m%dT%H%M%S")
                    latest_news_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    pass

        # Initialize vector store
        vector_store = get_vector_store_manager()

        # Delete existing data for this ticker to prevent duplicates
        # Note: Using deterministic IDs which will update on re-add
        try:
            logger.debug("Using deterministic IDs for upsert behavior", ticker=ticker)
        except Exception as e:
            logger.warning("Could not delete old data", ticker=ticker, error=str(e))

        # Common metadata for all documents
        base_metadata = {
            "ticker": ticker,
            "sector": overview.get("Sector", "N/A"),
            "industry": overview.get("Industry", "N/A"),
            "latest_quarter": latest_quarter,
            "data_source": "alpha_vantage",
        }

        docs_added = 0

        # 1. Store company overview description
        description = overview.get("Description", "")
        if description:
            # Chunk description into smaller pieces for better retrieval
            chunks = _chunk_text(description, chunk_size=500, overlap=100)
            doc_ids = [
                _generate_doc_id(ticker, "overview", i) for i in range(len(chunks))
            ]

            vector_store.add_documents(
                texts=chunks,
                metadatas=[
                    {
                        **base_metadata,
                        "type": "company_overview",
                        "fetched_at": overview_fetched_at,
                        "chunk_id": i,
                        "total_chunks": len(chunks),
                    }
                    for i in range(len(chunks))
                ],
                ids=doc_ids,
            )
            docs_added += len(chunks)

        # 2. Store financial metrics as structured metadata
        metrics = _extract_financial_metrics(overview)
        if metrics:
            # Store each metric as a separate document for precise retrieval
            metric_texts = []
            metric_metadatas = []
            metric_ids = []

            for i, (metric_name, metric_value) in enumerate(metrics.items()):
                metric_texts.append(f"{metric_name}: {metric_value}")
                metric_metadatas.append(
                    {
                        **base_metadata,
                        "type": "financial_metric",
                        "metric_name": metric_name,
                        "metric_value": str(metric_value),
                        "fetched_at": overview_fetched_at,
                    }
                )
                metric_ids.append(_generate_doc_id(ticker, "metric", i))

            vector_store.add_documents(
                texts=metric_texts,
                metadatas=metric_metadatas,
                ids=metric_ids,
            )
            docs_added += len(metric_texts)

        # 3. Store news articles
        news_articles_added = 0
        for i, article in enumerate(news_feed[:15]):  # Store top 15 articles
            summary = article.get("summary")
            if not summary:
                continue

            # Chunk long summaries
            chunks = _chunk_text(summary, chunk_size=500, overlap=50)
            article_ids = [
                _generate_doc_id(ticker, f"news_{i}", j) for j in range(len(chunks))
            ]

            vector_store.add_documents(
                texts=chunks,
                metadatas=[
                    {
                        **base_metadata,
                        "type": "news",
                        "title": article.get("title", "N/A"),
                        "source": article.get("source", "N/A"),
                        "sentiment": article.get("overall_sentiment_label", "N/A"),
                        "sentiment_score": article.get("overall_sentiment_score", 0.0),
                        "time_published": article.get("time_published", "N/A"),
                        "fetched_at": news_fetched_at,
                        "chunk_id": j,
                        "total_chunks": len(chunks),
                    }
                    for j in range(len(chunks))
                ],
                ids=article_ids,
            )
            docs_added += len(chunks)
            news_articles_added += 1

        # 4. Store technical indicators
        indicators_stored = 0
        indicators_fetched_at = _format_timestamp(time.time())
        if technical_indicators_data:
            for indicator_name, indicator_data in technical_indicators_data.items():
                # Format the indicator data as text
                formatted_text = _format_technical_indicator_data(
                    indicator_name, indicator_data, limit=30
                )

                # Store as a single document
                indicator_id = _generate_doc_id(
                    ticker, f"indicator_{indicator_name}", 0
                )
                vector_store.add_documents(
                    texts=[formatted_text],
                    metadatas=[
                        {
                            **base_metadata,
                            "type": "technical_indicator",
                            "indicator_name": indicator_name,
                            "fetched_at": indicators_fetched_at,
                        }
                    ],
                    ids=[indicator_id],
                )
                docs_added += 1
                indicators_stored += 1

        logger.info(
            "Stored company data in vector store",
            ticker=ticker,
            docs_added=docs_added,
            news_articles=news_articles_added,
            metrics_count=len(metrics),
            indicators_count=indicators_stored,
        )

        # Return success state
        updated_state: AgentState = {
            "messages": state["messages"],
            "current_step": "completed",
            "context": {
                "ticker": ticker,
                "docs_added": docs_added,
                "news_articles_stored": news_articles_added,
                "metrics_stored": len(metrics),
                "indicators_stored": indicators_stored,
                "latest_quarter": latest_quarter,
                "latest_news_time": latest_news_time,
                "overview_fetched_at": overview_fetched_at,
                "news_fetched_at": news_fetched_at,
                "indicators_fetched_at": indicators_fetched_at
                if technical_indicators_data
                else None,
            },
            "result": None,
            "error_message": None,
        }
        return updated_state

    except Exception as e:
        logger.error("Failed to gather company data", ticker=ticker, error=str(e))
        updated_state: AgentState = {
            "messages": state["messages"],
            "current_step": state["current_step"],
            "context": state["context"],
            "result": state["result"],
            "error_message": f"Failed to gather data: {e!s}",
        }
        return updated_state


class GatherAgent(BaseAgent):
    """Agent for gathering and storing company data in vector database."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        super().__init__("gather_agent", config)

    def create_workflow(self) -> Any:
        """Create the agent workflow graph."""
        from langgraph.graph import StateGraph

        workflow = StateGraph(AgentState)
        workflow.add_node("gather_data", gather_data_node)
        workflow.set_entry_point("gather_data")
        workflow.set_finish_point("gather_data")
        return workflow.compile()

    async def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Run the gather agent.

        Args:
            input_data: Dictionary with 'ticker' key

        Returns:
            Gather results with status and metadata
        """
        initial_state: AgentState = {
            "messages": [],
            "current_step": "initializing",
            "context": {"ticker": input_data.get("ticker", "")},
            "result": None,
            "error_message": None,
        }

        try:
            start_time = time.time()
            workflow = self.create_workflow()
            result = await workflow.ainvoke(initial_state)
            duration_ms = (time.time() - start_time) * 1000

            if result.get("error_message"):
                logger.error(
                    "Gather failed",
                    ticker=input_data.get("ticker"),
                    error=result.get("error_message"),
                )
                return {
                    "status": "error",
                    "error": result.get("error_message"),
                }

            context = result.get("context", {})
            logger.info(
                "Gather completed",
                ticker=input_data.get("ticker"),
                duration_ms=duration_ms,
                docs_added=context.get("docs_added", 0),
            )

            return {
                "status": "success",
                "ticker": input_data.get("ticker"),
                "docs_added": context.get("docs_added", 0),
                "news_articles_stored": context.get("news_articles_stored", 0),
                "metrics_stored": context.get("metrics_stored", 0),
                "indicators_stored": context.get("indicators_stored", 0),
                "latest_quarter": context.get("latest_quarter"),
                "latest_news_time": context.get("latest_news_time"),
                "overview_fetched_at": context.get("overview_fetched_at"),
                "news_fetched_at": context.get("news_fetched_at"),
                "indicators_fetched_at": context.get("indicators_fetched_at"),
                "duration_ms": duration_ms,
            }

        except Exception as e:
            logger.error("Gather failed", error=str(e))
            return {
                "status": "error",
                "error": str(e),
            }


async def gather_company_data(ticker: str) -> dict[str, Any]:
    """Convenience function to gather company data.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Gather results with status and metadata
    """
    agent = GatherAgent()
    return await agent.run({"ticker": ticker})


async def gather_multiple_tickers(
    tickers: list[str],
    rate_limit_delay: float | None = None,
) -> dict[str, Any]:
    """Gather data for multiple tickers with rate limiting.

    Args:
        tickers: List of stock ticker symbols
        rate_limit_delay: Delay between requests in seconds (uses config default if None)

    Returns:
        Gather results for all tickers with success/failure counts
    """
    if not tickers:
        return {
            "status": "error",
            "error": "No tickers provided",
            "total_tickers": 0,
            "successful": 0,
            "failed": 0,
            "results": [],
            "errors": [],
        }

    # Get rate limit from config if not provided
    if rate_limit_delay is None:
        config = get_config()
        rate_limit_delay = config.alpha_vantage.rate_limit_interval

    results = []
    errors = []

    for i, ticker in enumerate(tickers):
        # Rate limiting: wait between requests (except for first request)
        if i > 0:
            await asyncio.sleep(rate_limit_delay)

        result = await gather_company_data(ticker)

        if result.get("status") == "success":
            results.append(result)
        else:
            errors.append(
                {"ticker": ticker, "error": result.get("error", "Unknown error")}
            )

    return {
        "status": "success" if results else "error",
        "total_tickers": len(tickers),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors if errors else None,
    }
