"""Research agent for company analysis."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

from alpha_gen.core.agents.base import AgentConfig, AgentState, BaseAgent
from alpha_gen.core.config.settings import get_config
from alpha_gen.core.data_sources.alpha_vantage import (
    fetch_company_overview,
    fetch_news_sentiment,
)
from alpha_gen.core.rag import DocumentProcessor, get_vector_store_manager
from alpha_gen.core.utils.logging import get_langfuse_handler, get_logger

logger = get_logger(__name__)


RESEARCH_SYSTEM_PROMPT = """You are an expert investment research analyst. Your task is to analyze company data
and provide comprehensive investment research reports.

Your analysis should include:
1. Company overview and business model
2. Financial health assessment (revenue, profitability, debt)
3. Competitive positioning
4. Recent news and sentiment analysis - IMPORTANT: Analyze each news article provided, noting sentiment trends and key developments
5. Risk assessment
6. Investment recommendation with confidence level

Be thorough, objective, and base your analysis on the provided data.
Provide specific metrics and figures where available.
When analyzing news sentiment, reference specific articles and their sentiment scores to support your conclusions.
"""


async def fetch_company_data_node(state: AgentState) -> AgentState:
    """Fetch company data from Alpha Vantage or vector store."""
    ticker = state["context"].get("ticker", "").upper()
    skip_gather = state["context"].get("skip_gather", False)

    if not ticker:
        updated_state: AgentState = {
            "messages": state["messages"],
            "current_step": state["current_step"],
            "context": state["context"],
            "result": state["result"],
            "error_message": "No ticker provided",
        }
        return updated_state

    # If skip_gather is True, retrieve data from vector store
    if skip_gather:
        logger.info("Using pre-gathered data from vector store", ticker=ticker)
        return await _retrieve_from_vector_store(state, ticker)

    logger.info("Fetching company data from Alpha Vantage", ticker=ticker)

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
        # Fetch company overview and news sentiment
        # Rate limiting is handled automatically by the AlphaVantageClient
        overview_data = await fetch_company_overview(
            api_key=config.alpha_vantage.api_key,  # type: ignore[arg-type]
            symbol=ticker,
            timeout=config.alpha_vantage.timeout_seconds,
            rate_limit_interval=config.alpha_vantage.rate_limit_interval,
            base_url=config.alpha_vantage.base_url,
        )

        news_data = await fetch_news_sentiment(
            api_key=config.alpha_vantage.api_key,  # type: ignore[arg-type]
            tickers=ticker,
            limit=20,
            timeout=config.alpha_vantage.timeout_seconds,
            rate_limit_interval=config.alpha_vantage.rate_limit_interval,
            base_url=config.alpha_vantage.base_url,
        )

        # Extract timestamps from the actual data
        # Company overview includes LatestQuarter field
        latest_quarter = overview_data.content.get("LatestQuarter", "N/A")

        # News data includes time_published for each article
        news_feed = news_data.content.get("feed", [])
        latest_news_time = "N/A"
        if news_feed and len(news_feed) > 0:
            # Get the most recent news timestamp (format: YYYYMMDDTHHMMSS)
            latest_news_time = news_feed[0].get("time_published", "N/A")
            # Format it nicely if available
            if latest_news_time != "N/A" and len(latest_news_time) >= 8:
                try:
                    # Parse YYYYMMDDTHHMMSS format
                    dt = datetime.strptime(latest_news_time[:15], "%Y%m%dT%H%M%S")
                    latest_news_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    pass  # Keep original if parsing fails

        overview_fetched_at = datetime.fromtimestamp(
            overview_data.timestamp, tz=UTC
        ).strftime("%Y-%m-%d %H:%M:%S UTC")
        news_fetched_at = datetime.fromtimestamp(news_data.timestamp, tz=UTC).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )

        # Combine all data into context
        combined_context: dict[str, Any] = {
            "ticker": ticker,
            "company_overview": overview_data.content,
            "news_sentiment": news_data.content,
            "latest_quarter": latest_quarter,
            "latest_news_time": latest_news_time,
            "overview_fetched_at": overview_fetched_at,
            "news_fetched_at": news_fetched_at,
        }

        updated_state: AgentState = {
            "messages": state["messages"],
            "current_step": "analyzing",
            "context": combined_context,
            "result": state["result"],
            "error_message": state["error_message"],
        }
        return updated_state
    except Exception as e:
        logger.error("Failed to fetch company data", ticker=ticker, error=str(e))
        updated_state: AgentState = {
            "messages": state["messages"],
            "current_step": state["current_step"],
            "context": state["context"],
            "result": state["result"],
            "error_message": f"Failed to fetch data: {e!s}",
        }
        return updated_state


async def _retrieve_from_vector_store(state: AgentState, ticker: str) -> AgentState:
    """Retrieve pre-gathered data from vector store using retrieval module.

    Args:
        state: Current agent state
        ticker: Stock ticker symbol

    Returns:
        Updated state with retrieved data or error
    """
    from alpha_gen.core.agents.retrieval import retrieve_gathered_data

    data = retrieve_gathered_data(ticker)

    if not data:
        logger.warning("No pre-gathered data found", ticker=ticker)
        updated_state: AgentState = {
            "messages": state["messages"],
            "current_step": state["current_step"],
            "context": state["context"],
            "result": state["result"],
            "error_message": (
                f"No pre-gathered data found for {ticker}. "
                f"Run 'alpha-gen gather {ticker}' first."
            ),
        }
        return updated_state

    # Data retrieved successfully
    updated_state: AgentState = {
        "messages": state["messages"],
        "current_step": "analyzing",
        "context": data,
        "result": state["result"],
        "error_message": None,
    }
    return updated_state


async def analyze_data_node(state: AgentState) -> AgentState:
    """Analyze the company data from Alpha Vantage with RAG enhancement."""
    # Check if there was an error in previous step
    if state.get("error_message"):
        return state

    context = state["context"]
    overview = context.get("company_overview", {})
    news_sentiment = context.get("news_sentiment", {})
    ticker = context.get("ticker", "")

    # Initialize RAG components
    vector_store = get_vector_store_manager()
    doc_processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)

    # Store current company data in vector store for future retrieval
    try:
        company_description = overview.get("Description", "")
        if company_description:
            docs = doc_processor.process(
                content=company_description,
                source=f"company_overview_{ticker}",
                metadata={
                    "ticker": ticker,
                    "type": "company_overview",
                    "sector": overview.get("Sector", ""),
                    "industry": overview.get("Industry", ""),
                    "latest_quarter": context.get("latest_quarter"),
                    "fetched_at": context.get("overview_fetched_at"),
                },
            )
            vector_store.add_documents(
                texts=[doc.page_content for doc in docs],
                metadatas=[doc.metadata for doc in docs],
            )

        # Store news articles in vector store
        news_feed = news_sentiment.get("feed", [])
        for article in news_feed[:10]:  # Store top 10 news articles
            if article.get("summary"):
                news_docs = doc_processor.process(
                    content=article["summary"],
                    source=f"news_{ticker}_{article.get('time_published', '')}",
                    metadata={
                        "ticker": ticker,
                        "type": "news",
                        "title": article.get("title", ""),
                        "source": article.get("source", ""),
                        "sentiment": article.get("overall_sentiment_label", ""),
                        "time_published": article.get("time_published", ""),
                        "fetched_at": context.get("news_fetched_at"),
                    },
                )
                vector_store.add_documents(
                    texts=[doc.page_content for doc in news_docs],
                    metadatas=[doc.metadata for doc in news_docs],
                )

        logger.info("Stored company data in vector store", ticker=ticker)
    except Exception as e:
        logger.warning("Failed to store data in vector store", error=str(e))

    # Retrieve relevant historical context
    rag_context = ""
    try:
        query = (
            f"Investment analysis for {ticker} in {overview.get('Sector', '')} sector"
        )
        similar_docs = vector_store.similarity_search(
            query=query,
            k=5,
            metadata_filter={"type": "company_overview"},
        )

        filtered_docs = [
            doc for doc in similar_docs if doc.metadata.get("ticker") != ticker
        ]

        if filtered_docs:
            rag_context = "\n\nRelevant Historical Context:\n"
            for i, doc in enumerate(filtered_docs, 1):
                rag_context += (
                    f"\n{i}. {doc.metadata.get('ticker', 'Unknown')} "
                    f"({doc.metadata.get('sector', '')}): "
                    f"{doc.content[:200]}...\n"
                )

            logger.info(
                "Retrieved RAG context",
                ticker=ticker,
                docs_found=len(filtered_docs),
            )
    except Exception as e:
        logger.warning("Failed to retrieve RAG context", error=str(e))

    # Format news articles for better readability
    news_feed = news_sentiment.get("feed", [])
    formatted_news = ""
    if news_feed:
        formatted_news = "\n"
        for i, article in enumerate(news_feed[:10], 1):
            title = article.get("title", "N/A")
            summary = article.get("summary", "N/A")
            sentiment = article.get("overall_sentiment_label", "N/A")
            sentiment_score = article.get("overall_sentiment_score", "N/A")
            source = article.get("source", "N/A")
            time_published = article.get("time_published", "N/A")

            # Format timestamp if available
            if time_published != "N/A" and len(time_published) >= 8:
                try:
                    dt = datetime.strptime(time_published[:15], "%Y%m%dT%H%M%S")
                    time_published = dt.strftime("%Y-%m-%d")
                except ValueError:
                    pass

            formatted_news += f"""
Article {i}:
- Title: {title}
- Source: {source}
- Date: {time_published}
- Sentiment: {sentiment} (Score: {sentiment_score})
- Summary: {summary[:300]}...
"""
    else:
        formatted_news = "\nNo recent news articles available."

    # Build a comprehensive analysis prompt with RAG context
    analysis_prompt = f"""Analyze the following company data for {ticker}:

Data timestamps:
- Overview fetched at: {context.get("overview_fetched_at", "N/A")}
- Latest reported quarter: {context.get("latest_quarter", "N/A")}
- News fetched at: {context.get("news_fetched_at", "N/A")}
- Latest news time: {context.get("latest_news_time", "N/A")}

Company Overview:
- Name: {overview.get("Name", "N/A")}
- Sector: {overview.get("Sector", "N/A")}
- Industry: {overview.get("Industry", "N/A")}
- Market Cap: {overview.get("MarketCapitalization", "N/A")}
- P/E Ratio: {overview.get("PERatio", "N/A")}
- EPS: {overview.get("EPS", "N/A")}
- Revenue TTM: {overview.get("RevenueTTM", "N/A")}
- Profit Margin: {overview.get("ProfitMargin", "N/A")}
- 52 Week High: {overview.get("52WeekHigh", "N/A")}
- 52 Week Low: {overview.get("52WeekLow", "N/A")}
- Analyst Target Price: {overview.get("AnalystTargetPrice", "N/A")}

Description:
{overview.get("Description", "N/A")[:1000]}

Recent News & Sentiment:
{formatted_news}
{rag_context}

Please provide a comprehensive investment research report including:
1. Company overview and business model
2. Financial health assessment
3. Valuation analysis
4. News sentiment and market perception (analyze the news articles above)
5. Investment recommendation with confidence level

Note: Use the historical context above to compare with similar companies and identify patterns. If any historical context conflicts with the current data or news above, prioritize the current Alpha Vantage data.
"""

    config = get_config()

    # Get Langfuse callback handler
    langfuse_handler = get_langfuse_handler()
    callbacks = [langfuse_handler] if langfuse_handler else []

    llm = ChatOpenAI(
        model=config.llm.model_name,
        temperature=config.llm.temperature,
        api_key=config.llm.api_key,  # type: ignore[arg-type]
        base_url=config.llm.base_url,
    )

    # Create messages directly with the formatted prompt
    from langchain_core.messages import HumanMessage as HumanMsg
    from langchain_core.messages import SystemMessage

    messages = [
        SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
        HumanMsg(content=analysis_prompt),
    ]

    try:
        # Invoke LLM directly with messages and Langfuse callback
        response = await llm.ainvoke(messages, config={"callbacks": callbacks})  # type: ignore[arg-type]
        analysis_content = response.content
        if isinstance(analysis_content, list):
            analysis = "\n".join(str(item) for item in analysis_content)
        else:
            analysis = str(analysis_content)

        updated_state: AgentState = {
            "messages": state["messages"] + [HumanMessage(content=analysis)],
            "current_step": "completed",
            "context": state["context"],
            "result": analysis,
            "error_message": state["error_message"],
        }
        return updated_state
    except Exception as e:
        logger.error("Analysis failed", ticker=ticker, error=str(e))
        updated_state: AgentState = {
            "messages": state["messages"],
            "current_step": state["current_step"],
            "context": state["context"],
            "result": state["result"],
            "error_message": f"Analysis failed: {e!s}",
        }
        return updated_state


def should_continue(state: AgentState) -> str:
    """Determine if workflow should continue or terminate due to error."""
    if state.get("error_message"):
        return "end"
    return "continue"


def create_research_workflow() -> Any:
    """Create the research agent workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node("fetch_data", fetch_company_data_node)
    workflow.add_node("analyze", analyze_data_node)

    workflow.set_entry_point("fetch_data")

    # Add conditional edge that checks for errors
    workflow.add_conditional_edges(
        "fetch_data",
        should_continue,
        {
            "continue": "analyze",
            "end": "__end__",
        },
    )
    workflow.set_finish_point("analyze")

    return workflow.compile()


class ResearchAgent(BaseAgent):
    """Agent for conducting deep-dive company research."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        super().__init__("research_agent", config)
        self._workflow = create_research_workflow()

    def create_workflow(self) -> Any:
        """Create the agent workflow graph."""
        return self._workflow

    async def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Run the research agent.

        Args:
            input_data: Dictionary with 'ticker' and optional 'skip_gather' keys

        Returns:
            Research analysis results
        """
        initial_state: AgentState = {
            "messages": [],
            "current_step": "initializing",
            "context": {
                "ticker": input_data.get("ticker", ""),
                "skip_gather": input_data.get("skip_gather", False),
            },
            "result": None,
            "error_message": None,
        }

        try:
            start_time = time.time()

            result = await self._workflow.ainvoke(initial_state)

            duration_ms = (time.time() - start_time) * 1000

            # Check if workflow ended with an error
            if result.get("error_message"):
                logger.error(
                    "Research failed",
                    ticker=input_data.get("ticker"),
                    error=result.get("error_message"),
                )
                return {
                    "status": "error",
                    "error": result.get("error_message"),
                }

            logger.info(
                "Research completed",
                ticker=input_data.get("ticker"),
                duration_ms=duration_ms,
            )

            context = result.get("context", {})
            return {
                "status": "success",
                "ticker": input_data.get("ticker"),
                "analysis": result.get("result"),
                "context": context,
                "duration_ms": duration_ms,
                "latest_quarter": context.get("latest_quarter"),
                "latest_news_time": context.get("latest_news_time"),
            }

        except Exception as e:
            logger.error("Research failed", error=str(e))
            return {
                "status": "error",
                "error": str(e),
            }


async def research_company(ticker: str, skip_gather: bool = False) -> dict[str, Any]:
    """Convenience function to research a company.

    Args:
        ticker: Stock ticker symbol
        skip_gather: If True, use pre-gathered data from vector store instead of fetching from API

    Returns:
        Research analysis results
    """
    agent = ResearchAgent()
    return await agent.run({"ticker": ticker, "skip_gather": skip_gather})
