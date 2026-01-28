"""News agent for analyzing news-based investment opportunities."""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

from alpha_gen.core.agents.base import AgentConfig, AgentState, BaseAgent
from alpha_gen.core.config.settings import get_config
from alpha_gen.core.data_sources.alpha_vantage import fetch_news_sentiment
from alpha_gen.core.rag import DocumentProcessor, get_vector_store_manager
from alpha_gen.core.utils.logging import get_langfuse_handler, get_logger

logger = get_logger(__name__)


NEWS_SYSTEM_PROMPT = """You are an expert investment analyst specializing in identifying
investment opportunities from news and market sentiment. Your task is to analyze
recent news articles and identify stocks with potential for significant price movement.

For each news article, analyze:
1. The potential impact on the company's stock price
2. Whether the market reaction seems justified
3. Any emerging trends or patterns
4. Short-term vs long-term implications

Focus on identifying:
- Companies with positive news catalysts
- Earnings surprises (positive or negative)
- Product launches or innovations
- Merger and acquisition speculation
- Regulatory changes
- Analyst upgrades/downgrades

Provide a structured analysis with specific tickers and confidence levels.
"""


async def fetch_market_news_node(state: AgentState) -> AgentState:
    """Fetch market news and sentiment from Alpha Vantage."""
    logger.info("Fetching market news from Alpha Vantage")

    config = get_config()
    if not config.alpha_vantage.is_configured:
        return {
            **state,
            "error_message": "Alpha Vantage API key not configured",
        }

    try:
        # Fetch news for major market indices and tech stocks
        tickers = "SPY,QQQ,DIA,AAPL,MSFT,GOOGL,AMZN,NVDA,TSLA,META"

        news_data = await fetch_news_sentiment(
            api_key=config.alpha_vantage.api_key,  # type: ignore[arg-type]
            tickers=tickers,
            limit=100,
            timeout=config.alpha_vantage.timeout_seconds,
            rate_limit_interval=config.alpha_vantage.rate_limit_interval,
            base_url=config.alpha_vantage.base_url,
        )

        return {
            **state,
            "context": {
                **state["context"],
                "news_data": news_data.content,
            },
            "current_step": "analyzing_sentiment",
        }
    except Exception as e:
        logger.error("Failed to fetch news", error=str(e))
        return {
            **state,
            "error_message": f"Failed to fetch news: {e!s}",
        }


async def analyze_sentiment_node(state: AgentState) -> AgentState:
    """Analyze sentiment of collected news with RAG enhancement."""
    # Check if there was an error in previous step
    if state.get("error_message"):
        return state

    news_data = state["context"].get("news_data", {})
    feed = news_data.get("feed", [])[:20]  # Top 20 articles

    # Initialize RAG components
    vector_store = get_vector_store_manager()
    doc_processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)

    # Store news articles in vector store
    try:
        for article in feed:
            if article.get("summary"):
                docs = doc_processor.process(
                    content=article["summary"],
                    source=f"market_news_{article.get('time_published', '')}",
                    metadata={
                        "type": "market_news",
                        "title": article.get("title", ""),
                        "source": article.get("source", ""),
                        "sentiment": article.get("overall_sentiment_label", ""),
                        "topics": str(article.get("topics", [])),
                    },
                )
                vector_store.add_documents(
                    texts=[doc.page_content for doc in docs],
                    metadatas=[doc.metadata for doc in docs],
                )
        logger.info("Stored news articles in vector store", count=len(feed))
    except Exception as e:
        logger.warning("Failed to store news articles", error=str(e))

    # Retrieve similar news patterns
    rag_context = ""
    try:
        query = "Market news with positive sentiment and investment opportunities"
        similar_docs = vector_store.similarity_search(
            query=query,
            k=5,
            metadata_filter={"type": "market_news"},
        )

        if similar_docs:
            rag_context = "\n\nSimilar Historical News Patterns:\n"
            for i, doc in enumerate(similar_docs, 1):
                rag_context += f"\n{i}. {doc.metadata.get('title', 'Unknown')} ({doc.metadata.get('sentiment', '')}): {doc.content[:150]}...\n"

            logger.info("Retrieved similar news patterns", docs_found=len(similar_docs))
    except Exception as e:
        logger.warning("Failed to retrieve news patterns", error=str(e))

    analysis_prompt = f"""Analyze the sentiment and investment implications of the following news articles:

{str(feed)[:5000]}
{rag_context}

For each company mentioned, provide:
1. Overall sentiment (positive/negative/neutral) based on sentiment scores
2. Key news drivers and catalysts
3. Potential stock price impact (short-term and long-term)
4. Investment opportunity rating (1-5 stars)

Aggregate the findings into a market sentiment overview with actionable insights.
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
        SystemMessage(content=NEWS_SYSTEM_PROMPT),
        HumanMsg(content=analysis_prompt),
    ]

    try:
        # Invoke LLM directly with messages and Langfuse callback
        response = await llm.ainvoke(messages, config={"callbacks": callbacks})  # type: ignore[arg-type]
        sentiment_analysis = response.content

        return {
            **state,
            "messages": state["messages"] + [HumanMessage(content=sentiment_analysis)],
            "context": {
                **state["context"],
                "sentiment_analysis": sentiment_analysis,
            },
            "current_step": "identifying_opportunities",
        }
    except Exception as e:
        logger.error("Sentiment analysis failed", error=str(e))
        return {
            **state,
            "error_message": f"Analysis failed: {e!s}",
        }


async def identify_opportunities_node(state: AgentState) -> AgentState:
    """Identify specific investment opportunities from news."""
    # Check if there was an error in previous step
    if state.get("error_message"):
        return state

    sentiment = state["context"].get("sentiment_analysis", "")
    news_data = state["context"].get("news_data", {})
    feed = news_data.get("feed", [])[:15]

    opportunities_prompt = f"""Based on the following news sentiment analysis and raw news data,
identify specific investment opportunities:

Sentiment Analysis:
{sentiment}

Recent News Articles with Sentiment Scores:
{str(feed)[:3000]}

For each investment opportunity, provide:
1. Ticker symbol
2. Company name
3. News catalyst
4. Expected impact (short-term/long-term)
5. Risk level (high/medium/low)
6. Confidence score (1-10)
7. Recommended action (buy/sell/hold/wait)

Focus on the most actionable opportunities with clear catalysts.
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
        SystemMessage(content=NEWS_SYSTEM_PROMPT),
        HumanMsg(content=opportunities_prompt),
    ]

    try:
        # Invoke LLM directly with messages and Langfuse callback
        response = await llm.ainvoke(messages, config={"callbacks": callbacks})  # type: ignore[arg-type]
        opportunities = response.content

        return {
            **state,
            "messages": state["messages"] + [HumanMessage(content=opportunities)],
            "result": opportunities,
            "current_step": "completed",
        }
    except Exception as e:
        logger.error("Opportunity identification failed", error=str(e))
        return {
            **state,
            "error_message": f"Analysis failed: {e!s}",
        }


def should_continue(state: AgentState) -> str:
    """Determine if workflow should continue or terminate due to error."""
    if state.get("error_message"):
        return "end"
    return "continue"


def create_news_workflow() -> Any:
    """Create the news agent workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node("fetch_news", fetch_market_news_node)
    workflow.add_node("analyze_sentiment", analyze_sentiment_node)
    workflow.add_node("identify", identify_opportunities_node)

    workflow.set_entry_point("fetch_news")

    # Add conditional edges that check for errors
    workflow.add_conditional_edges(
        "fetch_news",
        should_continue,
        {
            "continue": "analyze_sentiment",
            "end": "__end__",
        },
    )
    workflow.add_conditional_edges(
        "analyze_sentiment",
        should_continue,
        {
            "continue": "identify",
            "end": "__end__",
        },
    )
    workflow.set_finish_point("identify")

    return workflow.compile()


class NewsAgent(BaseAgent):
    """Agent for analyzing news-based investment opportunities."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        super().__init__("news_agent", config)
        self._workflow = create_news_workflow()

    def create_workflow(self) -> Any:
        """Create the agent workflow graph."""
        return self._workflow

    async def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Run the news agent.

        Args:
            input_data: Optional configuration

        Returns:
            News-based investment opportunities analysis
        """
        initial_state: AgentState = {
            "messages": [],
            "current_step": "initializing",
            "context": {},
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
                    "News analysis failed",
                    error=result.get("error_message"),
                )
                return {
                    "status": "error",
                    "error": result.get("error_message"),
                }

            logger.info(
                "News analysis completed",
                duration_ms=duration_ms,
            )

            return {
                "status": "success",
                "analysis": result.get("result"),
                "duration_ms": duration_ms,
            }

        except Exception as e:
            logger.error("News analysis failed", error=str(e))
            return {
                "status": "error",
                "error": str(e),
            }


async def analyze_news() -> dict[str, Any]:
    """Convenience function to analyze news for investment opportunities.

    Returns:
        News-based investment opportunities analysis
    """
    agent = NewsAgent()
    return await agent.run({})
