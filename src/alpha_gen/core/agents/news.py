"""News agent for analyzing news-based investment opportunities."""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

from alpha_gen.core.agents.base import AgentConfig, AgentState, BaseAgent
from alpha_gen.core.config.settings import get_config
from alpha_gen.core.data_sources.alpha_vantage import fetch_news_sentiment
from alpha_gen.core.utils.logging import get_logger

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
    """Analyze sentiment of collected news."""
    news_data = state["context"].get("news_data", {})
    feed = news_data.get("feed", [])[:20]  # Top 20 articles

    analysis_prompt = f"""Analyze the sentiment and investment implications of the following news articles:

{str(feed)[:5000]}

For each company mentioned, provide:
1. Overall sentiment (positive/negative/neutral) based on sentiment scores
2. Key news drivers and catalysts
3. Potential stock price impact (short-term and long-term)
4. Investment opportunity rating (1-5 stars)

Aggregate the findings into a market sentiment overview with actionable insights.
"""

    config = get_config()
    llm = ChatOpenAI(
        model=config.llm.model_name,
        temperature=config.llm.temperature,
        api_key=config.llm.api_key,  # type: ignore[arg-type]
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", NEWS_SYSTEM_PROMPT),
            ("human", analysis_prompt),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    try:
        sentiment_analysis = await chain.ainvoke({})

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
    llm = ChatOpenAI(
        model=config.llm.model_name,
        temperature=config.llm.temperature,
        api_key=config.llm.api_key,  # type: ignore[arg-type]
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", NEWS_SYSTEM_PROMPT),
            ("human", opportunities_prompt),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    try:
        opportunities = await chain.ainvoke({})

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


def create_news_workflow() -> StateGraph:
    """Create the news agent workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node("fetch_news", fetch_market_news_node)
    workflow.add_node("analyze_sentiment", analyze_sentiment_node)
    workflow.add_node("identify", identify_opportunities_node)

    workflow.set_entry_point("fetch_news")
    workflow.add_edge("fetch_news", "analyze_sentiment")
    workflow.add_edge("analyze_sentiment", "identify")
    workflow.add_edge("identify", "final")

    return workflow


class NewsAgent(BaseAgent):
    """Agent for analyzing news-based investment opportunities."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        super().__init__("news_agent", config)
        self._workflow = create_news_workflow()

    def create_workflow(self) -> StateGraph:
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

            result = await self._workflow.ainvoke(initial_state)  # type: ignore[attr-defined]

            duration_ms = (time.time() - start_time) * 1000

            logger.info(
                "News analysis completed",
                duration_ms=duration_ms,
            )

            return {
                "status": "success",
                "analysis": result.get("result"),
                "sentiment": result.get("context", {}).get("sentiment_analysis"),
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
