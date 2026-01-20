"""Research agent for company analysis."""

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
from alpha_gen.core.data_sources.alpha_vantage import (
    fetch_company_overview,
    fetch_news_sentiment,
)
from alpha_gen.core.utils.logging import get_logger

logger = get_logger(__name__)


RESEARCH_SYSTEM_PROMPT = """You are an expert investment research analyst. Your task is to analyze company data
and provide comprehensive investment research reports.

Your analysis should include:
1. Company overview and business model
2. Financial health assessment (revenue, profitability, debt)
3. Competitive positioning
4. Recent news and sentiment analysis
5. Risk assessment
6. Investment recommendation with confidence level

Be thorough, objective, and base your analysis on the provided data.
Provide specific metrics and figures where available.
"""


async def fetch_company_data_node(state: AgentState) -> AgentState:
    """Fetch company data from Alpha Vantage."""
    ticker = state["context"].get("ticker", "").upper()

    if not ticker:
        return {
            **state,
            "error_message": "No ticker provided",
        }

    logger.info("Fetching company data from Alpha Vantage", ticker=ticker)

    config = get_config()
    if not config.alpha_vantage.is_configured:
        return {
            **state,
            "error_message": "Alpha Vantage API key not configured",
        }

    try:
        # Fetch company overview and news sentiment
        overview_data = await fetch_company_overview(
            api_key=config.alpha_vantage.api_key,  # type: ignore[arg-type]
            symbol=ticker,
            timeout=config.alpha_vantage.timeout_seconds,
        )

        news_data = await fetch_news_sentiment(
            api_key=config.alpha_vantage.api_key,  # type: ignore[arg-type]
            tickers=ticker,
            limit=20,
            timeout=config.alpha_vantage.timeout_seconds,
        )

        # Combine all data into context
        combined_context: dict[str, Any] = {
            "ticker": ticker,
            "company_overview": overview_data.content,
            "news_sentiment": news_data.content,
        }

        return {
            **state,
            "context": combined_context,
            "current_step": "analyzing",
        }
    except Exception as e:
        logger.error("Failed to fetch company data", ticker=ticker, error=str(e))
        return {
            **state,
            "error_message": f"Failed to fetch data: {e!s}",
        }


async def analyze_data_node(state: AgentState) -> AgentState:
    """Analyze the company data from Alpha Vantage."""
    context = state["context"]
    overview = context.get("company_overview", {})
    news_sentiment = context.get("news_sentiment", {})
    ticker = context.get("ticker", "")

    # Build a comprehensive analysis prompt
    analysis_prompt = f"""Analyze the following company data for {ticker}:

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
{str(news_sentiment.get("feed", [])[:10])[:3000]}

Please provide a comprehensive investment research report including:
1. Company overview and business model
2. Financial health assessment
3. Valuation analysis
4. News sentiment and market perception
5. Investment recommendation with confidence level
"""

    config = get_config()
    llm = ChatOpenAI(
        model=config.llm.model_name,
        temperature=config.llm.temperature,
        api_key=config.llm.api_key,  # type: ignore[arg-type]
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESEARCH_SYSTEM_PROMPT),
            ("human", analysis_prompt),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    try:
        analysis = await chain.ainvoke({})

        return {
            **state,
            "messages": state["messages"] + [HumanMessage(content=analysis)],
            "result": analysis,
            "current_step": "completed",
        }
    except Exception as e:
        logger.error("Analysis failed", ticker=ticker, error=str(e))
        return {
            **state,
            "error_message": f"Analysis failed: {e!s}",
        }


def create_research_workflow() -> StateGraph:
    """Create the research agent workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node("fetch_data", fetch_company_data_node)
    workflow.add_node("analyze", analyze_data_node)

    workflow.set_entry_point("fetch_data")
    workflow.add_edge("fetch_data", "analyze")
    workflow.add_edge("analyze", "final")

    return workflow


class ResearchAgent(BaseAgent):
    """Agent for conducting deep-dive company research."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        super().__init__("research_agent", config)
        self._workflow = create_research_workflow()

    def create_workflow(self) -> StateGraph:
        """Create the agent workflow graph."""
        return self._workflow

    async def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Run the research agent.

        Args:
            input_data: Dictionary with 'ticker' key

        Returns:
            Research analysis results
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

            result = await self._workflow.ainvoke(initial_state)  # type: ignore[attr-defined]

            duration_ms = (time.time() - start_time) * 1000

            logger.info(
                "Research completed",
                ticker=input_data.get("ticker"),
                duration_ms=duration_ms,
            )

            return {
                "status": "success",
                "ticker": input_data.get("ticker"),
                "analysis": result.get("result"),
                "context": result.get("context"),
                "duration_ms": duration_ms,
            }

        except Exception as e:
            logger.error("Research failed", error=str(e))
            return {
                "status": "error",
                "error": str(e),
            }


async def research_company(ticker: str) -> dict[str, Any]:
    """Convenience function to research a company.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Research analysis results
    """
    agent = ResearchAgent()
    return await agent.run({"ticker": ticker})
