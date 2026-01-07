"""Research agent for company analysis."""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

from ..config.settings import get_config
from ..scrapers.yahoo_finance import scrape_company_data
from ..utils.logging import get_logger
from .base import AgentConfig, AgentState, BaseAgent

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
    """Fetch company data from Yahoo Finance."""
    ticker = state["context"].get("ticker", "").upper()

    if not ticker:
        return {
            **state,
            "error_message": "No ticker provided",
        }

    logger.info("Fetching company data", ticker=ticker)

    try:
        scraped_data = await scrape_company_data(ticker)

        # Combine all scraped data into context
        combined_context: dict[str, Any] = {
            "ticker": ticker,
            "scraped_data": {
                source: {
                    "source": data.source,
                    "url": data.url,
                    "content": data.content,
                }
                for source, data in scraped_data.items()
            },
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
    """Analyze the scraped company data."""
    context = state["context"]
    scraped_data = context.get("scraped_data", {})
    ticker = context.get("ticker", "")

    # Build a comprehensive analysis prompt
    analysis_prompt = f"""Analyze the following company data for {ticker}:

Company Info:
{scraped_data.get("company_info", {}).get("content", {}).get("raw_html", "N/A")[:2000]}

Financials:
{scraped_data.get("financials", {}).get("content", {}).get("financials_html", "N/A")[:2000]}

Competitors:
{scraped_data.get("competitors", {}).get("content", {}).get("competitors_html", "N/A")[:2000]}

Recent News:
{scraped_data.get("news", {}).get("content", {}).get("articles", [])[:10]}

Please provide a comprehensive investment research report including:
1. Company overview
2. Financial health
3. Competitive analysis
4. News sentiment
5. Investment recommendation
"""

    config = get_config()
    llm = ChatOpenAI(
        model=config.llm.model_name,
        temperature=config.llm.temperature,
        api_key=config.llm.api_key,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", RESEARCH_SYSTEM_PROMPT),
        ("human", analysis_prompt),
    ])

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

            result = await self._workflow.ainvoke(initial_state)

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
