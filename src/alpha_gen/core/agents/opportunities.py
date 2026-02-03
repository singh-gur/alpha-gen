"""Opportunities agent for finding investment opportunities."""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

from alpha_gen.core.agents.base import AgentConfig, AgentState, BaseAgent
from alpha_gen.core.config.settings import get_config
from alpha_gen.core.data_sources.alpha_vantage import fetch_company_overview
from alpha_gen.core.data_sources.yahoo_finance import fetch_yahoo_losers
from alpha_gen.core.rag import DocumentProcessor, get_vector_store_manager
from alpha_gen.core.utils.logging import get_langfuse_handler, get_logger

logger = get_logger(__name__)


OPPORTUNITIES_SYSTEM_PROMPT = """You are an expert investment analyst specializing in identifying
value investment opportunities. Your task is to analyze market data and
identify stocks that may be undervalued or have strong fundamentals despite
recent underperformance.

Key criteria for analysis:
1. Price drop magnitude and recent trend
2. Trading volume and liquidity
3. Potential catalysts for recovery
4. Fundamental metrics (P/E, revenue growth, etc.)
5. Sector performance and trends
6. Risk assessment

For each opportunity identified, provide:
- Ticker and company name
- Why it appears undervalued or has recovery potential
- Key metrics and data points
- Risk factors
- Confidence level (high/medium/low)
"""


async def fetch_losers_node(state: AgentState) -> AgentState:
    """Fetch top losers from Yahoo Finance."""
    logger.info("Fetching top losers from Yahoo Finance")

    try:
        limit = state["context"].get("limit", 25)

        # Fetch losers from Yahoo Finance
        data = await fetch_yahoo_losers(limit=limit, timeout=60)

        # Transform Yahoo Finance data to match expected format
        losers = []
        for stock in data.content.get("stocks", []):
            losers.append(
                {
                    "ticker": stock["symbol"],
                    "name": stock["name"],
                    "price": str(stock["price"]),
                    "change": f"{stock['percent_change']}%",
                    "change_amount": str(stock["change"]),
                    "change_percentage": f"{stock['percent_change']}%",
                    "volume": str(stock["volume"]),
                }
            )

        return {
            **state,
            "context": {
                **state["context"],
                "losers_data": losers,
            },
            "current_step": "analyzing",
        }

    except Exception as e:
        logger.error("Failed to fetch losers from Yahoo Finance", error=str(e))
        return {
            **state,
            "error_message": f"Failed to fetch losers: {e!s}",
        }


async def fetch_detailed_data_node(state: AgentState) -> AgentState:
    """Fetch detailed company overview data for top losers."""
    # Check if there was an error in previous step
    if state.get("error_message"):
        return state

    losers = state["context"].get("losers_data", [])
    tickers_to_analyze = [loser["ticker"] for loser in losers[:5]]  # Top 5 losers

    logger.info(
        "Fetching detailed company data from Alpha Vantage", tickers=tickers_to_analyze
    )

    config = get_config()
    if not config.alpha_vantage.is_configured:
        return {
            **state,
            "error_message": "Alpha Vantage API key not configured",
        }

    detailed_data: dict[str, Any] = {}

    for ticker in tickers_to_analyze:
        try:
            overview_data = await fetch_company_overview(
                api_key=config.alpha_vantage.api_key,  # type: ignore[arg-type]
                symbol=ticker,
                timeout=config.alpha_vantage.timeout_seconds,
                rate_limit_interval=config.alpha_vantage.rate_limit_interval,
                base_url=config.alpha_vantage.base_url,
            )
            detailed_data[ticker] = {
                "overview": overview_data.content,
            }
        except Exception as e:
            logger.warning(f"Failed to fetch data for {ticker}", error=str(e))

    return {
        **state,
        "context": {
            **state["context"],
            "detailed_data": detailed_data,
        },
        "current_step": "identifying_opportunities",
    }


async def identify_opportunities_node(state: AgentState) -> AgentState:
    """Analyze data to identify investment opportunities with RAG enhancement."""
    # Check if there was an error in previous step
    if state.get("error_message"):
        return state

    context = state["context"]
    losers_data = context.get("losers_data", [])
    detailed_data = context.get("detailed_data", {})

    # Initialize RAG components
    vector_store = get_vector_store_manager()
    doc_processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)

    # Store loser data in vector store
    try:
        for ticker, data in detailed_data.items():
            if data.get("Description"):
                docs = doc_processor.process(
                    content=data["Description"],
                    source=f"loser_analysis_{ticker}",
                    metadata={
                        "ticker": ticker,
                        "type": "loser_opportunity",
                        "sector": data.get("Sector", ""),
                        "industry": data.get("Industry", ""),
                    },
                )
                vector_store.add_documents(
                    texts=[doc.page_content for doc in docs],
                    metadatas=[doc.metadata for doc in docs],
                )
        logger.info("Stored loser data in vector store", count=len(detailed_data))
    except Exception as e:
        logger.warning("Failed to store loser data", error=str(e))

    # Retrieve similar recovery patterns
    rag_context = ""
    try:
        query = "Companies that recovered from significant losses and became good investments"
        similar_docs = vector_store.similarity_search(
            query=query,
            k=3,
            metadata_filter={"type": "loser_opportunity"},
        )

        if similar_docs:
            rag_context = "\n\nHistorical Recovery Patterns:\n"
            for i, doc in enumerate(similar_docs, 1):
                rag_context += f"\n{i}. {doc.metadata.get('ticker', 'Unknown')}: {doc.content[:150]}...\n"

            logger.info("Retrieved recovery patterns", docs_found=len(similar_docs))
    except Exception as e:
        logger.warning("Failed to retrieve recovery patterns", error=str(e))

    analysis_prompt = f"""Analyze the following market data to identify investment opportunities:

Losers List Data:
{losers_data[:10]!s}

Detailed Analysis for Top Losers:
{str(detailed_data)[:3000]}
{rag_context}

Based on this data and historical recovery patterns, identify the most promising investment opportunities.
Focus on companies that:
1. Have strong fundamentals but are oversold
2. Have positive news catalysts or recovery potential
3. Show unusual trading activity
4. Are in sectors with favorable trends

For each opportunity, provide:
1. Ticker and brief company description
2. Why it's a potential opportunity
3. Key metrics supporting the thesis
4. Risk factors
5. Confidence level

Format the output as a structured report.
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
    from langchain_core.messages import SystemMessage

    messages = [
        SystemMessage(content=OPPORTUNITIES_SYSTEM_PROMPT),
        HumanMessage(content=analysis_prompt),
    ]

    try:
        # Invoke LLM directly with messages and Langfuse callback
        response = await llm.ainvoke(messages, config={"callbacks": callbacks})  # type: ignore[arg-type]
        analysis = response.content

        return {
            **state,
            "messages": state["messages"] + [HumanMessage(content=analysis)],
            "result": analysis,
            "current_step": "completed",
        }
    except Exception as e:
        logger.error("Opportunity analysis failed", error=str(e))
        return {
            **state,
            "error_message": f"Analysis failed: {e!s}",
        }


def should_continue(state: AgentState) -> str:
    """Determine if workflow should continue or terminate due to error."""
    if state.get("error_message"):
        return "end"
    return "continue"


def create_opportunities_workflow() -> Any:
    """Create the opportunities agent workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node("fetch_losers", fetch_losers_node)
    workflow.add_node("fetch_detailed", fetch_detailed_data_node)
    workflow.add_node("identify", identify_opportunities_node)

    workflow.set_entry_point("fetch_losers")

    # Add conditional edges that check for errors
    workflow.add_conditional_edges(
        "fetch_losers",
        should_continue,
        {
            "continue": "fetch_detailed",
            "end": "__end__",
        },
    )
    workflow.add_conditional_edges(
        "fetch_detailed",
        should_continue,
        {
            "continue": "identify",
            "end": "__end__",
        },
    )
    workflow.set_finish_point("identify")

    return workflow.compile()


class OpportunitiesAgent(BaseAgent):
    """Agent for identifying investment opportunities."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        super().__init__("opportunities_agent", config)
        self._workflow = create_opportunities_workflow()

    def create_workflow(self) -> Any:
        """Create the agent workflow graph."""
        return self._workflow

    async def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Run the opportunities agent.

        Args:
            input_data: Dictionary with optional 'limit' key

        Returns:
            Investment opportunities analysis
        """
        initial_state: AgentState = {
            "messages": [],
            "current_step": "initializing",
            "context": {
                "limit": input_data.get("limit", 25),
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
                    "Opportunities analysis failed",
                    error=result.get("error_message"),
                )
                return {
                    "status": "error",
                    "error": result.get("error_message"),
                }

            logger.info(
                "Opportunities analysis completed",
                duration_ms=duration_ms,
            )

            return {
                "status": "success",
                "losers_data": result.get("context", {}).get("losers_data", []),
                "analysis": result.get("result"),
                "duration_ms": duration_ms,
            }

        except Exception as e:
            logger.error("Opportunities analysis failed", error=str(e))
            return {
                "status": "error",
                "error": str(e),
            }


async def find_opportunities(limit: int = 25) -> dict[str, Any]:
    """Convenience function to find investment opportunities.

    Args:
        limit: Number of losers to analyze

    Returns:
        Investment opportunities analysis
    """
    agent = OpportunitiesAgent()
    return await agent.run({"limit": limit})
