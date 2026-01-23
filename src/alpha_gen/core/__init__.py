"""Core package for Alpha Gen - contains all business logic.

This package contains:
- agents: LangGraph agents for research, news analysis, and opportunities
- config: Application configuration and settings
- data_sources: Alpha Vantage API client and data models
- rag: Vector store and RAG functionality
- utils: Logging, observability, and utilities
- tools: Agent tools and utilities
"""

from __future__ import annotations

from alpha_gen.core.agents import (
    AgentConfig,
    AgentState,
    AgentStatus,
    BaseAgent,
    NewsAgent,
    OpportunitiesAgent,
    ResearchAgent,
    analyze_news,
    find_opportunities,
    research_company,
)
from alpha_gen.core.config import AppConfig, get_config
from alpha_gen.core.data_sources import (
    AlphaVantageClient,
    BaseDataSource,
    CompanyOverview,
    NewsArticle,
    SourceData,
    TopGainerLoser,
    fetch_company_overview,
    fetch_news_sentiment,
    fetch_top_gainers_losers,
)
from alpha_gen.core.utils import (
    LogContext,
    ObservabilityManager,
    ObservableContext,
    get_logger,
    get_observability_manager,
    log_execution,
    observe_agent_execution,
    setup_logging,
)

__all__ = [  # noqa: RUF022 - intentionally grouped by category
    # Agents
    "AgentConfig",
    "AgentState",
    "AgentStatus",
    "BaseAgent",
    "NewsAgent",
    "OpportunitiesAgent",
    "ResearchAgent",
    "analyze_news",
    "find_opportunities",
    "research_company",
    # Config
    "AppConfig",
    "get_config",
    # Data Sources
    "AlphaVantageClient",
    "BaseDataSource",
    "CompanyOverview",
    "NewsArticle",
    "SourceData",
    "TopGainerLoser",
    "fetch_company_overview",
    "fetch_news_sentiment",
    "fetch_top_gainers_losers",
    # Utils
    "LogContext",
    "ObservabilityManager",
    "ObservableContext",
    "get_logger",
    "get_observability_manager",
    "log_execution",
    "observe_agent_execution",
    "setup_logging",
]
