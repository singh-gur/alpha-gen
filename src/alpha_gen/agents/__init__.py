"""Agents module for Alpha Gen."""

from alpha_gen.agents.base import AgentConfig, AgentState, AgentStatus, BaseAgent
from alpha_gen.agents.news import NewsAgent, analyze_news
from alpha_gen.agents.opportunities import OpportunitiesAgent, find_opportunities
from alpha_gen.agents.research import ResearchAgent, research_company

__all__ = [
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
]
