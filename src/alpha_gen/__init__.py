"""Alpha Gen - AI-powered investment research assistant."""

__version__ = "0.1.0"

from .agents import analyze_news, find_opportunities, research_company
from .cli import app

__all__ = [
    "__version__",
    "analyze_news",
    "app",
    "find_opportunities",
    "research_company",
]
