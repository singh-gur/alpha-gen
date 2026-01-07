"""CLI commands subpackage."""

from .analyze import analyze_app
from .news import news_app
from .opportunities import opportunities_app
from .research import research_app

__all__ = [
    "analyze_app",
    "news_app",
    "opportunities_app",
    "research_app",
]
