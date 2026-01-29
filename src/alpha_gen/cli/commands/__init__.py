"""CLI commands subpackage."""

from alpha_gen.cli.commands.analyze import analyze_app
from alpha_gen.cli.commands.gather import gather_command
from alpha_gen.cli.commands.news import news_app
from alpha_gen.cli.commands.opportunities import opportunities_app
from alpha_gen.cli.commands.research import research_command

__all__ = [
    "analyze_app",
    "gather_command",
    "news_app",
    "opportunities_app",
    "research_command",
]
