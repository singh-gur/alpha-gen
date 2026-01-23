"""Base utilities for CLI commands."""

from __future__ import annotations

from typing import Annotated

import typer

# Common CLI options as reusable Annotated types
OutputOption = Annotated[
    str,
    typer.Option(
        "--output",
        "-o",
        help="Output format: 'text' (rich console), 'json' (structured data), 'markdown' (formatted report)",
    ),
]

SaveOption = Annotated[
    bool,
    typer.Option(
        "--save",
        "-s",
        help="Save report as markdown file in output directory (default: .out)",
    ),
]


def create_output_options(
    default_output: str = "text",
    default_save: bool = False,
) -> tuple[OutputOption, SaveOption]:
    """Create standard output and save options with defaults.

    Args:
        default_output: Default output format
        default_save: Default save behavior

    Returns:
        Tuple of (output, save) with proper defaults
    """
    # Note: Typer doesn't support dynamic defaults with Annotated,
    # so we return the types and let commands set defaults
    return OutputOption, SaveOption  # type: ignore[return-value]
