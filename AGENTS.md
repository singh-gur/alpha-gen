# AGENTS.md - Alpha Gen Development Guide

This document provides guidelines and commands for agentic coding agents working on Alpha Gen.

## Quick Start Commands

### Setup and Dependencies

```bash
# Install dependencies (creates .venv automatically)
just sync
just sync-dev  # Includes dev dependencies (linters, type checkers)

# Update dependencies
just update
```

### Development Commands

```bash
# Run all quality checks
just check

# Run linting (ruff)
just lint
just lint-fix  # Auto-fix linting issues

# Format code (ruff)
just format
just format-check

# Type checking (basedpyright - configured to warn-only for LangChain/LangGraph)
just type-check

# Run tests
just test
just test-coverage  # With coverage report

# Run single test file
uv run pytest tests/test_config.py -v

# Run single test
uv run pytest tests/test_config.py::TestLLMConfig::test_default_values -v

# Run tests with markers
uv run pytest -m "unit" -v
uv run pytest -m "not slow" -v  # Exclude slow tests
```

### Application Commands

```bash
# CLI help
just dev

# Research command
just research AAPL

# Find opportunities
just opps
just opps --limit 50

# Analyze news
just news

# Quick analysis
just analyze NVDA
```

### Docker

```bash
just docker-build
just docker-run-env .env
```

## Code Style Guidelines

### General Principles

- **Simple over complex**: Prefer straightforward Pythonic solutions
- **Explicit over implicit**: Avoid hidden behaviors (PEP 20)
- **DRY**: Eliminate duplication through abstraction
- **Single Responsibility**: Each function/class should have one clear purpose

### Imports

```python
# Standard library first, then third-party, then local
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import pydantic
import structlog
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
```

- Use `from __future__ import annotations` for all files
- Group imports: stdlib, third-party, local (separated by blank line)
- Configure ruff isort with `known-first-party = ["alpha_gen"]`

### Formatting

- **Line length**: 88 characters (ruff default)
- **Formatter**: ruff format
- Run `just format` before committing

### Type Hints

- **Required**: All function signatures must have type annotations
- **Python 3.13+**: Use latest syntax (`dict[str, Any]` over `Dict[str, Any]`)
- **Union types**: Use `|` syntax (`str | None` over `Optional[str]`)
- **Protocols**: Use `typing.Protocol` for structural typing
- **Generic types**: Use `TypeVar`, `Generic` for reusable generic classes

```python
from typing import Protocol, TypeVar, Generic

T = TypeVar("T")

class Repository(Protocol[T]):
    async def get(self, id: int) -> T | None: ...
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `AgentConfig`, `BaseDataSource`)
- **Functions/Methods**: `snake_case` (e.g., `get_config`, `fetch_data`)
- **Constants**: `SCREAMING_SNAKE_CASE` (e.g., `DEFAULT_TIMEOUT`)
- **Private Methods/Attributes**: `_leading_underscore` (e.g., `_set_status`)
- **Module Private**: `__dunder__` for special methods
- **Variables**: `snake_case` (e.g., `user_id`, `config_path`)

### Error Handling

- **Use exceptions**: For error conditions, not return codes
- **Custom exceptions**: Create specific exception classes for domain errors
- **Logging**: Use `structlog` for structured logging
- **Error context**: Log errors with relevant context before re-raising
- **No bare except**: Catch specific exceptions or use `finally` for cleanup

```python
import structlog

logger = structlog.get_logger(__name__)

class ConfigurationError(Exception):
    """Base exception for configuration errors."""

async def load_config(path: Path) -> AppConfig:
    try:
        return AppConfig.from_file(path)
    except (yaml.YAMLError, ValueError) as e:
        logger.error("Failed to load config", path=str(path), error=str(e))
        raise ConfigurationError(f"Invalid config at {path}") from e
```

### Pydantic Usage

- Use `frozen=True` for configuration classes
- Use `Field` with validators for constraints
- Use `model_validator` for cross-field validation
- Prefer `default_factory` for mutable defaults

```python
from pydantic import BaseModel, Field

class LLMConfig(BaseModel):
    """OpenAI-compatible LLM configuration. Provider is inferred from base_url."""
    model_name: str = "openrouter/default"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)
    api_key: str | None = None
    base_url: str | None = None

    model_config = {"frozen": True}

    @property
    def provider(self) -> str:
        """Infer provider from base_url."""
        if self.base_url is None:
            return "openrouter"
        if "localhost" in self.base_url or "127.0.0.1" in self.base_url:
            return "ollama"
        if "openrouter" in self.base_url:
            return "openrouter"
        return "openai-compatible"
```

### Async/Await

- Use `async`/`await` for I/O-bound operations
- Prefer `AsyncMock` for testing async code
- Use `asyncio.timeout` for async timeouts
- Handle async context managers properly

### Testing

- **Test structure**: `Test*` classes, `test_*` functions
- **Pytest markers**: `@pytest.mark.unit`, `@pytest.mark.slow`, `@pytest.mark.integration`
- **Arrange-Act-Assert**: Clear test structure
- **Mocking**: Mock external dependencies, not internal code
- **Coverage**: Aim for 90%+ coverage on business logic

```python
import pytest
from unittest.mock import AsyncMock

class TestLLMConfig:
    def test_default_values(self) -> None:
        config = LLMConfig()
        assert config.provider == "openai"
```

### Project Structure

```
alpha-gen/
├── src/alpha_gen/
│   ├── core/            # Core business logic (interface-agnostic)
│   │   ├── agents/      # LangGraph agents for research, news, opportunities
│   │   ├── data_sources/# Alpha Vantage API client and data models
│   │   ├── rag/         # Vector store & RAG functionality
│   │   ├── config/      # Application configuration and settings
│   │   ├── utils/       # Logging, observability, utilities
│   │   └── tools/       # Agent tools and utilities
│   ├── cli/             # CLI interface (Typer)
│   │   ├── commands/    # CLI command implementations
│   │   ├── helpers.py   # CLI output formatting helpers
│   │   └── main.py      # CLI entry point
│   └── main.py          # Application entry point
├── tests/
│   ├── test_*.py        # Unit tests
│   └── *_test.py        # Integration tests
└── pyproject.toml
```

**Architecture Notes:**
- **core/**: Contains all business logic, completely independent of interface
- **cli/**: CLI-specific code (Typer commands, output formatting)
- Future interfaces (MCP, API) will be siblings to `cli/` and import from `core/`
- All imports from interfaces should use `from alpha_gen.core import ...`

### Key Tooling

| Tool | Purpose |
|------|---------|
| `uv` | Package management (fast, modern) |
| `ruff` | Linting + formatting |
| `basedpyright` | Type checking (warn-only for LangChain/LangGraph) |
| `pytest` | Testing with asyncio support |
| `structlog` | Structured logging |
| `pydantic` | Data validation + settings |

### Pre-commit Hooks

```bash
just precommit-install
just precommit  # Run on all files
```

### CI Pipeline

```bash
just ci  # Run check + test
```

## Important Notes

- **LangChain/LangGraph types**: Type checking is configured as warn-only because these libraries have incomplete type stubs
- **Environment variables**: Use `.env` file for local development, see `.env.example`
- **Project requires**: Python 3.13+ and uv
- **Test paths**: Tests are in `tests/` directory, pattern: `test_*.py` or `*_test.py`

### Gather Command

The `gather` command pre-fetches and stores financial data in the vector database for faster research.

```bash
# Gather data for a single ticker
just gather AAPL
alpha-gen gather AAPL

# Gather data for multiple tickers
alpha-gen gather AAPL,MSFT,TSLA

# Use gathered data in research (faster, no API calls)
alpha-gen gather AAPL
alpha-gen research AAPL --skip-gather

# Save gather report
alpha-gen gather AAPL --save
```

**Data Freshness:**
- Gathered data is valid for 24 hours by default
- Stale data is automatically detected and rejected
- Re-run `gather` to refresh data for a ticker

**Benefits:**
- Faster research (no API calls)
- Avoid rate limits when analyzing multiple times
- Batch data collection for multiple tickers

**Storage:**
- Data is stored in vector database (Chroma by default)
- Location: `./data/vector_store/`
- Includes: company overview, financial metrics, news articles
- Uses deterministic IDs for automatic updates (no duplicates)
