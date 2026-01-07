# Alpha Gen

Multi-agentic AI investment research assistant using Yahoo Finance data.

## Features

- **Deep-dive Research**: Analyze company financials, competitors, and news
- **Opportunities Finder**: Identify undervalued stocks from underperformers
- **News Analysis**: Find investment opportunities from recent news
- **Agentic RAG**: Intelligent data retrieval and analysis

## Installation

### Prerequisites

- Python 3.13+
- uv (fast Python package manager)

### Setup

```bash
# Clone the repository
cd alpha-gen

# Sync dependencies (creates .venv automatically)
just sync

# Or using uv directly
uv sync
```

## Usage

### CLI Commands

```bash
# Research a company
just research AAPL
just research MSFT

# Find opportunities from losers list
just opportunities
just opportunities --limit 50

# Analyze recent news
just news

# Quick analysis
just analyze NVDA
just analyze TSLA --news

# With custom log level
just run-with-log-level DEBUG research AAPL
```

### Using uv run directly

```bash
uv run python -m alpha_gen.cli.main research AAPL
uv run python -m alpha_gen.cli.main opportunities
uv run python -m alpha_gen.cli.main news
```

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
OPENAI_API_KEY=your-api-key-here
LANGFUSE_PUBLIC_KEY=your-langfuse-key
LANGFUSE_SECRET_KEY=your-langfuse-secret
LOG_LEVEL=INFO
```

## Development

```bash
# Install dev dependencies
just sync-dev

# Run linting
just lint
just lint-fix

# Run type checking
just type-check

# Run all checks
just check

# Run tests
just test
just test-coverage

# Format code
just format
just format-check

# Run pre-commit hooks
just precommit-install
just precommit
```

## Docker

```bash
# Build image
just docker-build

# Run container
docker run --rm -e OPENAI_API_KEY=xxx alpha-gen research AAPL

# With env file
just docker-run-env .env
```

## Project Structure

```
alpha-gen/
├── src/alpha_gen/
│   ├── agents/          # LangGraph agents
│   ├── scrapers/        # Playwright scrapers
│   ├── rag/             # Vector store & RAG
│   ├── cli/             # Typer CLI
│   ├── config/          # Configuration
│   └── utils/           # Logging, observability
├── tests/               # Test suite
├── pyproject.toml       # Project config
└── justfile             # Task runner
```

## License

MIT
