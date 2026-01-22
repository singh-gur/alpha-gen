# Alpha Gen

Alpha Gen is a multi-agentic AI investment research assistant powered by LangGraph and Alpha Vantage financial data. It uses specialized AI agents to analyze stocks, identify investment opportunities, and provide comprehensive research reports based on real-time market data, company fundamentals, and news sentiment.

## Overview

Alpha Gen leverages multiple autonomous AI agents, each specialized for different aspects of investment research:

- **Research Agent**: Conducts deep-dive analysis of individual companies with financial metrics, news sentiment, and investment recommendations
- **Opportunities Agent**: Scans market losers to identify undervalued stocks with recovery potential
- **News Agent**: Analyzes recent market news to find trading opportunities based on sentiment and catalysts

Each agent follows a multi-step workflow orchestrated by LangGraph, fetching real-time data from Alpha Vantage and using LLMs (OpenAI/OpenRouter) for intelligent analysis.

## Key Features

- **Deep-dive Research**: Comprehensive company analysis including financials, valuation metrics, news sentiment, and investment recommendations
- **Opportunities Finder**: Identifies undervalued stocks from market losers by analyzing fundamentals, trading patterns, and recovery potential
- **News Analysis**: Discovers investment opportunities from recent news by analyzing sentiment scores, market catalysts, and potential price impacts
- **Agentic Architecture**: Built with LangGraph for orchestrating multi-step agent workflows with structured state management
- **Real-time Data**: Fetches live market data, company overviews, and news sentiment from Alpha Vantage API
- **Configurable LLM Backend**: Supports OpenAI, OpenRouter, and local Ollama models via OpenAI-compatible API
- **Structured Logging**: Uses structlog for detailed observability and debugging
- **LangFuse Integration**: Optional tracing and observability for agent workflows

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
just build-img

# Push to registry
just push-img

# Run locally
just run-local
```

## CI/CD with Concourse

Automated testing, building, and deployment pipeline with image signing.

### Quick Setup

```bash
# 1. Generate Cosign keys
just ci-generate-keys

# 2. Setup credentials
just ci-setup
vim ci/credentials.yml

# 3. Set pipeline
just ci-set prod

# 4. Unpause and trigger
just ci-unpause prod
just ci-test prod
```

### Common Commands

```bash
# Trigger jobs
just ci-test prod
just ci-build prod

# Watch execution
just ci-watch-test prod
just ci-watch-build prod

# Manage pipeline
just ci-status prod
just ci-pause prod
just ci-destroy prod
```

See [ci/README.md](ci/README.md) for details.

## Architecture & Logic

### Multi-Agent System

Alpha Gen uses a modular agent architecture where each agent is responsible for a specific investment research task. All agents inherit from `BaseAgent` and implement a LangGraph workflow with multiple nodes.

#### 1. Research Agent (`research_agent`)

**Purpose**: Perform deep-dive analysis on a specific company ticker

**Workflow**:
```
fetch_data → analyze → final
```

**Steps**:
1. **Fetch Data Node**: Retrieves company overview and news sentiment from Alpha Vantage API
   - Company financials: Market cap, P/E ratio, EPS, profit margin, revenue, etc.
   - News sentiment: Recent articles with sentiment scores and ticker relevance
2. **Analyze Node**: Uses LLM to generate comprehensive investment report
   - Business model assessment
   - Financial health analysis (revenue, profitability, debt)
   - Valuation metrics (P/E, target price vs current price)
   - News sentiment interpretation
   - Risk assessment
   - Investment recommendation with confidence level

**Usage**: `just research AAPL` or `just analyze NVDA`

#### 2. Opportunities Agent (`opportunities_agent`)

**Purpose**: Identify undervalued stocks from market losers

**Workflow**:
```
fetch_losers → fetch_detailed → identify → final
```

**Steps**:
1. **Fetch Losers Node**: Gets top losers, gainers, and most active stocks from Alpha Vantage
   - Retrieves price changes, volumes, and percentage moves
   - Filters top N losers (default: 25)
2. **Fetch Detailed Node**: Enriches top 5 losers with company overview data
   - Fundamentals: P/E ratio, market cap, revenue growth, margins
   - Allows deeper analysis beyond just price movement
3. **Identify Opportunities Node**: LLM analyzes losers for recovery potential
   - Identifies oversold stocks with strong fundamentals
   - Looks for positive catalysts or favorable sector trends
   - Assesses risk factors and confidence levels
   - Provides structured opportunity reports

**Usage**: `just opportunities` or `just opportunities --limit 50`

#### 3. News Agent (`news_agent`)

**Purpose**: Find trading opportunities from recent market news

**Workflow**:
```
fetch_news → analyze_sentiment → identify → final
```

**Steps**:
1. **Fetch News Node**: Retrieves recent news for major indices and tech stocks
   - Fetches 100 recent articles with sentiment scores
   - Targets: SPY, QQQ, DIA, AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META
2. **Analyze Sentiment Node**: LLM processes news sentiment data
   - Aggregates sentiment scores across articles
   - Identifies key news drivers and catalysts
   - Assesses short-term vs long-term impacts
3. **Identify Opportunities Node**: Generates actionable investment ideas
   - Ranks opportunities by confidence score
   - Provides recommended actions (buy/sell/hold/wait)
   - Highlights clear catalysts and risk levels

**Usage**: `just news`

### Data Sources

#### Alpha Vantage API
The primary data provider for Alpha Gen, accessed via `AlphaVantageClient`:

- **Company Overview**: Fundamental data including financials, ratios, margins, analyst targets
- **News Sentiment**: Real-time news with AI-generated sentiment scores per ticker
- **Top Gainers/Losers**: Daily market movers with price changes and volumes

All API calls include:
- Configurable timeouts (default: 30s)
- Structured error handling with detailed logging
- Rate limiting consideration (Alpha Vantage free tier: 25 requests/day)

### Configuration System

Alpha Gen uses Pydantic models for type-safe configuration management:

```python
# Loaded from environment variables or config.yaml
AppConfig
├── LLMConfig          # Model, temperature, API keys
├── AlphaVantageConfig # API key, timeout
├── VectorStoreConfig  # Vector DB settings (future RAG)
└── ObservabilityConfig # LangFuse tracing
```

**Priority**: Environment variables override `config.yaml` values

**Key Environment Variables**:
- `LLM_API_KEY` / `LLM_MODEL` / `LLM_BASE_URL`: LLM provider configuration
- `ALPHA_VANTAGE_API_KEY`: Required for market data access
- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`: Optional agent tracing

### Agent State Management

Each agent maintains a `TypedDict` state throughout execution:

```python
AgentState = {
    "messages": list[BaseMessage],    # Conversation history
    "current_step": str,               # Current workflow node
    "context": dict[str, Any],         # Data passed between nodes
    "result": str | None,              # Final analysis output
    "error_message": str | None,       # Error tracking
}
```

State flows through the LangGraph workflow, with each node transforming and enriching it.

### LangGraph Workflows

Agents use LangGraph `StateGraph` to orchestrate multi-step processes:

1. **Nodes**: Async functions that transform agent state (fetch data, analyze, etc.)
2. **Edges**: Define execution flow between nodes
3. **Entry Point**: Starting node for the workflow
4. **Final State**: Terminal node with complete results

Each workflow is compiled into an executable graph and invoked asynchronously.

## Project Structure

```
alpha-gen/
├── src/alpha_gen/
│   ├── core/                   # Core business logic (interface-agnostic)
│   │   ├── agents/             # LangGraph agents (research, opportunities, news)
│   │   │   ├── base.py         # BaseAgent, AgentState, AgentConfig
│   │   │   ├── research.py     # Research agent workflow
│   │   │   ├── opportunities.py # Opportunities agent workflow
│   │   │   └── news.py         # News agent workflow
│   │   ├── data_sources/       # Data fetching clients
│   │   │   ├── base.py         # BaseDataSource interface
│   │   │   └── alpha_vantage.py # Alpha Vantage API client
│   │   ├── rag/                # Vector store & RAG (future)
│   │   │   └── vector_store.py # Chroma/Pinecone integration
│   │   ├── config/             # Configuration management
│   │   │   └── settings.py     # Pydantic config models
│   │   └── utils/              # Utilities
│   │       ├── logging.py      # Structlog setup
│   │       └── observability.py # LangFuse integration
│   ├── cli/                    # CLI interface (Typer)
│   │   ├── main.py             # CLI entry point
│   │   ├── commands/           # Command implementations
│   │   │   ├── research.py     # Research command
│   │   │   ├── opportunities.py # Opportunities command
│   │   │   ├── news.py         # News command
│   │   │   └── analyze.py      # Quick analysis command
│   │   └── helpers.py          # Output formatting
│   └── main.py                 # Application entry point
├── tests/                      # Test suite
│   ├── test_config.py          # Configuration tests
│   ├── test_agents.py          # Agent workflow tests
│   └── test_data_sources.py   # Data source tests
├── pyproject.toml              # Project dependencies and config
├── justfile                    # Task runner recipes
├── .env.example                # Example environment variables
└── AGENTS.md                   # Development guide for AI agents
```

**Architecture Principles**:
- **Separation of Concerns**: Core logic in `core/`, interface in `cli/`
- **Interface Agnostic**: Business logic has no CLI dependencies
- **Extensible**: Easy to add MCP server, web API, or other interfaces
- **Type Safe**: Full type hints with basedpyright checking

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Agents** | LangGraph | Multi-step agent workflow orchestration |
| **LLM** | OpenAI/OpenRouter/Ollama | Natural language analysis and reasoning |
| **Data** | Alpha Vantage API | Real-time market data, financials, news |
| **Async** | asyncio + httpx | Concurrent API calls and I/O operations |
| **Config** | Pydantic | Type-safe settings and validation |
| **Logging** | structlog | Structured, contextual logging |
| **Tracing** | LangFuse (optional) | Agent workflow observability |
| **CLI** | Typer | Command-line interface |
| **Package Manager** | uv | Fast dependency resolution and installation |
| **Linting** | ruff | Fast Python linter and formatter |
| **Type Checking** | basedpyright | Static type analysis |
| **Testing** | pytest | Unit and integration tests |

## How It Works: Example Flow

### Research Agent Flow

```bash
$ just research AAPL
```

1. **CLI Command**: Parses ticker symbol and invokes research agent
2. **Agent Initialization**: Creates `ResearchAgent` with default config
3. **Workflow Execution**:
   ```
   Initial State: {"ticker": "AAPL", "current_step": "initializing"}
   
   → fetch_data node:
     - Calls Alpha Vantage API for company overview
     - Fetches recent news with sentiment scores
     - Updates state.context with all data
   
   → analyze node:
     - Builds analysis prompt with financial metrics
     - Invokes LLM (GPT-4/OpenRouter) with system prompt
     - Generates comprehensive investment report
     - Updates state.result with analysis
   
   → final node:
     - Returns complete state
   ```
4. **Output Formatting**: CLI formats markdown report with rich console output
5. **Logging**: Structured logs capture duration, errors, and key metrics

### Opportunities Agent Flow

```bash
$ just opportunities --limit 50
```

1. **Fetch Market Movers**: Alpha Vantage TOP_GAINERS_LOSERS endpoint
   - Returns top 50 losers with price changes and volumes
2. **Enrich Top 5**: Fetches detailed company overview for top 5 losers
   - Gets fundamentals: P/E, market cap, margins, revenue growth
3. **LLM Analysis**: 
   - Prompt includes both loser list and detailed fundamentals
   - LLM identifies oversold stocks with strong fundamentals
   - Generates opportunity report with confidence scores
4. **Structured Output**: Formatted report with tickers, metrics, and recommendations

## Development Guidelines

This project follows strict Python best practices:

- **Type Hints**: All functions have complete type annotations
- **Async/Await**: All I/O operations are async for performance
- **Error Handling**: Structured exception handling with detailed logging
- **Immutability**: Pydantic models use `frozen=True` where appropriate
- **Testing**: Pytest with async support and mocking
- **Documentation**: Comprehensive docstrings and inline comments

See `AGENTS.md` for detailed development guidelines for AI coding agents.

## Roadmap

- [ ] **RAG Integration**: Semantic search over historical research and documents
- [ ] **Portfolio Tracking**: Monitor positions and generate alerts
- [ ] **Backtesting**: Test investment strategies on historical data
- [ ] **MCP Server**: Model Context Protocol server for Claude integration
- [ ] **Web API**: REST API for programmatic access
- [ ] **Web UI**: Interactive dashboard for research and analysis
- [ ] **Real-time Alerts**: Webhook notifications for opportunities
- [ ] **Multi-timeframe Analysis**: Short, medium, long-term perspectives
- [ ] **Sector Rotation**: Identify sector trends and rotation opportunities
- [ ] **Options Analysis**: Volatility, Greeks, and options strategies

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Run `just check` before committing
4. Submit a pull request with clear description

## License

MIT
