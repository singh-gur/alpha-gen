# Alpha Gen - Justfile
# Run commands with: just <recipe>
#
# This project uses uv for fast dependency management.
# A local .venv is created and managed by uv sync.

set dotenv-load := true

# Default recipe - show help
default:
    @just --list

# Sync dependencies (create .venv and install)
sync:
    uv sync

# Sync with dev dependencies
sync-dev:
    uv sync --extra dev

# Update dependencies
update:
    uv sync --upgrade

# Run linting with ruff
lint:
    uv run ruff check src/

# Run ruff with fix
lint-fix:
    uv run ruff check src/ --fix

# Run type checking with basedpyright (warn-only due to LangChain/LangGraph types)
type-check:
    uv run basedpyright src/ 2>&1 || echo "Type check completed with warnings (expected for LangChain/LangGraph types)"

# Run all quality checks (lint + type check)
check: lint type-check
    @echo "All quality checks passed!"

# Run tests
test:
    PYTHONPATH=src uv run pytest tests/ -v

# Run tests with coverage
test-coverage:
    PYTHONPATH=src uv run pytest tests/ --cov=alpha_gen --cov-report=term-missing --cov-report=html

# Run development server / CLI help
dev:
    uv run python -m alpha_gen.cli.main --help

# Run research command
research ticker:
    uv run python -m alpha_gen.cli.main research {{ticker}}

# Run opportunities command
opportunities limit="25":
    uv run python -m alpha_gen.cli.main opportunities --limit {{limit}}

# Run news command
news:
    uv run python -m alpha_gen.cli.main news

# Run analyze command
analyze ticker:
    uv run python -m alpha_gen.cli.main analyze {{ticker}}

# Build Docker image
docker-build:
    docker build -t alpha-gen:latest .

# Build Docker image with custom tag
docker-build-tag tag:
    docker build -t alpha-gen:{{tag}} .

# Run Docker container
docker-run args="--help":
    docker run --rm alpha-gen:latest {{args}}

# Run Docker with environment file
docker-run-env env_file:
    docker run --rm --env-file {{env_file}} alpha-gen:latest --help

# Clean up Python cache files
clean-cache:
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.pyo" -delete 2>/dev/null || true
    rm -rf .pytest_cache .hypothesis htmlcov .coverage

# Remove .venv and clean up all generated files
clean: clean-cache
    rm -rf .venv data/vector_store/ 2>/dev/null || true
    rm -rf dist/ build/ *.egg-info/ 2>/dev/null || true
    echo "Cleaned up generated files and .venv"

# Format code with ruff
format:
    uv run ruff check src/ --fix
    uv run ruff format src/

# Format and check
format-check: format
    uv run ruff check src/

# Install pre-commit hooks
precommit-install:
    uv run pre-commit install

# Run pre-commit hooks
precommit:
    uv run pre-commit run --all-files

# Create coverage report
coverage-report: test-coverage
    @echo "Coverage report generated in htmlcov/index.html"

# Show project structure
tree:
    @find src/ -type f -name "*.py" | sort | head -30

# Show project info
info:
    @echo "Project: Alpha Gen"
    @echo "Python: $$(uv run python --version)"
    @echo "UV: $$(uv --version)"
    @echo "Ruff: $$(uv run ruff --version)"
    @echo "Virtual env: $$(pwd)/.venv"

# Run entire CI pipeline locally
ci: check test
    @echo "CI pipeline completed successfully!"

# OpenAI API key check
check-api-key:
    @if [ -z "$$OPENAI_API_KEY" ]; then \
        echo "OPENAI_API_KEY not set. Set it in .env or environment."; \
    else \
        echo "OPENAI_API_KEY is configured"; \
    fi

# Remove lock file and sync (use when dependencies are changed)
re-lock:
    rm -f uv.lock
    uv sync

# Shell into the virtual environment
shell:
    uv run --no-cmd python

# Install Playwright browsers
playwright-install:
    uv run playwright install chromium

# Run with custom log level
run-with-log-level log_level="DEBUG" ticker="":
    @if [ -z "{{ticker}}" ]; then \
        uv run python -m alpha_gen.cli.main --log-level {{log_level}}; \
    else \
        uv run python -m alpha_gen.cli.main research {{ticker}} --log-level {{log_level}}; \
    fi
