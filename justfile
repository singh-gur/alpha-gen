# Alpha Gen - Justfile
# Run commands with: just <recipe>

set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
set dotenv-load := true

# Docker image configuration
IMAGE := env("DOCKER_IMAGE", "alpha-gen")
REGISTRY := env("DOCKER_REGISTRY", "")
TAG := `git describe --tags --exact-match 2>/dev/null || git rev-parse --short HEAD`
BRANCH := `git rev-parse --abbrev-ref HEAD | tr '/' '-'`

# Show all available recipes
[group('help')]
default:
    @just --list

# Show project info
[group('help')]
info:
    @echo "Project: Alpha Gen"
    @uv run python --version | sed 's/^/Python: /'
    @uv --version | sed 's/^/UV: /'
    @uv run ruff --version | sed 's/^/Ruff: /'
    @echo "Docker Image: {{ IMAGE }}:{{ TAG }}"

# ============================================================================
# Setup & Dependencies
# ============================================================================

# Sync dependencies from lockfile
[group('setup')]
sync:
    uv sync

# Sync with dev dependencies
[group('setup')]
sync-dev:
    uv sync --extra dev

# Update dependencies
[group('setup')]
update:
    uv sync --upgrade

# Show dependency tree
[group('setup')]
deps-tree:
    uv tree

# Show outdated dependencies
[group('setup')]
outdated:
    uv pip list --outdated

# Remove lock file and resync
[group('setup')]
re-lock:
    rm -f uv.lock
    uv sync

# ============================================================================
# Code Quality
# ============================================================================

# Format code with ruff
[group('quality')]
format:
    uv run ruff format src/

# Check code with ruff (linting)
[group('quality')]
check:
    uv run ruff check src/

# Fix linting issues automatically
[group('quality')]
fix:
    uv run ruff check src/ --fix

# Run type checking with basedpyright
[group('quality')]
type-check:
    uv run basedpyright src/

# Run all quality checks (format + check + type-check)
[group('quality')]
lint: format check type-check
    @echo "✅ All quality checks passed!"

# Audit dependencies for security vulnerabilities
[group('quality')]
audit:
    uv pip install pip-audit
    uv run pip-audit

# Install and run pre-commit hooks
[group('quality')]
precommit:
    uv run pre-commit run --all-files

# Install pre-commit hooks
[group('quality')]
precommit-install:
    uv run pre-commit install

# ============================================================================
# Testing
# ============================================================================

# Run all tests
[group('test')]
test:
    PYTHONPATH=src uv run pytest tests/ -v

# Run tests with coverage
[group('test')]
test-cov:
    PYTHONPATH=src uv run pytest tests/ --cov=alpha_gen --cov-report=term-missing --cov-report=html

# Run a single test file
[group('test')]
test-file FILE:
    PYTHONPATH=src uv run pytest {{ FILE }} -v

# Run tests matching a pattern
[group('test')]
test-name PATTERN:
    PYTHONPATH=src uv run pytest -k "{{ PATTERN }}" -v

# Create coverage report
[group('test')]
coverage-report: test-cov
    @echo "Coverage report generated in htmlcov/index.html"

# ============================================================================
# Application Commands
# ============================================================================

# Show CLI help
[group('app')]
dev:
    uv run python -m alpha_gen.cli.main --help

# Run research command for a ticker
[group('app')]
research ticker:
    uv run python -m alpha_gen.cli.main research {{ ticker }}

# Gather financial data for tickers
[group('app')]
gather tickers:
    uv run python -m alpha_gen.cli.main gather {{ tickers }}


# Find investment opportunities
[group('app')]
opps limit="25":
    uv run python -m alpha_gen.cli.main opps --limit {{ limit }}

# Analyze recent news
[group('app')]
news:
    uv run python -m alpha_gen.cli.main news

# Quick analysis of a ticker
[group('app')]
analyze ticker:
    uv run python -m alpha_gen.cli.main analyze {{ ticker }}

# Run with custom log level
[group('app')]
run-with-log-level log_level="DEBUG" command="--help":
    LOG_LEVEL={{ log_level }} uv run python -m alpha_gen.cli.main {{ command }}

# Check if API keys are configured
[group('app')]
check-api-key:
    @if [ -z "$${LLM_API_KEY:-}" ] && [ -z "$${OPENAI_API_KEY:-}" ]; then \
        echo "⚠️  LLM_API_KEY or OPENAI_API_KEY not set"; \
    else \
        echo "✅ LLM API key is configured"; \
    fi
    @if [ -z "$${ALPHA_VANTAGE_API_KEY:-}" ]; then \
        echo "⚠️  ALPHA_VANTAGE_API_KEY not set"; \
    else \
        echo "✅ Alpha Vantage API key is configured"; \
    fi

# ============================================================================
# Docker
# ============================================================================

# Build docker image
[group('docker')]
build-img:
    docker build --load -t {{ IMAGE }}:{{ TAG }} -t {{ IMAGE }}:{{ BRANCH }} -t {{ IMAGE }}:latest .

# Push docker image to registry
[group('docker')]
push-img: build-img
    @if [ -n "{{ REGISTRY }}" ]; then \
        docker tag {{ IMAGE }}:{{ TAG }} {{ REGISTRY }}/{{ IMAGE }}:{{ TAG }}; \
        docker tag {{ IMAGE }}:{{ BRANCH }} {{ REGISTRY }}/{{ IMAGE }}:{{ BRANCH }}; \
        docker tag {{ IMAGE }}:latest {{ REGISTRY }}/{{ IMAGE }}:latest; \
        docker push {{ REGISTRY }}/{{ IMAGE }}:{{ TAG }}; \
        docker push {{ REGISTRY }}/{{ IMAGE }}:{{ BRANCH }}; \
        docker push {{ REGISTRY }}/{{ IMAGE }}:latest; \
    else \
        docker push {{ IMAGE }}:{{ TAG }}; \
        docker push {{ IMAGE }}:{{ BRANCH }}; \
        docker push {{ IMAGE }}:latest; \
    fi

# Build and run locally with Docker
[group('docker')]
run-local: build-img
    docker run --rm {{ IMAGE }}:latest --help

# Run Docker with environment file
[group('docker')]
run-env env_file:
    docker run --rm --env-file {{ env_file }} {{ IMAGE }}:latest

# ============================================================================
# Concourse CI/CD
# ============================================================================

# Set up credentials file from example
[group('ci')]
ci-setup:
    @if [ -f ci/credentials.yml ]; then \
        echo "⚠️  ci/credentials.yml already exists"; \
    else \
        cp ci/credentials.yml.example ci/credentials.yml; \
        echo "✅ Created ci/credentials.yml from example"; \
        echo "📝 Edit ci/credentials.yml with your actual credentials"; \
    fi

# Generate Cosign keys for image signing
[group('ci')]
ci-generate-keys:
    ./ci/generate-cosign-keys.sh

# Set Concourse pipeline
[group('ci')]
ci-set TARGET:
    @if [ ! -f ci/credentials.yml ]; then \
        echo "❌ ci/credentials.yml not found. Run 'just ci-setup' first."; \
        exit 1; \
    fi
    fly -t {{ TARGET }} set-pipeline \
        -p alpha-gen \
        -c ci/pipeline.yml \
        -l ci/credentials.yml
    @echo ""
    @echo "Pipeline set! Next steps:"
    @echo "  just ci-unpause {{ TARGET }}"
    @echo "  just ci-expose {{ TARGET }}"

# Unpause pipeline
[group('ci')]
ci-unpause TARGET:
    fly -t {{ TARGET }} unpause-pipeline -p alpha-gen

# Pause pipeline
[group('ci')]
ci-pause TARGET:
    fly -t {{ TARGET }} pause-pipeline -p alpha-gen

# Expose pipeline (make public)
[group('ci')]
ci-expose TARGET:
    fly -t {{ TARGET }} expose-pipeline -p alpha-gen

# Hide pipeline (make private)
[group('ci')]
ci-hide TARGET:
    fly -t {{ TARGET }} hide-pipeline -p alpha-gen

# Trigger test job
[group('ci')]
ci-test TARGET:
    fly -t {{ TARGET }} trigger-job -j alpha-gen/test

# Trigger build job
[group('ci')]
ci-build TARGET:
    fly -t {{ TARGET }} trigger-job -j alpha-gen/build-and-push

# Trigger release job
[group('ci')]
ci-release TARGET:
    fly -t {{ TARGET }} trigger-job -j alpha-gen/release

# Watch test job
[group('ci')]
ci-watch-test TARGET:
    fly -t {{ TARGET }} watch -j alpha-gen/test

# Watch build job
[group('ci')]
ci-watch-build TARGET:
    fly -t {{ TARGET }} watch -j alpha-gen/build-and-push

# View pipeline status
[group('ci')]
ci-status TARGET:
    fly -t {{ TARGET }} pipelines | grep alpha-gen

# View recent builds
[group('ci')]
ci-builds TARGET:
    fly -t {{ TARGET }} builds | grep alpha-gen | head -10

# Check pipeline resources
[group('ci')]
ci-check TARGET:
    fly -t {{ TARGET }} check-resource -r alpha-gen/repo

# Validate pipeline config
[group('ci')]
ci-validate:
    @echo "Validating pipeline configuration..."
    @fly validate-pipeline -c ci/pipeline.yml && echo "✅ Pipeline is valid"

# Destroy pipeline
[group('ci')]
ci-destroy TARGET:
    fly -t {{ TARGET }} destroy-pipeline -p alpha-gen

# ============================================================================
# Cleanup
# ============================================================================

# Remove Python cache files
[group('cleanup')]
clean-cache:
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.pyo" -delete 2>/dev/null || true
    rm -rf .pytest_cache .hypothesis .ruff_cache htmlcov .coverage coverage.xml

# Remove all generated files
[group('cleanup')]
clean: clean-cache
    rm -rf .venv data/vector_store/ 2>/dev/null || true
    rm -rf dist/ build/ *.egg-info/ 2>/dev/null || true
    @echo "✅ Cleanup complete!"

# Full cleanup (cache + venv)
[group('cleanup')]
distclean: clean
    rm -f uv.lock
    @echo "✅ Full cleanup complete!"

# ============================================================================
# Utilities
# ============================================================================

# Show project structure
[group('utils')]
tree:
    @find src/ -type f -name "*.py" | sort | head -30

# Activate virtual environment (print command to eval)
[group('utils')]
activate:
    @echo "source .venv/bin/activate"

# Shell into the virtual environment
[group('utils')]
shell:
    uv run --no-cmd python

# Install Playwright browsers
[group('utils')]
playwright-install:
    uv run playwright install chromium

# Run entire CI pipeline locally
[group('utils')]
ci-local: lint test
    @echo "✅ Local CI pipeline completed successfully!"
