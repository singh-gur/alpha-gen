# syntax=docker/dockerfile:1

# ============================================
# Alpha Gen - Multi-agentic AI Investment Research
# ============================================

FROM python:3.13-slim-bookworm

WORKDIR /app

# Install build dependencies and upgrade pip in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy dependency files and install dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# Copy source code
COPY src/ ./src/

# Set Python environment variables
ENV PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LLM_API_KEY="" \
    ALPHA_VANTAGE_API_KEY="" \
    LANGFUSE_PUBLIC_KEY="" \
    LANGFUSE_SECRET_KEY="" \
    LOG_LEVEL=INFO


# Create non-root user and switch to it
RUN groupadd --gid 1000 appgroup \
    && useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser \
    && chown -R appuser:appgroup /app

USER appuser
WORKDIR /home/appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import alpha_gen; import sys; sys.exit(0)"

# Default command - uses the installed CLI entrypoint
CMD ["alpha-gen", "--help"]

# Metadata
LABEL org.opencontainers.image.title="Alpha Gen" \
    org.opencontainers.image.description="Multi-agentic AI investment research assistant" \
    org.opencontainers.image.version="0.1.0" \
    org.opencontainers.image.source="https://github.com/team/alpha-gen" \
    maintainer="team@alphagen.dev"
