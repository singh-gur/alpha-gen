# syntax=docker/dockerfile:1

# ============================================
# Alpha Gen - Multi-agentic AI Investment Research
# ============================================

# Build stage - creates .venv with uv
FROM ghcr.io/astral-sh/uv:python3.13-slim-bookworm AS builder

# Set working directory
WORKDIR /app

# Copy UV configuration files first for better layer caching
COPY pyproject.toml uv.lock* ./

# Create virtual environment and install dependencies
# Using --no-device to avoid pip cache, --frozen for reproducible builds
RUN uv venv /app/.venv && \
    uv pip install --no-device -r pyproject.toml

# Copy source code
COPY src/ ./src/

# Production stage
FROM ghcr.io/astral-sh/uv:python3.13-slim-bookworm AS production

# Install Playwright system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    libatspi2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source code
COPY --from=builder /app/src/ /app/src/
COPY --from=builder /app/pyproject.toml /app/

# Set PYTHONPATH to use the local .venv
ENV PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Install Playwright browsers
RUN playwright install chromium

# Create non-root user for security
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Change ownership of app directory to non-root user
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Set working directory to home
WORKDIR /home/appuser

# Default environment variables (can be overridden)
ENV OPENAI_API_KEY="" \
    LANGFUSE_PUBLIC_KEY="" \
    LANGFUSE_SECRET_KEY="" \
    LOG_LEVEL=INFO

# Health check - verify Python is working
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    python -c "import sys; sys.exit(0)"

# Expose port (if running as server)
EXPOSE 8080

# Default command - show help
CMD ["python", "-m", "alpha_gen.cli.main", "--help"]

# Labels for container metadata
LABEL org.opencontainers.image.title="Alpha Gen" \
      org.opencontainers.image.description="Multi-agentic AI investment research assistant" \
      org.opencontainers.image.version="0.1.0" \
      maintainer="team@alphagen.dev"
