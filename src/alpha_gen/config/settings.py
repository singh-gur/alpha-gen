"""
Configuration management for Alpha Gen application.
Loads configuration from environment variables and config files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, validator


class LLMConfig(BaseModel):
    """Configuration for LLM (OpenAI-compatible API)."""

    model_name: str = "openrouter/default"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)
    api_key: str | None = None
    base_url: str | None = None

    model_config = {"frozen": True}

    @property
    def provider(self) -> str:
        """Infer provider from base_url for logging/debugging."""
        if self.base_url is None:
            return "openrouter"
        if "localhost" in self.base_url or "127.0.0.1" in self.base_url:
            return "ollama"
        if "openrouter" in self.base_url:
            return "openrouter"
        return "openai-compatible"


class VectorStoreConfig(BaseModel):
    """Configuration for vector store."""

    provider: Literal["chroma", "pinecone", "weaviate"] = "chroma"  # type: ignore[assignment]
    persist_directory: Path = Path("./data/vector_store")
    collection_name: str = "alpha_gen_docs"
    embedding_model: str = "all-MiniLM-L6-v2"

    model_config = {"frozen": True}

    @validator("provider")
    def validate_provider(cls, v: str) -> str:  # type: ignore[no-untyped-def]
        """Validate vector store provider."""
        valid_providers = ["chroma", "pinecone", "weaviate"]
        if v not in valid_providers:
            raise ValueError(f"Invalid provider: {v}. Must be one of {valid_providers}")
        return v

    def __post_init__(self) -> None:
        self.persist_directory.mkdir(parents=True, exist_ok=True)


class ScrapingConfig(BaseModel):
    """Configuration for web scraping."""

    timeout_seconds: int = 30
    retry_attempts: int = 3
    delay_between_retries: float = 1.0
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    headless: bool = True

    model_config = {"frozen": True}


class ObservabilityConfig(BaseModel):
    """Configuration for observability (LangFuse)."""

    enabled: bool = True
    public_key: str | None = None
    secret_key: str | None = None
    host: str = "https://cloud.langfuse.com"
    flush_interval: float = 1.0

    model_config = {"frozen": True}

    @property
    def is_configured(self) -> bool:
        """Check if LangFuse is properly configured."""
        return bool(self.public_key and self.secret_key)


class AppConfig(BaseModel):
    """Main application configuration."""

    app_name: str = "Alpha Gen"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"  # type: ignore[assignment]

    llm: LLMConfig = Field(default_factory=LLMConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    scraping: ScrapingConfig = Field(default_factory=ScrapingConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    model_config = {"frozen": True}

    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:  # type: ignore[no-untyped-def]
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        if v not in valid_levels:
            raise ValueError(f"Invalid log_level: {v}. Must be one of {valid_levels}")
        return v

    @classmethod
    def from_file(cls, config_path: Path | str | None = None) -> AppConfig:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path("config.yaml")
        elif isinstance(config_path, str):
            config_path = Path(config_path)

        if not config_path.exists():
            return cls()

        try:
            with open(config_path) as f:
                config_data = yaml.safe_load(f) or {}
            return cls(**config_data)
        except (yaml.YAMLError, ValueError) as e:
            raise ValueError(
                f"Failed to load configuration from {config_path}: {e}"
            ) from e

    @classmethod
    def from_env(cls) -> AppConfig:
        """Load configuration from environment variables."""
        import os

        llm_config = LLMConfig(
            model_name=os.getenv("LLM_MODEL", "openrouter/default"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL") or None,
        )

        vector_store_config = VectorStoreConfig(
            provider=os.getenv("VECTOR_STORE_PROVIDER", "chroma"),  # type: ignore[arg-type]
            persist_directory=Path(
                os.getenv("VECTOR_STORE_DIR", "./data/vector_store")
            ),
            collection_name=os.getenv("VECTOR_STORE_COLLECTION", "alpha_gen_docs")
            or "alpha_gen_docs",
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            or "all-MiniLM-L6-v2",
        )

        scraping_config = ScrapingConfig(
            timeout_seconds=int(os.getenv("SCRAPING_TIMEOUT", "30")),
            retry_attempts=int(os.getenv("SCRAPING_RETRIES", "3")),
            delay_between_retries=float(os.getenv("SCRAPING_DELAY", "1.0")),
            headless=os.getenv("SCRAPING_HEADLESS", "True").lower() == "true",
        )

        observability_config = ObservabilityConfig(
            enabled=os.getenv("LANGFUSE_ENABLED", "True").lower() == "true",
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )

        return cls(
            debug=os.getenv("DEBUG", "False").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO") or "INFO",  # type: ignore[arg-type]
            llm=llm_config,
            vector_store=vector_store_config,
            scraping=scraping_config,
            observability=observability_config,
        )


def get_config(config_path: Path | str | None = None) -> AppConfig:
    """Get application configuration, loading from file first, then environment."""
    if config_path and Path(config_path).exists():
        return AppConfig.from_file(config_path)
    return AppConfig.from_env()
