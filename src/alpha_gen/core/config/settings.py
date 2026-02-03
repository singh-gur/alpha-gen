"""
Configuration management for Alpha Gen application.
Loads configuration from environment variables and config files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator

# Load environment variables from .env file
load_dotenv()


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

    provider: Literal["chroma", "pgvector"] = "chroma"  # type: ignore[assignment]

    # Chroma settings
    persist_directory: Path = Path("./data/vector_store")
    collection_name: str = "alpha_gen_docs"

    # pgvector settings
    postgres_url: str | None = None  # postgresql://user:pass@host:port/dbname

    # Common settings
    embedding_model: str = "all-MiniLM-L6-v2"

    model_config = {"frozen": True}

    @validator("provider")
    def validate_provider(cls, v: str) -> str:  # type: ignore[no-untyped-def]
        """Validate vector store provider."""
        valid_providers = ["chroma", "pgvector"]
        if v not in valid_providers:
            raise ValueError(f"Invalid provider: {v}. Must be one of {valid_providers}")
        return v

    @validator("postgres_url")
    def validate_postgres_url(cls, v: str | None, values: dict[str, Any]) -> str | None:  # type: ignore[no-untyped-def]
        """Validate postgres_url is provided when using pgvector."""
        provider = values.get("provider")
        if provider == "pgvector" and not v:
            raise ValueError("postgres_url is required when provider is 'pgvector'")
        return v

    def __post_init__(self) -> None:
        if self.provider == "chroma":
            self.persist_directory.mkdir(parents=True, exist_ok=True)


class TechnicalIndicatorConfig(BaseModel):
    """Configuration for a single technical indicator."""

    enabled: bool = True
    time_period: int = Field(default=14, gt=0)
    series_type: Literal["close", "open", "high", "low"] = "close"  # type: ignore[assignment]
    interval: Literal["daily", "weekly", "monthly"] = "daily"  # type: ignore[assignment]

    model_config = {"frozen": True}


class TechnicalIndicatorsConfig(BaseModel):
    """Configuration for technical indicators to fetch from Alpha Vantage."""

    # Moving Averages
    sma: TechnicalIndicatorConfig | None = None  # Simple Moving Average
    ema: TechnicalIndicatorConfig | None = None  # Exponential Moving Average

    # Momentum Indicators
    rsi: TechnicalIndicatorConfig | None = None  # Relative Strength Index
    macd: TechnicalIndicatorConfig | None = (
        None  # Moving Average Convergence Divergence
    )
    stoch: TechnicalIndicatorConfig | None = None  # Stochastic Oscillator

    # Volatility Indicators
    bbands: TechnicalIndicatorConfig | None = None  # Bollinger Bands
    atr: TechnicalIndicatorConfig | None = None  # Average True Range

    # Trend Indicators
    adx: TechnicalIndicatorConfig | None = None  # Average Directional Index
    aroon: TechnicalIndicatorConfig | None = None  # Aroon Indicator
    cci: TechnicalIndicatorConfig | None = None  # Commodity Channel Index

    # Volume Indicators
    obv: TechnicalIndicatorConfig | None = None  # On Balance Volume
    ad: TechnicalIndicatorConfig | None = None  # Accumulation/Distribution

    model_config = {"frozen": True}

    @property
    def enabled_indicators(self) -> dict[str, TechnicalIndicatorConfig]:
        """Get all enabled indicators."""
        indicators = {}
        for field_name in self.__class__.model_fields:
            indicator = getattr(self, field_name)
            if indicator is not None and indicator.enabled:
                indicators[field_name.upper()] = indicator
        return indicators


class AlphaVantageConfig(BaseModel):
    """Configuration for Alpha Vantage API."""

    api_key: str | None = None
    timeout_seconds: int = 30
    rate_limit_interval: float = 1.2  # Seconds between requests (free tier: 1 req/sec)
    base_url: str | None = None  # Custom base URL (defaults to Alpha Vantage)
    technical_indicators: TechnicalIndicatorsConfig = Field(
        default_factory=TechnicalIndicatorsConfig
    )

    model_config = {"frozen": True}

    @property
    def is_configured(self) -> bool:
        """Check if Alpha Vantage is properly configured."""
        return bool(self.api_key)


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


class OutputConfig(BaseModel):
    """Configuration for output file generation."""

    output_dir: Path = Path(".out")

    model_config = {"frozen": True}

    def ensure_output_dir(self) -> None:
        """Ensure output directory exists."""
        self.output_dir.mkdir(parents=True, exist_ok=True)


class AppConfig(BaseModel):
    """Main application configuration."""

    app_name: str = "Alpha Gen"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"  # type: ignore[assignment]

    llm: LLMConfig = Field(default_factory=LLMConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    alpha_vantage: AlphaVantageConfig = Field(default_factory=AlphaVantageConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

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
            postgres_url=os.getenv("POSTGRES_URL"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            or "all-MiniLM-L6-v2",
        )

        # Load technical indicators config from environment
        def _load_indicator_config(
            indicator_name: str,
        ) -> TechnicalIndicatorConfig | None:
            """Load indicator config from environment variables."""
            enabled_key = f"ALPHA_VANTAGE_INDICATOR_{indicator_name.upper()}_ENABLED"
            enabled = os.getenv(enabled_key, "false").lower() == "true"

            if not enabled:
                return None

            time_period = int(
                os.getenv(
                    f"ALPHA_VANTAGE_INDICATOR_{indicator_name.upper()}_TIME_PERIOD",
                    "14",
                )
            )
            series_type = os.getenv(
                f"ALPHA_VANTAGE_INDICATOR_{indicator_name.upper()}_SERIES_TYPE", "close"
            )
            interval = os.getenv(
                f"ALPHA_VANTAGE_INDICATOR_{indicator_name.upper()}_INTERVAL", "daily"
            )

            return TechnicalIndicatorConfig(
                enabled=True,
                time_period=time_period,
                series_type=series_type,  # type: ignore[arg-type]
                interval=interval,  # type: ignore[arg-type]
            )

        technical_indicators_config = TechnicalIndicatorsConfig(
            sma=_load_indicator_config("sma"),
            ema=_load_indicator_config("ema"),
            rsi=_load_indicator_config("rsi"),
            macd=_load_indicator_config("macd"),
            stoch=_load_indicator_config("stoch"),
            bbands=_load_indicator_config("bbands"),
            atr=_load_indicator_config("atr"),
            adx=_load_indicator_config("adx"),
            aroon=_load_indicator_config("aroon"),
            cci=_load_indicator_config("cci"),
            obv=_load_indicator_config("obv"),
            ad=_load_indicator_config("ad"),
        )

        alpha_vantage_config = AlphaVantageConfig(
            api_key=os.getenv("ALPHA_VANTAGE_API_KEY"),
            timeout_seconds=int(os.getenv("ALPHA_VANTAGE_TIMEOUT", "30")),
            rate_limit_interval=float(
                os.getenv("ALPHA_VANTAGE_RATE_LIMIT_INTERVAL", "1.2")
            ),
            base_url=os.getenv("ALPHA_VANTAGE_BASE_URL") or None,
            technical_indicators=technical_indicators_config,
        )

        observability_config = ObservabilityConfig(
            enabled=os.getenv("LANGFUSE_ENABLED", "True").lower() == "true",
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )

        output_config = OutputConfig(
            output_dir=Path(os.getenv("OUTPUT_DIR", ".out")),
        )

        return cls(
            debug=os.getenv("DEBUG", "False").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO") or "INFO",  # type: ignore[arg-type]
            llm=llm_config,
            vector_store=vector_store_config,
            alpha_vantage=alpha_vantage_config,
            observability=observability_config,
            output=output_config,
        )


def get_config(config_path: Path | str | None = None) -> AppConfig:
    """Get application configuration, loading from file first, then environment."""
    if config_path and Path(config_path).exists():
        return AppConfig.from_file(config_path)
    return AppConfig.from_env()
