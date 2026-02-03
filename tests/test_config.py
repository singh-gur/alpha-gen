"""Tests for configuration module."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from alpha_gen.core.config.settings import (
    AlphaVantageConfig,
    AppConfig,
    LLMConfig,
    OutputConfig,
    TechnicalIndicatorConfig,
    TechnicalIndicatorsConfig,
    get_config,
)


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = LLMConfig()

        # provider is now a property inferred from base_url
        assert config.provider == "openrouter"  # default when base_url is None
        assert config.model_name == "openrouter/default"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.api_key is None
        assert config.base_url is None

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = LLMConfig(
            model_name="llama3",
            temperature=0.5,
            max_tokens=8192,
            api_key="test-key",
            base_url="http://localhost:11434/v1",
        )

        assert config.provider == "ollama"  # inferred from localhost base_url
        assert config.model_name == "llama3"
        assert config.temperature == 0.5
        assert config.max_tokens == 8192
        assert config.api_key == "test-key"

    def test_provider_inference(self) -> None:
        """Test provider inference from base_url."""
        # No base_url defaults to openrouter
        config = LLMConfig()
        assert config.provider == "openrouter"

        # localhost implies ollama
        config = LLMConfig(base_url="http://localhost:11434/v1")
        assert config.provider == "ollama"

        config = LLMConfig(base_url="http://127.0.0.1:11434/v1")
        assert config.provider == "ollama"

        # openrouter in URL implies openrouter
        config = LLMConfig(base_url="https://openrouter.ai/api/v1")
        assert config.provider == "openrouter"

        # unknown URL falls back to openai-compatible
        config = LLMConfig(base_url="https://api.example.com/v1")
        assert config.provider == "openai-compatible"

    def test_temperature_validation(self) -> None:
        """Test temperature validation."""
        with pytest.raises(ValueError):
            LLMConfig(temperature=-0.1)

        with pytest.raises(ValueError):
            LLMConfig(temperature=2.1)


class TestAppConfig:
    """Tests for AppConfig."""

    def test_default_values(self) -> None:
        """Test default app configuration."""
        config = AppConfig()

        assert config.app_name == "Alpha Gen"
        assert config.app_version == "0.1.0"
        assert config.debug is False
        assert config.log_level == "INFO"

    def test_from_env(self) -> None:
        """Test loading configuration from environment."""
        os.environ["LLM_MODEL"] = "gpt-4o-mini"
        os.environ["DEBUG"] = "true"
        os.environ["LOG_LEVEL"] = "DEBUG"

        try:
            config = AppConfig.from_env()

            assert config.llm.model_name == "gpt-4o-mini"
            assert config.debug is True
            assert config.log_level == "DEBUG"
        finally:
            del os.environ["LLM_MODEL"]
            del os.environ["DEBUG"]
            del os.environ["LOG_LEVEL"]

    def test_from_file_nonexistent(self) -> None:
        """Test loading from non-existent file returns defaults."""
        config = AppConfig.from_file("/nonexistent/path/config.yaml")

        assert config.app_name == "Alpha Gen"

    def test_frozen_config(self) -> None:
        """Test that configuration is immutable."""
        config = AppConfig()

        with pytest.raises(Exception):
            config.app_name = "Modified"


class TestOutputConfig:
    """Tests for OutputConfig."""

    def test_default_values(self) -> None:
        """Test default output configuration."""
        config = OutputConfig()

        assert config.output_dir == Path(".out")

    def test_custom_output_dir(self) -> None:
        """Test custom output directory."""
        config = OutputConfig(output_dir=Path("/tmp/reports"))

        assert config.output_dir == Path("/tmp/reports")

    def test_ensure_output_dir(self, tmp_path: Path) -> None:
        """Test that ensure_output_dir creates the directory."""
        output_dir = tmp_path / "test_output"
        config = OutputConfig(output_dir=output_dir)

        assert not output_dir.exists()
        config.ensure_output_dir()
        assert output_dir.exists()
        assert output_dir.is_dir()


class TestTechnicalIndicatorConfig:
    """Tests for TechnicalIndicatorConfig."""

    def test_default_values(self) -> None:
        """Test default technical indicator configuration."""
        config = TechnicalIndicatorConfig()

        assert config.enabled is True
        assert config.time_period == 14
        assert config.series_type == "close"
        assert config.interval == "daily"

    def test_custom_values(self) -> None:
        """Test custom technical indicator configuration."""
        config = TechnicalIndicatorConfig(
            enabled=False,
            time_period=20,
            series_type="open",
            interval="weekly",
        )

        assert config.enabled is False
        assert config.time_period == 20
        assert config.series_type == "open"
        assert config.interval == "weekly"

    def test_time_period_validation(self) -> None:
        """Test time_period validation."""
        with pytest.raises(ValueError):
            TechnicalIndicatorConfig(time_period=0)

        with pytest.raises(ValueError):
            TechnicalIndicatorConfig(time_period=-1)


class TestTechnicalIndicatorsConfig:
    """Tests for TechnicalIndicatorsConfig."""

    def test_default_values(self) -> None:
        """Test default technical indicators configuration."""
        config = TechnicalIndicatorsConfig()

        assert config.sma is None
        assert config.ema is None
        assert config.rsi is None
        assert config.macd is None
        assert config.stoch is None
        assert config.bbands is None
        assert config.atr is None
        assert config.adx is None
        assert config.aroon is None
        assert config.cci is None
        assert config.obv is None
        assert config.ad is None

    def test_enabled_indicators(self) -> None:
        """Test enabled_indicators property."""
        config = TechnicalIndicatorsConfig(
            sma=TechnicalIndicatorConfig(enabled=True, time_period=20),
            ema=TechnicalIndicatorConfig(enabled=True, time_period=12),
            rsi=TechnicalIndicatorConfig(enabled=False, time_period=14),
        )

        enabled = config.enabled_indicators
        assert len(enabled) == 2
        assert "SMA" in enabled
        assert "EMA" in enabled
        assert "RSI" not in enabled
        assert enabled["SMA"].time_period == 20
        assert enabled["EMA"].time_period == 12

    def test_no_enabled_indicators(self) -> None:
        """Test enabled_indicators when all are disabled or None."""
        config = TechnicalIndicatorsConfig()
        assert len(config.enabled_indicators) == 0

        config = TechnicalIndicatorsConfig(
            sma=TechnicalIndicatorConfig(enabled=False),
            ema=TechnicalIndicatorConfig(enabled=False),
        )
        assert len(config.enabled_indicators) == 0


class TestAlphaVantageConfig:
    """Tests for AlphaVantageConfig."""

    def test_default_values(self) -> None:
        """Test default Alpha Vantage configuration."""
        config = AlphaVantageConfig()

        assert config.api_key is None
        assert config.timeout_seconds == 30
        assert config.rate_limit_interval == 1.2
        assert config.base_url is None
        assert isinstance(config.technical_indicators, TechnicalIndicatorsConfig)

    def test_is_configured(self) -> None:
        """Test is_configured property."""
        config = AlphaVantageConfig()
        assert config.is_configured is False

        config = AlphaVantageConfig(api_key="test-key")
        assert config.is_configured is True

    def test_with_technical_indicators(self) -> None:
        """Test Alpha Vantage config with technical indicators."""
        indicators_config = TechnicalIndicatorsConfig(
            sma=TechnicalIndicatorConfig(enabled=True, time_period=20),
            rsi=TechnicalIndicatorConfig(enabled=True, time_period=14),
        )

        config = AlphaVantageConfig(
            api_key="test-key",
            technical_indicators=indicators_config,
        )

        assert config.is_configured is True
        assert len(config.technical_indicators.enabled_indicators) == 2
        assert "SMA" in config.technical_indicators.enabled_indicators
        assert "RSI" in config.technical_indicators.enabled_indicators
