"""Tests for configuration module."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from alpha_gen.config.settings import AppConfig, LLMConfig, get_config


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = LLMConfig()

        assert config.provider == "openai"
        assert config.model_name == "gpt-4o"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.api_key is None
        assert config.base_url is None

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = LLMConfig(
            provider="anthropic",
            model_name="claude-3-opus-20240229",
            temperature=0.5,
            max_tokens=8192,
            api_key="test-key",
        )

        assert config.provider == "anthropic"
        assert config.model_name == "claude-3-opus-20240229"
        assert config.temperature == 0.5
        assert config.max_tokens == 8192
        assert config.api_key == "test-key"

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
