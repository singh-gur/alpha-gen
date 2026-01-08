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
