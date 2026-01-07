"""Observability module for Alpha Gen using LangFuse."""

from __future__ import annotations

from typing import Any

from alpha_gen.config.settings import get_config


class ObservabilityManager:
    """Manager for observability features (LangFuse)."""

    def __init__(self) -> None:
        self.config = get_config()
        self._client: Any = None

    @property
    def is_enabled(self) -> bool:
        """Check if observability is enabled."""
        return (
            self.config.observability.enabled
            and self.config.observability.is_configured
        )

    def get_client(self) -> Any | None:
        """Get the LangFuse client."""
        if not self.is_enabled:
            return None

        if self._client is None:
            try:
                from langfuse import Langfuse

                self._client = Langfuse(
                    public_key=self.config.observability.public_key,
                    secret_key=self.config.observability.secret_key,
                    host=self.config.observability.host,
                )
            except Exception:
                return None

        return self._client

    def get_callbacks(self) -> list[Any]:
        """Get list of callbacks for LangChain."""
        client = self.get_client()
        if client is None:
            return []

        try:
            return [client]
        except Exception:
            return []

    def flush(self) -> None:
        """Flush any pending traces."""
        if self._client:
            try:
                self._client.flush()
            except Exception:
                pass

    def create_generation(
        self,
        name: str,
        input_data: Any,
        metadata: dict[str, Any] | None = None,
    ) -> LangfuseSpan | DummySpan:
        """Create a new generation for tracking."""
        if not self.is_enabled:
            return DummySpan()

        try:
            client = self.get_client()
            if client is None:
                return DummySpan()

            generation = client.generation(
                name=name,
                input=input_data,
                metadata=metadata,
            )
            return LangfuseSpan(generation=generation)
        except Exception:
            return DummySpan()


class LangfuseSpan:
    """Wrapper for LangFuse span/generation."""

    def __init__(self, generation: Any) -> None:
        self.generation = generation

    def end(self, output: Any, metadata: dict[str, Any] | None = None) -> None:
        """End the span/generation."""
        try:
            self.generation.end(output=output, metadata=metadata)
        except Exception:
            pass

    def score(self, name: str, value: float) -> None:
        """Add a score to the span/generation."""
        try:
            self.generation.score(name=name, value=value)
        except Exception:
            pass


class DummySpan:
    """Dummy span for when LangFuse is not configured."""

    def end(self, output: Any, metadata: dict[str, Any] | None = None) -> None:
        pass

    def score(self, name: str, value: float) -> None:
        pass


# Global observability manager instance
_observability_manager: ObservabilityManager | None = None


def get_observability_manager() -> ObservabilityManager:
    """Get the global observability manager instance."""
    global _observability_manager
    if _observability_manager is None:
        _observability_manager = ObservabilityManager()
    return _observability_manager


def observe_agent_execution(
    agent_name: str,
    input_data: dict[str, Any],
) -> ObservableContext:
    """Context manager for observing agent execution."""
    manager = get_observability_manager()
    return ObservableContext(manager, agent_name, input_data)


class ObservableContext:
    """Context manager for observability."""

    def __init__(
        self,
        manager: ObservabilityManager,
        agent_name: str,
        input_data: dict[str, Any],
    ) -> None:
        self.manager = manager
        self.agent_name = agent_name
        self.input_data = input_data
        self._span: LangfuseSpan | DummySpan = DummySpan()

    def __enter__(self) -> ObservableContext:
        self._span = self.manager.create_generation(
            name=self.agent_name,
            input_data=self.input_data,
        )
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def end(self, output: Any, metadata: dict[str, Any] | None = None) -> None:
        """End observation with output."""
        self._span.end(output=output, metadata=metadata)

    def score(self, name: str, value: float) -> None:
        """Add a score."""
        self._span.score(name=name, value=value)
