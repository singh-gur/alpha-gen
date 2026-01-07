"""Base agent module for Alpha Gen."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph
from typing_extensions import TypedDict


class AgentStatus(str, Enum):
    """Agent execution status."""

    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    ERROR = "error"
    COMPLETED = "completed"


class AgentState(TypedDict):
    """Base state for agents."""

    messages: list[BaseMessage]
    current_step: str
    context: dict[str, Any]
    result: str | None
    error_message: str | None


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for an agent."""

    model_name: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 30


class BaseAgent(ABC):
    """Base class for all agents."""

    def __init__(self, name: str, config: AgentConfig | None = None) -> None:
        self.name = name
        self.config = config or AgentConfig()
        self._status = AgentStatus.IDLE

    @property
    def status(self) -> AgentStatus:
        """Get current agent status."""
        return self._status

    def _set_status(self, status: AgentStatus) -> None:
        """Set agent status."""
        self._status = status

    @abstractmethod
    def create_workflow(self) -> StateGraph:
        """Create the agent workflow graph."""
        pass

    @abstractmethod
    async def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Run the agent with input data."""
        pass
