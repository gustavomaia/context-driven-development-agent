"""Agent factory for creating specialized CDD agents.

This module provides the AgentFactory class that centralizes agent creation,
making it easier to add new agent types and maintain consistent initialization.
"""

from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import Type

from ..session.base_agent import BaseAgent
from .executor import ExecutorAgent
from .planner import PlannerAgent
from .socrates import SocratesAgent
from .writer import WriterAgent


if TYPE_CHECKING:
    from ..session.chat_session import ChatSession


# Registry of available agent types
AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {
    "socrates": SocratesAgent,
    "planner": PlannerAgent,
    "executor": ExecutorAgent,
}


class AgentFactory:
    """Factory for creating specialized CDD agents.

    This factory:
    - Centralizes agent creation logic
    - Provides a registry of available agent types
    - Validates prerequisites before agent creation
    - Makes it easy to add new agent types

    Example:
        factory = AgentFactory(session, provider_config, tool_registry)
        agent = factory.create("socrates", target_path)
    """

    def __init__(
        self,
        session: "ChatSession",
        provider_config: Any,
        tool_registry: Any,
    ):
        """Initialize factory.

        Args:
            session: Parent ChatSession instance
            provider_config: LLM provider configuration
            tool_registry: Available tools for agents
        """
        self.session = session
        self.provider_config = provider_config
        self.tool_registry = tool_registry

    def create(self, agent_type: str, target_path: Path) -> BaseAgent:
        """Create an agent by type name.

        Args:
            agent_type: Agent type name (socrates, planner, executor)
            target_path: Path to ticket/doc to work on

        Returns:
            Initialized agent instance

        Raises:
            ValueError: If agent type is unknown
        """
        agent_type = agent_type.lower()

        if agent_type not in AGENT_REGISTRY:
            available = ", ".join(AGENT_REGISTRY.keys())
            raise ValueError(
                f"Unknown agent type: '{agent_type}'. " f"Available types: {available}"
            )

        agent_class = AGENT_REGISTRY[agent_type]

        return agent_class(
            target_path=target_path,
            session=self.session,
            provider_config=self.provider_config,
            tool_registry=self.tool_registry,
        )

    def create_writer(self, target_path: Path) -> WriterAgent:
        """Create a WriterAgent (special case - not a BaseAgent).

        WriterAgent is a utility class for file I/O, not a conversational
        agent, so it has a separate creation method.

        Args:
            target_path: Path where content should be saved

        Returns:
            Initialized WriterAgent instance
        """
        return WriterAgent(target_path=target_path)

    @staticmethod
    def list_agent_types() -> list:
        """List available agent types.

        Returns:
            List of agent type names
        """
        return list(AGENT_REGISTRY.keys())

    @staticmethod
    def get_agent_class(agent_type: str) -> Type[BaseAgent]:
        """Get agent class by type name.

        Args:
            agent_type: Agent type name

        Returns:
            Agent class

        Raises:
            ValueError: If agent type is unknown
        """
        agent_type = agent_type.lower()

        if agent_type not in AGENT_REGISTRY:
            available = ", ".join(AGENT_REGISTRY.keys())
            raise ValueError(
                f"Unknown agent type: '{agent_type}'. " f"Available types: {available}"
            )

        return AGENT_REGISTRY[agent_type]
