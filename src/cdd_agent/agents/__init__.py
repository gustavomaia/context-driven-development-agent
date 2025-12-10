"""Specialized CDD agents.

Agents:
- SocratesAgent: Refine ticket requirements through dialogue
- PlannerAgent: Generate implementation plans
- ExecutorAgent: Execute autonomous coding
- WriterAgent: File persistence (utility, not conversational)

Factory:
- AgentFactory: Centralized agent creation
"""

from .executor import ExecutorAgent
from .factory import AgentFactory
from .planner import PlannerAgent
from .socrates import SocratesAgent
from .test_agent import TestAgent
from .writer import WriterAgent


__all__ = [
    "AgentFactory",
    "ExecutorAgent",
    "PlannerAgent",
    "SocratesAgent",
    "TestAgent",
    "WriterAgent",
]
