"""Base protocol and types for LLM providers.

This module defines the ProviderStrategy protocol that all LLM providers
must implement, enabling the Strategy pattern for provider selection.
"""

from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import Protocol
from typing import runtime_checkable


@runtime_checkable
class ProviderStrategy(Protocol):
    """Protocol defining the interface for LLM providers.

    All LLM providers (Anthropic, OpenAI, custom) must implement this
    interface to be used with the Agent class.

    The Strategy pattern allows:
    - Clean separation of authentication logic (OAuth vs API key)
    - Easy addition of new providers (OpenAI, local models)
    - Testable provider implementations
    - Agent class remains provider-agnostic
    """

    @property
    def model(self) -> str:
        """Get the model name for this provider.

        Returns:
            Model identifier string (e.g., "claude-sonnet-4-20250514")
        """
        ...

    def create_message(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system: str,
        max_tokens: int = 4096,
    ) -> Any:
        """Create a message (non-streaming).

        Args:
            messages: Conversation history
            tools: Available tools in Anthropic format
            system: System prompt
            max_tokens: Maximum tokens in response

        Returns:
            Provider-specific response object with:
            - stop_reason: "end_turn" | "tool_use" | "max_tokens"
            - content: List of content blocks (text, tool_use)
        """
        ...

    def stream_message(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system: str,
        max_tokens: int = 4096,
    ) -> Generator[Dict[str, Any], None, Any]:
        """Stream a message response.

        Args:
            messages: Conversation history
            tools: Available tools in Anthropic format
            system: System prompt
            max_tokens: Maximum tokens in response

        Yields:
            Event dictionaries with type and data:
            - {"type": "text_delta", "text": str}
            - {"type": "tool_use_start", "id": str, "name": str}
            - {"type": "tool_use_delta", "partial_json": str}
            - {"type": "message_stop", "message": response}

        Returns:
            Final message object after stream completes
        """
        ...
