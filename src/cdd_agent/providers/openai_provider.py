"""OpenAI provider implementation.

This module provides the OpenAIProvider class that implements
the ProviderStrategy protocol using OpenAI's API.
"""

from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional

from ..config import ProviderConfig
from ..logging import get_logger


logger = get_logger(__name__)


class OpenAIProvider:
    """OpenAI API provider using API key authentication.

    This provider implements the ProviderStrategy protocol for OpenAI models
    (GPT-4, GPT-4o, o1, etc.).

    Attributes:
        model: The model name to use for API calls
    """

    def __init__(
        self,
        provider_config: ProviderConfig,
        model_tier: str = "mid",
        max_retries: int = 5,
        timeout: float = 600.0,
    ):
        """Initialize OpenAI provider.

        Args:
            provider_config: Provider configuration with API key and models
            model_tier: Model tier to use (small/mid/big)
            max_retries: Maximum retry attempts for API calls
            timeout: Request timeout in seconds
        """
        self._provider_config = provider_config
        self._model_tier = model_tier
        self._max_retries = max_retries
        self._timeout = timeout
        self._client: Optional[Any] = None

    @property
    def model(self) -> str:
        """Get the model name for this provider."""
        return self._provider_config.get_model(self._model_tier)

    @property
    def client(self) -> Any:
        """Get or create the OpenAI client (lazy initialization).

        Returns:
            OpenAI client instance
        """
        if self._client is None:
            # Lazy import to avoid loading openai unless needed
            import openai

            self._client = openai.OpenAI(
                api_key=self._provider_config.get_api_key(),
                base_url=self._provider_config.base_url,
                max_retries=self._max_retries,
                timeout=self._timeout,
            )
            logger.debug(f"OpenAI client initialized with model: {self.model}")

        return self._client

    def create_message(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system: str,
        max_tokens: int = 4096,
    ) -> Any:
        """Create a message using OpenAI API.

        Converts Anthropic-style messages/tools to OpenAI format and
        returns a response object compatible with the Agent's expectations.

        Args:
            messages: Conversation messages (Anthropic format)
            tools: Tool definitions (Anthropic format)
            system: System prompt
            max_tokens: Maximum tokens in response

        Returns:
            Response object with Anthropic-compatible interface
        """
        # Convert messages to OpenAI format
        openai_messages = self._convert_messages(messages, system)

        # Convert tools to OpenAI format
        openai_tools = self._convert_tools(tools) if tools else None

        logger.debug(
            f"OpenAI API call: model={self.model} "
            f"messages={len(openai_messages)} tools={len(tools) if tools else 0}"
        )

        # Make API call
        kwargs = {
            "model": self.model,
            "messages": openai_messages,
            "max_tokens": max_tokens,
        }
        if openai_tools:
            kwargs["tools"] = openai_tools

        response = self.client.chat.completions.create(**kwargs)

        # Convert response to Anthropic-compatible format
        return self._convert_response(response)

    def stream_message(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system: str,
        max_tokens: int = 4096,
    ) -> Generator[Any, None, None]:
        """Stream a message using OpenAI API.

        Args:
            messages: Conversation messages (Anthropic format)
            tools: Tool definitions (Anthropic format)
            system: System prompt
            max_tokens: Maximum tokens in response

        Yields:
            Response chunks
        """
        # Convert to OpenAI format
        openai_messages = self._convert_messages(messages, system)
        openai_tools = self._convert_tools(tools) if tools else None

        kwargs = {
            "model": self.model,
            "messages": openai_messages,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if openai_tools:
            kwargs["tools"] = openai_tools

        stream = self.client.chat.completions.create(**kwargs)

        for chunk in stream:
            yield chunk

    def _convert_messages(
        self, messages: List[Dict[str, Any]], system: str
    ) -> List[Dict[str, Any]]:
        """Convert Anthropic-style messages to OpenAI format.

        Args:
            messages: Anthropic-format messages
            system: System prompt

        Returns:
            OpenAI-format messages
        """
        openai_messages: List[Dict[str, Any]] = []

        # Add system message first
        if system:
            openai_messages.append({"role": "system", "content": system})

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "user":
                # Handle tool results (user messages with tool_result content)
                if isinstance(content, list):
                    # Check if this is a tool result
                    tool_results = [
                        item for item in content if item.get("type") == "tool_result"
                    ]
                    if tool_results:
                        # Convert to OpenAI tool message format
                        for result in tool_results:
                            openai_messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": result.get("tool_use_id"),
                                    "content": result.get("content", ""),
                                }
                            )
                    else:
                        # Regular content list
                        text_content = " ".join(
                            item.get("text", "")
                            for item in content
                            if item.get("type") == "text"
                        )
                        openai_messages.append(
                            {
                                "role": "user",
                                "content": text_content,
                            }
                        )
                else:
                    openai_messages.append({"role": "user", "content": content})

            elif role == "assistant":
                # Handle assistant messages with tool use
                if isinstance(content, list):
                    # Check for tool use blocks
                    tool_calls = []
                    text_content = ""

                    for block in content:
                        if hasattr(block, "type"):
                            # This is an Anthropic content block object
                            if block.type == "tool_use":
                                import json

                                tool_calls.append(
                                    {
                                        "id": block.id,
                                        "type": "function",
                                        "function": {
                                            "name": block.name,
                                            "arguments": json.dumps(block.input),
                                        },
                                    }
                                )
                            elif block.type == "text":
                                text_content += block.text
                        elif isinstance(block, dict):
                            if block.get("type") == "tool_use":
                                import json

                                tool_calls.append(
                                    {
                                        "id": block.get("id"),
                                        "type": "function",
                                        "function": {
                                            "name": block.get("name"),
                                            "arguments": json.dumps(
                                                block.get("input", {})
                                            ),
                                        },
                                    }
                                )
                            elif block.get("type") == "text":
                                text_content += block.get("text", "")

                    msg_dict: Dict[str, Any] = {
                        "role": "assistant",
                        "content": text_content or None,
                    }
                    if tool_calls:
                        msg_dict["tool_calls"] = tool_calls
                    openai_messages.append(msg_dict)
                else:
                    openai_messages.append({"role": "assistant", "content": content})

        return openai_messages

    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Anthropic-style tools to OpenAI format.

        Args:
            tools: Anthropic-format tool definitions

        Returns:
            OpenAI-format tool definitions
        """
        openai_tools = []

        for tool in tools:
            # Anthropic format: {name, description, input_schema}
            # OpenAI format: {type: "function", function: {name, description, params}}
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            }
            openai_tools.append(openai_tool)

        return openai_tools

    def _convert_response(self, response: Any) -> Any:
        """Convert OpenAI response to Anthropic-compatible format.

        Args:
            response: OpenAI ChatCompletion response

        Returns:
            Object with Anthropic-compatible interface
        """
        from dataclasses import dataclass
        from dataclasses import field
        from typing import List as ListType

        @dataclass
        class TextBlock:
            type: str = "text"
            text: str = ""

        @dataclass
        class ToolUseBlock:
            type: str = "tool_use"
            id: str = ""
            name: str = ""
            input: Dict[str, Any] = field(default_factory=dict)

        @dataclass
        class AnthropicResponse:
            content: ListType[Any] = field(default_factory=list)
            stop_reason: Optional[str] = None

        choice = response.choices[0]
        message = choice.message
        content: List[Any] = []

        # Add text content if present
        if message.content:
            content.append(TextBlock(text=message.content))

        # Add tool calls if present
        if message.tool_calls:
            import json

            for tool_call in message.tool_calls:
                content.append(
                    ToolUseBlock(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        input=json.loads(tool_call.function.arguments),
                    )
                )

        # Convert finish_reason to Anthropic stop_reason
        finish_reason = choice.finish_reason
        if finish_reason == "stop":
            stop_reason = "end_turn"
        elif finish_reason == "tool_calls":
            stop_reason = "tool_use"
        elif finish_reason == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = finish_reason

        return AnthropicResponse(content=content, stop_reason=stop_reason)
