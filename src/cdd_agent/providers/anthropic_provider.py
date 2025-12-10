"""Anthropic provider with standard API key authentication.

This provider implements the ProviderStrategy protocol for Anthropic's
Claude models using API key authentication.
"""

from typing import Any
from typing import Dict
from typing import Generator
from typing import List

from ..config import ProviderConfig
from ..logging import get_logger


logger = get_logger(__name__)


class AnthropicProvider:
    """Anthropic provider with API key authentication.

    This provider handles:
    - Lazy initialization of the Anthropic client
    - API key authentication
    - Message creation (streaming and non-streaming)
    - Retry logic and timeout configuration

    Example:
        provider = AnthropicProvider(config, model_tier="mid")
        response = provider.create_message(messages, tools, system)
    """

    def __init__(
        self,
        provider_config: ProviderConfig,
        model_tier: str = "mid",
        max_retries: int = 5,
        timeout: float = 600.0,
    ):
        """Initialize Anthropic provider.

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
        self._client = None

    @property
    def model(self) -> str:
        """Get the model name for this provider.

        Returns:
            Model identifier string
        """
        return self._provider_config.get_model(self._model_tier)

    @property
    def client(self):
        """Lazy initialization of Anthropic client.

        The Anthropic SDK is only imported and initialized when needed,
        saving startup time.

        Returns:
            Initialized Anthropic client

        Raises:
            ImportError: If anthropic package is not installed
        """
        if self._client is None:
            try:
                logger.debug("Initializing Anthropic client with API key")
                import anthropic

                self._client = anthropic.Anthropic(
                    api_key=self._provider_config.get_api_key(),
                    base_url=self._provider_config.base_url,
                    max_retries=self._max_retries,
                    timeout=self._timeout,
                )
                logger.info("Anthropic client initialized successfully")
            except ImportError as e:
                logger.error("Failed to import anthropic SDK", exc_info=True)
                raise ImportError(
                    "Failed to import anthropic. Please install it with: "
                    "pip install anthropic"
                ) from e
            except Exception as e:
                logger.error(f"Error initializing Anthropic client: {e}", exc_info=True)
                raise

        return self._client

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
            Anthropic Message response object
        """
        logger.debug(
            f"API request: model={self.model} "
            f"messages_count={len(messages)} "
            f"tools_count={len(tools)}"
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=messages,
            tools=tools,
            system=system,
        )

        logger.debug(f"API response: stop_reason={response.stop_reason}")
        return response

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
            Event dictionaries with streaming data

        Returns:
            Final message object after stream completes
        """
        logger.debug(
            f"Streaming API request: model={self.model} "
            f"messages_count={len(messages)} "
            f"tools_count={len(tools)}"
        )

        with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            messages=messages,
            tools=tools,
            system=system,
        ) as stream:
            for event in stream:
                # Convert Anthropic events to our standard format
                if event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        yield {"type": "text_delta", "text": event.delta.text}
                    elif hasattr(event.delta, "partial_json"):
                        yield {
                            "type": "tool_use_delta",
                            "partial_json": event.delta.partial_json,
                        }

                elif event.type == "content_block_start":
                    if hasattr(event.content_block, "type"):
                        if event.content_block.type == "tool_use":
                            yield {
                                "type": "tool_use_start",
                                "id": event.content_block.id,
                                "name": event.content_block.name,
                            }

            # Get and return final message
            final_message = stream.get_final_message()
            logger.debug(
                f"Streaming API response: stop_reason={final_message.stop_reason}"
            )

            yield {"type": "message_stop", "message": final_message}
            return final_message
