"""Anthropic provider with OAuth authentication for Claude Pro/Max plans.

This provider implements the ProviderStrategy protocol using OAuth tokens
for authentication, enabling zero-cost API usage for Pro/Max subscribers.
"""

import time
from typing import Any
from typing import Dict
from typing import Generator
from typing import List

from ..config import ProviderConfig
from ..logging import get_logger


logger = get_logger(__name__)


class OAuthTransport:
    """Custom HTTP transport that adds OAuth headers.

    This transport:
    - Removes the default x-api-key header
    - Adds OAuth Bearer token in Authorization header
    - Adds required OAuth beta headers
    """

    def __init__(self, access_token: str):
        """Initialize OAuth transport.

        Args:
            access_token: OAuth access token
        """
        import httpx

        self._transport = httpx.HTTPTransport()
        self.access_token = access_token

    def handle_request(self, request) -> Any:
        """Handle request by adding OAuth headers.

        Args:
            request: HTTP request object

        Returns:
            HTTP response
        """
        # Remove x-api-key header if present (SDK adds it by default)
        if "x-api-key" in request.headers:
            del request.headers["x-api-key"]

        # Remove x-stainless-* headers that identify this as the Python SDK
        # OAuth tokens may be restricted to specific client types
        headers_to_remove = [
            k for k in request.headers.keys() if k.lower().startswith("x-stainless")
        ]
        for header in headers_to_remove:
            del request.headers[header]

        # Set User-Agent similar to AI SDK
        request.headers["User-Agent"] = "ai-sdk/anthropic"

        # Add OAuth Bearer token
        request.headers["Authorization"] = f"Bearer {self.access_token}"

        # Add required OAuth beta headers (matching OpenCode's implementation)
        request.headers["anthropic-beta"] = (
            "oauth-2025-04-20,"
            "claude-code-20250219,"
            "interleaved-thinking-2025-05-14,"
            "fine-grained-tool-streaming-2025-05-14"
        )

        return self._transport.handle_request(request)


class AnthropicOAuthProvider:
    """Anthropic provider with OAuth authentication.

    This provider handles:
    - OAuth token management and auto-refresh
    - Custom HTTP transport for OAuth headers
    - Message creation (streaming and non-streaming)

    OAuth tokens are automatically refreshed when they expire within
    5 minutes of the current time.

    Example:
        provider = AnthropicOAuthProvider(config, model_tier="mid")
        response = provider.create_message(messages, tools, system)
    """

    # Refresh token if it expires within this many seconds
    TOKEN_REFRESH_THRESHOLD = 300  # 5 minutes

    def __init__(
        self,
        provider_config: ProviderConfig,
        model_tier: str = "mid",
        max_retries: int = 5,
        timeout: float = 600.0,
    ):
        """Initialize Anthropic OAuth provider.

        Args:
            provider_config: Provider configuration with OAuth tokens
            model_tier: Model tier to use (small/mid/big)
            max_retries: Maximum retry attempts for API calls
            timeout: Request timeout in seconds

        Raises:
            ValueError: If provider_config doesn't have OAuth configured
        """
        if not provider_config.oauth:
            raise ValueError(
                "AnthropicOAuthProvider requires OAuth configuration. "
                "Use AnthropicProvider for API key authentication."
            )

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

    def _refresh_token_if_needed(self) -> None:
        """Refresh OAuth token if it's expiring soon.

        Checks if the token expires within TOKEN_REFRESH_THRESHOLD seconds
        and refreshes it if necessary. Updated tokens are saved to config.
        """
        oauth_config = self._provider_config.oauth
        if oauth_config is None:
            return

        # Check if token needs refresh
        if time.time() < (oauth_config.expires_at - self.TOKEN_REFRESH_THRESHOLD):
            return  # Token still valid

        logger.info("OAuth token expiring soon, refreshing...")

        try:
            import asyncio

            from ..oauth import AnthropicOAuth

            oauth_handler = AnthropicOAuth()
            new_tokens = asyncio.run(
                oauth_handler.refresh_access_token(oauth_config.refresh_token)
            )

            if new_tokens:
                logger.info("OAuth token refreshed successfully")

                # Update config with new tokens
                oauth_config.access_token = new_tokens["access_token"]
                oauth_config.expires_at = new_tokens["expires_at"]
                if "refresh_token" in new_tokens:
                    oauth_config.refresh_token = new_tokens["refresh_token"]

                # Save updated tokens to config file
                from ..config import ConfigManager

                config_manager = ConfigManager()
                settings = config_manager.load()

                # Find and update the provider with OAuth
                for provider_name, provider_cfg in settings.providers.items():
                    if provider_cfg.oauth == oauth_config:
                        settings.providers[provider_name].oauth = oauth_config
                        break

                config_manager.save(settings)
                logger.debug("Updated OAuth tokens saved to config")

                # Clear cached client to force re-creation with new token
                self._client = None
            else:
                logger.warning("Failed to refresh OAuth token - using existing token")

        except Exception as e:
            logger.error(f"Error refreshing OAuth token: {e}", exc_info=True)
            logger.warning("Continuing with existing OAuth token")

    @property
    def client(self):
        """Lazy initialization of Anthropic client with OAuth.

        The client is created with a custom HTTP transport that handles
        OAuth authentication headers.

        Returns:
            Initialized Anthropic client

        Raises:
            ImportError: If required packages are not installed
        """
        # Check and refresh token before creating/using client
        self._refresh_token_if_needed()

        if self._client is None:
            try:
                logger.debug("Initializing Anthropic client with OAuth")
                import anthropic
                import httpx

                oauth_config = self._provider_config.oauth
                if oauth_config is None:
                    raise ValueError("OAuth configuration is missing")

                # Create custom httpx client with OAuth transport
                http_client = httpx.Client(
                    transport=OAuthTransport(oauth_config.access_token),
                    timeout=self._timeout,
                )

                # Create Anthropic client with custom HTTP client
                # The API key is a dummy - it will be replaced by OAuth transport
                self._client = anthropic.Anthropic(
                    api_key="dummy-key-replaced-by-oauth",
                    base_url=self._provider_config.base_url,
                    max_retries=self._max_retries,
                    timeout=self._timeout,
                    http_client=http_client,
                )

                logger.info("Anthropic client initialized with OAuth authentication")

            except ImportError as e:
                logger.error("Failed to import required packages", exc_info=True)
                raise ImportError(
                    "Failed to import anthropic or httpx. Please install with: "
                    "pip install anthropic httpx"
                ) from e
            except Exception as e:
                logger.error(f"Error initializing OAuth client: {e}", exc_info=True)
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

        Note: OAuth API rejects custom fields in tool schemas, so
        risk_level should be excluded from tools before calling.

        Args:
            messages: Conversation history
            tools: Available tools in Anthropic format (without risk_level)
            system: System prompt
            max_tokens: Maximum tokens in response

        Returns:
            Anthropic Message response object
        """
        logger.debug(
            f"API request (OAuth): model={self.model} "
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

        Note: OAuth API rejects custom fields in tool schemas, so
        risk_level should be excluded from tools before calling.

        Args:
            messages: Conversation history
            tools: Available tools in Anthropic format (without risk_level)
            system: System prompt
            max_tokens: Maximum tokens in response

        Yields:
            Event dictionaries with streaming data

        Returns:
            Final message object after stream completes
        """
        logger.debug(
            f"Streaming API request (OAuth): model={self.model} "
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

    @property
    def uses_oauth(self) -> bool:
        """Check if this provider uses OAuth authentication.

        Returns:
            True (always, for OAuth provider)
        """
        return True
