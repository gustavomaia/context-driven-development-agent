"""Factory for creating LLM providers.

This module provides a factory function that creates the appropriate
provider based on the configuration's provider_type using a registry pattern.
"""

from typing import Callable
from typing import Dict
from typing import Union

from ..config import ProviderConfig
from ..logging import get_logger
from .anthropic_oauth_provider import AnthropicOAuthProvider
from .anthropic_provider import AnthropicProvider
from .openai_provider import OpenAIProvider


logger = get_logger(__name__)


# Type alias for all provider types
Provider = Union[AnthropicProvider, AnthropicOAuthProvider, OpenAIProvider]

# Type for provider factory functions
ProviderFactory = Callable[[ProviderConfig, str, int, float], Provider]


def _create_anthropic_provider(
    provider_config: ProviderConfig,
    model_tier: str,
    max_retries: int,
    timeout: float,
) -> Provider:
    """Create Anthropic provider (API key or OAuth)."""
    oauth_config = getattr(provider_config, "oauth", None)
    if oauth_config is not None:
        logger.info("Creating AnthropicOAuthProvider (OAuth authentication)")
        return AnthropicOAuthProvider(
            provider_config=provider_config,
            model_tier=model_tier,
            max_retries=max_retries,
            timeout=timeout,
        )

    logger.info("Creating AnthropicProvider (API key authentication)")
    return AnthropicProvider(
        provider_config=provider_config,
        model_tier=model_tier,
        max_retries=max_retries,
        timeout=timeout,
    )


def _create_openai_provider(
    provider_config: ProviderConfig,
    model_tier: str,
    max_retries: int,
    timeout: float,
) -> Provider:
    """Create OpenAI provider."""
    logger.info("Creating OpenAIProvider")
    return OpenAIProvider(
        provider_config=provider_config,
        model_tier=model_tier,
        max_retries=max_retries,
        timeout=timeout,
    )


# Registry mapping provider_type to factory function
PROVIDER_REGISTRY: Dict[str, ProviderFactory] = {
    "anthropic": _create_anthropic_provider,
    "openai": _create_openai_provider,
}


def register_provider(provider_type: str, factory: ProviderFactory) -> None:
    """Register a new provider factory.

    Args:
        provider_type: Provider type name (e.g., "anthropic", "openai")
        factory: Factory function that creates the provider
    """
    PROVIDER_REGISTRY[provider_type] = factory
    logger.debug(f"Registered provider factory: {provider_type}")


def create_provider(
    provider_config: ProviderConfig,
    model_tier: str = "mid",
    max_retries: int = 5,
    timeout: float = 600.0,
) -> Provider:
    """Create the appropriate provider based on configuration.

    Uses the PROVIDER_REGISTRY to look up the factory function for
    the configured provider_type.

    Args:
        provider_config: Provider configuration with provider_type
        model_tier: Model tier to use (small/mid/big)
        max_retries: Maximum retry attempts for API calls
        timeout: Request timeout in seconds

    Returns:
        Appropriate provider instance

    Raises:
        ValueError: If provider_type is not registered

    Example:
        config = config_manager.get_effective_config()
        provider = create_provider(config, model_tier="mid")
        response = provider.create_message(messages, tools, system)
    """
    # Get provider type (default to "anthropic" for backward compatibility)
    provider_type = getattr(provider_config, "provider_type", None) or "anthropic"

    factory = PROVIDER_REGISTRY.get(provider_type)
    if factory is None:
        supported = ", ".join(f"'{t}'" for t in PROVIDER_REGISTRY.keys())
        raise ValueError(
            f"Unsupported provider_type: '{provider_type}'. "
            f"Supported types: {supported}"
        )

    return factory(provider_config, model_tier, max_retries, timeout)


def is_oauth_provider(provider: Provider) -> bool:
    """Check if a provider uses OAuth authentication.

    Args:
        provider: Provider instance

    Returns:
        True if provider uses OAuth, False otherwise
    """
    return isinstance(provider, AnthropicOAuthProvider)


def is_openai_provider(provider: Provider) -> bool:
    """Check if a provider is an OpenAI provider.

    Args:
        provider: Provider instance

    Returns:
        True if provider is OpenAI, False otherwise
    """
    return isinstance(provider, OpenAIProvider)
