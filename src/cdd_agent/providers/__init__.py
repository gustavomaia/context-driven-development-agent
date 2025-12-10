"""LLM Provider implementations.

This module provides a Strategy pattern implementation for LLM providers,
enabling clean separation between different authentication methods and
multi-provider support.

Available Providers:
- AnthropicProvider: Anthropic API with API key authentication
- AnthropicOAuthProvider: Anthropic API with OAuth (Claude Pro/Max plans)
- OpenAIProvider: OpenAI API (GPT-4, GPT-4o, o1, etc.)

Usage:
    from cdd_agent.providers import create_provider

    provider = create_provider(provider_config)
    response = provider.create_message(messages, tools, system)
"""

from .anthropic_oauth_provider import AnthropicOAuthProvider
from .anthropic_provider import AnthropicProvider
from .base import ProviderStrategy
from .factory import create_provider
from .openai_provider import OpenAIProvider


__all__ = [
    "AnthropicOAuthProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "ProviderStrategy",
    "create_provider",
]
