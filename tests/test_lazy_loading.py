"""Test lazy loading of LLM providers to ensure fast CLI startup."""

import sys
import time
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from cdd_agent.config import ProviderConfig


class TestAnthropicLazyLoading:
    """Test that Anthropic SDK is only loaded when needed."""

    def test_agent_creation_does_not_import_anthropic(self):
        """Test that creating an Agent doesn't import anthropic."""
        # Mock modules to track imports
        original_modules = sys.modules.copy()

        try:
            # Remove anthropic from sys.modules if it exists
            if "anthropic" in sys.modules:
                del sys.modules["anthropic"]

            # Mock the ToolRegistry, ApprovalManager and ConversationManager
            with (
                patch("cdd_agent.agent.ToolRegistry") as mock_registry,
                patch("cdd_agent.agent.ApprovalManager") as mock_approval,
                patch("cdd_agent.conversation.ContextLoader") as mock_context,
            ):
                mock_registry.return_value.get_tool_schema.return_value = {
                    "type": "function"
                }
                mock_approval.return_value.check_tool_approval.return_value = True
                mock_context.return_value.load_context.return_value = ""

                # Create mock provider config
                provider_config = Mock(spec=ProviderConfig)
                provider_config.get_api_key.return_value = "test-key"
                provider_config.base_url = "https://api.anthropic.com"
                provider_config.provider_type = "anthropic"
                provider_config.get_model.return_value = "claude-3-haiku-20240307"
                provider_config.oauth = None  # No OAuth for this test

                # Import Agent class fresh
                from importlib import reload

                import cdd_agent.agent

                reload(cdd_agent.agent)
                from cdd_agent.agent import Agent

                # Create agent - this should NOT import anthropic
                agent = Agent(
                    provider_config=provider_config,
                    tool_registry=mock_registry.return_value,
                    model_tier="mid",
                    enable_context=False,
                )

                # Verify anthropic is not imported yet
                assert (
                    "anthropic" not in sys.modules
                ), "anthropic should not be imported on Agent creation"

                # Access client property - this should import anthropic
                with patch("anthropic.Anthropic") as mock_anthropic:
                    mock_client = Mock()
                    mock_anthropic.return_value = mock_client

                    # Access client to trigger lazy loading
                    _ = agent.client

                    # Now anthropic should be imported
                    assert (
                        "anthropic" in sys.modules
                    ), "anthropic should be imported when accessing client"
                    mock_anthropic.assert_called_once_with(
                        api_key="test-key",
                        base_url="https://api.anthropic.com",
                        max_retries=5,
                        timeout=600.0,
                    )

        finally:
            # Restore original modules
            sys.modules.clear()
            sys.modules.update(original_modules)

    def test_simple_agent_lazy_loading(self):
        """Test that SimpleAgent also uses lazy loading."""
        original_modules = sys.modules.copy()

        try:
            if "anthropic" in sys.modules:
                del sys.modules["anthropic"]

            provider_config = Mock(spec=ProviderConfig)
            provider_config.get_api_key.return_value = "test-key"
            provider_config.base_url = "https://api.anthropic.com"

            from importlib import reload

            import cdd_agent.agent

            reload(cdd_agent.agent)
            from cdd_agent.agent import SimpleAgent

            # Create SimpleAgent - should not import anthropic
            agent = SimpleAgent(provider_config=provider_config)

            assert (
                "anthropic" not in sys.modules
            ), "anthropic should not be imported on SimpleAgent creation"

            # Access client property - should import anthropic
            with patch("anthropic.Anthropic") as mock_anthropic:
                mock_client = Mock()
                mock_anthropic.return_value = mock_client

                # Access client to trigger lazy loading
                _ = agent.client

                assert (
                    "anthropic" in sys.modules
                ), "anthropic should be imported when accessing SimpleAgent client"
                mock_anthropic.assert_called_once_with(
                    api_key="test-key",
                    base_url="https://api.anthropic.com",
                    max_retries=5,
                    timeout=600.0,
                )

        finally:
            sys.modules.clear()
            sys.modules.update(original_modules)

    def test_cli_commands_do_not_import_providers_eagerly(self):
        """Test that CLI commands don't import providers until needed."""
        original_modules = sys.modules.copy()

        try:
            # Remove provider modules
            for module in ["anthropic", "openai", "cdd_agent.agent"]:
                if module in sys.modules:
                    del sys.modules[module]

            # Import CLI module - should not import providers
            from importlib import reload

            import cdd_agent.cli

            reload(cdd_agent.cli)

            # Check that provider modules are not imported
            assert (
                "anthropic" not in sys.modules
            ), "anthropic should not be imported on CLI import"
            assert (
                "openai" not in sys.modules
            ), "openai should not be imported on CLI import"
            assert (
                "cdd_agent.agent" not in sys.modules
            ), "agent module should not be imported on CLI import"

        finally:
            sys.modules.clear()
            sys.modules.update(original_modules)


class TestStartupPerformance:
    """Test that lazy loading improves startup performance."""

    def test_startup_time_with_lazy_loading(self):
        """Measure that startup time is reasonable with lazy loading."""
        original_modules = sys.modules.copy()

        try:
            # Remove provider modules
            for module in ["anthropic", "openai"]:
                if module in sys.modules:
                    del sys.modules[module]

            # Time the import of CLI module
            start_time = time.time()

            from importlib import reload

            import cdd_agent.cli

            reload(cdd_agent.cli)

            end_time = time.time()
            startup_time = end_time - start_time

            # Should be fast (< 100ms for import)
            assert (
                startup_time < 0.1
            ), f"CLI import took {startup_time:.3f}s, expected < 0.1s"

            # Verify providers are not loaded
            assert "anthropic" not in sys.modules
            assert "openai" not in sys.modules

        finally:
            sys.modules.clear()
            sys.modules.update(original_modules)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
