"""Authentication management for CDD Agent.

This module handles:
- Interactive provider setup
- API key validation
- OAuth flow for Claude Pro/Max plans
- Configuration display
"""

import asyncio
import webbrowser
from typing import Dict

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.prompt import Prompt
from rich.table import Table

from .config import ConfigManager
from .config import OAuthTokens
from .config import ProviderConfig
from .config import Settings


console = Console()


class AuthManager:
    """Manages authentication setup and validation."""

    def __init__(self, config_manager: ConfigManager):
        """Initialize auth manager.

        Args:
            config_manager: ConfigManager instance
        """
        self.config = config_manager

    def interactive_setup(self) -> Settings:
        """Guide user through initial setup.

        Returns:
            Settings object after setup
        """
        console.print(
            Panel.fit(
                "[bold cyan]Welcome to CDD Agent![/bold cyan]\n\n"
                "Let's set up your LLM provider authentication.",
                border_style="cyan",
            )
        )

        # Choose provider
        provider_choice = Prompt.ask(
            "\nWhich LLM provider do you want to use?",
            choices=["anthropic", "openai", "custom"],
            default="anthropic",
        )

        if provider_choice == "anthropic":
            return self._setup_anthropic()
        elif provider_choice == "openai":
            return self._setup_openai()
        else:
            return self._setup_custom()

    def _setup_anthropic(self) -> Settings:
        """Set up Anthropic configuration.

        Returns:
            Settings object with Anthropic provider
        """
        console.print("\n[bold]Anthropic Setup[/bold]")
        console.print(
            "Get your API key from: [link]https://console.anthropic.com/[/link]"
        )

        api_key = Prompt.ask("Enter your Anthropic API key", password=True)

        # Test the key
        if self._validate_anthropic_key(api_key):
            console.print("[green]✓ API key validated successfully![/green]")
        else:
            console.print(
                "[yellow]⚠ Could not validate API key (but saving anyway)[/yellow]"
            )

        # Model selection
        use_defaults = Confirm.ask("Use default model mappings?", default=True)

        if use_defaults:
            models = {
                "small": "claude-3-5-haiku-20241022",
                "mid": "claude-sonnet-4-5-20250929",
                "big": "claude-opus-4-20250514",
            }
        else:
            models = self._prompt_models()

        provider_config = ProviderConfig(
            auth_token=api_key,
            base_url="https://api.anthropic.com",
            models=models,
            provider_type="anthropic",
        )

        settings = Settings(
            default_provider="anthropic",
            providers={"anthropic": provider_config},
        )

        self.config.save(settings)
        console.print(
            f"[green]✓ Configuration saved to {self.config.config_file}[/green]"
        )
        return settings

    def _setup_openai(self) -> Settings:
        """Set up OpenAI configuration.

        Returns:
            Settings object with OpenAI provider
        """
        console.print("\n[bold]OpenAI Setup[/bold]")
        console.print(
            "Get your API key from: [link]https://platform.openai.com/[/link]"
        )

        api_key = Prompt.ask("Enter your OpenAI API key", password=True)

        # Model selection
        use_defaults = Confirm.ask("Use default model mappings?", default=True)

        if use_defaults:
            models = {
                "small": "gpt-4o-mini",
                "mid": "gpt-4o",
                "big": "o1-preview",
            }
        else:
            models = self._prompt_models()

        provider_config = ProviderConfig(
            api_key=api_key,
            base_url="https://api.openai.com/v1",
            models=models,
            provider_type="openai",
        )

        settings = Settings(
            default_provider="openai", providers={"openai": provider_config}
        )

        self.config.save(settings)
        console.print(
            f"[green]✓ Configuration saved to {self.config.config_file}[/green]"
        )
        return settings

    def _setup_custom(self) -> Settings:
        """Set up custom provider (like z.ai).

        Returns:
            Settings object with custom provider
        """
        console.print("\n[bold]Custom Provider Setup[/bold]")
        console.print(
            "This is for alternative providers (like z.ai, local servers, proxies)"
        )

        base_url = Prompt.ask(
            "\nEnter base URL",
            default="https://api.z.ai/api/anthropic",
        )

        api_key = Prompt.ask("Enter your API key/token", password=True)

        provider_type = Prompt.ask(
            "\nAPI compatibility (which API format does it use?)",
            choices=["anthropic", "openai"],
            default="anthropic",
        )

        console.print("\n[bold]Model Configuration[/bold]")
        console.print("Map model tiers to actual model names:")

        models = {
            "small": Prompt.ask("  Small model (fast/cheap)", default="glm-4.5-air"),
            "mid": Prompt.ask("  Mid model (balanced)", default="glm-4.6"),
            "big": Prompt.ask("  Big model (powerful)", default="glm-4.6"),
        }

        provider_config = ProviderConfig(
            auth_token=api_key,
            base_url=base_url,
            models=models,
            provider_type=provider_type,
        )

        settings = Settings(
            default_provider="custom", providers={"custom": provider_config}
        )

        self.config.save(settings)
        console.print(
            f"[green]✓ Custom provider configured and saved to "
            f"{self.config.config_file}[/green]"
        )
        return settings

    def _prompt_models(self) -> Dict[str, str]:
        """Prompt user for custom model mappings.

        Returns:
            Dictionary mapping tier names to model names
        """
        console.print("\n[bold]Custom Model Configuration[/bold]")
        return {
            "small": Prompt.ask("  Small model (fast/cheap)"),
            "mid": Prompt.ask("  Mid model (balanced)"),
            "big": Prompt.ask("  Big model (powerful)"),
        }

    def _validate_anthropic_key(self, api_key: str) -> bool:
        """Test Anthropic API key.

        Args:
            api_key: API key to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            import anthropic

            client = anthropic.Anthropic(
                api_key=api_key,
                max_retries=5,  # Increase from default 2 to handle overloaded errors
                timeout=600.0,  # 10 minutes timeout for long-running requests
            )
            # Try a minimal request
            client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True
        except Exception as e:
            console.print(f"[dim]Validation error: {e}[/dim]")
            return False

    def display_current_config(self) -> None:
        """Show current configuration in a table."""
        if not self.config.exists():
            console.print(
                "[yellow]No configuration found. "
                "Run 'cdd-agent auth setup' first.[/yellow]"
            )
            return

        settings = self.config.load()

        table = Table(title="Current Configuration")
        table.add_column("Provider", style="cyan")
        table.add_column("Base URL", style="green")
        table.add_column("Default Model", style="yellow")
        table.add_column("Status", style="magenta")

        for name, provider in settings.providers.items():
            is_default = "⭐ Default" if name == settings.default_provider else ""
            has_key = "✓ Configured" if provider.get_api_key() else "✗ Missing Key"

            table.add_row(
                name,
                provider.base_url,
                provider.get_model(),
                f"{is_default} {has_key}",
            )

        console.print(table)

        # Show config file location
        console.print(f"\n[dim]Config file: {self.config.config_file}[/dim]")

    def setup_oauth_interactive(self, provider_name: str = "anthropic") -> None:
        """Set up OAuth authentication for Claude Pro/Max plans.

        This method guides the user through OAuth authentication flow:
        1. Opens browser to Anthropic's OAuth page
        2. User authorizes and copies authorization code
        3. Exchanges code for OAuth tokens
        4. Saves tokens to settings.json

        Args:
            provider_name: Provider to configure (default: anthropic)

        Supports two modes:
        - "max": OAuth for Claude Pro/Max plan (zero-cost API access)
        - "api-key": Create permanent API key via OAuth
        """
        from .oauth import AnthropicOAuth

        console.print("\n[bold cyan]Anthropic OAuth Setup[/bold cyan]")
        console.print(
            "This will authenticate with your Claude Pro or Max plan "
            "for zero-cost API access.\n"
        )

        # Ask which authentication mode
        mode_choice = Prompt.ask(
            "Choose authentication mode",
            choices=["max", "api-key"],
            default="max",
        )

        console.print()
        if mode_choice == "max":
            console.print(
                "[yellow]Mode:[/yellow] OAuth (Claude Pro/Max)\n"
                "[dim]Uses OAuth tokens that auto-refresh. "
                "Best for plan subscribers.[/dim]\n"
            )
        else:
            console.print(
                "[yellow]Mode:[/yellow] API Key Generation\n"
                "[dim]Creates a permanent API key via OAuth. More convenient but "
                "counts toward API usage.[/dim]\n"
            )

        oauth_handler = AnthropicOAuth()

        if mode_choice == "max":
            # OAuth flow for Claude Max plan
            auth_url, verifier = asyncio.run(oauth_handler.start_auth_flow(mode="max"))

            console.print("[bold]Step 1: Authorize in browser[/bold]")
            console.print(f"Opening: [link]{auth_url}[/link]\n")

            # Try to open browser automatically
            try:
                webbrowser.open(auth_url)
                console.print("[green]✓ Browser opened[/green]")
            except Exception:
                console.print(
                    "[yellow]⚠ Could not open browser automatically.[/yellow]"
                )
                console.print("Please copy the URL above and open it manually.\n")

            # Get authorization code from user
            console.print(
                "\n[bold]Step 2: Paste authorization code[/bold]\n"
                "[dim]After authorizing, you'll receive a code. Paste it here:[/dim]"
            )
            auth_code = Prompt.ask("[cyan]Authorization code[/cyan]")

            # Exchange for tokens
            console.print("\n[yellow]Exchanging code for OAuth tokens...[/yellow]")
            tokens = asyncio.run(oauth_handler.exchange_code(auth_code, verifier))

            if not tokens:
                console.print(
                    "[red]✗ Failed to exchange authorization code.[/red]\n"
                    "[dim]Please try again or check that you copied the code "
                    "correctly.[/dim]"
                )
                return

            # Create OAuth config
            oauth_config = OAuthTokens(
                refresh_token=tokens["refresh_token"],
                access_token=tokens["access_token"],
                expires_at=tokens["expires_at"],
            )

            # Load or create settings
            if self.config.exists():
                settings = self.config.load()
            else:
                settings = self.config.create_default()

            # Update provider config with OAuth
            if provider_name not in settings.providers:
                settings.providers[provider_name] = ProviderConfig(
                    base_url="https://api.anthropic.com",
                    models={
                        "small": "claude-3-5-haiku-20241022",
                        "mid": "claude-sonnet-4-5-20250929",
                        "big": "claude-opus-4-20250514",
                    },
                )

            settings.providers[provider_name].oauth = oauth_config
            settings.default_provider = provider_name
            self.config.save(settings)

            console.print(
                "\n[green]✓ OAuth setup successful![/green]\n"
                "[dim]Your Claude Pro/Max plan is now connected.\n"
                "Tokens will auto-refresh when needed.[/dim]\n"
            )

        else:
            # Create permanent API key via OAuth
            auth_url, verifier = asyncio.run(
                oauth_handler.start_auth_flow(mode="console")
            )

            console.print("[bold]Step 1: Authorize in browser[/bold]")
            console.print(f"Opening: [link]{auth_url}[/link]\n")

            try:
                webbrowser.open(auth_url)
                console.print("[green]✓ Browser opened[/green]")
            except Exception:
                console.print(
                    "[yellow]⚠ Could not open browser automatically.[/yellow]"
                )
                console.print("Please copy the URL above and open it manually.\n")

            console.print(
                "\n[bold]Step 2: Paste authorization code[/bold]\n"
                "[dim]After authorizing, you'll receive a code. Paste it here:[/dim]"
            )
            auth_code = Prompt.ask("[cyan]Authorization code[/cyan]")

            console.print("\n[yellow]Creating permanent API key...[/yellow]")
            api_key = asyncio.run(
                oauth_handler.create_api_key_from_oauth(auth_code, verifier)
            )

            if not api_key:
                console.print(
                    "[red]✗ Failed to create API key.[/red]\n"
                    "[dim]Please try again or check that you copied the code "
                    "correctly.[/dim]"
                )
                return

            # Load or create settings
            if self.config.exists():
                settings = self.config.load()
            else:
                settings = self.config.create_default()

            # Save API key
            if provider_name not in settings.providers:
                settings.providers[provider_name] = ProviderConfig(
                    base_url="https://api.anthropic.com",
                    models={
                        "small": "claude-3-5-haiku-20241022",
                        "mid": "claude-sonnet-4-5-20250929",
                        "big": "claude-opus-4-20250514",
                    },
                )

            settings.providers[provider_name].auth_token = api_key
            settings.default_provider = provider_name
            self.config.save(settings)

            console.print(
                "\n[green]✓ API key created and saved![/green]\n"
                "[dim]You can now use cdd-agent with this API key.[/dim]\n"
            )
