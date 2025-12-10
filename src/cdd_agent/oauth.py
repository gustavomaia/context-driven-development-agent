"""OAuth 2.0 authentication handler for Anthropic Claude Pro/Max plans.

This module implements the OAuth flow for authenticating with Claude Pro/Max
subscriptions, allowing zero-cost API usage for plan subscribers.

Based on OpenCode's implementation (opencode-anthropic-auth@0.0.2).
"""

import secrets
import time
from typing import Optional
from typing import Tuple
from urllib.parse import quote
from urllib.parse import urlencode

import httpx
from authlib.oauth2.rfc7636 import create_s256_code_challenge


# Anthropic OAuth configuration
CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
OAUTH_BASE = "https://console.anthropic.com"
API_BASE = "https://api.anthropic.com"


class AnthropicOAuth:
    """Handle Anthropic OAuth 2.0 authentication flow.

    Supports two authentication modes:
    1. Claude Pro/Max OAuth: Direct OAuth tokens for plan-based API access
    2. API Key Generation: Create permanent API key via OAuth
    """

    def __init__(self):
        """Initialize OAuth handler."""
        self.client_id = CLIENT_ID

    async def start_auth_flow(self, mode: str = "max") -> Tuple[str, str]:
        """Start OAuth authorization flow with PKCE.

        Args:
            mode: Authentication mode
                - "max": Claude Pro/Max OAuth (uses claude.ai)
                - "console": API key generation (uses console.anthropic.com)

        Returns:
            Tuple of (authorization_url, code_verifier)

        Example:
            oauth = AnthropicOAuth()
            url, verifier = await oauth.start_auth_flow("max")
            # Open url in browser, user authorizes
            # User pastes code back
            tokens = await oauth.exchange_code(code, verifier)
        """
        # Generate PKCE challenge and verifier
        code_verifier = secrets.token_urlsafe(32)
        code_challenge = create_s256_code_challenge(code_verifier)

        # Build authorization URL
        base_host = "claude.ai" if mode == "max" else "console.anthropic.com"
        auth_url = f"https://{base_host}/oauth/authorize"

        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": f"{OAUTH_BASE}/oauth/code/callback",
            "scope": "org:create_api_key user:profile user:inference",
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": code_verifier,  # State doubles as verifier for validation
            "code": "true",
        }

        full_url = f"{auth_url}?{urlencode(params, quote_via=quote)}"

        return full_url, code_verifier

    async def exchange_code(
        self, authorization_code: str, code_verifier: str
    ) -> Optional[dict]:
        """Exchange authorization code for OAuth tokens.

        Args:
            authorization_code: The authorization code from OAuth callback
                Can be in format "code" or "code#state"
            code_verifier: PKCE code verifier from start_auth_flow()

        Returns:
            Dictionary with:
                - refresh_token: Long-lived refresh token
                - access_token: Short-lived access token
                - expires_at: Unix timestamp when access token expires
            Returns None if exchange fails.

        Example:
            tokens = await oauth.exchange_code("abc123#xyz", verifier)
            if tokens:
                print(f"Access token: {tokens['access_token']}")
                print(f"Expires at: {tokens['expires_at']}")
        """
        # Parse authorization code (may contain state)
        code_parts = authorization_code.split("#")
        code = code_parts[0]
        state = code_parts[1] if len(code_parts) > 1 else None

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{OAUTH_BASE}/v1/oauth/token",
                    json={
                        "code": code,
                        "state": state,
                        "grant_type": "authorization_code",
                        "client_id": self.client_id,
                        "redirect_uri": f"{OAUTH_BASE}/oauth/code/callback",
                        "code_verifier": code_verifier,
                    },
                    headers={"Content-Type": "application/json"},
                )

                if not response.is_success:
                    return None

                data = response.json()

                return {
                    "refresh_token": data["refresh_token"],
                    "access_token": data["access_token"],
                    "expires_at": int(time.time()) + data["expires_in"],
                }

        except Exception:
            return None

    async def refresh_access_token(self, refresh_token: str) -> Optional[dict]:
        """Refresh an expired access token.

        Args:
            refresh_token: The refresh token from initial OAuth flow

        Returns:
            Dictionary with:
                - refresh_token: New refresh token (may be rotated)
                - access_token: New access token
                - expires_at: New expiration timestamp
            Returns None if refresh fails.

        Example:
            new_tokens = await oauth.refresh_access_token(old_refresh_token)
            if new_tokens:
                # Save new tokens to config
                config.oauth.access_token = new_tokens['access_token']
                config.oauth.expires_at = new_tokens['expires_at']
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{OAUTH_BASE}/v1/oauth/token",
                    json={
                        "grant_type": "refresh_token",
                        "refresh_token": refresh_token,
                        "client_id": self.client_id,
                    },
                    headers={"Content-Type": "application/json"},
                )

                if not response.is_success:
                    return None

                data = response.json()

                # Refresh token may or may not rotate
                return {
                    "refresh_token": data.get("refresh_token", refresh_token),
                    "access_token": data["access_token"],
                    "expires_at": int(time.time()) + data["expires_in"],
                }

        except Exception:
            return None

    async def create_api_key_from_oauth(
        self, authorization_code: str, code_verifier: str
    ) -> Optional[str]:
        """Create a permanent API key using OAuth authentication.

        This is an alternative to storing OAuth tokens. The API key is
        permanent and doesn't require token refresh.

        Args:
            authorization_code: The authorization code from OAuth callback
            code_verifier: PKCE code verifier from start_auth_flow()

        Returns:
            API key string, or None if creation fails

        Example:
            api_key = await oauth.create_api_key_from_oauth(code, verifier)
            if api_key:
                # Save API key to config
                config.auth_token = api_key
        """
        # First exchange code for tokens
        tokens = await self.exchange_code(authorization_code, code_verifier)
        if not tokens:
            return None

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{API_BASE}/api/oauth/claude_cli/create_api_key",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {tokens['access_token']}",
                    },
                )

                if not response.is_success:
                    return None

                data = response.json()
                return str(data.get("raw_key")) if data.get("raw_key") else None

        except Exception:
            return None
