"""Conversation management for Agent.

This module provides the ConversationManager class that handles:
- Message history management
- Context window management (pruning/compaction)
- Project context injection
"""

from typing import Any
from typing import Dict
from typing import List

from rich.console import Console

from .context import ContextLoader
from .logging import get_logger


logger = get_logger(__name__)
console = Console()


class ConversationManager:
    """Manages conversation history and context window.

    This class handles:
    - Message history storage
    - Context injection for first message
    - Context window management (pruning old messages)
    - Conversation compaction (summarizing old exchanges)

    Example:
        manager = ConversationManager(enable_context=True)
        manager.add_user_message("Hello")
        manager.add_assistant_message(response.content)
        manager.manage_context_window()
    """

    def __init__(
        self,
        enable_context: bool = True,
        max_messages: int = 100,
        max_chars: int = 500000,
    ):
        """Initialize conversation manager.

        Args:
            enable_context: Whether to load hierarchical context files
            max_messages: Maximum number of messages before compaction
            max_chars: Maximum total characters before compaction
        """
        self.messages: List[Dict[str, Any]] = []
        self.enable_context = enable_context
        self.max_messages = max_messages
        self.max_chars = max_chars

        # Initialize context loader
        self.context_loader = ContextLoader() if enable_context else None

        # Load project context once at initialization
        self.project_context = self._load_project_context()

    def _load_project_context(self) -> str:
        """Load hierarchical context files.

        Uses ContextLoader to load and merge contexts from:
        - Global: ~/.cdd/CDD.md or ~/.claude/CLAUDE.md
        - Project: CDD.md or CLAUDE.md at project root

        Returns:
            Merged context string, or fallback message if no context found
        """
        if not self.enable_context or not self.context_loader:
            return "Context loading disabled (--no-context flag)."

        context = self.context_loader.load_context()

        if context:
            return context
        return "No context files found (create ~/.cdd/CDD.md or ./CDD.md)."

    def add_user_message(self, content: str) -> None:
        """Add a user message to history.

        If this is the first message and project context exists,
        the context is injected into the message.

        Args:
            content: User message content
        """
        # Inject project context into first message only
        if len(self.messages) == 0 and self.project_context:
            enhanced_content = f"{self.project_context}\n\n{'─' * 80}\n\n{content}"
        else:
            enhanced_content = content

        self.messages.append({"role": "user", "content": enhanced_content})

    def add_assistant_message(self, content: Any) -> None:
        """Add an assistant message to history.

        Args:
            content: Assistant message content (can be string or list of blocks)
        """
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_results(self, tool_results: List[Dict[str, Any]]) -> None:
        """Add tool results to history.

        Args:
            tool_results: List of tool result blocks
        """
        self.messages.append({"role": "user", "content": tool_results})

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages = []

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get current message history.

        Returns:
            List of message dictionaries
        """
        return self.messages

    def calculate_size(self) -> int:
        """Calculate total character size of conversation.

        Returns:
            Total character count
        """
        total = 0
        for msg in self.messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total += len(str(block.get("content", "")))
                        total += len(str(block.get("text", "")))
        return total

    def manage_context_window(self) -> None:
        """Prune old messages to stay within context limits.

        Strategy:
        - Keep first message (often contains important context)
        - Keep last N messages (recent context)
        - Log when limits are reached (compaction disabled for testing)
        """
        total_chars = self.calculate_size()
        logger.debug(
            f"Context check: {len(self.messages)} messages, "
            f"{total_chars} chars (limits: {self.max_messages} msgs, "
            f"{self.max_chars})"
        )

        if len(self.messages) < self.max_messages and total_chars <= self.max_chars:
            return

        if total_chars > self.max_chars or len(self.messages) >= self.max_messages:
            logger.warning(
                f"Context limits reached ({len(self.messages)} msgs, "
                f"{total_chars} chars) - compaction disabled for testing"
            )

    def compact(self) -> bool:
        """Manually trigger conversation compaction.

        Similar to Claude Code's /compact command, this:
        1. Keeps the initial user message (important context)
        2. Keeps the last 4 messages (recent context)
        3. Summarizes middle messages

        Returns:
            True if compaction was performed, False if too short
        """
        if len(self.messages) <= 6:
            console.print(
                "[yellow]⚠ Conversation too short to compact "
                "(need at least 6 messages)[/yellow]"
            )
            return False

        # Keep first message and last 4 messages
        first_message = self.messages[0]
        last_messages = self.messages[-4:]
        middle_messages = self.messages[1:-4]

        # Generate summary of middle messages
        summary_parts = []
        tools_used = set()

        for msg in middle_messages:
            content = msg.get("content", "")

            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tools_used.add("tools")

            if msg.get("role") == "assistant":
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            if text and len(text) > 100:
                                summary_parts.append(text[:100] + "...")
                elif isinstance(content, str) and content:
                    if len(content) > 100:
                        summary_parts.append(content[:100] + "...")

        # Create compact summary message
        summary_text = "Previous conversation summary:\n"
        if tools_used:
            summary_text += "- Used various tools to analyze and modify code\n"
        if summary_parts:
            summary_text += "- Key actions: " + "; ".join(summary_parts[:3]) + "\n"
        summary_text += f"[{len(middle_messages)} messages compacted]"

        summary_message = {"role": "user", "content": summary_text}

        # Replace message history
        old_count = len(self.messages)
        self.messages = [first_message, summary_message] + last_messages

        # Calculate size reduction
        old_size = sum(len(str(msg.get("content", ""))) for msg in middle_messages)
        new_size = len(summary_text)

        logger.info(
            f"Compacted {old_count} messages to {len(self.messages)} "
            f"(saved ~{old_size - new_size} chars)"
        )
        console.print(
            f"[dim]💾 Conversation compacted: {old_count} → "
            f"{len(self.messages)} messages "
            f"(saved ~{(old_size - new_size) / 1024:.1f}KB)[/dim]"
        )
        return True

    @property
    def is_empty(self) -> bool:
        """Check if conversation is empty.

        Returns:
            True if no messages
        """
        return len(self.messages) == 0
