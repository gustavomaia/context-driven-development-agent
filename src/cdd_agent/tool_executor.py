"""Tool execution for Agent.

This module provides the ToolExecutor class that handles:
- Tool execution with approval checking
- Result enrichment and formatting
- Error handling and logging
- Result truncation for API limits
"""

from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import Optional

from rich.console import Console

from .logging import get_logger


if TYPE_CHECKING:
    from .approval import ApprovalManager
    from .background_manager import BackgroundProcessManager
    from .tool_formatter import ToolResultFormatter
    from .tools import ToolRegistry


console = Console()
logger = get_logger("tool_executor")


# Maximum size for tool results to prevent API errors
MAX_TOOL_RESULT_SIZE = 50000


class ToolExecutor:
    """Executes tools with approval, formatting, and error handling.

    This class is responsible for:
    - Formatting tool announcements
    - Checking approval before execution
    - Executing tools via the registry
    - Enriching results with metadata
    - Truncating large results
    - Handling errors gracefully

    Example:
        executor = ToolExecutor(
            tool_registry=registry,
            formatter=formatter,
            approval_manager=approval,
            background_manager=background,
        )
        result = executor.execute("read_file", {"path": "foo.py"}, "tool-123")
    """

    def __init__(
        self,
        tool_registry: "ToolRegistry",
        formatter: "ToolResultFormatter",
        approval_manager: Optional["ApprovalManager"] = None,
        background_manager: Optional["BackgroundProcessManager"] = None,
    ):
        """Initialize tool executor.

        Args:
            tool_registry: Registry of available tools
            formatter: Formatter for announcements and results
            approval_manager: Optional approval manager for safety checks
            background_manager: Optional background process manager
        """
        self.tool_registry = tool_registry
        self.formatter = formatter
        self.approval_manager = approval_manager
        self.background_manager = background_manager

    def execute(self, name: str, args: dict, tool_use_id: str) -> Dict[str, Any]:
        """Execute a tool and return result.

        Args:
            name: Tool name
            args: Tool arguments
            tool_use_id: ID from LLM's tool_use block

        Returns:
            Tool result in Anthropic format with enriched metadata
        """
        # Format tool announcement with friendly display
        announcement = self.formatter.format_announcement(name, args)
        console.print(f"[cyan]  → {announcement}[/cyan]")

        # Check approval if approval manager is configured
        if self.approval_manager:
            denial_result = self._check_approval(name, args, tool_use_id)
            if denial_result:
                return denial_result

        # Execute the tool
        try:
            return self._execute_and_format(name, args, tool_use_id)
        except Exception as e:
            return self._handle_error(name, args, tool_use_id, e)

    def _check_approval(
        self, name: str, args: dict, tool_use_id: str
    ) -> Optional[Dict[str, Any]]:
        """Check if tool execution is approved.

        Args:
            name: Tool name
            args: Tool arguments
            tool_use_id: Tool use ID

        Returns:
            Error result if denied, None if approved
        """
        try:
            risk_level = self.tool_registry.get_risk_level(name)
            # approval_manager is guaranteed non-None by caller check
            approved = self.approval_manager.should_approve(name, args, risk_level)  # type: ignore[union-attr]

            if not approved:
                console.print("[yellow]  ⚠ User denied tool execution[/yellow]")
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": "Tool execution denied by user",
                    "is_error": True,
                }
        except Exception as e:
            logger.error(
                f"Approval check failed for tool '{name}': {e}",
                exc_info=True,
            )
            console.print(f"[red]  ✗ Approval error: {e}[/red]")
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": f"Approval check failed: {str(e)}",
                "is_error": True,
            }

        return None

    def _execute_and_format(
        self, name: str, args: dict, tool_use_id: str
    ) -> Dict[str, Any]:
        """Execute tool and format result.

        Args:
            name: Tool name
            args: Tool arguments
            tool_use_id: Tool use ID

        Returns:
            Formatted tool result
        """
        logger.debug(f"Executing tool '{name}' with args: {args}")
        result = self.tool_registry.execute(name, args)
        console.print("[green]  ✓ Success[/green]")
        logger.info(f"Tool '{name}' executed successfully")

        # Enrich result based on tool type
        enriched_result = self._enrich_result(name, args, result)

        # Ensure content is a valid string
        if not isinstance(enriched_result, str):
            enriched_result = str(enriched_result)

        # Truncate extremely large results
        enriched_result = self._truncate_if_needed(name, enriched_result, len(result))

        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": enriched_result,
        }

    def _enrich_result(self, name: str, args: dict, result: str) -> str:
        """Enrich result with metadata.

        Args:
            name: Tool name
            args: Tool arguments
            result: Raw result

        Returns:
            Enriched result string
        """
        # Check if this is a background tool
        if self.background_manager and self.background_manager.is_background_tool(name):
            return self.background_manager.enrich_result(name, args, result)

        # Use standard formatter for other tools
        return self.formatter.enrich_result(name, args, result)

    def _truncate_if_needed(self, name: str, result: str, original_size: int) -> str:
        """Truncate result if it exceeds maximum size.

        Args:
            name: Tool name for logging
            result: Result to potentially truncate
            original_size: Original result size for logging

        Returns:
            Potentially truncated result
        """
        if len(result) > MAX_TOOL_RESULT_SIZE:
            truncated_size = MAX_TOOL_RESULT_SIZE - 500  # Leave room for message
            truncated = (
                f"{result[:truncated_size]}\n\n"
                f"... [TRUNCATED: {len(result) - truncated_size} more characters]"
            )
            logger.warning(
                f"Tool '{name}' result truncated from {original_size} "
                f"to {MAX_TOOL_RESULT_SIZE} chars"
            )
            return truncated
        return result

    def _handle_error(
        self, name: str, args: dict, tool_use_id: str, error: Exception
    ) -> Dict[str, Any]:
        """Handle tool execution error.

        Args:
            name: Tool name
            args: Tool arguments
            tool_use_id: Tool use ID
            error: The exception that occurred

        Returns:
            Error result dictionary
        """
        logger.error(f"Tool '{name}' execution failed: {error}", exc_info=True)
        logger.debug(f"Failed tool args: {args}")
        console.print(f"[red]  ✗ Error: {error}[/red]")

        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": f"Error: {str(error)}",
            "is_error": True,
        }
