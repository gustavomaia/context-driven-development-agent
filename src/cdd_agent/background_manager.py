"""Background process management for Agent.

This module provides the BackgroundProcessManager class that handles:
- Tracking background processes started by the agent
- Enriching background tool results with context
- Managing process lifecycle and state
"""

import re
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from .logging import get_logger


logger = get_logger("background_manager")


# Background tools that should be handled specially
BACKGROUND_TOOLS = frozenset(
    {
        "run_bash_background",
        "get_background_status",
        "interrupt_background_process",
        "get_background_output",
        "list_background_processes",
    }
)


class BackgroundProcessManager:
    """Manages background process tracking and result enrichment.

    This class is responsible for:
    - Registering background processes started by the agent
    - Tracking process state (id, command, start time, status)
    - Enriching tool results with process context
    - Providing process listing and lookup

    Example:
        manager = BackgroundProcessManager()
        manager.register("abc-123", "pytest tests/")
        info = manager.get("abc-123")
        enriched = manager.enrich_result("run_bash_background", args, result)
    """

    def __init__(self):
        """Initialize background process manager."""
        self._processes: Dict[str, Dict[str, Any]] = {}
        self._counter = 0

    @property
    def processes(self) -> Dict[str, Dict[str, Any]]:
        """Get all tracked processes.

        Returns:
            Dictionary mapping process IDs to process info
        """
        return self._processes

    @property
    def counter(self) -> int:
        """Get the process counter.

        Returns:
            Number of processes registered
        """
        return int(self._counter)

    def register(self, process_id: str, command: str) -> None:
        """Register a background process for tracking.

        Args:
            process_id: Unique identifier for the background process
            command: Command being executed
        """
        self._processes[process_id] = {
            "process_id": process_id,
            "command": command,
            "start_time": time.time(),
            "status": "starting",
            "args": {},
        }

        self._counter += 1
        logger.info(
            f"Registered background process: {process_id[:12]}... "
            f"for command: {command[:100]}"
        )

    def get(self, process_id: Optional[str]) -> Dict[str, Any]:
        """Get information about a background process.

        Args:
            process_id: Process identifier

        Returns:
            Process information dict or empty dict if not found
        """
        if process_id is None:
            return {}
        return self._processes.get(process_id, {})

    def list_all(self) -> List[Dict[str, Any]]:
        """List all tracked background processes.

        Returns:
            List of process information dictionaries
        """
        return list(self._processes.values())

    def is_background_tool(self, tool_name: str) -> bool:
        """Check if a tool is a background tool.

        Args:
            tool_name: Name of the tool

        Returns:
            True if tool is a background tool
        """
        return tool_name in BACKGROUND_TOOLS

    def enrich_result(self, tool_name: str, args: dict, result: str) -> str:
        """Enrich background tool results with context.

        Args:
            tool_name: Name of the background tool
            args: Arguments passed to the tool
            result: Result from tool execution

        Returns:
            Enhanced result string with process information
        """
        if tool_name == "run_bash_background":
            return self._enrich_run_result(args, result)

        elif tool_name in ("get_background_status", "get_background_output"):
            return self._enrich_status_or_output(tool_name, args, result)

        elif tool_name == "list_background_processes":
            return self._enrich_list_result(result)

        # Default: return as-is for unknown background tools
        return result

    def _enrich_run_result(self, args: dict, result: str) -> str:
        """Enrich run_bash_background result.

        Args:
            args: Tool arguments
            result: Raw result

        Returns:
            Enriched result string
        """
        command = args.get("command", "unknown")
        match = re.search(r"Background process started: (\w+-\w+-\w+-\w+-\w+)", result)

        if match:
            process_id = match.group(1)
            self.register(process_id, command)

            return (
                f"🚀 Background process started: {process_id[:12]}...\n"
                f"Command: {command}\n\n"
                f"{result}"
            )
        else:
            return (
                f"❌ Failed to start background process\n"
                f"Command: {command}\n\n"
                f"{result}"
            )

    def _enrich_status_or_output(self, tool_name: str, args: dict, result: str) -> str:
        """Enrich get_background_status or get_background_output result.

        Args:
            tool_name: Name of the tool
            args: Tool arguments
            result: Raw result

        Returns:
            Enriched result string
        """
        process_id = args.get("process_id")
        process_info = self.get(process_id)

        if not process_info:
            return result

        if tool_name == "get_background_status":
            runtime = time.time() - process_info.get("start_time", time.time())
            return (
                f"📊 Process Status: {process_info.get('process_id', 'unknown')}\n"
                f"Status: {process_info.get('status', 'unknown')}\n"
                f"Command: {process_info.get('command', 'unknown')}\n"
                f"Runtime: {runtime:.1f} s\n"
                f"{'─' * 40}\n{result}"
            )
        else:  # get_background_output
            return (
                f"📄 Process Output: {process_info.get('process_id', 'unknown')}\n"
                f"Command: {process_info.get('command', 'unknown')}\n"
                f"Status: {process_info.get('status', 'unknown')}\n"
                f"{'─' * 40}\n"
                f"{result}"
            )

    def _enrich_list_result(self, result: str) -> str:
        """Enrich list_background_processes result.

        Args:
            result: Raw result

        Returns:
            Enriched result string
        """
        # Import here to avoid circular imports
        from .background_executor import get_background_executor

        executor = get_background_executor()
        processes = executor.list_all_processes()

        if not processes:
            return result

        active_count = len(processes)
        running_count = len([p for p in processes if p.status.value == "running"])
        completed_count = len([p for p in processes if p.status.value == "completed"])

        return (
            f"📋 Background Processes Summary\n"
            f"Total: {active_count}\n"
            f"Running: {running_count}\n"
            f"Completed: {completed_count}\n"
            f"{'─' * 40}\n"
            f"{result}"
        )
