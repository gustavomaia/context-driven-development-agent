"""Tool result formatting for Agent.

This module provides the ToolResultFormatter class that handles:
- Formatting tool execution announcements for console output
- Enriching tool results with metadata for better LLM understanding
- Path abbreviation for readability
"""

from pathlib import Path
from typing import Any


class ToolResultFormatter:
    """Formats tool announcements and enriches tool results.

    This class provides:
    - Human-readable tool announcements for console output
    - Enriched tool results with metadata (line counts, file info, etc.)
    - Path abbreviation for better readability

    Example:
        formatter = ToolResultFormatter()
        args = {"path": "src/main.py"}
        announcement = formatter.format_announcement("read_file", args)
        enriched = formatter.enrich_result("read_file", args, content)
    """

    @staticmethod
    def abbreviate_path(path: str, max_components: int = 3) -> str:
        """Abbreviate a file path to show only the last N components.

        Args:
            path: Full file path
            max_components: Maximum path components to show

        Returns:
            Abbreviated path string

        Example:
            abbreviate_path("/home/user/project/src/main.py", 3)
            # Returns: ".../project/src/main.py"
        """
        parts = Path(path).parts
        if len(parts) <= max_components:
            return path
        return ".../" + "/".join(parts[-max_components:])

    def format_announcement(self, tool_name: str, args: dict) -> str:
        """Format tool execution announcement for console.

        Makes tool announcements more readable by showing key info
        instead of raw dict arguments.

        Args:
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            Formatted announcement string
        """
        if tool_name == "read_file":
            path = args.get("path", "?")
            short_path = self.abbreviate_path(path)
            return f"📖 Reading: {short_path}"

        elif tool_name == "write_file":
            path = args.get("path", "?")
            short_path = self.abbreviate_path(path)
            return f"📝 Writing: {short_path}"

        elif tool_name == "edit_file":
            path = args.get("path", "?")
            short_path = self.abbreviate_path(path)
            return f"✏️  Editing: {short_path}"

        elif tool_name == "list_files":
            path = args.get("path", ".")
            short_path = self.abbreviate_path(path)
            return f"📁 Listing: {short_path}"

        elif tool_name == "glob_files":
            pattern = args.get("pattern", "?")
            return f"🔍 Finding files: {pattern}"

        elif tool_name == "grep_files":
            pattern = args.get("pattern", "?")
            file_pattern = args.get("file_pattern", "**/*")
            return f"🔎 Searching: '{pattern}' in {file_pattern}"

        elif tool_name == "run_bash":
            command = args.get("command", "?")
            if len(command) > 50:
                command = command[:47] + "..."
            return f"⚡ Running: {command}"

        elif tool_name == "git_status":
            return "📊 Git status"

        elif tool_name == "git_diff":
            path = args.get("file_path", "")
            if path:
                short_path = self.abbreviate_path(path)
                return f"📊 Git diff: {short_path}"
            return "📊 Git diff (all changes)"

        elif tool_name == "git_log":
            max_commits = args.get("max_commits", 10)
            return f"📊 Git log (last {max_commits} commits)"

        elif tool_name == "git_commit":
            message = args.get("message", "")
            short_msg = message[:30] + "..." if len(message) > 30 else message
            return f"📦 Git commit: {short_msg}"

        # Background tool announcements
        elif tool_name == "run_bash_background":
            command = args.get("command", "?")
            if len(command) > 50:
                command = command[:47] + "..."
            return f"🚀 Starting background process: {command}"

        elif tool_name == "get_background_status":
            process_id = args.get("process_id", "?")
            short_id = process_id[:12] if len(process_id) > 12 else process_id
            return f"📊 Checking background process: {short_id}..."

        elif tool_name == "interrupt_background_process":
            process_id = args.get("process_id", "?")
            short_id = process_id[:12] if len(process_id) > 12 else process_id
            return f"⏹ Interrupting background process: {short_id}..."

        elif tool_name == "get_background_output":
            process_id = args.get("process_id", "?")
            lines = args.get("lines", 50)
            short_id = process_id[:12] if len(process_id) > 12 else process_id
            return f"📄 Retrieving output from {short_id}... (last {lines} lines)"

        elif tool_name == "list_background_processes":
            return "📋 Listing all background processes"

        # Default: show tool name and args
        return f"Executing: {tool_name}({args})"

    def enrich_result(self, tool_name: str, args: dict, result: Any) -> str:
        """Add context to tool results for better LLM understanding.

        Enriches tool results with metadata like file paths, line counts,
        match statistics, etc.

        Args:
            tool_name: Name of the tool that was executed
            args: Arguments passed to the tool
            result: Raw result from tool execution

        Returns:
            Enriched result string with metadata
        """
        result_str = str(result)

        if tool_name == "read_file":
            path = args.get("path", "unknown")
            line_count = len(result_str.splitlines())
            char_count = len(result_str)
            return (
                f"File: {path}\n"
                f"Lines: {line_count} | Characters: {char_count}\n"
                f"{'─' * 60}\n"
                f"{result_str}"
            )

        elif tool_name == "write_file":
            path = args.get("path", "unknown")
            return f"📝 {result_str}\n\nFile written: {path}"

        elif tool_name == "edit_file":
            path = args.get("path", "unknown")
            return f"✏️  {result_str}\n\nFile edited: {path}"

        elif tool_name == "glob_files":
            lines = result_str.splitlines()
            if lines and "Found" in lines[0]:
                return result_str
            else:
                file_count = len([line for line in lines if line.strip()])
                return f"Glob search results ({file_count} files):\n{result_str}"

        elif tool_name == "grep_files":
            lines = result_str.splitlines()
            if lines and "Found" in lines[0]:
                return result_str
            else:
                match_count = len(
                    [line for line in lines if ":" in line and line.startswith(" ")]
                )
                return f"Search results ({match_count} matches):\n{result_str}"

        elif tool_name == "list_files":
            path = args.get("path", ".")
            items = result_str.splitlines()
            item_count = len(items)

            if "(empty directory)" in result_str:
                return f"Directory: {path}\nStatus: Empty"

            dirs = len([i for i in items if "📁" in i])
            files = len([i for i in items if "📄" in i])

            return (
                f"Directory: {path}\n"
                f"Items: {item_count} total ({dirs} directories, {files} files)\n"
                f"{'─' * 60}\n"
                f"{result_str}"
            )

        elif tool_name == "git_status":
            if "clean" in result_str.lower():
                return f"✓ {result_str}"
            elif "Not a git repository" in result_str:
                return f"⚠ {result_str}"
            else:
                change_count = len(result_str.splitlines()) - 1
                return f"Git repository status ({change_count} changes):\n{result_str}"

        elif tool_name == "git_diff":
            if "No changes" in result_str:
                return f"✓ {result_str}"
            else:
                lines = result_str.splitlines()
                additions = len([line for line in lines if line.startswith("+")])
                deletions = len([line for line in lines if line.startswith("-")])
                return (
                    f"Git diff (+{additions} additions, -{deletions} deletions):\n"
                    f"{result_str}"
                )

        elif tool_name == "run_bash":
            command = args.get("command", "unknown")
            lines = result_str.splitlines()
            line_count = len(lines)
            return (
                f"Command: {command}\n"
                f"Output ({line_count} lines):\n"
                f"{'─' * 60}\n"
                f"{result_str}"
            )

        # Default: return as-is for unknown tools
        return result_str
