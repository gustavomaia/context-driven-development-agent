"""Integration tests for Agent background tool support (Phase 4).

This module tests:
- Agent properly routes background tools through special handler
- Background tool announcements are formatted correctly
- Process tracking works across agent interactions
- Enhanced result formatting for background tools
- Agent maintains process context across conversation turns
"""

import unittest
from unittest.mock import Mock
from unittest.mock import patch

from cdd_agent.agent import BACKGROUND_TOOLS
from cdd_agent.agent import Agent
from cdd_agent.config import ProviderConfig
from cdd_agent.tools import ToolRegistry


class TestAgentBackgroundToolRouting(unittest.TestCase):
    """Test that agent properly routes background tools."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock provider config
        self.mock_provider_config = Mock(spec=ProviderConfig)
        self.mock_provider_config.provider_name = "anthropic"
        self.mock_provider_config.model_name = "claude-3-5-sonnet-20241022"
        self.mock_provider_config.api_key = "test-key"

        # Create real tool registry
        self.tool_registry = ToolRegistry()

        # Create agent
        self.agent = Agent(
            provider_config=self.mock_provider_config,
            tool_registry=self.tool_registry,
            enable_context=False,  # Disable context loading for tests
        )

    def test_background_tools_set_defined(self):
        """Test that BACKGROUND_TOOLS set is properly defined."""
        # Accept both set and frozenset
        self.assertTrue(isinstance(BACKGROUND_TOOLS, (set, frozenset)))
        self.assertIn("run_bash_background", BACKGROUND_TOOLS)
        self.assertIn("get_background_status", BACKGROUND_TOOLS)
        self.assertIn("interrupt_background_process", BACKGROUND_TOOLS)
        self.assertIn("get_background_output", BACKGROUND_TOOLS)
        self.assertIn("list_background_processes", BACKGROUND_TOOLS)

    def test_agent_has_background_infrastructure(self):
        """Test that agent has background process tracking."""
        self.assertTrue(hasattr(self.agent, "background_processes"))
        self.assertTrue(hasattr(self.agent, "background_process_counter"))
        self.assertTrue(hasattr(self.agent, "background_executor"))

        self.assertIsInstance(self.agent.background_processes, dict)
        self.assertEqual(self.agent.background_process_counter, 0)

    def test_execute_tool_routes_background_tools(self):
        """Test that _execute_tool routes background tools to special handler."""
        # Use a valid UUID format
        process_id = "f8793d68-fd26-4716-980e-8d389b59abf2"

        # Mock the tool execution
        with patch.object(self.agent.tool_registry, "execute") as mock_execute:
            mock_execute.return_value = (
                f"Background process started: {process_id}\n"
                "Command: echo 'test'\n"
                "Status: Running"
            )

            # Execute a background tool
            result = self.agent._execute_tool(
                "run_bash_background",
                {"command": "echo 'test'", "timeout": 300},
                "tool-123",
            )

        # Verify result structure
        self.assertEqual(result["type"], "tool_result")
        self.assertEqual(result["tool_use_id"], "tool-123")
        self.assertIn("content", result)
        self.assertFalse(result.get("is_error", False))

        # Verify process was registered
        self.assertEqual(len(self.agent.background_processes), 1)
        self.assertIn(process_id, self.agent.background_processes)

    def test_execute_tool_routes_regular_tools_normally(self):
        """Test that regular tools still use standard enrichment."""
        # Mock file read
        with patch.object(self.agent.tool_registry, "execute") as mock_execute:
            mock_execute.return_value = "file content here"

            result = self.agent._execute_tool(
                "read_file", {"path": "/test/file.txt"}, "tool-456"
            )

        # Verify result structure
        self.assertEqual(result["type"], "tool_result")
        self.assertEqual(result["tool_use_id"], "tool-456")
        self.assertIn("content", result)
        # Standard enrichment should add file path context
        self.assertIn("file content here", result["content"])


class TestAgentBackgroundToolAnnouncements(unittest.TestCase):
    """Test background tool announcement formatting."""

    def setUp(self):
        """Set up test fixtures."""
        mock_provider_config = Mock(spec=ProviderConfig)
        mock_provider_config.provider_name = "anthropic"
        mock_provider_config.model_name = "claude-3-5-sonnet-20241022"
        mock_provider_config.api_key = "test-key"

        self.agent = Agent(
            provider_config=mock_provider_config,
            tool_registry=ToolRegistry(),
            enable_context=False,
        )

    def test_run_bash_background_announcement(self):
        """Test run_bash_background announcement formatting."""
        announcement = self.agent._formatter.format_announcement(
            "run_bash_background",
            {"command": "pytest tests/ -v", "timeout": 300},
        )

        self.assertIn("🚀", announcement)
        self.assertIn("Starting background process", announcement)
        self.assertIn("pytest tests/ -v", announcement)

    def test_run_bash_background_long_command_truncated(self):
        """Test that long commands are truncated in announcements."""
        long_command = "echo " + "x" * 100
        announcement = self.agent._formatter.format_announcement(
            "run_bash_background", {"command": long_command, "timeout": 300}
        )

        self.assertIn("🚀", announcement)
        self.assertIn("...", announcement)
        self.assertLess(len(announcement), 100)

    def test_get_background_status_announcement(self):
        """Test get_background_status announcement formatting."""
        announcement = self.agent._formatter.format_announcement(
            "get_background_status", {"process_id": "abc123-def456-ghi789"}
        )

        self.assertIn("📊", announcement)
        self.assertIn("Checking background process", announcement)
        self.assertIn("abc123-def45", announcement)  # First 12 chars

    def test_interrupt_background_process_announcement(self):
        """Test interrupt_background_process announcement formatting."""
        announcement = self.agent._formatter.format_announcement(
            "interrupt_background_process", {"process_id": "test-process-123"}
        )

        self.assertIn("⏹", announcement)
        self.assertIn("Interrupting background process", announcement)
        self.assertIn("test-process", announcement)

    def test_get_background_output_announcement(self):
        """Test get_background_output announcement formatting."""
        announcement = self.agent._formatter.format_announcement(
            "get_background_output", {"process_id": "proc-123", "lines": 20}
        )

        self.assertIn("📄", announcement)
        self.assertIn("Retrieving output", announcement)
        self.assertIn("20 lines", announcement)

    def test_list_background_processes_announcement(self):
        """Test list_background_processes announcement formatting."""
        announcement = self.agent._formatter.format_announcement(
            "list_background_processes", {}
        )

        self.assertIn("📋", announcement)
        self.assertIn("Listing all background processes", announcement)


class TestAgentBackgroundProcessTracking(unittest.TestCase):
    """Test agent background process tracking across interactions."""

    def setUp(self):
        """Set up test fixtures."""
        mock_provider_config = Mock(spec=ProviderConfig)
        mock_provider_config.provider_name = "anthropic"
        mock_provider_config.model_name = "claude-3-5-sonnet-20241022"
        mock_provider_config.api_key = "test-key"

        self.agent = Agent(
            provider_config=mock_provider_config,
            tool_registry=ToolRegistry(),
            enable_context=False,
        )

    def test_register_background_process(self):
        """Test process registration."""
        process_id = "test-proc-123"
        command = "echo 'test command'"

        self.agent._register_background_process(process_id, command)

        # Verify registration
        self.assertIn(process_id, self.agent.background_processes)
        process_info = self.agent.background_processes[process_id]

        self.assertEqual(process_info["command"], command)
        self.assertEqual(process_info["process_id"], process_id)
        self.assertIn("start_time", process_info)
        self.assertIn("status", process_info)

    def test_get_background_process(self):
        """Test retrieving process information."""
        process_id = "test-proc-456"
        command = "pytest tests/"

        self.agent._register_background_process(process_id, command)

        # Retrieve process info
        process_info = self.agent._get_background_process(process_id)

        self.assertIsNotNone(process_info)
        self.assertEqual(process_info["process_id"], process_id)
        self.assertEqual(process_info["command"], command)

    def test_get_nonexistent_background_process(self):
        """Test retrieving non-existent process returns empty dict."""
        process_info = self.agent._get_background_process("nonexistent-123")
        self.assertEqual(process_info, {})

    def test_list_background_processes(self):
        """Test listing all background processes."""
        # Register multiple processes
        self.agent._register_background_process("proc-1", "echo 'one'")
        self.agent._register_background_process("proc-2", "echo 'two'")
        self.agent._register_background_process("proc-3", "echo 'three'")

        # List all processes
        processes = self.agent._list_background_processes()

        self.assertEqual(len(processes), 3)
        process_ids = [p["process_id"] for p in processes]
        self.assertIn("proc-1", process_ids)
        self.assertIn("proc-2", process_ids)
        self.assertIn("proc-3", process_ids)

    def test_background_process_counter_increments(self):
        """Test that process counter increments."""
        initial_count = self.agent.background_process_counter

        self.agent._register_background_process("proc-1", "echo 'test'")

        self.assertEqual(self.agent.background_process_counter, initial_count + 1)


class TestAgentBackgroundToolResultHandling(unittest.TestCase):
    """Test enhanced result formatting for background tools."""

    def setUp(self):
        """Set up test fixtures."""
        mock_provider_config = Mock(spec=ProviderConfig)
        mock_provider_config.provider_name = "anthropic"
        mock_provider_config.model_name = "claude-3-5-sonnet-20241022"
        mock_provider_config.api_key = "test-key"

        self.agent = Agent(
            provider_config=mock_provider_config,
            tool_registry=ToolRegistry(),
            enable_context=False,
        )

    def test_handle_background_tool_result_extracts_process_id(self):
        """Test that process ID is extracted from tool result."""
        process_id = "abc123de-f456-gh78-ij90-klmn12345678"  # Valid UUID format
        result = f"""Background process started: {process_id}
Command: pytest tests/ -v
Status: Running
Started: 2025-11-08 10:30:00
Use get_background_status() to check progress"""

        # Use the background manager's enrich_result method
        enriched = self.agent._background_manager.enrich_result(
            "run_bash_background",
            {"command": "pytest tests/ -v"},
            result,
        )

        # Verify process was registered
        self.assertEqual(len(self.agent.background_processes), 1)

        # Verify enriched result contains useful info
        self.assertIn(process_id[:12], enriched)  # Check for shortened ID
        self.assertIsInstance(enriched, str)

    def test_handle_background_status_result(self):
        """Test handling of get_background_status results."""
        # First register a process
        self.agent._register_background_process("proc-123", "echo 'test'")

        result = """Process ID: proc-123
Status: running
Runtime: 5.2 seconds
Output: 10 lines
Command: echo 'test'"""

        # Use the background manager's enrich_result method
        enriched = self.agent._background_manager.enrich_result(
            "get_background_status",
            {"process_id": "proc-123"},
            result,
        )

        self.assertIn("proc-123", enriched)
        self.assertIsInstance(enriched, str)

    def test_handle_background_tool_result_with_error(self):
        """Test handling of background tool errors."""
        result = "Error: Process not found: nonexistent-123"

        # Use the background manager's enrich_result method
        enriched = self.agent._background_manager.enrich_result(
            "get_background_status",
            {"process_id": "nonexistent-123"},
            result,
        )

        # Should still return the error message
        self.assertIn("Error", enriched)
        self.assertIsInstance(enriched, str)


class TestAgentBackgroundIntegration(unittest.TestCase):
    """Integration tests for complete background workflow."""

    def setUp(self):
        """Set up test fixtures."""
        mock_provider_config = Mock(spec=ProviderConfig)
        mock_provider_config.provider_name = "anthropic"
        mock_provider_config.model_name = "claude-3-5-sonnet-20241022"
        mock_provider_config.api_key = "test-key"

        self.agent = Agent(
            provider_config=mock_provider_config,
            tool_registry=ToolRegistry(),
            enable_context=False,
        )

    @patch("cdd_agent.tools.get_background_executor")
    def test_full_background_workflow(self, mock_get_executor):
        """Test complete workflow: start -> check -> get output -> interrupt."""
        # Mock background executor
        mock_executor = Mock()
        mock_process = Mock()
        # Use valid UUID format for process_id
        mock_process.process_id = "f8793d68-fd26-4716-980e-8d389b59abf2"
        mock_process.status.value = "running"
        mock_process.get_runtime.return_value = 5.0
        mock_process.output_lines = ["line 1", "line 2", "line 3"]

        mock_executor.execute_command.return_value = mock_process
        mock_executor.get_process.return_value = mock_process
        mock_executor.interrupt_process.return_value = True
        mock_get_executor.return_value = mock_executor

        # 1. Start background process
        with patch.object(self.agent.tool_registry, "execute") as mock_execute:
            mock_execute.return_value = (
                f"Background process started: {mock_process.process_id}\n"
                "Command: echo 'test'\n"
                "Status: Running"
            )

            self.agent._execute_tool(
                "run_bash_background", {"command": "echo 'test'"}, "tool-1"
            )

        # Verify process was registered
        self.assertEqual(len(self.agent.background_processes), 1)
        self.assertIn(mock_process.process_id, self.agent.background_processes)

        # 2. Check status
        with patch.object(self.agent.tool_registry, "execute") as mock_execute:
            mock_execute.return_value = (
                f"Process ID: {mock_process.process_id}\nStatus: running\nRuntime: 5.0s"
            )

            result2 = self.agent._execute_tool(
                "get_background_status",
                {"process_id": mock_process.process_id},
                "tool-2",
            )

        self.assertIn("running", result2["content"].lower())

        # 3. Get output
        with patch.object(self.agent.tool_registry, "execute") as mock_execute:
            mock_execute.return_value = "line 1\nline 2\nline 3"

            result3 = self.agent._execute_tool(
                "get_background_output",
                {"process_id": mock_process.process_id, "lines": 10},
                "tool-3",
            )

        self.assertIn("line", result3["content"])

        # 4. Interrupt
        with patch.object(self.agent.tool_registry, "execute") as mock_execute:
            mock_execute.return_value = (
                f"Process {mock_process.process_id} interrupted successfully"
            )

            result4 = self.agent._execute_tool(
                "interrupt_background_process",
                {"process_id": mock_process.process_id},
                "tool-4",
            )

        self.assertIn("interrupted", result4["content"].lower())


if __name__ == "__main__":
    unittest.main()
