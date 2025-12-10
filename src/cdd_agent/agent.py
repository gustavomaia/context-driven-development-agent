"""Core agent conversation loop.

This module implements the main agentic loop:
1. User sends message
2. LLM processes with tool access
3. LLM decides to use tools or respond
4. If tool use: execute → feed back to LLM → loop
5. If done: return final response

The Agent class is provider-agnostic - it uses the ProviderStrategy pattern
to delegate LLM communication to specific provider implementations.
"""

from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional

from rich.console import Console

from .approval import ApprovalManager
from .background_manager import BACKGROUND_TOOLS  # noqa: F401 - re-exported
from .background_manager import BackgroundProcessManager
from .config import ProviderConfig
from .conversation import ConversationManager
from .logging import get_logger
from .prompts import PAIR_CODING_SYSTEM_PROMPT
from .prompts import REFLECTION_PROMPT
from .prompts import REFLECTION_SYSTEM_PROMPT
from .prompts import build_oauth_system_prompt
from .providers import create_provider
from .providers.factory import is_oauth_provider
from .tool_executor import ToolExecutor
from .tool_formatter import ToolResultFormatter
from .tools import ToolRegistry
from .utils.execution_state import ExecutionMode


if TYPE_CHECKING:
    from anthropic.types import Message


console = Console()
logger = get_logger("agent")


class Agent:
    """Main conversational agent with tool execution."""

    def __init__(
        self,
        provider_config: ProviderConfig,
        tool_registry: ToolRegistry,
        model_tier: str = "mid",
        max_iterations: int = 10,
        approval_manager: Optional[ApprovalManager] = None,
        enable_context: bool = True,
        execution_mode: ExecutionMode = ExecutionMode.NORMAL,
    ):
        """Initialize agent.

        Args:
            provider_config: Provider configuration
            tool_registry: Registry of available tools
            model_tier: Model tier to use (small/mid/big)
            max_iterations: Maximum conversation iterations
            approval_manager: Optional approval manager for tool execution safety
            enable_context: Whether to load hierarchical context files (default: True)
            execution_mode: Execution mode (NORMAL or PLAN for read-only)
        """
        self.provider_config = provider_config
        self.tool_registry = tool_registry
        self.model_tier = model_tier
        self.max_iterations = max_iterations
        self.approval_manager = approval_manager
        self.enable_context = enable_context
        self.execution_mode = execution_mode

        # Create provider using Strategy pattern
        # Provider handles all LLM communication (OAuth, API key, etc.)
        self._provider = create_provider(
            provider_config=provider_config,
            model_tier=model_tier,
        )

        # Create conversation manager (handles history and context)
        self._conversation = ConversationManager(enable_context=enable_context)

        # Create tool result formatter
        self._formatter = ToolResultFormatter()

        # Build system prompt (project context will be injected into first message)
        # For OAuth: system prompt must be an array with Claude Code header
        # Check if this provider config uses OAuth (has oauth tokens set)
        self._uses_oauth = getattr(provider_config, "oauth", None) is not None
        self.system_prompt = PAIR_CODING_SYSTEM_PROMPT

        # Create background process manager
        self._background_manager = BackgroundProcessManager()

        # Create tool executor (handles execution, approval, formatting)
        self._tool_executor = ToolExecutor(
            tool_registry=tool_registry,
            formatter=self._formatter,
            approval_manager=approval_manager,
            background_manager=self._background_manager,
        )

        # Create background executor reference for easy access
        from .background_executor import get_background_executor

        self.background_executor = get_background_executor

    def _get_system_prompt(self, custom_prompt: Optional[str] = None) -> Any:
        """Get system prompt in correct format for the provider.

        For OAuth providers, returns an array with Claude Code header as first block.
        For other providers, returns the prompt as a string.

        Args:
            custom_prompt: Optional custom prompt to use instead of default

        Returns:
            System prompt in correct format (str or list of content blocks)
        """
        prompt = custom_prompt or self.system_prompt
        if self._uses_oauth:
            return build_oauth_system_prompt(prompt)
        return prompt

    @property
    def client(self):
        """Get the underlying LLM client from the provider.

        This property provides backward compatibility for code that
        accesses the client directly. The client is lazily initialized
        by the provider.

        Returns:
            The provider's LLM client
        """
        return self._provider.client

    @property
    def messages(self) -> List[Dict[str, Any]]:
        """Get conversation messages (backward compatibility).

        Returns:
            List of message dictionaries
        """
        return self._conversation.messages

    @messages.setter
    def messages(self, value: List[Dict[str, Any]]) -> None:
        """Set conversation messages (backward compatibility)."""
        self._conversation.messages = value

    @property
    def project_context(self) -> str:
        """Get project context (backward compatibility).

        Returns:
            Project context string
        """
        return self._conversation.project_context

    @property
    def context_loader(self):
        """Get context loader (backward compatibility).

        Returns:
            ContextLoader instance or None
        """
        return self._conversation.context_loader

    def load_project_context(self) -> str:
        """Load hierarchical context files.

        Delegates to ConversationManager.

        Returns:
            Merged context string, or fallback message if no context found
        """
        return self._conversation._load_project_context()

    @property
    def background_processes(self) -> Dict[str, Dict[str, Any]]:
        """Get background processes (backward compatibility).

        Returns:
            Dictionary of tracked background processes
        """
        return self._background_manager.processes

    @property
    def background_process_counter(self) -> int:
        """Get background process counter (backward compatibility).

        Returns:
            Number of background processes registered
        """
        return self._background_manager.counter

    def run(self, user_message: str, system_prompt: Optional[str] = None) -> str:
        """Run conversation with user message.

        This is the main agentic loop:
        - Send message to LLM with tools
        - If LLM wants to use tools: execute them
        - Feed tool results back to LLM
        - Repeat until LLM is done

        Args:
            user_message: User's input message
            system_prompt: Optional system prompt for context

        Returns:
            Final text response from LLM

        Raises:
            RuntimeError: If max iterations reached
        """
        # Inject project context into first message only (not into system prompt)
        # This prevents repeating 8KB+ of context on every API call
        if len(self.messages) == 0 and self.project_context:
            enhanced_message = f"{self.project_context}\n\n{'─' * 80}\n\n{user_message}"
        else:
            enhanced_message = user_message

        # Add user message to history
        self.messages.append({"role": "user", "content": enhanced_message})

        # Get model name from tier
        model = self.provider_config.get_model(self.model_tier)

        # Agentic loop
        for iteration in range(self.max_iterations):
            console.print(
                f"[dim]Iteration {iteration + 1}/{self.max_iterations}...[/dim]"
            )

            # Manage context window before each LLM call
            self._conversation.manage_context_window()

            # Filter tools based on execution mode
            read_only = self.execution_mode.is_read_only()

            # Log API request details for debugging
            logger.debug(
                f"API request: model={model} "
                f"messages_count={len(self.messages)} "
                f"tools_count={len(self.tool_registry.get_schemas(read_only=read_only))}"
                f"execution_mode={self.execution_mode.value}"
            )

            # Call LLM with tools via provider
            try:
                # When using OAuth, exclude risk_level field (Anthropic
                # OAuth API rejects custom fields)
                include_risk = not is_oauth_provider(self._provider)

                response = self._provider.create_message(
                    messages=self.messages,
                    tools=self.tool_registry.get_schemas(
                        include_risk_level=include_risk, read_only=read_only
                    ),
                    system=self._get_system_prompt(system_prompt),
                    max_tokens=4096,
                )
                logger.debug(f"API response: stop_reason={response.stop_reason}")
            except Exception as e:
                logger.error(
                    f"API call failed: {e}",
                    exc_info=True,
                )
                logger.debug(f"Messages at time of error: {self.messages}")
                raise

            # Check stop reason
            if response.stop_reason == "end_turn":
                # LLM is done, extract text response
                return self._extract_text(response)

            elif response.stop_reason == "tool_use":
                # LLM wants to use tools
                console.print("[cyan]🔧 Agent using tools...[/cyan]")

                # Add assistant's response to history
                self.messages.append({"role": "assistant", "content": response.content})

                # Execute all tool calls
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = self._execute_tool(block.name, block.input, block.id)
                        tool_results.append(result)

                # Add tool results to history
                self.messages.append({"role": "user", "content": tool_results})

                # Loop continues - LLM will process tool results

            elif response.stop_reason == "max_tokens":
                console.print(
                    "[yellow]⚠ Response truncated (max tokens reached)[/yellow]"
                )
                return self._extract_text(response)

            elif response.stop_reason is None:
                # Some providers (like MiniMax M2) return None as stop_reason
                # This is valid - treat it as end_turn
                logger.debug("Stop reason is None, treating as end_turn")
                return self._extract_text(response)

            else:
                # Log unexpected stop reasons but don't show warning to user
                logger.warning(
                    f"Unexpected stop reason: {response.stop_reason}, "
                    "treating as end_turn"
                )
                return self._extract_text(response)

        raise RuntimeError(
            f"Max iterations ({self.max_iterations}) reached without completion"
        )

    def _execute_tool(self, name: str, args: dict, tool_use_id: str) -> dict:
        """Execute a tool and return result.

        Delegates to ToolExecutor for execution, approval, and formatting.

        Args:
            name: Tool name
            args: Tool arguments
            tool_use_id: ID from LLM's tool_use block

        Returns:
            Tool result in Anthropic format with enriched metadata
        """
        return self._tool_executor.execute(name, args, tool_use_id)

    def _register_background_process(self, process_id: str, command: str) -> None:
        """Register a background process (backward compatibility).

        Delegates to BackgroundProcessManager.
        """
        self._background_manager.register(process_id, command)

    def _get_background_process(self, process_id: Optional[str]) -> Dict[str, Any]:
        """Get background process info (backward compatibility).

        Delegates to BackgroundProcessManager.
        """
        return self._background_manager.get(process_id)

    def _list_background_processes(self) -> List[Dict[str, Any]]:
        """List background processes (backward compatibility).

        Delegates to BackgroundProcessManager.
        """
        return self._background_manager.list_all()

    def compact(self):
        """Manually trigger conversation compaction (like Claude Code's /compact).

        This is the public API for manual compaction that can be called
        from slash commands or user requests.

        Returns:
            True if compaction was performed, False if too short
        """
        return self._conversation.compact()

    def _extract_text(self, response: "Message") -> str:
        """Extract text content from response.

        Args:
            response: Anthropic API response

        Returns:
            Text content as string
        """
        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)

        return "\n".join(text_parts) if text_parts else ""

    def stream(
        self, user_message: str, system_prompt: Optional[str] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream conversation with user message.

        Similar to run() but yields chunks as they arrive for real-time display.

        Yields:
            Dict with 'type' and data:
            - {'type': 'text', 'content': str} - Text chunk from LLM
            - {'type': 'tool_use', 'name': str, 'args': dict} - Tool being called
            - {'type': 'tool_result', 'name': str, 'result': str} - Tool result
            - {'type': 'thinking', 'content': str} - Status messages
            - {'type': 'error', 'content': str} - Error messages

        Args:
            user_message: User's input message
            system_prompt: Optional system prompt for context
        """
        # Inject project context into first message only (not into system prompt)
        # This prevents repeating 8KB+ of context on every API call
        if len(self.messages) == 0 and self.project_context:
            enhanced_message = f"{self.project_context}\n\n{'─' * 80}\n\n{user_message}"
        else:
            enhanced_message = user_message

        # Add user message to history
        self.messages.append({"role": "user", "content": enhanced_message})

        # Get model name from tier
        model = self.provider_config.get_model(self.model_tier)

        # Agentic loop
        for iteration in range(self.max_iterations):
            yield {
                "type": "thinking",
                "content": f"Iteration {iteration + 1}/{self.max_iterations}",
            }

            # Manage context window before each LLM call
            self._conversation.manage_context_window()

            # Filter tools based on execution mode
            read_only = self.execution_mode.is_read_only()

            # Log API request details for debugging
            logger.debug(
                f"Streaming API request: model={model} "
                f"messages_count={len(self.messages)} "
                f"tools_count={len(self.tool_registry.get_schemas(read_only=read_only))}"
                f"execution_mode={self.execution_mode.value}"
            )

            # Stream LLM response
            try:
                # When using OAuth, exclude risk_level field (Anthropic
                # OAuth API rejects custom fields)
                include_risk = not is_oauth_provider(self._provider)

                with self.client.messages.stream(
                    model=model,
                    max_tokens=4096,
                    messages=self.messages,
                    tools=self.tool_registry.get_schemas(
                        include_risk_level=include_risk, read_only=read_only
                    ),
                    system=self._get_system_prompt(system_prompt),
                ) as stream:
                    # Accumulate response
                    accumulated_text = []
                    accumulated_tool_uses = []

                    for event in stream:
                        # Text delta - stream it immediately
                        if event.type == "content_block_delta":
                            if hasattr(event.delta, "text"):
                                chunk = event.delta.text
                                accumulated_text.append(chunk)
                                yield {"type": "text", "content": chunk}

                        # Tool use - accumulate and announce
                        elif event.type == "content_block_start":
                            if hasattr(event.content_block, "type"):
                                if event.content_block.type == "tool_use":
                                    tool_use = {
                                        "id": event.content_block.id,
                                        "name": event.content_block.name,
                                        "input": {},
                                    }
                                    accumulated_tool_uses.append(tool_use)
                                    yield {
                                        "type": "tool_use",
                                        "name": event.content_block.name,
                                    }

                        # Tool input delta - accumulate
                        elif event.type == "content_block_delta":
                            if hasattr(event.delta, "partial_json"):
                                # Update the last tool use input (streaming JSON)
                                if accumulated_tool_uses:
                                    # We'll get the full input at message_stop
                                    pass

                    # Get final message
                    final_message = stream.get_final_message()
                    logger.debug(
                        f"Streaming API response: "
                        f"stop_reason={final_message.stop_reason}"
                    )

                    # Add assistant response to history
                    self.messages.append(
                        {"role": "assistant", "content": final_message.content}
                    )

                    # Check stop reason
                    if final_message.stop_reason == "end_turn":
                        # Done!
                        return

                    elif final_message.stop_reason == "tool_use":
                        # Execute tools
                        tool_results = []
                        for block in final_message.content:
                            if block.type == "tool_use":
                                # Execute and yield result
                                result = self._execute_tool(
                                    block.name, block.input, block.id
                                )
                                tool_results.append(result)

                                yield {
                                    "type": "tool_result",
                                    "name": block.name,
                                    "result": result.get("content", ""),
                                    "is_error": result.get("is_error", False),
                                }

                        # Add tool results to history
                        self.messages.append({"role": "user", "content": tool_results})

                        # Log tool results for debugging
                        logger.debug(
                            f"Added {len(tool_results)} tool result(s) "
                            "to conversation history"
                        )

                        # Continue loop - LLM will process tool results

                    elif final_message.stop_reason == "max_tokens":
                        yield {
                            "type": "error",
                            "content": "Response truncated (max tokens reached)",
                        }
                        return

                    elif final_message.stop_reason is None:
                        # Some providers (like MiniMax M2) return None as stop_reason
                        # This is valid - treat it as end_turn
                        logger.debug("Stop reason is None, treating as end_turn")
                        return

                    else:
                        # Log unexpected stop reasons but don't treat as error
                        logger.warning(
                            f"Unexpected stop reason: {final_message.stop_reason} "
                            "treating as end_turn"
                        )
                        return
            except Exception as e:
                # Check if this is an overloaded error (529)
                error_message = str(e)
                is_overloaded = (
                    "overloaded" in error_message.lower() or "529" in error_message
                )

                logger.error(
                    f"Streaming API call failed: {e}",
                    exc_info=True,
                )
                logger.debug(f"Messages at time of error: {self.messages}")

                # Provide helpful error message
                if is_overloaded:
                    yield {
                        "type": "error",
                        "content": (
                            "⚠️ Anthropic API is temporarily overloaded "
                            "(all 5 retry attempts failed).\n\n"
                            "This is a service-level issue, not related to your "
                            "request size.\n"
                            "Please try again in a few moments."
                        ),
                    }
                else:
                    yield {
                        "type": "error",
                        "content": f"API error: {error_message}",
                    }
                return

        # Max iterations reached
        yield {
            "type": "error",
            "content": f"Max iterations ({self.max_iterations}) reached",
        }

    def run_with_reflection(
        self, user_message: str, system_prompt: Optional[str] = None
    ) -> str:
        """Run conversation with post-execution reflection summary.

        Similar to run() but adds an optional reflection summary at the end
        if the task involved significant tool usage. The reflection provides:
        - What was accomplished
        - Files that were modified
        - Potential issues or areas needing attention
        - Suggested next steps

        Args:
            user_message: User's input message
            system_prompt: Optional system prompt for context

        Returns:
            Final text response with optional reflection summary appended
        """
        # Execute normal agentic loop
        response = self.run(user_message, system_prompt)

        # After completion, check if reflection would be valuable
        if self._should_reflect():
            reflection = self._get_reflection()
            return f"{response}\n\n---\n## Summary\n{reflection}"

        return response

    def _should_reflect(self) -> bool:
        """Determine if reflection is needed.

        Reflection is useful when the agent has used multiple tools,
        indicating a complex task that would benefit from a summary.

        Returns:
            True if tools were used extensively (>2 tool calls)
        """
        # Count how many times tools were used
        tool_count = sum(
            1
            for msg in self.messages
            if msg.get("role") == "assistant"
            and any(
                b.get("type") == "tool_use"
                for b in (
                    msg.get("content", [])
                    if isinstance(msg.get("content"), list)
                    else []
                )
            )
        )

        # Reflect if we executed tools multiple times
        return tool_count > 2

    def _get_reflection(self) -> str:
        """Ask LLM to reflect on what was accomplished.

        Sends a reflection prompt to the LLM asking it to summarize
        the work that was just completed.

        Returns:
            Brief summary of accomplishments, files modified, and next steps
        """
        # Make a quick non-streaming call for reflection
        self.messages.append({"role": "user", "content": REFLECTION_PROMPT})

        model = self.provider_config.get_model(self.model_tier)

        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=500,
                messages=self.messages,
                system=REFLECTION_SYSTEM_PROMPT,
            )

            return self._extract_text(response)
        except Exception as e:
            # If reflection fails, just skip it
            console.print(f"[yellow]⚠ Reflection skipped: {e}[/yellow]")
            return ""

    def clear_history(self):
        """Clear conversation history."""
        self._conversation.clear()

    def set_execution_mode(self, mode: ExecutionMode):
        """Set execution mode (for runtime toggling).

        Args:
            mode: ExecutionMode (NORMAL or PLAN)
        """
        self.execution_mode = mode
        logger.debug(f"Execution mode changed to: {mode.value}")


class SimpleAgent:
    """Simplified agent for quick testing (no tool use)."""

    def __init__(self, provider_config: ProviderConfig, model_tier: str = "mid"):
        """Initialize simple agent.

        Args:
            provider_config: Provider configuration
            model_tier: Model tier to use
        """
        self.provider_config = provider_config
        self.model_tier = model_tier

        # Create provider using Strategy pattern
        self._provider = create_provider(
            provider_config=provider_config,
            model_tier=model_tier,
        )

        # Check if this provider uses OAuth
        self._uses_oauth = getattr(provider_config, "oauth", None) is not None

    @property
    def client(self):
        """Get the underlying LLM client from the provider.

        Returns:
            The provider's LLM client
        """
        return self._provider.client

    def _get_system_prompt(self, custom_prompt: Optional[str] = None) -> Any:
        """Get system prompt in correct format for the provider.

        For OAuth providers, returns an array with Claude Code header as first block.
        For other providers, returns the prompt as a string.

        Args:
            custom_prompt: Optional custom prompt to use

        Returns:
            System prompt in correct format (str or list of content blocks)
        """
        prompt = custom_prompt or "You are a helpful assistant."
        if self._uses_oauth:
            return build_oauth_system_prompt(prompt)
        return prompt

    def run(self, user_message: str, system_prompt: Optional[str] = None) -> str:
        """Run a simple conversation without tools.

        Args:
            user_message: User's message
            system_prompt: Optional system prompt

        Returns:
            LLM response
        """
        response = self._provider.create_message(
            messages=[{"role": "user", "content": user_message}],
            tools=[],
            system=self._get_system_prompt(system_prompt),
            max_tokens=1024,
        )

        # Extract text
        for block in response.content:
            if block.type == "text":
                return str(block.text)

        return ""
