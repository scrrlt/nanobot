"""Agent loop: the core processing engine."""

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, cast

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.router import classify_intent
from nanobot.agent.safety import OscillationDetector
from nanobot.agent.schemas import DraftingPhase
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.base import STATUS_FAILED
from nanobot.agent.tools.filesystem import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.planning import UpdatePlanTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.agent.tools.cron import CronTool
from nanobot.config.schema import ExecToolConfig
from nanobot.cron.service import CronService

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import SessionManager
from nanobot.utils.parsing import parse_llm_json


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    # Maximum number of cached oscillation detectors to retain. This is a
    # conservative default sized for typical interactive workloads; tuneable via
    # subclassing or future configuration if deployments require broader fan-out.
    MAX_OSCILLATION_DETECTORS = 100

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        fast_model: str | None = None,
        max_iterations: int = 20,
        brave_api_key: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
    ):
        """
        Initialize the AgentLoop.

        Args:
            bus: The message bus for communication.
            provider: The LLM provider.
            workspace: The agent's workspace directory.
            model: The primary LLM model to use.
            fast_model: A faster, cheaper model for routing and planning. If not
                provided, the primary model is used.
            max_iterations: Max tool execution iterations per message.
            brave_api_key: API key for Brave Search.
            exec_config: Configuration for the shell execution tool.
            cron_service: The cron service for scheduled tasks.
            restrict_to_workspace: Restrict file operations to workspace.
            session_manager: The session manager.
        """
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.fast_model = fast_model or self.model
        self.max_iterations = max_iterations
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        
        self.oscillation_detectors: OrderedDict[str, OscillationDetector] = (
            OrderedDict()
        )

        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        # Tracks invalid 'force' attempts per session for basic auditing
        self._force_attempts: dict[str, int] = {}
        self._register_default_tools()

    @staticmethod
    def _format_plan(analysis: str | None, strategy: list[str]) -> str:
        """Create a standard representation for the operational plan."""
        goal_text = analysis.strip() if analysis else "N/A"
        steps = strategy if strategy else ["No steps defined."]
        steps_text = "\n".join(f"- {step}" for step in steps)
        return (
            "## OPERATIONAL PLAN (MANAGED VIA update_plan)\n"
            f"GOAL: {goal_text}\n"
            "STEPS:\n"
            f"{steps_text}"
        )

    @staticmethod
    def _normalize_strategy(raw_strategy: Any) -> list[str]:
        """Normalize strategy input to a list of strings."""
        if isinstance(raw_strategy, list):
            sequence = cast(list[Any], raw_strategy)
        elif raw_strategy is None:
            sequence = []
        else:
            sequence = [raw_strategy]

        return [str(step) for step in sequence]

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools (restrict to workspace if configured)
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        self.tools.register(ReadFileTool(allowed_dir=allowed_dir))
        self.tools.register(WriteFileTool(allowed_dir=allowed_dir))
        self.tools.register(EditFileTool(allowed_dir=allowed_dir))
        self.tools.register(ListDirTool(allowed_dir=allowed_dir))

        # Shell tool
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ))

        # Web tools
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())

        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)

        # Spawn tool (for subagents)
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool)

        # Planning tool
        self.tools.register(UpdatePlanTool())
        
        # Cron tool (for scheduling)
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        if self._running:
            logger.warning("Agent loop already running")
            return

        self._running = True
        logger.info("Agent loop started")

        try:
            while self._running:
                inbound = await self.bus.consume_inbound()
                if inbound is None:
                    logger.info("Agent loop received shutdown signal")
                    break

                try:
                    response = await self._process_message(inbound)
                    if response:
                        await self.bus.publish_outbound(response)
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"Error processing message: {exc}")
                    await self.bus.publish_outbound(
                        OutboundMessage(
                            channel=inbound.channel,
                            chat_id=inbound.chat_id,
                            content=("Sorry, I encountered an error: " f"{str(exc)}"),
                        )
                    )
        finally:
            self._running = False
            logger.info("Agent loop stopped")

    def stop(self) -> None:
        """Stop the agent loop."""
        if not self._running:
            self.bus.stop()
            logger.info("Agent loop stopping")
            return

        self._running = False
        self.bus.stop()
        logger.info("Agent loop stopping")

    async def _process_message(self, msg: InboundMessage) -> OutboundMessage | None:
        # Handle system messages (subagent announces)
        if msg.channel == "system":
            return await self._process_system_message(msg)

        session = self.sessions.get_or_create(msg.session_key)
        
        # Log preview
        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}: {preview}")

        # Update tool contexts
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(msg.channel, msg.chat_id)
        
        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(msg.channel, msg.chat_id)
        
        cron_tool = self.tools.get("cron")
        if isinstance(cron_tool, CronTool):
            cron_tool.set_context(msg.channel, msg.chat_id)

        if msg.session_key not in self.oscillation_detectors:
            # Clean up old detectors if at capacity
            if len(self.oscillation_detectors) >= self.MAX_OSCILLATION_DETECTORS:
                # Remove least recently used detector
                self.oscillation_detectors.popitem(last=False)

            self.oscillation_detectors[msg.session_key] = OscillationDetector(
                self.workspace
            )
        else:
            # Move to end (most recently used)
            self.oscillation_detectors.move_to_end(msg.session_key)

        safety = self.oscillation_detectors[msg.session_key]

        # --- PHASE 1: ROUTING & DRAFTING ---
        active_plan = None
        plan: dict[str, Any] = {}
        try:
            is_direct = await classify_intent(
                msg.content, self.provider, self.fast_model
            )
        except Exception as err:
            logger.error(f"Error in classify_intent: {err}")
            is_direct = False  # Safe fallback - treat as non-direct intent

        if is_direct:
            try:
                # Detached Planning Call
                draft = await self.provider.chat(
                    messages=[{"role": "user", "content": msg.content}],
                    response_format={
                        "type": "json_object",
                        "schema": DraftingPhase.model_json_schema(),
                    },
                    model=self.fast_model,  # Use fast model for planning phase
                )
                if draft and draft.content:
                    parsed_plan = parse_llm_json(draft.content)
                    if isinstance(parsed_plan, dict):
                        plan = cast(dict[str, Any], parsed_plan)

                    strategy_steps = self._normalize_strategy(plan.get("strategy"))
                    plan["strategy"] = strategy_steps
                    analysis_value = plan.get("analysis")
                    analysis_text = (
                        analysis_value if isinstance(analysis_value, str) else None
                    )
                    active_plan = self._format_plan(analysis_text, strategy_steps)
            except Exception as e:
                logger.warning(f"Drafting failed: {e}")

        # --- PHASE 2: EXECUTION ---
        iteration = 0

        # Build the base messages, including the managed plan (if any), once.
        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=msg.content,
            media=msg.media if msg.media else None,
            system_suffix=active_plan,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )
        final_content = None

        while iteration < self.max_iterations:
            iteration += 1

            response = await self.provider.chat(
                messages,
                tools=self.tools.get_definitions(),
                model=self.model,
            )

            if response.has_tool_calls:
                # Add assistant message with tool calls
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)  # Must be JSON string
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )

                for tool in response.tool_calls:
                    # --- Plan Update Interceptor ---
                    if tool.name == "update_plan":
                        strategy_steps = self._normalize_strategy(
                            tool.arguments.get("strategy")
                        )
                        plan["strategy"] = strategy_steps

                        new_analysis = tool.arguments.get("analysis")
                        if isinstance(new_analysis, str) and new_analysis.strip():
                            plan["analysis"] = new_analysis.strip()

                        analysis_value = plan.get("analysis")
                        analysis_text = (
                            analysis_value if isinstance(analysis_value, str) else None
                        )

                        active_plan = self._format_plan(
                            analysis_text, plan.get("strategy", [])
                        )
                        # Update system suffix in existing messages instead of rebuilding
                        messages = self.context.update_system_suffix(
                            messages, active_plan
                        )
                        # Add the tool result for the update_plan call
                        messages = self.context.add_tool_result(
                            messages, tool.id, tool.name, "Plan updated successfully."
                        )
                        continue

                    # --- SAFETY INTERCEPTOR & FORCE HANDLING ---
                    is_forced = bool(tool.arguments.get("force", False))
                    reason = tool.arguments.get("reason")
                    safety_err = safety.check(tool.name, tool.arguments)

                    # Validate 'force' usage: require non-empty 'reason' (len>=10)
                    if is_forced:
                        if not isinstance(reason, str) or len(reason.strip()) < 10:
                            count = self._force_attempts.get(msg.session_key, 0) + 1
                            self._force_attempts[msg.session_key] = count
                            logger.warning(
                                "Force attempted without sufficient reason (session=%s tool=%s attempt=%d)",
                                msg.session_key,
                                tool.name,
                                count,
                            )
                            result_text = (
                                "Error: 'force' requires a non-empty 'reason' explanation "
                                "(min 10 characters). Request blocked."
                            )
                            messages = self.context.add_tool_result(
                                messages, tool.id, tool.name, result_text
                            )
                            continue
                        # Log approved force usage (truncate reason for logs)
                        safe_reason = " ".join(reason.splitlines())[:200]
                        logger.info(
                            "Force used with reason in session %s tool %s: %s",
                            msg.session_key,
                            tool.name,
                            safe_reason,
                        )

                    if safety_err and not is_forced:
                        # BLOCK ACTION
                        result_text = (
                            f"{safety_err}\n\n"
                            "ESCAPE HATCH: Retry with 'force': true and a 'reason' "
                            "if you are certain this is necessary."
                        )
                        messages = self.context.add_tool_result(messages, tool.id, tool.name, result_text)
                        continue

                    # --- ACTION EXECUTION (exception-safe) ---
                    args_str = json.dumps(tool.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tool.name}({args_str[:200]})")
                    try:
                        execution = await self.tools.execute(tool.name, tool.arguments)
                        result_text = execution.message if hasattr(execution, 'message') else str(execution)
                    except Exception as exc:
                        sanitized = f"{type(exc).__name__}: {str(exc)[:200]}"
                        logger.error(
                            "Tool execution error for session %s tool %s: %s",
                            msg.session_key,
                            tool.name,
                            sanitized,
                        )
                        result_text = f"{STATUS_FAILED} - Tool execution error: {type(exc).__name__}"

                    # --- REFLECTION TRIGGER ---
                    if result_text.startswith(STATUS_FAILED):
                        result_text += (
                            "\n\n[SYSTEM INTERVENTION]\n"
                            "PROTOCOL: STOP. Do not retry identical command.\n"
                            "ACTION: Analyze failure -> Update Plan -> Retry."
                        )

                    messages = self.context.add_tool_result(
                        messages, tool.id, tool.name, result_text
                    )
            else:
                final_content = response.content
                break

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        # Log response preview
        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info(f"Response to {msg.channel}:{msg.sender_id}: {preview}")

        # Save to session
        session.add_message("user", msg.content)
        session.add_message("assistant", final_content)
        self.sessions.save(session)

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=msg.metadata or {},  # Pass through for channel-specific needs (e.g. Slack thread_ts)
        )

    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).

        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")

        # Parse origin from chat_id (format: "channel:chat_id")
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            # Fallback
            origin_channel = "cli"
            origin_chat_id = msg.chat_id

        # Use the origin session for context
        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)

        # Update tool contexts
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(origin_channel, origin_chat_id)

        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(origin_channel, origin_chat_id)

        cron_tool = self.tools.get("cron")
        if isinstance(cron_tool, CronTool):
            cron_tool.set_context(origin_channel, origin_chat_id)

        # Build messages with the announce content
        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=msg.content,
            channel=origin_channel,
            chat_id=origin_chat_id,
        )

        # Agent loop (limited for announce handling)
        iteration = 0
        final_content = None

        while iteration < self.max_iterations:
            iteration += 1

            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model
            )

            if response.has_tool_calls:
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )

                for tool_call in response.tool_calls:
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")
                    
                    execution = await self.tools.execute(
                        tool_call.name, tool_call.arguments
                    )
                    
                    result_text = execution.message if hasattr(execution, 'message') else str(execution)
                    
                    messages = self.context.add_tool_result(
                        messages,
                        tool_call.id,
                        tool_call.name,
                        result_text,
                    )
            else:
                final_content = response.content
                break

        if final_content is None:
            final_content = "Background task completed."

        # Save to session (mark as system message in history)
        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message("assistant", final_content)
        self.sessions.save(session)

        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=final_content
        )

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
    ) -> str:
        """
        Process a message directly (for CLI or cron usage).

        Args:
            content: The message content.
            session_key: Session identifier.
            channel: Source channel (for context).
            chat_id: Source chat ID (for context).

        Returns:
            The agent's response.
        """
        
        if session_key and not (channel and chat_id != "direct"):
            if ":" in session_key:
                channel, chat_id = session_key.split(":", 1)
            else:
                chat_id = session_key

        msg = InboundMessage(
            channel=channel, sender_id="user", chat_id=chat_id, content=content
        )

        try:
            response = await self._process_message(msg)
        except Exception as err:
            logger.error(f"Error processing direct message: {err}")
            return f"Error processing message: {err}"

        return response.content if response else ""
