"""Agent loop: the core processing engine."""

import asyncio
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
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import ExecToolConfig
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
        """
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.fast_model = fast_model or self.model
        self.max_iterations = max_iterations
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.oscillation_detectors: OrderedDict[str, OscillationDetector] = (
            OrderedDict()
        )

        self.context = ContextBuilder(workspace)
        self.sessions = SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
        )

        self._running = False
        self._register_default_tools()

    @staticmethod
    def _format_plan(analysis: str | None, strategy: list[str]) -> str:
        """Create a standard representation for the operational plan."""
        goal_text = analysis.strip() if analysis else "N/A"
        steps = strategy if strategy else ["No steps defined."]
        steps_text = "\n".join(f"- {step}" for step in steps)
        return (
            "## OPERATIONAL PLAN (IMMUTABLE)\n"
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
        # File tools
        self.tools.register(ReadFileTool())
        self.tools.register(WriteFileTool())
        self.tools.register(EditFileTool())
        self.tools.register(ListDirTool())

        # Shell tool
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.exec_config.restrict_to_workspace,
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

    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        logger.info("Agent loop started")

        while self._running:
            try:
                # Wait for next message
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )

                # Process it
                try:
                    response = await self._process_message(msg)
                    if response:
                        await self.bus.publish_outbound(response)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    # Send error response
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}"
                    ))
            except asyncio.TimeoutError:
                continue

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(self, msg: InboundMessage) -> OutboundMessage | None:
        session = self.sessions.get_or_create(msg.session_key)

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
                        plan = parsed_plan

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

        # Build the base messages, including the immutable plan (if any), once.
        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=msg.content,
            system_suffix=active_plan,
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
                    messages, response.content, tool_call_dicts
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

                    # --- SAFETY INTERCEPTOR ---
                    is_forced = tool.arguments.get("force", False)
                    safety_err = safety.check(tool.name, tool.arguments)

                    if safety_err and not is_forced:
                        # BLOCK ACTION
                        result = (
                            f"{safety_err}\n\n"
                            "ESCAPE HATCH: Retry with 'force': true and a 'reason' "
                            "if you are certain this is necessary."
                        )
                    else:
                        # ALLOW ACTION
                        result = await self.tools.execute(tool.name, tool.arguments)

                    # --- REFLECTION TRIGGER ---
                    if result.startswith(STATUS_FAILED):
                        result += (
                            "\n\n[SYSTEM INTERVENTION]\n"
                            "PROTOCOL: STOP. Do not retry identical command.\n"
                            "ACTION: Analyze failure -> Update Plan -> Retry."
                        )

                    messages = self.context.add_tool_result(
                        messages, tool.id, tool.name, result
                    )
            else:
                final_content = response.content
                break

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        # Save to session
        session.add_message("user", msg.content)
        session.add_message("assistant", final_content)
        self.sessions.save(session)

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content
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

        # Build messages with the announce content
        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=msg.content
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
                    messages, response.content, tool_call_dicts
                )

                for tool_call in response.tool_calls:
                    args_str = json.dumps(tool_call.arguments)
                    logger.debug(f"Executing tool: {tool_call.name} with arguments: {args_str}")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
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

    async def process_direct(self, content: str, session_key: str = "cli:direct") -> str:
        """
        Process a message directly (for CLI usage).

        Args:
            content: The message content.
            session_key: Session identifier.

        Returns:
            The agent's response.
        """
        channel = "cli"
        chat_id = "direct"
        if session_key:
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
