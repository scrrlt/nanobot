"""Subagent manager for background task execution."""

import asyncio
import json
import uuid
import re
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool


SensitiveFieldMap = dict[str, frozenset[str]]

_SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "content",
        "command",
        "password",
        "secret",
        "token",
        "api_key",
    }
)

_TOOL_SENSITIVE_FIELDS_RAW: SensitiveFieldMap = {
    "write_file": frozenset({"content"}),
    "exec": frozenset({"command"}),
    "web_search": frozenset({"query"}),
}

_TOOL_SENSITIVE_FIELDS: SensitiveFieldMap = {
    tool_name.lower(): frozenset(
        field.lower() for field in fields if field.lower() not in _SENSITIVE_KEYS
    )
    for tool_name, fields in _TOOL_SENSITIVE_FIELDS_RAW.items()
}


def _redact_value(value: Any) -> str:
    if isinstance(value, str):
        return f"<REDACTED len={len(value)}>"
    if isinstance(value, (bytes, bytearray)):
        return f"<REDACTED bytes len={len(value)}>"
    return "<REDACTED>"


def _sanitize_exception_text(msg: Any) -> str:
    """Sanitize exception messages by redacting paths/keys and truncating."""
    s = str(msg)
    # Redact Windows paths like C:\Users\... (single backslash path separators)
    s = re.sub(r"[A-Za-z]:\\[^\\\s]+", "<REDACTED_PATH>", s)
    # Redact forward-slash filesystem paths but avoid URLs (negative lookbehind ensures
    # the preceding character is not ':' or '/') and match segmented path components
    s = re.sub(r"(?<![:/])/(?:[\w\-.]+/)*[\w\-.]+", "<REDACTED_PATH>", s)
    # Redact common secret-like key/value pairs
    s = re.sub(
        r"(?i)((?:api[_-]?key|token|password|secret))\s*[:=]\s*[^\s]+",
        lambda m: f"{m.group(1)}: <REDACTED>",
        s,
    )
    if len(s) > 400:
        s = s[:400] + "...[truncated]"
    return s


def _sanitize_arguments(tool_name: str, payload: Any) -> Any:
    tool_key = tool_name.lower()
    extra_sensitive = _TOOL_SENSITIVE_FIELDS.get(tool_key, frozenset())

    if isinstance(payload, dict):
        sanitized: dict[str, Any] = {}
        for key, value in payload.items():
            lower_key = key.lower()
            if lower_key in _SENSITIVE_KEYS or lower_key in extra_sensitive:
                sanitized[key] = _redact_value(value)
            else:
                sanitized[key] = _sanitize_arguments(tool_name, value)
        return sanitized
    if isinstance(payload, tuple):
        return tuple(_sanitize_arguments(tool_name, item) for item in payload)
    if isinstance(payload, list):
        return [_sanitize_arguments(tool_name, item) for item in payload]
    return payload


class SubagentManager:
    """
    Manages background subagent execution.

    Subagents are lightweight agent instances that run in the background
    to handle specific tasks. They share the same LLM provider but have
    isolated context and a focused system prompt.
    """

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        bus: MessageBus,
        model: str | None = None,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        restrict_to_workspace: bool = False,
    ):
        from nanobot.config.schema import ExecToolConfig

        self.provider = provider
        self.workspace = workspace
        self.bus = bus
        self.model = model or provider.get_default_model()
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.restrict_to_workspace = restrict_to_workspace
        self._running_tasks: dict[str, asyncio.Task[None]] = {}

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
    ) -> str:
        """
        Spawn a subagent to execute a task in the background.

        Args:
            task: The task description for the subagent.
            label: Optional human-readable label for the task.
            origin_channel: The channel to announce results to.
            origin_chat_id: The chat ID to announce results to.

        Returns:
            Status message indicating the subagent was started.
        """
        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")

        origin = {
            "channel": origin_channel,
            "chat_id": origin_chat_id,
        }

        # Create background task
        bg_task = asyncio.create_task(
            self._run_subagent(task_id, task, display_label, origin)
        )
        self._running_tasks[task_id] = bg_task

        # Cleanup when done
        bg_task.add_done_callback(lambda _: self._running_tasks.pop(task_id, None))

        logger.info(f"Spawned subagent [{task_id}]: {display_label}")
        return f"Subagent [{display_label}] started (id: {task_id}). I'll notify you when it completes."

    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, str],
    ) -> None:
        """Execute the subagent task and announce the result."""
        logger.info(f"Subagent [{task_id}] starting task: {label}")

        try:
            # Build subagent tools (no message tool, no spawn tool)
            tools = ToolRegistry()
            allowed_dir = self.workspace if self.restrict_to_workspace else None
            tools.register(ReadFileTool(allowed_dir=allowed_dir))
            tools.register(WriteFileTool(allowed_dir=allowed_dir))
            tools.register(ListDirTool(allowed_dir=allowed_dir))
            tools.register(ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.restrict_to_workspace,
            ))
            tools.register(WebSearchTool(api_key=self.brave_api_key))
            tools.register(WebFetchTool())

            # Build messages with subagent-specific prompt
            system_prompt = self._build_subagent_prompt(task)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ]

            # Run agent loop (limited iterations)
            max_iterations = 15
            iteration = 0
            final_result: str | None = None

            while iteration < max_iterations:
                iteration += 1

                response = await self.provider.chat(
                    messages=messages,
                    tools=tools.get_definitions(),
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
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in response.tool_calls
                    ]
                    messages.append({
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": tool_call_dicts,
                    })

                    # Execute tools
                    for tool_call in response.tool_calls:
                        safe_args = _sanitize_arguments(
                            tool_call.name, tool_call.arguments
                        )
                        logger.debug(
                            "Subagent [%s] executing %s with arguments %s",
                            task_id,
                            tool_call.name,
                            safe_args,
                        )
                        result = await tools.execute(tool_call.name, tool_call.arguments)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.name,
                            "content": result,
                        })
                else:
                    final_result = response.content
                    break

            if final_result is None:
                final_result = "Task completed but no final response was generated."

            logger.info(f"Subagent [{task_id}] completed successfully")
            await self._announce_result(task_id, label, task, final_result, origin, "ok")

        except Exception as e:
            sanitized = _sanitize_exception_text(e)
            error_msg = f"Error: {sanitized}"
            logger.error(f"Subagent [{task_id}] failed: {sanitized}")
            await self._announce_result(task_id, label, task, error_msg, origin, "error")

    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        origin: dict[str, str],
        status: str,
    ) -> None:
        """Announce the subagent result to the main agent via the message bus."""
        status_text = "completed successfully" if status == "ok" else "failed"

        announce_content = f"""[Subagent '{label}' {status_text}]

Task: {task}

Result:
{result}

Summarize this naturally for the user. Keep it brief (1-2 sentences). Do not mention technical details like "subagent" or task IDs."""

        # Inject as system message to trigger main agent
        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=announce_content,
        )

        await self.bus.publish_inbound(msg)
        logger.debug(f"Subagent [{task_id}] announced result to {origin['channel']}:{origin['chat_id']}")

    def _build_subagent_prompt(self, task: str) -> str:
        """Build a focused system prompt for the subagent."""
        return f"""# Subagent

You are a subagent spawned by the main agent to complete a specific task.

## Your Task
{task}

## Rules
1. Stay focused - complete only the assigned task, nothing else
2. Your final response will be reported back to the main agent
3. Do not initiate conversations or take on side tasks
4. Be concise but informative in your findings

## What You Can Do
- Read and write files in the workspace
- Execute shell commands
- Search the web and fetch web pages
- Complete the task thoroughly

## What You Cannot Do
- Send messages directly to users (no message tool available)
- Spawn other subagents
- Access the main agent's conversation history

## Workspace
Your workspace is at: {self.workspace}

When you have completed the task, provide a clear summary of your findings or actions."""

    def get_running_count(self) -> int:
        """Return the number of currently running subagents."""
        return len(self._running_tasks)
