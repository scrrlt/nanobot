"""Tool registry for dynamic tool management."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from nanobot.agent.tools.base import STATUS_FAILED, Tool


class ToolExecutionStatus(Enum):
    """Enumerates possible outcomes of a tool invocation."""

    SUCCESS = "success"
    VALIDATION_ERROR = "validation_error"
    TOOL_NOT_FOUND = "tool_not_found"
    EXECUTION_ERROR = "execution_error"


@dataclass(frozen=True)
class ToolExecutionResult:
    """Structured result returned by the tool registry."""

    status: ToolExecutionStatus
    message: str
    errors: list[str] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        """Return ``True`` when the execution completed successfully."""
        return self.status is ToolExecutionStatus.SUCCESS


class ToolRegistry:
    """
    Registry for agent tools.

    Allows dynamic registration and execution of tools.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions in OpenAI format."""
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(self, name: str, params: dict[str, Any]) -> ToolExecutionResult:
        """
        Execute a tool by name with given parameters.

        Args:
            name: Tool name.
            params: Tool parameters.
        """
        tool = self._tools.get(name)
        if not tool:
            return ToolExecutionResult(
                status=ToolExecutionStatus.TOOL_NOT_FOUND,
                message=f"{STATUS_FAILED}\nError: Tool '{name}' not found.",
            )

        try:
            errors = tool.validate_params(params)
            if errors:
                return ToolExecutionResult(
                    status=ToolExecutionStatus.VALIDATION_ERROR,
                    message=(
                        f"{STATUS_FAILED}\nInvalid parameters for tool '{name}': "
                        + "; ".join(errors)
                    ),
                    errors=errors,
                )

            output = await tool.execute(**params)
            return ToolExecutionResult(
                status=ToolExecutionStatus.SUCCESS,
                message=output,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolExecutionResult(
                status=ToolExecutionStatus.EXECUTION_ERROR,
                message=f"{STATUS_FAILED}\nError executing {name}: {str(exc)}",
            )

    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
