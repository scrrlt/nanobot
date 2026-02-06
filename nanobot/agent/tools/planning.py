from typing import Any

from nanobot.agent.tools.base import Tool


class UpdatePlanTool(Tool):
    @property
    def name(self) -> str:
        return "update_plan"

    @property
    def description(self) -> str:
        return "Update the current operational plan/strategy. Use this when the current plan is blocked or failed."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "strategy": {
                    "type": "array",
                    "items": {"type": ["string", "number", "boolean"]},
                    "description": (
                        "The new list of steps to execute. Non-string values "
                        "will be converted to strings."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": "Why the plan is being updated.",
                },
                "analysis": {
                    "type": "string",
                    "description": "Optional refreshed goal/analysis text for the plan.",
                },
            },
            "required": ["strategy", "reason"],
        }

    async def execute(self, **kwargs: Any) -> str:
        # The actual logic is intercepted by the Loop,
        # but this needs to return success for the log.
        return "Plan updated."
