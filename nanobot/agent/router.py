"""Utilities for routing requests based on estimated complexity."""

from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field

from nanobot.providers.base import LLMProvider


class RouteDecision(BaseModel):
    """Structured complexity classification returned by the LLM router."""

    complexity_level: Literal["simple", "complex"]
    rationale: str
    suggested_actions: list[str] = Field(  # Reserved for future routing logic
        default_factory=list,
    )


async def classify_intent(content: str, provider: "LLMProvider", model: str) -> bool:
    """Determine whether an incoming request requires complex handling.

    Args:
        content: Raw user content to classify.
        provider: Language model provider used for routing decisions.
        model: Provider-specific model identifier.

    Returns:
        True if the request should route to the complex executor, otherwise False.

    Raises:
        Nothing: This function handles provider errors internally and fails open.
    """
    if len(content) < 15 and " " not in content:
        return False
    try:
        from nanobot.providers.litellm_provider import LiteLLMProvider

        if not isinstance(provider, LiteLLMProvider):
            return True

        response = await provider.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the user's request as simple or complex.\n"
                        "Respond strictly with lowercase values.\n"
                        "simple: a single action, clear outcome, no external"
                        " dependencies, and no ambiguous requirements.\n"
                        "complex: multi-step reasoning, coordination across"
                        " files or tools, dependency ordering, ambiguous goals,"
                        " or need for external data.\n"
                        "Decision rules:\n"
                        "1. Multiple sequential steps or branching -> complex.\n"
                        "2. Requests needing research, external services, or"
                        " interpretation of large contexts -> complex.\n"
                        "3. Clear, atomic updates or questions with direct"
                        " answers -> simple.\n"
                        "Respond strictly with JSON:"
                        ' {"complexity_level": "simple|complex",'
                        ' "rationale": <string>,'
                        ' "suggested_actions": [<string>, ...]}'
                    ),
                },
                {"role": "user", "content": content},
            ],
            response_format={
                "type": "json_object",
                "schema": RouteDecision.model_json_schema(),
            },
            model=model,
            temperature=0,
        )
        if not response or not response.content:
            return True
        decision = RouteDecision.model_validate_json(response.content)
        return decision.complexity_level == "complex"
    except Exception as exc:  # noqa: BLE001
        logger.opt(exception=exc).debug("Intent classification failed; failing open")
        return True
