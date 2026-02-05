from pydantic import BaseModel
from nanobot.providers.base import LLMProvider


class RouteDecision(BaseModel):
    is_complex: bool
    reasoning: str


async def classify_intent(content: str, provider: "LLMProvider", model: str) -> bool:
    """Fast intent classification. Fails open (True) if unsure."""
    if len(content) < 15 and " " not in content:
        return False
    try:
        from nanobot.providers.litellm_provider import LiteLLMProvider

        if not isinstance(provider, LiteLLMProvider):
            return True

        response = await provider.chat(
            messages=[
                {"role": "system", "content": "Analyze complexity. JSON output."},
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
        return decision.is_complex
    except Exception:
        return True
