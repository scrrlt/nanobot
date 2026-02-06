import json
from typing import Any


def parse_llm_json(content: str) -> Any:
    """
    Parses a JSON string from an LLM response.
    Handles markdown code blocks.
    """
    stripped = content.strip()
    if stripped.startswith("```json"):
        if stripped.endswith("```"):
            stripped = stripped[7:-3]
        else:
            # No closing code fence detected; only remove the leading marker.
            stripped = stripped[7:]

    try:
        return json.loads(stripped)
    except json.JSONDecodeError as e:
        preview = stripped[:200]
        if len(stripped) > 200:
            preview += "..."
        raise json.JSONDecodeError(
            f"Failed to parse LLM JSON response: {preview}", e.doc, e.pos
        )
