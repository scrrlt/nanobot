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
            content = stripped[7:-3]
        else:
            # No closing code fence detected; only remove the leading marker.
            content = stripped[7:]

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Failed to parse LLM JSON response: {content}", e.doc, e.pos
        )
