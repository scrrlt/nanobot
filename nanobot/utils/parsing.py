"""Utility helpers for parsing LLM responses."""

import json
import re
from typing import Any

_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)

_DELIMITER_MAP = {
    "{": "}",
    "[": "]",
}


def _strip_markdown_fence(content: str) -> str:
    """Remove a surrounding ```json fence if present."""
    stripped = content.strip()
    match = _FENCE_PATTERN.fullmatch(stripped)
    if match:
        inner = match.group(1)
        return inner.strip()

    # Fallback for unterminated code fences: drop the leading marker only.
    if stripped.lower().startswith("```json"):
        without_marker = stripped[7:]
        if without_marker.endswith("```"):
            without_marker = without_marker[:-3]
        return without_marker.strip()

    return stripped


def _find_matching_delimiter(content: str, start: int) -> int | None:
    """Best-effort search for the closing delimiter balancing braces/brackets."""
    opening = content[start]
    closing = _DELIMITER_MAP.get(opening)
    if not closing:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(content)):
        char = content[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index
    return None


def _extract_json_segment(content: str) -> str | None:
    """Attempt to locate a JSON object/array within free-form text."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\{\[]", content):
        start = match.start()
        try:
            _, end = decoder.raw_decode(content, start)
            segment = content[start:end].strip()
            if segment:
                return segment
        except json.JSONDecodeError:
            possible_end = _find_matching_delimiter(content, start)
            if possible_end is not None:
                segment = content[start : possible_end + 1].strip()
                if segment:
                    return segment
    return None


def parse_llm_json(content: str) -> Any:
    """
    Parse JSON embedded in an LLM response.

    The parser first attempts to remove surrounding ```json fences, then falls
    back to locating the first decodable JSON object/array within arbitrary
    prose.
    """
    stripped = _strip_markdown_fence(content)
    if not stripped:
        stripped = content.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as e:
        candidate = _extract_json_segment(content)
        if candidate is not None:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as inner:
                candidate_preview = candidate[:200]
                if len(candidate) > 200:
                    candidate_preview += "..."
                raise json.JSONDecodeError(
                    f"Failed to parse embedded JSON segment: {candidate_preview}",
                    inner.doc,
                    inner.pos,
                ) from inner

        preview = stripped[:200]
        if len(stripped) > 200:
            preview += "..."
        raise json.JSONDecodeError(
            f"Failed to parse LLM JSON response: {preview}", e.doc, e.pos
        ) from e
