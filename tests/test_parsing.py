"""Tests for LLM JSON parsing helpers."""

import json

import pytest

from nanobot.utils.parsing import parse_llm_json


def test_parse_llm_json_handles_markdown_fence() -> None:
    content = """```json\n{\n  \"alpha\": 1\n}\n```"""
    result = parse_llm_json(content)
    assert result == {"alpha": 1}


def test_parse_llm_json_with_embedded_prose() -> None:
    content = 'The tool responded with {"ok": true} after processing.'
    result = parse_llm_json(content)
    assert result == {"ok": True}


def test_parse_llm_json_with_multiple_candidates_selects_first_valid() -> None:
    content = 'Noise [1, 2, 3] and then {"ignored": false}'
    result = parse_llm_json(content)
    assert result == [1, 2, 3]


def test_parse_llm_json_multiple_fences_uses_json_block() -> None:
    content = "```text\nhello\n```\n\n" '```json\n{"value": 42}\n```\n'
    result = parse_llm_json(content)
    assert result == {"value": 42}


def test_parse_llm_json_reports_embedded_failure() -> None:
    content = "Prose before {invalid json} and after."
    with pytest.raises(json.JSONDecodeError) as exc_info:
        parse_llm_json(content)

    message = str(exc_info.value)
    assert "Failed to parse embedded JSON segment" in message
