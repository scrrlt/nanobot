from typing import Any

import pytest

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry


class SampleTool(Tool):
    @property
    def name(self) -> str:
        return "sample"

    @property
    def description(self) -> str:
        return "sample tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 2},
                "count": {"type": "integer", "minimum": 1, "maximum": 10},
                "mode": {"type": "string", "enum": ["fast", "full"]},
                "meta": {
                    "type": "object",
                    "properties": {
                        "tag": {"type": "string"},
                        "flags": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["tag"],
                },
            },
            "required": ["query", "count"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return "ok"


class UnionTool(Tool):
    @property
    def name(self) -> str:
        return "union"

    @property
    def description(self) -> str:
        return "tool with union-typed parameters"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "value": {"type": ["string", "number"]},
                "items": {
                    "type": "array",
                    "items": {"type": ["string", "number", "boolean"]},
                },
                "payload": {
                    "anyOf": [
                        {"type": "string"},
                        {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                    ]
                },
                "marker": {
                    "oneOf": [
                        {"type": "string", "minLength": 1},
                        {"type": "integer", "minimum": 0},
                    ]
                },
            },
            "required": ["value", "items", "payload", "marker"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return "ok"


def test_validate_params_missing_required() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi"})
    assert "missing required count" in "; ".join(errors)


def test_validate_params_type_and_range() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi", "count": 0})
    assert any("count must be >= 1" in e for e in errors)

    errors = tool.validate_params({"query": "hi", "count": "2"})
    assert any("count should be integer" in e for e in errors)


def test_validate_params_enum_and_min_length() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "h", "count": 2, "mode": "slow"})
    assert any("query must be at least 2 chars" in e for e in errors)
    assert any("mode must be one of" in e for e in errors)


def test_validate_params_nested_object_and_array() -> None:
    tool = SampleTool()
    errors = tool.validate_params(
        {
            "query": "hi",
            "count": 2,
            "meta": {"flags": [1, "ok"]},
        }
    )
    assert any("missing required meta.tag" in e for e in errors)
    assert any("meta.flags[0] should be string" in e for e in errors)


def test_validate_params_ignores_unknown_fields() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi", "count": 2, "extra": "x"})
    assert errors == []


@pytest.mark.asyncio
async def test_registry_returns_validation_error() -> None:
    reg = ToolRegistry()
    reg.register(SampleTool())
    result = await reg.execute("sample", {"query": "hi"})
    assert "Invalid parameters" in result


def test_union_type_validation_supports_multiple_candidates() -> None:
    tool = UnionTool()
    assert (
        tool.validate_params(
            {
                "value": "alpha",
                "items": ["one", 2, True],
                "payload": [1, 2, 3],
                "marker": 1,
            }
        )
        == []
    )

    assert (
        tool.validate_params(
            {
                "value": 9,
                "items": ["x"],
                "payload": "ready",
                "marker": "tag",
            }
        )
        == []
    )


def test_union_type_validation_reports_errors() -> None:
    tool = UnionTool()
    errors = tool.validate_params(
        {
            "value": [1],
            "items": ["ok", {"oops": True}],
            "payload": [1, "bad"],
            "marker": 1,
        }
    )

    joined = "; ".join(errors)
    assert "value should match one of types" in joined
    assert "items[1]" in joined
    assert "payload did not match any allowed schema option" in joined


def test_one_of_requires_exact_match() -> None:
    tool = UnionTool()
    errors = tool.validate_params(
        {
            "value": "x",
            "items": ["x"],
            "payload": "ready",
            "marker": "",
        }
    )

    assert any("oneOf" in err for err in errors)

    errors = tool.validate_params(
        {
            "value": "x",
            "items": ["x"],
            "payload": "ready",
            "marker": 0,
        }
    )
    assert errors == []
