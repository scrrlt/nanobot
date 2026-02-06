"""Tests for ContextBuilder caching and suffix updates."""

from pathlib import Path
from unittest.mock import patch

import pytest

from nanobot.agent.context import ContextBuilder


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a minimal workspace for context builder tests."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


def test_update_system_suffix_uses_cached_prompt(workspace: Path) -> None:
    """Update system suffix should reuse cached base prompt without rebuilding."""
    builder = ContextBuilder(workspace)

    messages = builder.build_messages(
        history=[],
        current_message="Hello",
        system_suffix="## PLAN A",
    )

    assert messages[0]["role"] == "system"
    assert messages[0]["content"].endswith("## PLAN A")

    with patch.object(
        builder,
        "build_system_prompt",
        side_effect=AssertionError("System prompt should not rebuild"),
    ):
        updated = builder.update_system_suffix(messages, "## PLAN B")

    assert updated[0]["content"].endswith("## PLAN B")
    assert "## PLAN A" not in updated[0]["content"]
