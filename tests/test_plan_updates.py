"""Tests for plan update behavior in AgentLoop."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


@pytest.fixture
def mock_bus() -> MagicMock:
    """Provide a mocked message bus."""
    return MagicMock(spec=MessageBus)


@pytest.fixture
def mock_provider() -> MagicMock:
    """Provide a mocked LLMProvider."""
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "default/model"
    return provider


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace directory."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.mark.asyncio
@patch("nanobot.agent.loop.classify_intent", new_callable=AsyncMock)
async def test_plan_update_converts_non_string_steps_and_updates_analysis(
    mock_classify_intent: AsyncMock,
    mock_bus: MagicMock,
    mock_provider: MagicMock,
    workspace: Path,
) -> None:
    """Plan updates should stringify steps and refresh the goal text."""
    mock_classify_intent.return_value = False

    plan_tool_call = ToolCallRequest(
        id="tool-1",
        name="update_plan",
        arguments={
            "strategy": ["alpha", 123, False],
            "analysis": "Refined goal",
            "reason": "Updating plan based on refined analysis.",
        },
    )

    mock_provider.chat.side_effect = [
        LLMResponse(content=None, tool_calls=[plan_tool_call]),
        LLMResponse(content="All done."),
    ]

    agent_loop = AgentLoop(
        bus=mock_bus,
        provider=mock_provider,
        workspace=workspace,
        model="primary/model",
        fast_model="fast/model",
    )

    result = await agent_loop.process_direct("run plan update test")

    assert result == "All done."
    assert mock_provider.chat.call_count == 2

    second_call = mock_provider.chat.call_args_list[1]
    messages = second_call.args[0]
    system_message = messages[0]
    assert system_message["role"] == "system"
    system_content = system_message["content"]

    assert "Refined goal" in system_content
    assert "- alpha" in system_content
    assert "- 123" in system_content
    assert "- False" in system_content

    assert "tools" in second_call.kwargs
    assert "model" in second_call.kwargs
