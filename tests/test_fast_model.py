"""Tests for the AgentLoop fast_model logic."""


from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider, LLMResponse


@pytest.fixture
def mock_bus():
    """Fixture for a mocked MessageBus."""
    return MagicMock(spec=MessageBus)


@pytest.fixture
def mock_provider():
    """Fixture for a mocked LLMProvider."""
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "default/model"
    return provider


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Fixture for a temporary workspace directory."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


def _find_chat_call(mock_provider: MagicMock, **kwargs_filters: Any):
    """Find a specific call to the provider's chat method."""
    for call in mock_provider.chat.call_args_list:
        # call.kwargs is the dictionary of keyword arguments for the call
        match = True
        for key, value in kwargs_filters.items():
            if value is True:  # Check for presence of a key
                if key not in call.kwargs:
                    match = False
                    break
            elif call.kwargs.get(key) != value:
                match = False
                break
        if match:
            return call
    return None


@pytest.mark.asyncio
@patch("nanobot.agent.loop.classify_intent", new_callable=AsyncMock)
async def test_agent_loop_uses_fast_model_when_provided(
    mock_classify_intent: AsyncMock,
    mock_bus: MagicMock,
    mock_provider: MagicMock,
    workspace: Path,
):
    """Verify AgentLoop uses fast_model for routing if provided."""
    # Arrange
    mock_classify_intent.return_value = True
    mock_provider.chat.return_value = LLMResponse(content="{}")

    agent_loop = AgentLoop(
        bus=mock_bus,
        provider=mock_provider,
        workspace=workspace,
        model="primary/model",
        fast_model="fast/model",
    )

    # Act
    await agent_loop.process_direct("test message")

    # Assert
    assert mock_classify_intent.call_count == 1
    # Check that classify_intent was called with the fast_model
    assert mock_classify_intent.call_args[0][2] == "fast/model"

    # Check that the drafting call also used the fast_model
    drafting_call = _find_chat_call(mock_provider, response_format=True)
    assert drafting_call is not None, "Drafting call not found"
    assert drafting_call.kwargs["model"] == "fast/model"

    # Check that the main execution call used the primary model
    execution_call = _find_chat_call(mock_provider, tools=True)
    assert execution_call is not None, "Execution call with tools not found"
    assert execution_call.kwargs["model"] == "primary/model"


@pytest.mark.asyncio
@patch("nanobot.agent.loop.classify_intent", new_callable=AsyncMock)
async def test_agent_loop_falls_back_to_main_model(
    mock_classify_intent: AsyncMock,
    mock_bus: MagicMock,
    mock_provider: MagicMock,
    workspace: Path,
):
    """Verify AgentLoop falls back to the main model when fast_model is not set."""
    # Arrange
    mock_classify_intent.return_value = True
    mock_provider.chat.return_value = LLMResponse(content="{}")

    agent_loop = AgentLoop(
        bus=mock_bus,
        provider=mock_provider,
        workspace=workspace,
        model="primary/model",
        fast_model=None,  # Explicitly not provided
    )

    # Act
    await agent_loop.process_direct("test message")

    # Assert
    assert agent_loop.fast_model == "primary/model"
    assert mock_classify_intent.call_count == 1
    # Check that classify_intent was called with the primary model
    assert mock_classify_intent.call_args[0][2] == "primary/model"

    # Check that the drafting call also used the primary model
    drafting_call = _find_chat_call(mock_provider, response_format=True)
    assert drafting_call is not None, "Drafting call not found"
    assert drafting_call.kwargs["model"] == "primary/model"


@pytest.mark.asyncio
async def test_process_direct_honors_session_key(
    mock_bus: MagicMock,
    mock_provider: MagicMock,
    workspace: Path,
) -> None:
    """Ensure process_direct uses the provided session key for routing."""

    agent_loop = AgentLoop(
        bus=mock_bus,
        provider=mock_provider,
        workspace=workspace,
        model="primary/model",
        fast_model="primary/model",
    )

    async_mock = AsyncMock(
        return_value=OutboundMessage(
            channel="console",
            chat_id="session",
            content="ok",
        )
    )

    with patch.object(agent_loop, "_process_message", new=async_mock):
        await agent_loop.process_direct("hello", session_key="console:session")

    assert async_mock.await_count == 1
    call_args = async_mock.await_args
    msg = call_args.args[0]
    assert msg.channel == "console"
    assert msg.chat_id == "session"


@pytest.mark.asyncio
async def test_process_direct_returns_error_message_on_failure(
    mock_bus: MagicMock,
    mock_provider: MagicMock,
    workspace: Path,
) -> None:
    """process_direct should surface errors rather than returning empty text."""

    agent_loop = AgentLoop(
        bus=mock_bus,
        provider=mock_provider,
        workspace=workspace,
        model="primary/model",
        fast_model="primary/model",
    )

    async_mock = AsyncMock(side_effect=RuntimeError("boom"))

    with patch.object(agent_loop, "_process_message", new=async_mock):
        response = await agent_loop.process_direct("hello")

    assert "Error processing message" in response
    assert "boom" in response
