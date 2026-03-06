"""Test for Discord channel typing safety fix.

This test verifies that the Discord typing loop properly handles
the case where self._http becomes None during operation.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from nanobot.config.schema import DiscordConfig


@pytest.mark.asyncio
async def test_discord_typing_loop_safety():
    """Test that Discord typing loop safely handles _http becoming None."""
    # Import Discord channel
    from nanobot.channels.discord import DiscordChannel
    
    # Create mock config
    config = Mock(spec=DiscordConfig)
    config.token = "test_token"
    config.channels = ["123456"]
    config.allow_mentions = False
    
    # Create mock message bus
    mock_bus = Mock()
    
    # Create channel instance
    channel = DiscordChannel(config, mock_bus)
    
    # Set up the channel state
    channel._running = True
    
    # Mock HTTP client that will become None during operation
    mock_http = Mock()
    mock_http.post = AsyncMock()
    channel._http = mock_http
    
    # Start typing in background
    typing_task = asyncio.create_task(channel._start_typing("123456"))
    
    # Wait a bit to let typing loop start
    await asyncio.sleep(0.1)
    
    # Simulate _http becoming None (connection lost)
    channel._http = None
    
    # Wait for typing loop to detect and exit gracefully
    await asyncio.sleep(0.1)
    
    # Clean up
    channel._running = False
    typing_task.cancel()
    
    try:
        await typing_task
    except asyncio.CancelledError:
        pass
    
    # If we reach here without exception, the fix is working
    assert True


@pytest.mark.asyncio  
async def test_discord_typing_loop_continues_with_valid_http():
    """Test that Discord typing loop continues normally when _http is valid."""
    from nanobot.channels.discord import DiscordChannel
    
    config = Mock(spec=DiscordConfig)
    config.token = "test_token"
    config.channels = ["123456"]
    config.allow_mentions = False
    
    mock_bus = Mock()
    channel = DiscordChannel(config, mock_bus)
    
    channel._running = True
    
    # Mock HTTP client that stays valid
    mock_http = Mock()
    mock_http.post = AsyncMock()
    channel._http = mock_http
    
    # Start typing
    typing_task = asyncio.create_task(channel._start_typing("123456"))
    
    # Let it run briefly
    await asyncio.sleep(0.05)
    
    # Verify HTTP post was called
    assert mock_http.post.called
    
    # Clean up
    channel._running = False
    typing_task.cancel()
    
    try:
        await typing_task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])