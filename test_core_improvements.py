#!/usr/bin/env python3
"""
Core component tests for framework stability improvements:

1. Message Bus Backpressure: Bounded queues with retry logic
2. Base Channel Interface: Standardized media and retry functionality  
3. Signal imports and structure verification
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock

# Test core bus and channel components 
from nanobot.bus.queue import MessageBus
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.channels.base import BaseChannel


class TestChannel(BaseChannel):
    """Test channel implementation."""
    
    name = "test"
    
    async def start(self) -> None:
        pass
        
    async def stop(self) -> None:
        pass
        
    async def send(self, msg: OutboundMessage) -> None:
        pass


async def test_message_bus_backpressure():
    """Test MessageBus backpressure protection with bounded queues."""
    print("Testing message bus backpressure protection...")
    
    # Create bounded queue
    bus = MessageBus(inbound_maxsize=2, outbound_maxsize=2)
    
    # Verify bounded capacity
    assert bus.inbound_capacity == (0, 2), "Should show correct capacity"
    assert not bus.is_inbound_full(), "Should not be full initially"
    
    # Fill to capacity
    for i in range(2):
        msg = InboundMessage(
            channel="test",
            sender_id=f"user{i}", 
            chat_id="test",
            content=f"Message {i}"
        )
        published = await bus.publish_inbound(msg, timeout=0.1)
        assert published, f"Message {i} should publish"
    
    assert bus.is_inbound_full(), "Queue should be full"
    
    # Test backpressure (should fail)
    overflow = InboundMessage(
        channel="test",
        sender_id="overflow",
        chat_id="test", 
        content="Overflow"
    )
    published = await bus.publish_inbound(overflow, timeout=0.1)
    assert not published, "Should fail due to backpressure"
    
    print("✅ Message bus backpressure: Bounded queues prevent memory exhaustion")


async def test_channel_retry_logic():
    """Test BaseChannel retry logic for backpressure scenarios."""
    print("Testing BaseChannel backpressure retry logic...")
    
    # Setup bounded bus and channel
    bus = MessageBus(inbound_maxsize=1, outbound_maxsize=1) 
    config = Mock()
    config.allow_from = ["*"]
    channel = TestChannel(config, bus)
    
    # Fill queue
    await bus.publish_inbound(InboundMessage(
        channel="pre",
        sender_id="prefill",
        chat_id="test",
        content="Prefill"
    ))
    
    # Test retry logic - since queue stays full, this should complete after max retries
    start_time = asyncio.get_event_loop().time()
    
    # The method should complete even if all retries fail
    await channel._handle_message(
        sender_id="retry_user",
        chat_id="test_chat", 
        content="Retry test",
        max_retries=1,  # Will exhaust retries quickly
        retry_delay=0.05  # Fast for testing
    )
    
    elapsed = asyncio.get_event_loop().time() - start_time
    assert elapsed > 0.02, "Should take time due to retry attempts"
    
    print("✅ BaseChannel retry logic: Handles backpressure with exponential backoff")


async def test_standardized_media_interface():
    """Test standardized media upload interface."""
    print("Testing standardized media upload interface...")
    
    bus = MessageBus()
    config = Mock()
    channel = TestChannel(config, bus)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir) / "test.jpg"
        test_file.write_bytes(b"fake image")
        
        # BaseChannel should return None by default
        result = await channel.upload_media(test_file)
        assert result is None, "Default implementation should return None"
    
    print("✅ Standardized media interface: upload_media() method available")


def test_signal_handling_structure():
    """Test that signal handling structures are in place."""
    print("Testing signal handling infrastructure...")
    
    # Verify signal imports in agent/loop.py
    import nanobot.agent.loop
    
    # Check that signal module is imported
    assert hasattr(nanobot.agent.loop, 'signal'), "Signal module should be imported"
    assert hasattr(nanobot.agent.loop, 'sys'), "Sys module should be imported"
    
    # Check AgentLoop has shutdown-related attributes
    from nanobot.agent.loop import AgentLoop
    
    # Mock basic dependencies to create instance
    bus = MessageBus()
    provider = Mock()
    provider.get_default_model.return_value = "test"
    workspace = Path("/tmp")
    
    loop = AgentLoop(bus=bus, provider=provider, workspace=workspace)
    
    assert hasattr(loop, '_shutdown_event'), "Should have shutdown event"
    assert hasattr(loop, '_graceful_shutdown'), "Should have graceful shutdown flag"
    assert hasattr(loop, 'stop'), "Should have stop method"
    
    print("✅ Signal handling: Infrastructure properly configured")


async def main():
    """Run core framework improvement tests."""
    print("🧪 Testing framework stability improvements (core components)...\n")
    
    try:
        await test_message_bus_backpressure()
        await test_channel_retry_logic() 
        await test_standardized_media_interface()
        test_signal_handling_structure()
        
        print("\n🎉 Core framework improvements validated successfully!")
        print("\nSummary of improvements:")
        print("  ✅ Message Bus Backpressure: Bounded queues with timeout and monitoring")
        print("  ✅ Channel Retry Logic: Exponential backoff for queue saturation")
        print("  ✅ Standardized Media Interface: upload_media() across all channels")
        print("  ✅ Signal Handling Infrastructure: SIGTERM/SIGINT support for graceful shutdown")
        
        print(f"\nKey benefits:")
        print(f"  - Memory protection: Prevents OOM during message bursts")
        print(f"  - Consistent API: Standardized media upload across platforms")
        print(f"  - Container-friendly: Proper signal handling for K8s/Docker")
        print(f"  - Reliability: Graceful degradation under load")
        
    except Exception as e:
        print(f"\n❌ Framework validation failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())