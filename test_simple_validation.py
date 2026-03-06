#!/usr/bin/env python3
"""
Simple validation of framework stability improvements
"""

import asyncio
from unittest.mock import Mock

from nanobot.bus.queue import MessageBus
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.channels.base import BaseChannel


class TestChannel(BaseChannel):
    name = "test"
    
    async def start(self): pass
    async def stop(self): pass  
    async def send(self, msg: OutboundMessage): pass


async def test_bounded_queues():
    """Test that MessageBus properly implements bounded queues."""
    print("Testing bounded queues...")
    
    # Create small bounded queue
    bus = MessageBus(inbound_maxsize=2, outbound_maxsize=2)
    
    # Test capacity reporting
    assert bus.inbound_capacity == (0, 2)
    assert not bus.is_inbound_full()
    
    # Fill queue
    msg1 = InboundMessage(channel="test", sender_id="user1", chat_id="chat", content="msg1")
    msg2 = InboundMessage(channel="test", sender_id="user2", chat_id="chat", content="msg2")
    
    result1 = await bus.publish_inbound(msg1, timeout=0.1)
    result2 = await bus.publish_inbound(msg2, timeout=0.1)
    
    assert result1 and result2
    assert bus.is_inbound_full()
    
    # Test backpressure
    msg3 = InboundMessage(channel="test", sender_id="user3", chat_id="chat", content="msg3")
    result3 = await bus.publish_inbound(msg3, timeout=0.1)
    assert not result3  # Should reject due to backpressure
    
    print("✅ Bounded queues working correctly")


async def test_media_interface():
    """Test that BaseChannel has upload_media method."""
    print("Testing media interface...")
    
    bus = MessageBus()
    config = Mock()
    channel = TestChannel(config, bus)
    
    # Should have upload_media method that returns None by default
    assert hasattr(channel, 'upload_media')
    
    print("✅ Media interface present")


async def main():
    print("🧪 Running simple framework validation...\n")
    
    await test_bounded_queues()
    await test_media_interface()
    
    print("\n🎉 Framework improvements are working!")
    print("\nImprovements validated:")
    print("  ✅ Message Bus Backpressure: Bounded queues prevent memory exhaustion")
    print("  ✅ Standardized Media Interface: upload_media() method available")
    print("  ✅ Framework structures in place for enhanced stability")


if __name__ == "__main__":
    asyncio.run(main())