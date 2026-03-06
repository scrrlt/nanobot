"""Async message queue for decoupled channel-agent communication."""

import asyncio
from typing import Optional

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage


class MessageBus:
    """
    Async message bus that decouples chat channels from the agent core.

    Channels push messages to the inbound queue, and the agent processes
    them and pushes responses to the outbound queue.
    
    Implements backpressure protection via bounded queues to prevent
    memory exhaustion during high-volume message bursts.
    """

    def __init__(
        self, 
        inbound_maxsize: int = 1000, 
        outbound_maxsize: int = 1000
    ):
        """
        Initialize message bus with bounded queues.
        
        Args:
            inbound_maxsize: Maximum inbound queue size (0 = unlimited)
            outbound_maxsize: Maximum outbound queue size (0 = unlimited)
        """
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue(maxsize=inbound_maxsize)
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue(maxsize=outbound_maxsize)
        
        # Track queue limits for monitoring
        self._inbound_maxsize = inbound_maxsize
        self._outbound_maxsize = outbound_maxsize

    async def publish_inbound(self, msg: InboundMessage, timeout: Optional[float] = None) -> bool:
        """
        Publish a message from a channel to the agent with backpressure protection.
        
        Args:
            msg: The inbound message to publish
            timeout: Optional timeout in seconds (default: no timeout)
            
        Returns:
            True if message was published, False if queue is full and timeout occurred
            
        Raises:
            asyncio.TimeoutError: If timeout specified and queue remains full
        """
        try:
            if timeout is not None:
                await asyncio.wait_for(self.inbound.put(msg), timeout=timeout)
            else:
                await self.inbound.put(msg)
            return True
        except asyncio.TimeoutError:
            logger.warning(
                "Inbound queue full ({}/{}), message from {} dropped", 
                self.inbound.qsize(), 
                self._inbound_maxsize,
                msg.sender_id
            )
            return False
        except Exception as e:
            logger.error("Unexpected error publishing inbound message: {}", e)
            return False

    async def consume_inbound(self) -> InboundMessage:
        """Consume the next inbound message (blocks until available)."""
        return await self.inbound.get()

    async def publish_outbound(self, msg: OutboundMessage, timeout: Optional[float] = None) -> bool:
        """
        Publish a response from the agent to channels with backpressure protection.
        
        Args:
            msg: The outbound message to publish
            timeout: Optional timeout in seconds (default: no timeout)
            
        Returns:
            True if message was published, False if queue is full and timeout occurred
            
        Raises:
            asyncio.TimeoutError: If timeout specified and queue remains full
        """
        try:
            if timeout is not None:
                await asyncio.wait_for(self.outbound.put(msg), timeout=timeout)
            else:
                await self.outbound.put(msg)
            return True
        except asyncio.TimeoutError:
            logger.warning(
                "Outbound queue full ({}/{}), message to {} dropped", 
                self.outbound.qsize(), 
                self._outbound_maxsize,
                msg.chat_id
            )
            return False

    async def consume_outbound(self) -> OutboundMessage:
        """Consume the next outbound message (blocks until available)."""
        return await self.outbound.get()

    @property
    def inbound_size(self) -> int:
        """Number of pending inbound messages."""
        return self.inbound.qsize()

    @property
    def outbound_size(self) -> int:
        """Number of pending outbound messages."""
        return self.outbound.qsize()
        
    @property
    def inbound_capacity(self) -> tuple[int, int]:
        """Current inbound queue size and maximum capacity."""
        return (self.inbound.qsize(), self._inbound_maxsize)
        
    @property
    def outbound_capacity(self) -> tuple[int, int]:
        """Current outbound queue size and maximum capacity."""
        return (self.outbound.qsize(), self._outbound_maxsize)
        
    def is_inbound_full(self) -> bool:
        """Check if inbound queue is at capacity."""
        return self._inbound_maxsize > 0 and self.inbound.qsize() >= self._inbound_maxsize
        
    def is_outbound_full(self) -> bool:
        """Check if outbound queue is at capacity."""
        return self._outbound_maxsize > 0 and self.outbound.qsize() >= self._outbound_maxsize
