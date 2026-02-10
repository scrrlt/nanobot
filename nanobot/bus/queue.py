"""Async message queue for decoupled channel-agent communication."""

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage


@dataclass(frozen=True)
class _InboundShutdown:
    """Sentinel signalling that the inbound queue should shut down."""


@dataclass(frozen=True)
class _OutboundShutdown:
    """Sentinel signalling that the outbound queue should shut down."""


InboundQueueItem = InboundMessage | _InboundShutdown
OutboundQueueItem = OutboundMessage | _OutboundShutdown


class MessageBus:
    """Async message bus that decouples chat channels from the agent core."""

    def __init__(self) -> None:
        self.inbound: asyncio.Queue[InboundQueueItem] = asyncio.Queue()
        self.outbound: asyncio.Queue[OutboundQueueItem] = asyncio.Queue()
        self._outbound_subscribers: dict[str, list[Callable[[OutboundMessage], Awaitable[None]]]] = {}
        self._dead_letters: asyncio.Queue[OutboundMessage] = asyncio.Queue()
        self._inbound_shutdown_requested = False
        self._outbound_shutdown_requested = False

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """Publish a message from a channel to the agent."""
        if self._inbound_shutdown_requested:
            logger.warning("Inbound message dropped during shutdown: %s", msg)
            return
        await self.inbound.put(msg)

    async def consume_inbound(self) -> InboundMessage | None:
        """Consume the next inbound message.

        Returns ``None`` once shutdown has been requested.
        """
        item = await self.inbound.get()
        try:
            if isinstance(item, _InboundShutdown):
                return None
            return item
        finally:
            # Always pair get() with task_done() so inbound.join() does not hang
            try:
                self.inbound.task_done()
            except Exception:
                pass

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """Publish a response from the agent to channels."""
        if self._outbound_shutdown_requested:
            logger.warning("Outbound message dropped during shutdown: %s", msg)
            return
        await self.outbound.put(msg)

    async def consume_outbound(self) -> OutboundMessage | None:
        """Consume the next outbound message.

        Returns ``None`` once shutdown has been requested.
        """
        item = await self.outbound.get()
        try:
            if isinstance(item, _OutboundShutdown):
                return None
            return item
        finally:
            # Always pair get() with task_done() so outbound.join() does not hang
            try:
                self.outbound.task_done()
            except Exception:
                pass

    def subscribe_outbound(
        self,
        channel: str,
        callback: Callable[[OutboundMessage], Awaitable[None]],
    ) -> None:
        """Subscribe to outbound messages for a specific channel.

        Subscribers are invoked by the channel dispatcher when an outbound
        message is consumed; this avoids multiple concurrent consumers of the
        same outbound queue.
        """
        if channel not in self._outbound_subscribers:
            self._outbound_subscribers[channel] = []
        self._outbound_subscribers[channel].append(callback)

    async def notify_subscribers(self, msg: OutboundMessage) -> None:
        """Invoke callbacks for subscribed outbound consumers."""
        subscribers = self._outbound_subscribers.get(msg.channel, [])
        dead_lettered = False
        for callback in subscribers:
            try:
                await callback(msg)
            except Exception as exc:  # noqa: BLE001
                logger.error("Error dispatching to %s: %s", msg.channel, exc)
                if not dead_lettered:
                    await self._dead_letters.put(msg)
                    dead_lettered = True

    async def dispatch_outbound(self) -> None:
        """Deprecated: use ChannelManager to consume outbound messages and notify subscribers.

        Historically this function consumed the outbound queue directly and
        dispatched to subscribers. ChannelManager now acts as the single
        consumer and will call notify_subscribers() after sending to the
        destination channel. Calling this function will raise to avoid dual consumers.
        """
        raise NotImplementedError(
            "dispatch_outbound is deprecated; ChannelManager handles outbound dispatch"
        )

    def stop(self) -> None:
        """Signal the dispatcher and consumers to shut down."""
        # Flip flags first to avoid a race window where multiple callers enqueue
        # duplicate shutdown sentinels concurrently.
        if not self._inbound_shutdown_requested:
            self._inbound_shutdown_requested = True
            self.inbound.put_nowait(_InboundShutdown())
        if not self._outbound_shutdown_requested:
            self._outbound_shutdown_requested = True
            self.outbound.put_nowait(_OutboundShutdown())

    @property
    def inbound_size(self) -> int:
        """Number of pending inbound messages."""
        return self.inbound.qsize()

    @property
    def outbound_size(self) -> int:
        """Number of pending outbound messages."""
        return self.outbound.qsize()

    @property
    def dead_letters(self) -> asyncio.Queue[OutboundMessage]:
        """Queue containing messages that failed to dispatch."""
        return self._dead_letters
