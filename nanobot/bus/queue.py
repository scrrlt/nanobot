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
        if isinstance(item, _InboundShutdown):
            return None
        return item

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
        if isinstance(item, _OutboundShutdown):
            return None
        return item

    def subscribe_outbound(
        self,
        channel: str,
        callback: Callable[[OutboundMessage], Awaitable[None]],
    ) -> None:
        """Subscribe to outbound messages for a specific channel."""
        if channel not in self._outbound_subscribers:
            self._outbound_subscribers[channel] = []
        self._outbound_subscribers[channel].append(callback)

    async def dispatch_outbound(self) -> None:
        """Dispatch outbound messages to subscribed channels."""
        while True:
            item = await self.outbound.get()
            is_shutdown = isinstance(item, _OutboundShutdown)
            try:
                if not is_shutdown:
                    subscribers = self._outbound_subscribers.get(item.channel, [])
                    dead_lettered = False
                    for callback in subscribers:
                        try:
                            await callback(item)
                        except Exception as exc:  # noqa: BLE001
                            logger.error(
                                "Error dispatching to %s: %s",
                                item.channel,
                                exc,
                            )
                            if not dead_lettered:
                                await self._dead_letters.put(item)
                                dead_lettered = True
            finally:
                self.outbound.task_done()
            if is_shutdown:
                break

    def stop(self) -> None:
        """Signal the dispatcher and consumers to shut down."""
        if not self._inbound_shutdown_requested:
            self.inbound.put_nowait(_InboundShutdown())
            self._inbound_shutdown_requested = True
        if not self._outbound_shutdown_requested:
            self.outbound.put_nowait(_OutboundShutdown())
            self._outbound_shutdown_requested = True

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
