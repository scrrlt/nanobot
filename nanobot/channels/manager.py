"""Channel manager for coordinating chat channels."""

import asyncio
from typing import Any

from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import Config


class ChannelManager:
    """
    Manages chat channels and coordinates message routing.

    Responsibilities:
    - Initialize enabled channels (Telegram, WhatsApp, etc.)
    - Start/stop channels
    - Route outbound messages
    """

    def __init__(self, config: Config, bus: MessageBus):
        self.config = config
        self.bus = bus
        self.channels: dict[str, BaseChannel] = {}
        self._dispatch_task: asyncio.Task[Any] | None = None

        self._init_channels()

    def _init_channels(self) -> None:
        """Initialize channels based on config."""

        # Telegram channel
        if self.config.channels.telegram.enabled:
            try:
                from nanobot.channels.telegram import TelegramChannel
                self.channels["telegram"] = TelegramChannel(
                    self.config.channels.telegram,
                    self.bus,
                    groq_api_key=self.config.providers.groq.api_key,
                )
                logger.info("Telegram channel enabled")
            except ImportError as e:
                logger.warning(f"Telegram channel not available: {e}")

        # WhatsApp channel
        if self.config.channels.whatsapp.enabled:
            try:
                from nanobot.channels.whatsapp import WhatsAppChannel
                self.channels["whatsapp"] = WhatsAppChannel(
                    self.config.channels.whatsapp, self.bus
                )
                logger.info("WhatsApp channel enabled")
            except ImportError as e:
                logger.warning(f"WhatsApp channel not available: {e}")

        # Discord channel
        if self.config.channels.discord.enabled:
            try:
                from nanobot.channels.discord import DiscordChannel
                self.channels["discord"] = DiscordChannel(
                    self.config.channels.discord, self.bus
                )
                logger.info("Discord channel enabled")
            except ImportError as e:
                logger.warning(f"Discord channel not available: {e}")

        # Feishu channel
        if self.config.channels.feishu.enabled:
            try:
                from nanobot.channels.feishu import FeishuChannel
                self.channels["feishu"] = FeishuChannel(
                    self.config.channels.feishu, self.bus
                )
                logger.info("Feishu channel enabled")
            except ImportError as e:
                logger.warning(f"Feishu channel not available: {e}")

    async def start_all(self) -> None:
        """Start all enabled channels and the outbound dispatcher."""
        if not self.channels:
            logger.warning("No channels enabled")
            return

        # Start outbound dispatcher
        self._dispatch_task = asyncio.create_task(self._dispatch_outbound())

        # Start enabled channels
        task_pairs: list[tuple[str, asyncio.Task[Any]]] = []
        for name, channel in self.channels.items():
            logger.info(f"Starting {name} channel...")
            task_pairs.append((name, asyncio.create_task(channel.start())))

        # Wait for all to complete (they should run forever)
        results = await asyncio.gather(
            *(task for _, task in task_pairs), return_exceptions=True
        )
        for (name, _), result in zip(task_pairs, results):
            if isinstance(result, Exception):
                logger.opt(exception=result).error("Channel {} task failed", name)

    async def stop_all(self) -> None:
        """Stop all channels and the dispatcher."""
        logger.info("Stopping all channels...")

        # Stop all channels first to prevent new outbound messages
        for name, channel in self.channels.items():
            try:
                await channel.stop()
                logger.info(f"Stopped {name} channel")
            except Exception as exc:  # noqa: BLE001
                logger.opt(exception=exc).error("Error stopping %s", name)

        # Stop dispatcher task after channels have been halted
        dispatch_task = self._dispatch_task
        self._dispatch_task = None
        self.bus.stop()
        if dispatch_task is not None:
            try:
                await dispatch_task
            except asyncio.CancelledError:
                logger.debug("Outbound dispatcher task cancelled during shutdown")
            except Exception as exc:  # noqa: BLE001
                logger.opt(exception=exc).error(
                    "Outbound dispatcher encountered an error during shutdown"
                )

    async def _dispatch_outbound(self) -> None:
        """Dispatch outbound messages to the appropriate channel."""
        logger.info("Outbound dispatcher started")

        while True:
            try:
                message = await self.bus.consume_outbound()
            except asyncio.CancelledError:
                logger.debug("Outbound dispatcher cancelled")
                break

            if message is None:
                logger.info("Outbound dispatcher received shutdown signal")
                break

            channel = self.channels.get(message.channel)
            if channel:
                try:
                    await channel.send(message)
                except Exception as exc:
                    logger.error(f"Error sending to {message.channel}: {exc}")
            else:
                logger.warning(f"Unknown channel: {message.channel}")

    def get_channel(self, name: str) -> BaseChannel | None:
        """Get a channel by name."""
        return self.channels.get(name)

    def get_status(self) -> dict[str, Any]:
        """Get status of all channels."""
        return {
            name: {
                "enabled": True,
                "running": channel.is_running
            }
            for name, channel in self.channels.items()
        }

    @property
    def enabled_channels(self) -> list[str]:
        """Get list of enabled channel names."""
        return list(self.channels.keys())
