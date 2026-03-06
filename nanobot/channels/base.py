"""Base channel interface for chat platforms."""

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus


class BaseChannel(ABC):
    """
    Abstract base class for chat channel implementations.

    Each channel (Telegram, Discord, etc.) should implement this interface
    to integrate with the nanobot message bus.
    """

    name: str = "base"

    def __init__(self, config: Any, bus: MessageBus):
        """
        Initialize the channel.

        Args:
            config: Channel-specific configuration.
            bus: The message bus for communication.
        """
        self.config = config
        self.bus = bus
        self._running = False
        
        # Backpressure configuration with defaults
        backpressure_config = getattr(config, 'backpressure', None)
        if backpressure_config:
            self._max_retries = backpressure_config.max_retries
            self._timeout = backpressure_config.timeout_seconds  
            self._base_retry_delay = backpressure_config.base_retry_delay
            self._max_retry_delay = min(backpressure_config.max_retry_delay, 30.0)
        else:
            # Fallback defaults for backward compatibility
            self._max_retries = 3
            self._timeout = 5.0
            self._base_retry_delay = 1.0
            self._max_retry_delay = 30.0

    @abstractmethod
    async def start(self) -> None:
        """
        Start the channel and begin listening for messages.

        This should be a long-running async task that:
        1. Connects to the chat platform
        2. Listens for incoming messages
        3. Forwards messages to the bus via _handle_message()
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the channel and clean up resources."""
        pass

    @abstractmethod
    async def send(self, msg: OutboundMessage) -> None:
        """
        Send a message through this channel.

        Args:
            msg: The message to send.
        """
        pass
        
    async def upload_media(self, media_path: Path) -> Optional[str]:
        """
        Upload media file and return platform-specific media ID or URL.
        
        This provides a standardized interface for media uploads across
        all channel types (Telegram, Discord, Mochat, etc.).
        
        Args:
            media_path: Path to the media file to upload
            
        Returns:
            Platform-specific media identifier or None if upload failed
            
        Note:
            Default implementation returns None (no media support).
            Channels that support media should override this method.
        """
        logger.warning(
            "Channel {} does not implement media upload support", 
            self.name
        )
        return None

    def is_allowed(self, sender_id: str) -> bool:
        """Check if *sender_id* is permitted.  Empty list → deny all; ``"*"`` → allow all."""
        allow_list = getattr(self.config, "allow_from", [])
        if not allow_list:
            logger.warning("{}: allow_from is empty — all access denied", self.name)
            return False
        if "*" in allow_list:
            return True
        sender_str = str(sender_id)
        return sender_str in allow_list or any(
            p in allow_list for p in sender_str.split("|") if p
        )

    async def _handle_message(
        self,
        sender_id: str,
        chat_id: str,
        content: str,
        media: list[str] | None = None,
        metadata: dict[str, Any] | None = None, 
        session_key: str | None = None,
        max_retries: int | None = None,
        retry_delay: float | None = None,
    ) -> None:
        """
        Handle an incoming message from the chat platform with configurable backpressure protection.

        This method checks permissions and forwards to the bus with retry logic
        to handle queue backpressure during high-volume scenarios.

        Args:
            sender_id: The sender's identifier.
            chat_id: The chat/channel identifier.
            content: Message text content.
            media: Optional list of media URLs.
            metadata: Optional channel-specific metadata.
            session_key: Optional session key override (e.g. thread-scoped sessions).
            max_retries: Override max retry attempts (uses config default if None).
            retry_delay: Override base retry delay (uses config default if None).
        """
        if not self.is_allowed(sender_id):
            logger.warning(
                "Access denied for sender {} on channel {}. "
                "Add them to allowFrom list in config to grant access.",
                sender_id, self.name,
            )
            return

        msg = InboundMessage(
            channel=self.name,
            sender_id=str(sender_id),
            chat_id=str(chat_id),
            content=content,
            media=media or [],
            metadata=metadata or {},
            session_key_override=session_key,
        )

        # Use configured values or method overrides
        retries = max_retries if max_retries is not None else self._max_retries
        base_delay = retry_delay if retry_delay is not None else self._base_retry_delay

        # Attempt to publish with retry logic for backpressure
        for attempt in range(retries + 1):
            try:
                # Use configured timeout to avoid blocking indefinitely
                timeout = self._timeout if attempt < retries else None
                published = await self.bus.publish_inbound(msg, timeout=timeout)
                
                if published:
                    return  # Success
                    
                # Queue is full, implement retry with exponential backoff
                if attempt < retries:
                    delay = min(base_delay * (2 ** attempt), self._max_retry_delay)
                    logger.info(
                        "Message bus backpressure detected, retrying in {:.1f}s (attempt {}/{})",
                        delay, attempt + 1, retries
                    )
                    await asyncio.sleep(delay)
                else:
                    # Final attempt failed
                    logger.error(
                        "Failed to publish message from {} after {} attempts due to persistent backpressure",
                        sender_id, retries + 1
                    )
                    
            except Exception as e:
                logger.exception("Error publishing message from {}: {}", sender_id, e)
                break

    @property
    def is_running(self) -> bool:
        """Check if the channel is running."""
        return self._running
