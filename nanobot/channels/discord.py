"""Discord channel implementation using Discord Gateway websocket."""

import asyncio
import json
from pathlib import Path, PurePath
from typing import Any

import aiofiles
import httpx
import websockets
from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import DiscordConfig


DISCORD_API_BASE = "https://discord.com/api/v10"
MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024  # 20MB


class DiscordChannel(BaseChannel):
    """Discord channel using Gateway websocket."""

    name = "discord"

    def __init__(self, config: DiscordConfig, bus: MessageBus):
        super().__init__(config, bus)
        self.config: DiscordConfig = config
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._seq: int | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._typing_tasks: dict[str, asyncio.Task] = {}
        self._http: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """Start the Discord gateway connection."""
        if not self.config.token:
            logger.error("Discord bot token not configured")
            return

        self._running = True
        self._http = httpx.AsyncClient(timeout=30.0)

        while self._running:
            try:
                logger.info("Connecting to Discord gateway...")
                async with websockets.connect(self.config.gateway_url) as ws:
                    self._ws = ws
                    await self._gateway_loop()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Discord gateway error: {e}")
                if self._running:
                    logger.info("Reconnecting to Discord gateway in 5 seconds...")
                    await asyncio.sleep(5)

    async def stop(self) -> None:
        """Stop the Discord channel."""
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
        for task in self._typing_tasks.values():
            task.cancel()
        self._typing_tasks.clear()
        if self._ws:
            await self._ws.close()
            self._ws = None
        if self._http:
            await self._http.aclose()
            self._http = None

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Discord REST API."""
        if not self._http:
            logger.warning("Discord HTTP client not initialized")
            return

        url = f"{DISCORD_API_BASE}/channels/{msg.chat_id}/messages"
        payload: dict[str, Any] = {"content": msg.content}

        if msg.reply_to:
            payload["message_reference"] = {"message_id": msg.reply_to}
            payload["allowed_mentions"] = {"replied_user": False}

        headers = {"Authorization": f"Bot {self.config.token}"}

        try:
            backoff = 1.0
            last_response: httpx.Response | None = None
            for attempt in range(3):
                try:
                    response = await self._http.post(url, headers=headers, json=payload)
                except (asyncio.TimeoutError, httpx.TransportError) as exc:
                    if attempt == 2:
                        logger.error(
                            f"Error sending Discord message after retries: {exc}"
                        )
                        break
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"Non-retryable Discord send failure: {exc}")
                    break

                status = response.status_code
                last_response = response
                if 200 <= status < 300:
                    return
                if status == 429:
                    retry_after = 1.0
                    try:
                        data = response.json()
                        retry_after = float(data.get("retry_after", retry_after))
                    except (ValueError, json.JSONDecodeError, TypeError):
                        logger.debug(
                            f"Failed to parse retry_after from Discord response: {response.text}"
                        )
                    logger.warning(
                        f"Discord rate limited, retrying in {retry_after:.1f}s"
                    )
                    if attempt == 2:
                        logger.error(
                            f"Discord rate limited after retries (429): {response.text}"
                        )
                        break
                    await asyncio.sleep(retry_after)
                    continue
                if 500 <= status < 600:
                    if attempt == 2:
                        logger.error(f"Discord server error {status}: {response.text}")
                        break
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                if 400 <= status < 500:
                    logger.error(
                        f"Discord rejected message ({status}): {response.text}"
                    )
                    break
                logger.error(
                    f"Unexpected Discord response status {status}: {response.text}"
                )
                break

            error_status = last_response.status_code if last_response else "unknown"
            raise RuntimeError(
                f"Failed to send Discord message after retries (status {error_status})"
            )
        finally:
            await self._stop_typing(msg.chat_id)

    async def _gateway_loop(self) -> None:
        """Main gateway loop: identify, heartbeat, dispatch events."""
        if not self._ws:
            return

        async for raw in self._ws:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from Discord gateway: {raw[:100]}")
                continue

            op = data.get("op")
            event_type = data.get("t")
            seq = data.get("s")
            payload = data.get("d")

            if seq is not None:
                self._seq = seq

            if op == 10:
                # HELLO: start heartbeat and identify
                interval_ms = payload.get("heartbeat_interval", 45000)
                await self._start_heartbeat(interval_ms / 1000)
                await self._identify()
            elif op == 0 and event_type == "READY":
                logger.info("Discord gateway READY")
            elif op == 0 and event_type == "MESSAGE_CREATE":
                await self._handle_message_create(payload)
            elif op == 7:
                # RECONNECT: exit loop to reconnect
                logger.info("Discord gateway requested reconnect")
                break
            elif op == 9:
                # INVALID_SESSION: reconnect
                logger.warning("Discord gateway invalid session")
                break

    async def _identify(self) -> None:
        """Send IDENTIFY payload."""
        if not self._ws:
            return

        identify = {
            "op": 2,
            "d": {
                "token": self.config.token,
                "intents": self.config.intents,
                "properties": {
                    "os": "nanobot",
                    "browser": "nanobot",
                    "device": "nanobot",
                },
            },
        }
        await self._ws.send(json.dumps(identify))

    async def _start_heartbeat(self, interval_s: float) -> None:
        """Start or restart the heartbeat loop."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()

        async def heartbeat_loop() -> None:
            while self._running and self._ws:
                payload = {"op": 1, "d": self._seq}
                try:
                    await self._ws.send(json.dumps(payload))
                except Exception as e:
                    logger.warning(f"Discord heartbeat failed: {e}")
                    break
                await asyncio.sleep(interval_s)

        self._heartbeat_task = asyncio.create_task(heartbeat_loop())

    async def _handle_message_create(self, payload: dict[str, Any]) -> None:
        """Handle incoming Discord messages."""
        author = payload.get("author") or {}
        if author.get("bot"):
            return

        sender_id = str(author.get("id", ""))
        channel_id = str(payload.get("channel_id", ""))
        content = payload.get("content") or ""

        if not sender_id or not channel_id:
            return

        if not self.is_allowed(sender_id):
            return

        content_parts = [content] if content else []
        media_paths: list[str] = []
        media_dir = Path.home() / ".nanobot" / "media"

        for attachment in payload.get("attachments") or []:
            url = attachment.get("url")
            raw_filename = attachment.get("filename") or "attachment"
            size = attachment.get("size") or 0
            if not url or not self._http:
                continue
            if size and size > MAX_ATTACHMENT_BYTES:
                content_parts.append(f"[attachment: {raw_filename} - too large]")
                continue

            candidate_name = PurePath(raw_filename).name.replace("\x00", "")
            safe_name = "".join(
                char
                for char in candidate_name
                if char.isalnum() or char in {"-", "_", "."}
            )
            if not safe_name:
                safe_name = str(attachment.get("id", "file"))
            final_name = f"{attachment.get('id', 'file')}_{safe_name}"
            file_path = media_dir / final_name

            try:
                media_dir.mkdir(parents=True, exist_ok=True)
                async with self._http.stream("GET", url) as resp:
                    resp.raise_for_status()
                    try:
                        async with aiofiles.open(file_path, "wb") as file_obj:
                            bytes_written = 0
                            async for chunk in resp.aiter_bytes():
                                if not chunk:
                                    continue
                                bytes_written += len(chunk)
                                if bytes_written > MAX_ATTACHMENT_BYTES:
                                    raise ValueError(
                                        "Attachment exceeds maximum allowed size"
                                    )
                                await file_obj.write(chunk)
                    except Exception:
                        file_path.unlink(missing_ok=True)
                        raise
                media_paths.append(str(file_path))
                content_parts.append(f"[attachment: {file_path}]")
            except Exception as e:
                logger.opt(exception=e).warning("Failed to download Discord attachment")
                content_parts.append(f"[attachment: {safe_name} - download failed]")

        reply_to = (payload.get("referenced_message") or {}).get("id")

        await self._start_typing(channel_id)

        await self._handle_message(
            sender_id=sender_id,
            chat_id=channel_id,
            content="\n".join(p for p in content_parts if p) or "[empty message]",
            media=media_paths,
            metadata={
                "message_id": str(payload.get("id", "")),
                "guild_id": payload.get("guild_id"),
                "reply_to": reply_to,
            },
        )

    async def _start_typing(self, channel_id: str) -> None:
        """Start periodic typing indicator for a channel."""
        await self._stop_typing(channel_id)

        async def typing_loop() -> None:
            url = f"{DISCORD_API_BASE}/channels/{channel_id}/typing"
            headers = {"Authorization": f"Bot {self.config.token}"}
            while self._running:
                try:
                    client = self._http
                    if client is None:
                        break
                    await client.post(url, headers=headers)
                except Exception:
                    if not self._running:
                        break
                    logger.opt(exception=True).debug("Typing indicator request failed")
                await asyncio.sleep(8)

        self._typing_tasks[channel_id] = asyncio.create_task(typing_loop())

    async def _stop_typing(self, channel_id: str) -> None:
        """Stop typing indicator for a channel."""
        task = self._typing_tasks.pop(channel_id, None)
        if task:
            task.cancel()
