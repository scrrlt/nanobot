"""Mochat channel implementation using Socket.IO with HTTP polling fallback.

This module provides a comprehensive Mochat integration with:
- Type-safe async Socket.IO connections with HTTP fallback
- Robust error handling and retry mechanisms  
- Message buffering and deduplication
- Session and panel management
- Comprehensive resource management
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from collections import deque
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, Set, Union
from uuid import uuid4

import httpx
from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import MochatConfig
from nanobot.utils.helpers import get_data_path

try:
    import socketio
    SOCKETIO_AVAILABLE = True
except ImportError:
    socketio = None
    SOCKETIO_AVAILABLE = False

try:
    import msgpack  # noqa: F401
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False

# Configuration constants
MAX_SEEN_MESSAGE_IDS = 2000
CURSOR_SAVE_DEBOUNCE_S = 0.5
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_DELAY_MS = 1000
MAX_RETRY_DELAY_MS = 30000
CONNECTION_TIMEOUT_S = 30.0
DEFAULT_HEALTH_CHECK_INTERVAL_S = 60.0


# ---------------------------------------------------------------------------
# Type definitions and protocols
# ---------------------------------------------------------------------------

class ConnectionState(Enum):
    """Connection state enumeration."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting" 
    CONNECTED = "connected"
    READY = "ready"
    ERROR = "error"


class TargetKind(Enum):
    """Target kind enumeration."""
    SESSION = "session"
    PANEL = "panel"


@dataclass(frozen=True)
class CorrelationId:
    """Unique identifier for request correlation."""
    value: str = field(default_factory=lambda: uuid4().hex[:8])
    
    def __str__(self) -> str:
        return self.value


class EventPayload(Protocol):
    """Protocol for event payloads."""
    messageId: str
    author: str
    content: Any
    meta: Dict[str, Any]
    groupId: str
    converseId: str
    

class SocketEvent(Protocol):
    """Protocol for socket events."""
    type: str
    timestamp: Optional[str]
    payload: EventPayload
    seq: Optional[int]


class WatchPayload(Protocol):
    """Protocol for watch payloads."""
    sessionId: str
    cursor: Optional[int]
    events: List[SocketEvent]


class AuthorInfo(Protocol):
    """Protocol for author information."""
    nickname: Optional[str]
    email: Optional[str]
    agentId: Optional[str]


# ---------------------------------------------------------------------------
# Exception classes
# ---------------------------------------------------------------------------

class MochatError(Exception):
    """Base exception for Mochat-related errors."""
    def __init__(self, message: str, correlation_id: Optional[CorrelationId] = None) -> None:
        super().__init__(message)
        self.correlation_id = correlation_id or CorrelationId()


class ConnectionError(MochatError):
    """Raised when connection operations fail."""
    pass


class AuthenticationError(MochatError):
    """Raised when authentication fails."""
    pass


class SubscriptionError(MochatError):
    """Raised when subscription operations fail."""
    pass


class APIError(MochatError):
    """Raised when API calls fail."""
    def __init__(self, message: str, status_code: Optional[int] = None, 
                 correlation_id: Optional[CorrelationId] = None) -> None:
        super().__init__(message, correlation_id)
        self.status_code = status_code


class RetryExhaustedError(MochatError):
    """Raised when retry attempts are exhausted."""
    pass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetryConfig:
    """Configuration for retry mechanisms."""
    max_attempts: int = DEFAULT_RETRY_ATTEMPTS
    base_delay_ms: int = DEFAULT_RETRY_DELAY_MS
    max_delay_ms: int = MAX_RETRY_DELAY_MS
    exponential_base: float = 2.0
    jitter: bool = True
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number (0-based)."""
        delay_ms = min(
            self.base_delay_ms * (self.exponential_base ** attempt),
            self.max_delay_ms
        )
        if self.jitter:
            import random
            delay_ms *= (0.5 + random.random() * 0.5)  # 50-100% of calculated delay
        return delay_ms / 1000.0


@dataclass
class MochatBufferedEntry:
    """Buffered inbound entry for delayed dispatch."""
    raw_body: str
    author: str
    sender_name: str = ""
    sender_username: str = ""
    timestamp: Optional[int] = None
    message_id: str = ""
    group_id: str = ""
    correlation_id: CorrelationId = field(default_factory=CorrelationId)
    
    def __post_init__(self) -> None:
        """Validate required fields."""
        if not self.raw_body.strip():
            raise ValueError("raw_body cannot be empty")
        if not self.author.strip():
            raise ValueError("author cannot be empty")


@dataclass
class DelayState:
    """Per-target delayed message state with proper lifecycle management."""
    entries: List[MochatBufferedEntry] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    timer: Optional[asyncio.Task[None]] = None
    
    async def cancel_timer(self) -> None:
        """Cancel any active timer."""
        if self.timer and not self.timer.done():
            self.timer.cancel()
            try:
                await self.timer
            except asyncio.CancelledError:
                pass
            self.timer = None


@dataclass(frozen=True)
class MochatTarget:
    """Outbound target resolution result."""
    id: str
    is_panel: bool
    
    def __post_init__(self) -> None:
        """Validate target ID."""
        if not self.id.strip():
            raise ValueError("Target ID cannot be empty")


@dataclass
class ConnectionMetrics:
    """Connection health and performance metrics."""
    connected_at: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    reconnect_count: int = 0
    message_count: int = 0
    error_count: int = 0
    last_error: Optional[Exception] = None
    
    def record_connection(self) -> None:
        """Record a successful connection."""
        self.connected_at = datetime.utcnow()
        self.last_heartbeat = datetime.utcnow()
        
    def record_heartbeat(self) -> None:
        """Record a heartbeat."""
        self.last_heartbeat = datetime.utcnow()
        
    def record_message(self) -> None:
        """Record a processed message."""
        self.message_count += 1
        
    def record_error(self, error: Exception) -> None:
        """Record an error."""
        self.error_count += 1
        self.last_error = error
        
    def record_reconnect(self) -> None:
        """Record a reconnection."""
        self.reconnect_count += 1
        self.record_connection()


@dataclass
class HealthStatus:
    """Health check status."""
    is_healthy: bool
    connection_state: ConnectionState
    metrics: ConnectionMetrics
    issues: List[str] = field(default_factory=list)
    checked_at: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Utility classes and functions
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """Circuit breaker for handling repeated failures."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state: str = "closed"  # closed, open, half-open
        
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == "closed":
            return True
        if self.state == "open":
            if self.last_failure_time and (asyncio.get_event_loop().time() - self.last_failure_time) > self.recovery_timeout:
                self.state = "half-open"
                return True
            return False
        return True  # half-open
        
    def record_success(self) -> None:
        """Record a successful operation."""
        self.failure_count = 0
        self.state = "closed"
        
    def record_failure(self) -> None:
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = asyncio.get_event_loop().time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


# ---------------------------------------------------------------------------
# Pure helper functions with type safety
# ---------------------------------------------------------------------------

def safe_dict(value: Any) -> Dict[str, Any]:
    """Return value if it's a dict, else empty dict.
    
    Args:
        value: Value to check and convert
        
    Returns:
        Dict if value is dict, otherwise empty dict
    """
    return value if isinstance(value, dict) else {}


def str_field(src: Dict[str, Any], *keys: str) -> str:
    """Return the first non-empty str value found for keys, stripped.
    
    Args:
        src: Source dictionary to search
        *keys: Keys to search for in order
        
    Returns:
        First non-empty string value found, or empty string
    """
    for key in keys:
        value = src.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def make_synthetic_event(
    message_id: str, 
    author: str, 
    content: Any,
    meta: Any, 
    group_id: str, 
    converse_id: str,
    timestamp: Optional[Any] = None, 
    *, 
    author_info: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build a synthetic 'message.add' event dict with validation.
    
    Args:
        message_id: Unique message identifier
        author: Message author ID
        content: Message content (any type)
        meta: Message metadata
        group_id: Group/panel ID if applicable
        converse_id: Conversation ID
        timestamp: Optional timestamp
        author_info: Optional author information
        
    Returns:
        Synthetic event dictionary
        
    Raises:
        ValueError: If required fields are empty
    """
    if not message_id.strip():
        raise ValueError("message_id cannot be empty")
    if not author.strip():
        raise ValueError("author cannot be empty")
    if not converse_id.strip():
        raise ValueError("converse_id cannot be empty")
    
    payload: Dict[str, Any] = {
        "messageId": message_id.strip(), 
        "author": author.strip(),
        "content": content, 
        "meta": safe_dict(meta),
        "groupId": group_id.strip(), 
        "converseId": converse_id.strip(),
    }
    
    if author_info is not None:
        payload["authorInfo"] = safe_dict(author_info)
        
    return {
        "type": "message.add",
        "timestamp": timestamp or datetime.utcnow().isoformat(),
        "payload": payload,
    }


def normalize_mochat_content(content: Any) -> str:
    """Normalize content payload to text with proper error handling.
    
    Args:
        content: Content of any type to normalize
        
    Returns:
        String representation of content
    """
    if isinstance(content, str):
        return content.strip()
    if content is None:
        return ""
    try:
        return json.dumps(content, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        logger.warning("Failed to JSON encode content: {}", e)
        return str(content)


def resolve_mochat_target(raw: str) -> MochatTarget:
    """Resolve id and target kind from user-provided target string.
    
    Args:
        raw: Raw target string (may include prefixes)
        
    Returns:
        Resolved target with ID and panel flag
        
    Raises:
        ValueError: If target resolution fails
    """
    trimmed = (raw or "").strip()
    if not trimmed:
        raise ValueError("Target string cannot be empty")
    
    lowered = trimmed.lower()
    cleaned, forced_panel = trimmed, False
    
    # Check for prefixes
    for prefix in ("mochat:", "group:", "channel:", "panel:"):
        if lowered.startswith(prefix):
            cleaned = trimmed[len(prefix):].strip()
            forced_panel = prefix in {"group:", "channel:", "panel:"}
            break
    
    if not cleaned:
        raise ValueError("Target ID cannot be empty after prefix removal")
        
    return MochatTarget(
        id=cleaned, 
        is_panel=forced_panel or not cleaned.startswith("session_")
    )


def extract_mention_ids(value: Any) -> List[str]:
    """Extract mention ids from heterogeneous mention payload.
    
    Args:
        value: Value that may contain mention IDs
        
    Returns:
        List of extracted mention ID strings
    """
    if not isinstance(value, list):
        return []
        
    ids: List[str] = []
    for item in value:
        if isinstance(item, str):
            if item.strip():
                ids.append(item.strip())
        elif isinstance(item, dict):
            for key in ("id", "userId", "_id"):
                candidate = item.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    ids.append(candidate.strip())
                    break
    return ids


def resolve_was_mentioned(payload: Dict[str, Any], agent_user_id: Optional[str]) -> bool:
    """Resolve mention state from payload metadata and text fallback.
    
    Args:
        payload: Message payload to check
        agent_user_id: Agent's user ID to check for mentions
        
    Returns:
        True if agent was mentioned, False otherwise
    """
    if not agent_user_id:
        return False
        
    meta = payload.get("meta")
    if isinstance(meta, dict):
        # Check explicit mention flags
        if meta.get("mentioned") is True or meta.get("wasMentioned") is True:
            return True
            
        # Check mention arrays
        for field in ("mentions", "mentionIds", "mentionedUserIds", "mentionedUsers"):
            if agent_user_id in extract_mention_ids(meta.get(field)):
                return True
    
    # Fallback to content text search
    content = payload.get("content")
    if not isinstance(content, str) or not content:
        return False
        
    return f"<@{agent_user_id}>" in content or f"@{agent_user_id}" in content


def resolve_require_mention(
    config: MochatConfig, 
    session_id: str, 
    group_id: str
) -> bool:
    """Resolve mention requirement for group/panel conversations.
    
    Args:
        config: Mochat configuration
        session_id: Session ID to check
        group_id: Group ID to check
        
    Returns:
        True if mention is required, False otherwise
    """
    groups = config.groups or {}
    
    # Check specific session/group config first, then wildcard, then global
    for key in (group_id, session_id, "*"):
        if key and key in groups:
            return bool(groups[key].require_mention)
            
    return bool(config.mention.require_in_groups)


def build_buffered_body(entries: List[MochatBufferedEntry], is_group: bool) -> str:
    """Build text body from one or more buffered entries.
    
    Args:
        entries: List of buffered entries to process
        is_group: Whether this is a group conversation
        
    Returns:
        Combined message body
    """
    if not entries:
        return ""
        
    if len(entries) == 1:
        return entries[0].raw_body
        
    lines: List[str] = []
    for entry in entries:
        if not entry.raw_body:
            continue
            
        if is_group:
            label = (
                entry.sender_name.strip() or 
                entry.sender_username.strip() or 
                entry.author
            )
            if label:
                lines.append(f"{label}: {entry.raw_body}")
                continue
                
        lines.append(entry.raw_body)
        
    return "\n".join(lines).strip()


def parse_timestamp(value: Any) -> Optional[int]:
    """Parse event timestamp to epoch milliseconds.
    
    Args:
        value: Timestamp value to parse
        
    Returns:
        Epoch milliseconds or None if parsing fails
    """
    if not isinstance(value, str) or not value.strip():
        return None
        
    try:
        # Handle ISO format with Z suffix
        cleaned = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
        return int(dt.timestamp() * 1000)
    except (ValueError, OverflowError) as e:
        logger.debug("Failed to parse timestamp '{}': {}", value, e)
        return None

# ---------------------------------------------------------------------------
# Connection Management Classes
# ---------------------------------------------------------------------------

class ConnectionManager:
    """Manages websocket and HTTP connections with health monitoring."""
    
    def __init__(
        self,
        config: MochatConfig,
        retry_config: Optional[RetryConfig] = None
    ) -> None:
        self.config = config
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker = CircuitBreaker()
        self.metrics = ConnectionMetrics()
        
        self._http_client: Optional[httpx.AsyncClient] = None
        self._socket_client: Optional[Any] = None
        self._connection_state = ConnectionState.DISCONNECTED
        self._health_check_task: Optional[asyncio.Task[None]] = None
        self._state_lock = asyncio.Lock()
        
        # Event handlers
        self._on_connect: Optional[Callable[[], Awaitable[None]]] = None
        self._on_disconnect: Optional[Callable[[], Awaitable[None]]] = None
        self._on_error: Optional[Callable[[Exception], Awaitable[None]]] = None
    
    async def __aenter__(self) -> 'ConnectionManager':
        """Async context manager entry."""
        await self.start()
        return self
        
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()
    
    def set_event_handlers(
        self,
        on_connect: Optional[Callable[[], Awaitable[None]]] = None,
        on_disconnect: Optional[Callable[[], Awaitable[None]]] = None,
        on_error: Optional[Callable[[Exception], Awaitable[None]]] = None,
    ) -> None:
        """Set event handlers for connection events."""
        self._on_connect = on_connect
        self._on_disconnect = on_disconnect
        self._on_error = on_error
    
    @property
    def connection_state(self) -> ConnectionState:
        """Current connection state."""
        return self._connection_state
    
    @property
    def is_connected(self) -> bool:
        """Whether connection is established and ready."""
        return self._connection_state in {ConnectionState.CONNECTED, ConnectionState.READY}
    
    @property
    def socket_client(self) -> Optional[Any]:
        """Get the socket client (if available)."""
        return self._socket_client
    
    async def start(self) -> None:
        """Start connection manager and establish connections."""
        if not self.config.claw_token:
            raise AuthenticationError("Mochat claw_token not configured")
            
        async with self._state_lock:
            if self._connection_state != ConnectionState.DISCONNECTED:
                return
                
            self._connection_state = ConnectionState.CONNECTING
            
        try:
            # Initialize HTTP client
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(CONNECTION_TIMEOUT_S),
                headers={"User-Agent": f"nanobot-mochat/{self.config.claw_token[:8]}"}
            )
            
            # Test HTTP connectivity first
            await self._test_http_connectivity()
            
            # Attempt WebSocket connection
            websocket_ok = await self._start_websocket_connection()
            
            if websocket_ok:
                self._connection_state = ConnectionState.CONNECTED
                self.metrics.record_connection()
            else:
                logger.warning("WebSocket connection failed, using HTTP-only mode")
                self._connection_state = ConnectionState.CONNECTED  # Can still work with HTTP
                self.metrics.record_connection()
                
            # Start health monitoring
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            if self._on_connect:
                await self._on_connect()
                
        except Exception as e:
            self._connection_state = ConnectionState.ERROR
            self.metrics.record_error(e)
            await self._cleanup_connections()
            if self._on_error:
                await self._on_error(e)
            raise ConnectionError(f"Failed to establish connections: {e}") from e
    
    async def stop(self) -> None:
        """Stop connection manager and cleanup resources."""
        async with self._state_lock:
            if self._connection_state == ConnectionState.DISCONNECTED:
                return
                
            self._connection_state = ConnectionState.DISCONNECTED
            
        try:
            if self._on_disconnect:
                await self._on_disconnect()
        except Exception as e:
            logger.warning("Error in disconnect handler: {}", e)
            
        await self._cleanup_connections()
    
    async def _test_http_connectivity(self) -> None:
        """Test basic HTTP connectivity to the API."""
        if not self._http_client:
            raise ConnectionError("HTTP client not initialized")
            
        url = f"{self.config.base_url.strip().rstrip('/')}/api/health"
        correlation_id = CorrelationId()
        
        for attempt in range(self.retry_config.max_attempts):
            try:
                response = await self._http_client.get(
                    url,
                    headers={"X-Claw-Token": self.config.claw_token},
                )
                
                if response.is_success:
                    logger.debug("HTTP connectivity test passed [{}]", correlation_id)
                    return
                    
                if response.status_code == 401:
                    raise AuthenticationError(
                        "Invalid claw_token", 
                        correlation_id
                    )
                    
                if attempt == self.retry_config.max_attempts - 1:
                    raise APIError(
                        f"HTTP connectivity test failed: {response.status_code}",
                        response.status_code,
                        correlation_id
                    )
                    
            except (httpx.RequestError, httpx.TimeoutException) as e:
                if attempt == self.retry_config.max_attempts - 1:
                    raise ConnectionError(
                        f"HTTP connectivity test failed: {e}",
                        correlation_id
                    ) from e
                    
                await asyncio.sleep(self.retry_config.calculate_delay(attempt))
    
    async def _start_websocket_connection(self) -> bool:
        """Start WebSocket connection with retry logic."""
        if not SOCKETIO_AVAILABLE:
            logger.info("python-socketio not installed, skipping WebSocket")
            return False
            
        serializer = "default"
        if not self.config.socket_disable_msgpack and MSGPACK_AVAILABLE:
            serializer = "msgpack"
        elif not self.config.socket_disable_msgpack:
            logger.warning("msgpack not available, using JSON serializer")
            
        correlation_id = CorrelationId()
        
        for attempt in range(self.retry_config.max_attempts):
            if not self.circuit_breaker.can_execute():
                logger.warning("Circuit breaker open, skipping WebSocket attempt [{}]", correlation_id)
                return False
                
            try:
                client = socketio.AsyncClient(
                    reconnection=True,
                    reconnection_attempts=self.config.max_retry_attempts or None,
                    reconnection_delay=max(0.1, self.config.socket_reconnect_delay_ms / 1000.0),
                    reconnection_delay_max=max(0.1, self.config.socket_max_reconnect_delay_ms / 1000.0),
                    logger=False,
                    engineio_logger=False,
                    serializer=serializer,
                )
                
                # Set up direct event handlers
                await self._setup_socket_handlers(client)
                
                socket_url = (self.config.socket_url or self.config.base_url).strip().rstrip("/")
                socket_path = (self.config.socket_path or "/socket.io").strip().lstrip("/")
                
                await client.connect(
                    socket_url,
                    transports=["websocket"],
                    socketio_path=socket_path,
                    auth={"token": self.config.claw_token},
                    wait_timeout=max(1.0, self.config.socket_connect_timeout_ms / 1000.0),
                )
                
                self._socket_client = client
                self.circuit_breaker.record_success()
                logger.info("WebSocket connection established [{}]", correlation_id)
                return True
                
            except Exception as e:
                self.circuit_breaker.record_failure()
                logger.warning(
                    "WebSocket connection attempt {} failed [{}]: {}",
                    attempt + 1, correlation_id, e
                )
                
                if attempt == self.retry_config.max_attempts - 1:
                    return False
                    
                await asyncio.sleep(self.retry_config.calculate_delay(attempt))
                
        return False
    
    async def _setup_socket_handlers(self, client: Any) -> None:
        """Setup socket.io event handlers."""
        
        @client.event
        async def connect() -> None:
            logger.info("WebSocket connected")
            self._connection_state = ConnectionState.CONNECTED
            self.metrics.record_connection()
            
        @client.event
        async def disconnect() -> None:
            logger.warning("WebSocket disconnected")
            if self._connection_state != ConnectionState.DISCONNECTED:
                self._connection_state = ConnectionState.ERROR
                self.metrics.record_reconnect()
                
        @client.event
        async def connect_error(data: Any) -> None:
            error = ConnectionError(f"WebSocket connect error: {data}")
            logger.error("WebSocket connect error: {}", data)
            self.metrics.record_error(error)
            if self._on_error:
                await self._on_error(error)
    
    async def _health_check_loop(self) -> None:
        """Continuous health monitoring loop."""
        interval = DEFAULT_HEALTH_CHECK_INTERVAL_S
        
        while self._connection_state != ConnectionState.DISCONNECTED:
            try:
                await asyncio.sleep(interval)
                
                if self._connection_state == ConnectionState.DISCONNECTED:
                    break
                    
                health = await self.get_health_status()
                
                if not health.is_healthy:
                    logger.warning(
                        "Health check failed: {}", 
                        ", ".join(health.issues)
                    )
                    
                self.metrics.record_heartbeat()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Health check error: {}", e)
                self.metrics.record_error(e)
    
    async def get_health_status(self) -> HealthStatus:
        """Get current health status."""
        issues: List[str] = []
        is_healthy = True
        
        # Check HTTP client
        if not self._http_client:
            issues.append("HTTP client not available")
            is_healthy = False
        
        # Check WebSocket if expected
        if SOCKETIO_AVAILABLE and not self._socket_client:
            issues.append("WebSocket not connected (fallback mode)")
            # This is not necessarily unhealthy - we can work with HTTP only
        
        # Check recent errors
        if self.metrics.error_count > 10:
            issues.append(f"High error count: {self.metrics.error_count}")
            is_healthy = False
        
        # Check connection age (detect stale connections)
        if self.metrics.connected_at:
            connection_age = datetime.utcnow() - self.metrics.connected_at
            if connection_age.total_seconds() > 86400:  # 24 hours
                issues.append("Connection is stale (>24h)")
        
        return HealthStatus(
            is_healthy=is_healthy,
            connection_state=self._connection_state,
            metrics=self.metrics,
            issues=issues
        )
    
    async def _cleanup_connections(self) -> None:
        """Clean up all connections and resources."""
        # Cancel health check
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
        
        # Close WebSocket
        if self._socket_client:
            try:
                await self._socket_client.disconnect()
            except Exception as e:
                logger.debug("Error disconnecting WebSocket: {}", e)
            self._socket_client = None
        
        # Close HTTP client
        if self._http_client:
            try:
                await self._http_client.aclose()
            except Exception as e:
                logger.debug("Error closing HTTP client: {}", e)
            self._http_client = None
    
    async def http_request(
        self, 
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[CorrelationId] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic and error handling."""
        if not self._http_client:
            raise ConnectionError("HTTP client not available")
            
        correlation_id = correlation_id or CorrelationId()
        url = f"{self.config.base_url.strip().rstrip('/')}{path}"
        
        headers = {
            "Content-Type": "application/json",
            "X-Claw-Token": self.config.claw_token,
            "X-Correlation-ID": str(correlation_id),
        }
        
        for attempt in range(self.retry_config.max_attempts):
            try:
                if method.upper() == "GET":
                    response = await self._http_client.get(url, headers=headers)
                elif method.upper() == "POST":
                    response = await self._http_client.post(
                        url, headers=headers, json=data or {}
                    )
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                if response.is_success:
                    try:
                        result = response.json()
                        return self._process_api_response(result, correlation_id)
                    except json.JSONDecodeError as e:
                        raise APIError(
                            f"Invalid JSON response: {e}",
                            response.status_code,
                            correlation_id
                        ) from e
                
                if response.status_code == 401:
                    raise AuthenticationError(
                        "Authentication failed - invalid token",
                        correlation_id
                    )
                
                if attempt == self.retry_config.max_attempts - 1:
                    raise APIError(
                        f"HTTP {response.status_code}: {response.text[:200]}",
                        response.status_code,
                        correlation_id
                    )
                
            except (httpx.RequestError, httpx.TimeoutException) as e:
                if attempt == self.retry_config.max_attempts - 1:
                    raise ConnectionError(
                        f"HTTP request failed: {e}",
                        correlation_id
                    ) from e
                
                logger.debug(
                    "HTTP request attempt {} failed [{}]: {}",
                    attempt + 1, correlation_id, e
                )
                
                await asyncio.sleep(self.retry_config.calculate_delay(attempt))
        
        raise RetryExhaustedError(
            f"HTTP request failed after {self.retry_config.max_attempts} attempts",
            correlation_id
        )
    
    def _process_api_response(
        self, 
        response: Any, 
        correlation_id: CorrelationId
    ) -> Dict[str, Any]:
        """Process and validate API response."""
        if isinstance(response, dict):
            # Handle structured API responses
            if isinstance(response.get("code"), int):
                if response["code"] != 200:
                    message = str(
                        response.get("message") or 
                        response.get("name") or 
                        "API request failed"
                    )
                    raise APIError(
                        f"API error: {message} (code={response['code']})",
                        response["code"],
                        correlation_id
                    )
                
                data = response.get("data")
                return data if isinstance(data, dict) else {}
            
            return response
        

class MessageBuffer:
    """Manages message buffering, deduplication, and delayed processing."""
    
    def __init__(self, config: MochatConfig) -> None:
        self.config = config
        
        # Deduplication tracking
        self._seen_sets: Dict[str, Set[str]] = {}
        self._seen_queues: Dict[str, deque[str]] = {}
        
        # Delayed message processing
        self._delay_states: Dict[str, DelayState] = {}
        self._processing_locks: Dict[str, asyncio.Lock] = {}
        
    async def __aenter__(self) -> 'MessageBuffer':
        """Async context manager entry."""
        return self
        
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.cleanup()
    
    def is_duplicate_message(self, target_key: str, message_id: str) -> bool:
        """Check if message has been seen before.
        
        Args:
            target_key: Target identifier (e.g., 'session:123')
            message_id: Message ID to check
            
        Returns:
            True if message is duplicate, False if new
        """
        if not message_id.strip():
            return False
            
        seen_set = self._seen_sets.setdefault(target_key, set())
        seen_queue = self._seen_queues.setdefault(target_key, deque())
        
        if message_id in seen_set:
            return True
            
        # Add to tracking
        seen_set.add(message_id)
        seen_queue.append(message_id)
        
        # Maintain size limit
        while len(seen_queue) > MAX_SEEN_MESSAGE_IDS:
            old_id = seen_queue.popleft()
            seen_set.discard(old_id)
            
        return False
    
    async def process_entry(
        self,
        target_key: str,
        entry: MochatBufferedEntry,
        is_group: bool,
        was_mentioned: bool,
        require_mention: bool,
        use_delay: bool,
        dispatch_callback: Callable[[str, TargetKind, List[MochatBufferedEntry], bool], Awaitable[None]]
    ) -> None:
        """Process a message entry with appropriate buffering/delay logic.
        
        Args:
            target_key: Target identifier 
            entry: Message entry to process
            is_group: Whether this is a group conversation
            was_mentioned: Whether agent was mentioned
            require_mention: Whether mention is required
            use_delay: Whether to use delayed processing
            dispatch_callback: Callback to dispatch messages
        """
        # Check mention requirements
        if require_mention and not was_mentioned and not use_delay:
            logger.debug("Skipping message - mention required but not found")
            return
        
        # Determine target kind from key
        target_kind = TargetKind.PANEL if target_key.startswith("panel:") else TargetKind.SESSION
        target_id = target_key.split(":", 1)[1] if ":", target_key else target_key
        
        if use_delay:
            if was_mentioned:
                # Immediate dispatch for mentions, flush any pending
                await self._flush_delayed_entries(
                    target_key, target_id, target_kind, "mention", entry, dispatch_callback
                )
            else:
                # Add to delayed queue
                await self._enqueue_delayed_entry(
                    target_key, target_id, target_kind, entry, dispatch_callback
                )
        else:
            # Immediate dispatch
            await dispatch_callback(target_id, target_kind, [entry], was_mentioned)
    
    async def _enqueue_delayed_entry(
        self,
        target_key: str,
        target_id: str, 
        target_kind: TargetKind,
        entry: MochatBufferedEntry,
        dispatch_callback: Callable[[str, TargetKind, List[MochatBufferedEntry], bool], Awaitable[None]]
    ) -> None:
        """Add entry to delayed processing queue."""
        lock = self._processing_locks.setdefault(target_key, asyncio.Lock())
        
        async with lock:
            state = self._delay_states.setdefault(target_key, DelayState())
            
            async with state.lock:
                state.entries.append(entry)
                
                # Cancel existing timer
                if state.timer and not state.timer.done():
                    state.timer.cancel()
                    
                # Start new timer
                state.timer = asyncio.create_task(
                    self._delay_timer(
                        target_key, target_id, target_kind, dispatch_callback
                    )
                )
    
    async def _delay_timer(
        self,
        target_key: str,
        target_id: str,
        target_kind: TargetKind, 
        dispatch_callback: Callable[[str, TargetKind, List[MochatBufferedEntry], bool], Awaitable[None]]
    ) -> None:
        """Timer for delayed message processing."""
        try:
            delay_s = max(0, self.config.reply_delay_ms) / 1000.0
            await asyncio.sleep(delay_s)
            
            await self._flush_delayed_entries(
                target_key, target_id, target_kind, "timer", None, dispatch_callback
            )
        except asyncio.CancelledError:
            # Timer was cancelled, this is expected
            pass
    
    async def _flush_delayed_entries(
        self,
        target_key: str,
        target_id: str,
        target_kind: TargetKind,
        trigger: str,
        additional_entry: Optional[MochatBufferedEntry],
        dispatch_callback: Callable[[str, TargetKind, List[MochatBufferedEntry], bool], Awaitable[None]]
    ) -> None:
        """Flush all delayed entries for a target."""
        lock = self._processing_locks.setdefault(target_key, asyncio.Lock())
        
        async with lock:
            state = self._delay_states.setdefault(target_key, DelayState())
            
            async with state.lock:
                # Add additional entry if provided
                if additional_entry:
                    state.entries.append(additional_entry)
                
                # Cancel timer if it wasn't the trigger
                current_task = asyncio.current_task()
                if state.timer and state.timer is not current_task:
                    state.timer.cancel()
                state.timer = None
                
                # Get entries and clear
                entries = state.entries[:]
                state.entries.clear()
        
        # Dispatch if we have entries
        if entries:
            was_mentioned = trigger == "mention"
            await dispatch_callback(target_id, target_kind, entries, was_mentioned)
    
    async def cleanup(self) -> None:
        """Clean up all timers and resources."""
        # Cancel all delay timers
        for state in self._delay_states.values():
            await state.cancel_timer()
        
        # Clear all data
        self._delay_states.clear()
        self._processing_locks.clear()
        self._seen_sets.clear()
        self._seen_queues.clear()


class StateManager:
    """Manages persistent state including session cursors."""
    
    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.cursor_path = state_dir / "session_cursors.json"
        
        self.session_cursors: Dict[str, int] = {}
        self._save_task: Optional[asyncio.Task[None]] = None
        self._save_lock = asyncio.Lock()
        
    async def __aenter__(self) -> 'StateManager':
        """Async context manager entry."""
        await self.load()
        return self
        
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.save(force=True)
    
    async def load(self) -> None:
        """Load state from disk."""
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            
            if not self.cursor_path.exists():
                logger.debug("No existing cursor file found")
                return
                
            content = self.cursor_path.read_text("utf-8")
            data = json.loads(content)
            
            if not isinstance(data, dict):
                logger.warning("Invalid cursor file format")
                return
                
            cursors = data.get("cursors")
            if isinstance(cursors, dict):
                for session_id, cursor in cursors.items():
                    if isinstance(session_id, str) and isinstance(cursor, int) and cursor >= 0:
                        self.session_cursors[session_id] = cursor
                        
            logger.info("Loaded {} session cursors", len(self.session_cursors))
            
        except Exception as e:
            logger.warning("Failed to load state: {}", e)
    
    def get_cursor(self, session_id: str) -> int:
        """Get cursor for session."""
        return self.session_cursors.get(session_id, 0)
    
    def update_cursor(self, session_id: str, cursor: int) -> None:
        """Update cursor for session."""
        if cursor < 0 or cursor < self.session_cursors.get(session_id, 0):
            logger.debug(
                "Ignoring cursor update for {}: {} (current: {})",
                session_id, cursor, self.session_cursors.get(session_id, 0)
            )
            return
            
        self.session_cursors[session_id] = cursor
        
        # Schedule save (debounced)
        if not self._save_task or self._save_task.done():
            self._save_task = asyncio.create_task(self._save_debounced())
    
    async def _save_debounced(self) -> None:
        """Save with debouncing to avoid excessive disk writes."""
        await asyncio.sleep(CURSOR_SAVE_DEBOUNCE_S)
        await self.save()
    
    async def save(self, force: bool = False) -> None:
        """Save state to disk."""
        async with self._save_lock:
            try:
                # Cancel pending save task if forcing
                if force and self._save_task and not self._save_task.done():
                    self._save_task.cancel()
                    try:
                        await self._save_task
                    except asyncio.CancelledError:
                        pass
                    self._save_task = None
                
                self.state_dir.mkdir(parents=True, exist_ok=True)
                
                data = {
                    "schemaVersion": 1,
                    "updatedAt": datetime.utcnow().isoformat(),
                    "cursors": self.session_cursors.copy(),
                }
                
                content = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
                self.cursor_path.write_text(content, "utf-8")
                
                logger.debug("Saved {} session cursors", len(self.session_cursors))
                


class TargetManager:
    """Manages session and panel discovery and subscription."""
    
    def __init__(
        self, 
        config: MochatConfig,
        connection_manager: ConnectionManager
    ) -> None:
        self.config = config
        self.connection_manager = connection_manager
        
        # Target tracking
        self.session_set: Set[str] = set()
        self.panel_set: Set[str] = set()
        self.session_by_converse: Dict[str, str] = {}
        
        # Discovery settings
        self.auto_discover_sessions = False
        self.auto_discover_panels = False
        
        # Cold start tracking
        self.cold_sessions: Set[str] = set()
        
        # Locks for thread-safe operations
        self._target_locks: Dict[str, asyncio.Lock] = {}
        
        # Initialize from config
        self._init_from_config()
    
    def _init_from_config(self) -> None:
        """Initialize targets from configuration."""
        sessions, self.auto_discover_sessions = self._normalize_id_list(self.config.sessions)
        panels, self.auto_discover_panels = self._normalize_id_list(self.config.panels)
        
        self.session_set.update(sessions)
        self.panel_set.update(panels)
        
        # Mark sessions as cold (need history backfill)
        for session_id in sessions:
            self.cold_sessions.add(session_id)
            
        logger.info(
            "Initialized targets: {} sessions, {} panels (auto-discover: sessions={}, panels={})",
            len(sessions), len(panels), self.auto_discover_sessions, self.auto_discover_panels
        )
    
    @staticmethod
    def _normalize_id_list(values: List[str]) -> tuple[List[str], bool]:
        """Normalize ID list and check for auto-discovery wildcard."""
        cleaned = [str(v).strip() for v in values if str(v).strip()]
        return (
            sorted({v for v in cleaned if v != "*"}),
            "*" in cleaned
        )
    
    def get_target_lock(self, target_kind: str, target_id: str) -> asyncio.Lock:
        """Get or create a lock for target operations."""
        key = f"{target_kind}:{target_id}"
        if key not in self._target_locks:
            self._target_locks[key] = asyncio.Lock()
        return self._target_locks[key]
    
    def is_cold_session(self, session_id: str) -> bool:
        """Check if session is in cold start state."""
        return session_id in self.cold_sessions
    
    def mark_session_warm(self, session_id: str) -> None:
        """Mark session as warmed up (history loaded)."""
        self.cold_sessions.discard(session_id)
    
    async def refresh_targets(self, subscribe_new: bool = False) -> None:
        """Refresh session and panel discovery."""
        try:
            if self.auto_discover_sessions:
                await self._refresh_sessions(subscribe_new)
            if self.auto_discover_panels:
                await self._refresh_panels(subscribe_new)
        except Exception as e:
            logger.warning("Target refresh failed: {}", e)
    
    async def _refresh_sessions(self, subscribe_new: bool) -> None:
        """Refresh session discovery."""
        try:
            response = await self.connection_manager.http_request(
                "POST", "/api/claw/sessions/list", 
                {}
            )
        except Exception as e:
            logger.warning("Session list request failed: {}", e)
            return
        
        sessions = response.get("sessions")
        if not isinstance(sessions, list):
            return
        
        new_sessions: List[str] = []
        
        for session_data in sessions:
            if not isinstance(session_data, dict):
                continue
                
            session_id = str_field(session_data, "sessionId")
            if not session_id:
                continue
                
            if session_id not in self.session_set:
                self.session_set.add(session_id)
                new_sessions.append(session_id) 
                self.cold_sessions.add(session_id)
                
            # Track conversation mapping
            converse_id = str_field(session_data, "converseId")
            if converse_id:
                self.session_by_converse[converse_id] = session_id
        
        if new_sessions:
            logger.info("Discovered {} new sessions", len(new_sessions))
            
            if subscribe_new and self.connection_manager.socket_client:
                await self._subscribe_sessions(new_sessions)
    
    async def _refresh_panels(self, subscribe_new: bool) -> None:
        """Refresh panel discovery."""
        try:
            response = await self.connection_manager.http_request(
                "POST",
                "/api/claw/groups/get",
                {}
            )
        except Exception as e:
            logger.warning("Panel list request failed: {}", e)
            return
        
        raw_panels = response.get("panels")
        if not isinstance(raw_panels, list):
            return
        
        new_panels: List[str] = []
        
        for panel_data in raw_panels:
            if not isinstance(panel_data, dict):
                continue
                
            # Only include type 0 panels (regular channels)
            panel_type = panel_data.get("type")
            if isinstance(panel_type, int) and panel_type != 0:
                continue
                
            panel_id = str_field(panel_data, "id", "_id")
            if panel_id and panel_id not in self.panel_set:
                self.panel_set.add(panel_id)
                new_panels.append(panel_id)
        
        if new_panels:
            logger.info("Discovered {} new panels", len(new_panels))
            
            if subscribe_new and self.connection_manager.socket_client:
                await self._subscribe_panels(new_panels)
    
    async def subscribe_all(self) -> bool:
        """Subscribe to all known targets."""
        if not self.connection_manager.socket_client:
            return False
            
        success = True
        success &= await self._subscribe_sessions(sorted(self.session_set))
        success &= await self._subscribe_panels(sorted(self.panel_set))
        
        return success
    
    async def _subscribe_sessions(self, session_ids: List[str]) -> bool:
        """Subscribe to session events."""
        if not session_ids or not self.connection_manager.socket_client:
            return True
            
        # Get cursors from state manager (we'll need to inject this)
        cursors = {sid: 0 for sid in session_ids}  # Placeholder
        
        try:
            result = await self._socket_call(
                "com.claw.im.subscribeSessions",
                {
                    "sessionIds": session_ids,
                    "cursors": cursors,
                    "limit": self.config.watch_limit,
                }
            )
            
            if not result.get("result"):
                logger.error(
                    "Session subscription failed: {}",
                    result.get("message", "unknown error")
                )
                return False
            
            logger.info("Subscribed to {} sessions", len(session_ids))
            return True
            
        except Exception as e:
            logger.error("Session subscription error: {}", e)
            return False
    
    async def _subscribe_panels(self, panel_ids: List[str]) -> bool:
        """Subscribe to panel events."""
        if not self.connection_manager.socket_client:
            return True
            
        try:
            result = await self._socket_call(
                "com.claw.im.subscribePanels",
                {"panelIds": panel_ids}
            )
            
            if not result.get("result"):
                logger.error(
                    "Panel subscription failed: {}", 
                    result.get("message", "unknown error")
                )
                return False
            
            logger.info("Subscribed to {} panels", len(panel_ids)) 
            return True
            
        except Exception as e:
            logger.error("Panel subscription error: {}", e)
            return False
    
    async def _socket_call(
        self, 
        event_name: str, 
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make a socket.io call with timeout."""
        if not self.connection_manager.socket_client:
            return {"result": False, "message": "socket not connected"}
            
        try:
            response = await self.connection_manager.socket_client.call(
                event_name, 
                payload, 
                timeout=10
            )
            return response if isinstance(response, dict) else {"result": True, "data": response}
            
        except Exception as e:
            return {"result": False, "message": str(e)}


class EventProcessor:
    """Processes inbound events and coordinates message handling."""
    
    def __init__(
        self,
        config: MochatConfig,
        target_manager: TargetManager,
        message_buffer: MessageBuffer,
        state_manager: StateManager,
        dispatch_callback: Callable[[str, str, str, Dict[str, Any]], Awaitable[None]]
    ) -> None:
        self.config = config
        self.target_manager = target_manager
        self.message_buffer = message_buffer
        self.state_manager = state_manager
        self.dispatch_callback = dispatch_callback
    
    async def handle_watch_payload(
        self, 
        payload: Dict[str, Any], 
        target_kind: TargetKind
    ) -> None:
        """Handle watch payload from websocket or polling."""
        if not isinstance(payload, dict):
            logger.debug("Invalid watch payload type: {}", type(payload))
            return
            
        target_id = str_field(payload, "sessionId")
        if not target_id:
            logger.debug("Watch payload missing sessionId")
            return
        
        # Get target lock for thread safety
        lock = self.target_manager.get_target_lock(target_kind.value, target_id)
        
        async with lock:
            # Update cursor for sessions
            if target_kind == TargetKind.SESSION:
                cursor = payload.get("cursor")
                if isinstance(cursor, int) and cursor >= 0:
                    self.state_manager.update_cursor(target_id, cursor)
                    
            # Process events
            events = payload.get("events")
            if not isinstance(events, list):
                return
                
            # Skip history for cold sessions
            if (target_kind == TargetKind.SESSION and 
                self.target_manager.is_cold_session(target_id)):
                self.target_manager.mark_session_warm(target_id)
                logger.debug("Warmed up session {}, skipping history", target_id)
                return
            
            for event in events:
                if not isinstance(event, dict):
                    continue
                    
                # Update cursor from event sequence
                if target_kind == TargetKind.SESSION:
                    seq = event.get("seq")
                    if isinstance(seq, int) and seq > self.state_manager.get_cursor(target_id):
                        self.state_manager.update_cursor(target_id, seq)
                
                # Process message events
                if event.get("type") == "message.add":
                    await self._process_message_event(target_id, event, target_kind)
    
    async def _process_message_event(
        self,
        target_id: str,
        event: Dict[str, Any],
        target_kind: TargetKind
    ) -> None:
        """Process a single message.add event."""
        payload = event.get("payload")
        if not isinstance(payload, dict):
            return
        
        # Extract message info
        author = str_field(payload, "author")
        if not author:
            return
            
        # Skip own messages
        if self.config.agent_user_id and author == self.config.agent_user_id:
            return
            
        message_id = str_field(payload, "messageId")
        target_key = f"{target_kind.value}:{target_id}"
        
        # Check for duplicates
        if message_id and self.message_buffer.is_duplicate_message(target_key, message_id):
            return
        
        # Build message entry
        raw_content = normalize_mochat_content(payload.get("content")) or "[empty message]"
        
        author_info = safe_dict(payload.get("authorInfo"))
        sender_name = str_field(author_info, "nickname", "email")
        sender_username = str_field(author_info, "agentId")
        
        entry = MochatBufferedEntry(
            raw_body=raw_content,
            author=author,
            sender_name=sender_name,
            sender_username=sender_username,
            timestamp=parse_timestamp(event.get("timestamp")),
            message_id=message_id,
            group_id=str_field(payload, "groupId")
        )
        
        # Determine processing logic
        group_id = str_field(payload, "groupId")
        is_group = bool(group_id)
        was_mentioned = resolve_was_mentioned(payload, self.config.agent_user_id)
        require_mention = (
            target_kind == TargetKind.PANEL and 
            is_group and 
            resolve_require_mention(self.config, target_id, group_id)
        )
        use_delay = (
            target_kind == TargetKind.PANEL and 
            self.config.reply_delay_mode == "non-mention"
        )
        
        # Process through buffer
        await self.message_buffer.process_entry(
            target_key=target_key,
            entry=entry,
            is_group=is_group,
            was_mentioned=was_mentioned,
            require_mention=require_mention,
            use_delay=use_delay,
            dispatch_callback=self._dispatch_buffered_messages
        )
    
    async def _dispatch_buffered_messages(
        self,
        target_id: str,
        target_kind: TargetKind,
        entries: List[MochatBufferedEntry],
        was_mentioned: bool
    ) -> None:
        """Dispatch buffered messages to the main handler."""
        if not entries:
            return
            
        last_entry = entries[-1]
        is_group = bool(last_entry.group_id)
        combined_body = build_buffered_body(entries, is_group)
        
        metadata = {
            "message_id": last_entry.message_id,
            "timestamp": last_entry.timestamp,
            "is_group": is_group,
            "group_id": last_entry.group_id,
            "sender_name": last_entry.sender_name,
            "sender_username": last_entry.sender_username,
            "target_kind": target_kind.value,
            "was_mentioned": was_mentioned,
            "buffered_count": len(entries),
            "correlation_id": str(last_entry.correlation_id),
        }
        
        await self.dispatch_callback(
            last_entry.author,
            target_id,
            combined_body,
            metadata
        )


# ---------------------------------------------------------------------------
# Main Channel Implementation
# ---------------------------------------------------------------------------

class MochatChannel(BaseChannel):
    """Mochat channel with comprehensive error handling and resource management.
    
    This implementation uses a modular architecture with separate components for:
    - Connection management (WebSocket + HTTP with retry logic)
    - Message buffering and deduplication 
    - Target/session management
    - Event processing
    - State persistence
    
    Features:
    - Type-safe async operations
    - Circuit breaker pattern for reliability
    - Comprehensive health monitoring
    - Graceful degradation (WebSocket -> HTTP polling)
    - Resource lifecycle management
    """
    
    name = "mochat"
    
    def __init__(self, config: MochatConfig, bus: MessageBus) -> None:
        super().__init__(config, bus)
        self.config: MochatConfig = config
        
        # Component initialization
        self._state_dir = get_data_path() / "mochat"
        self._connection_manager: Optional[ConnectionManager] = None
        self._message_buffer: Optional[MessageBuffer] = None
        self._target_manager: Optional[TargetManager] = None
        self._event_processor: Optional[EventProcessor] = None
        self._state_manager: Optional[StateManager] = None
        
        # Background tasks
        self._refresh_task: Optional[asyncio.Task[None]] = None
        self._fallback_tasks: Dict[str, asyncio.Task[None]] = {}
        
        # State tracking
        self._fallback_mode = False
        self._initialization_complete = False
        
        # Configuration validation
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration at startup."""
        if not self.config.claw_token:
            raise ValueError("Mochat claw_token is required")
        
        if not self.config.base_url:
            raise ValueError("Mochat base_url is required")
            
        # Validate URL format
        if not self.config.base_url.startswith(("http://", "https://")):
            raise ValueError("Mochat base_url must start with http:// or https://")
    
    async def start(self) -> None:
        """Start Mochat channel with all components."""
        try:
            self._running = True
            
            # Initialize components in order
            await self._initialize_components()
            
            # Setup event handlers
            await self._setup_event_handlers()
            
            # Start connections
            await self._start_connections()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self._initialization_complete = True
            logger.info("Mochat channel started successfully")
            
            # Keep running
            while self._running:
                await asyncio.sleep(1.0)
                
        except Exception as e:
            logger.error("Failed to start Mochat channel: {}", e)
            await self.stop()
            raise
    
    async def _initialize_components(self) -> None:
        """Initialize all component instances."""
        # State manager
        self._state_manager = StateManager(self._state_dir)
        await self._state_manager.load()
        
        # Connection manager
        retry_config = RetryConfig(
            max_attempts=self.config.max_retry_attempts or DEFAULT_RETRY_ATTEMPTS,
            base_delay_ms=self.config.retry_delay_ms or DEFAULT_RETRY_DELAY_MS
        )
        
        self._connection_manager = ConnectionManager(self.config, retry_config)
        
        # Target manager
        self._target_manager = TargetManager(self.config, self._connection_manager)
        
        # Message buffer
        self._message_buffer = MessageBuffer(self.config)
        
        # Event processor
        self._event_processor = EventProcessor(
            config=self.config,
            target_manager=self._target_manager,
            message_buffer=self._message_buffer,
            state_manager=self._state_manager,
            dispatch_callback=self._handle_processed_message
        )
    
    async def _setup_event_handlers(self) -> None:
        """Setup event handlers for connection events."""
        if not self._connection_manager:
            raise RuntimeError("Connection manager not initialized")
            
        self._connection_manager.set_event_handlers(
            on_connect=self._on_connection_established,
            on_disconnect=self._on_connection_lost,
            on_error=self._on_connection_error
        )
        
        # Setup socket event handlers if available
        if self._connection_manager.socket_client:
            await self._setup_socket_handlers()
    
    async def _setup_socket_handlers(self) -> None:
        """Setup socket.io event handlers."""
        if not self._connection_manager or not self._connection_manager.socket_client:
            return
            
        client = self._connection_manager.socket_client
        
        @client.on("claw.session.events")
        async def on_session_events(payload: Dict[str, Any]) -> None:
            if self._event_processor:
                await self._event_processor.handle_watch_payload(payload, TargetKind.SESSION)
        
        @client.on("claw.panel.events")
        async def on_panel_events(payload: Dict[str, Any]) -> None:
            if self._event_processor:
                await self._event_processor.handle_watch_payload(payload, TargetKind.PANEL)
        
        # Notify event handlers
        for event_name in (
            "notify:chat.inbox.append",
            "notify:chat.message.add", 
            "notify:chat.message.update",
            "notify:chat.message.recall",
            "notify:chat.message.delete"
        ):
            client.on(event_name, self._create_notify_handler(event_name))
    
    def _create_notify_handler(self, event_name: str) -> Callable[[Any], Awaitable[None]]:
        """Create notify event handler for specific event type."""
        async def handler(payload: Any) -> None:
            try:
                if event_name == "notify:chat.inbox.append":
                    await self._handle_notify_inbox_append(payload)
                elif event_name.startswith("notify:chat.message."):
                    await self._handle_notify_chat_message(payload)
            except Exception as e:
                logger.exception("Error handling notify event {}: {}", event_name, e)
        
        return handler
    
    async def _start_connections(self) -> None:
        """Start connection manager."""
        if not self._connection_manager:
            raise RuntimeError("Connection manager not initialized")
            
        await self._connection_manager.start()
    
    async def _start_background_tasks(self) -> None:
        """Start background refresh and monitoring tasks."""
        if not self._running:
            return
            
        # Start target refresh loop
        self._refresh_task = asyncio.create_task(self._refresh_loop())
        
        # Start fallback workers if needed
        if not self._connection_manager or not self._connection_manager.socket_client:
            await self._ensure_fallback_workers()
    
    async def _on_connection_established(self) -> None:
        """Handle successful connection establishment."""
        try:
            logger.info("Connection established, subscribing to targets")
            
            if self._target_manager:
                await self._target_manager.subscribe_all()
                await self._target_manager.refresh_targets(subscribe_new=True)
            
            # Stop fallback workers if WebSocket is connected
            if (self._connection_manager and 
                self._connection_manager.socket_client and 
                self._fallback_mode):
                await self._stop_fallback_workers()
                
        except Exception as e:
            logger.exception("Error in connection established handler: {}", e)
    
    async def _on_connection_lost(self) -> None:
        """Handle connection loss."""
        try:
            logger.warning("Connection lost, starting fallback mode")
            await self._ensure_fallback_workers()
        except Exception as e:
            logger.exception("Error in connection lost handler: {}", e)
    
    async def _on_connection_error(self, error: Exception) -> None:
        """Handle connection errors."""
        logger.error("Connection error: {}", error)
        
        # Ensure fallback workers are running
        try:
            await self._ensure_fallback_workers()
        except Exception as e:
            logger.exception("Error starting fallback workers: {}", e)
    
    async def _refresh_loop(self) -> None:
        """Background loop for target refresh and health monitoring."""
        interval = max(1.0, self.config.refresh_interval_ms / 1000.0)
        
        while self._running:
            try:
                await asyncio.sleep(interval)
                
                if not self._running:
                    break
                    
                # Refresh targets
                if self._target_manager:
                    ws_ready = (
                        self._connection_manager and 
                        self._connection_manager.socket_client and
                        self._connection_manager.connection_state == ConnectionState.READY
                    )
                    await self._target_manager.refresh_targets(subscribe_new=ws_ready)
                
                # Ensure fallback workers if needed
                if self._fallback_mode:
                    await self._ensure_fallback_workers()
                    
            except Exception as e:
                logger.warning("Refresh loop error: {}", e)
    
    async def _ensure_fallback_workers(self) -> None:
        """Ensure HTTP polling fallback workers are running."""
        if not self._running or not self._target_manager:
            return
            
        self._fallback_mode = True
        
        # Start session workers
        for session_id in self._target_manager.session_set:
            task_key = f"session:{session_id}"
            if task_key not in self._fallback_tasks or self._fallback_tasks[task_key].done():
                self._fallback_tasks[task_key] = asyncio.create_task(
                    self._session_fallback_worker(session_id)
                )
        
        # Start panel workers  
        for panel_id in self._target_manager.panel_set:
            task_key = f"panel:{panel_id}"
            if task_key not in self._fallback_tasks or self._fallback_tasks[task_key].done():
                self._fallback_tasks[task_key] = asyncio.create_task(
                    self._panel_fallback_worker(panel_id)
                )
    
    async def _stop_fallback_workers(self) -> None:
        """Stop all fallback polling workers."""
        self._fallback_mode = False
        
        tasks = list(self._fallback_tasks.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
        self._fallback_tasks.clear()
    
    async def _session_fallback_worker(self, session_id: str) -> None:
        """HTTP polling worker for a specific session."""
        while self._running and self._fallback_mode:
            try:
                if not self._connection_manager or not self._state_manager:
                    break
                    
                cursor = self._state_manager.get_cursor(session_id)
                
                response = await self._connection_manager.http_request(
                    "POST",
                    "/api/claw/sessions/watch",
                    {
                        "sessionId": session_id,
                        "cursor": cursor,
                        "timeoutMs": self.config.watch_timeout_ms,
                        "limit": self.config.watch_limit,
                    }
                )
                
                if self._event_processor:
                    await self._event_processor.handle_watch_payload(
                        response, TargetKind.SESSION
                    )
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(
                    "Session fallback worker error for {}: {}",
                    session_id, e
                )
                await asyncio.sleep(max(0.1, self.config.retry_delay_ms / 1000.0))
    
    async def _panel_fallback_worker(self, panel_id: str) -> None:
        """HTTP polling worker for a specific panel."""
        sleep_interval = max(1.0, self.config.refresh_interval_ms / 1000.0)
        
        while self._running and self._fallback_mode:
            try:
                if not self._connection_manager:
                    break
                    
                response = await self._connection_manager.http_request(
                    "POST",
                    "/api/claw/groups/panels/messages",
                    {
                        "panelId": panel_id,
                        "limit": min(100, max(1, self.config.watch_limit)),
                    }
                )
                
                messages = response.get("messages")
                if isinstance(messages, list):
                    for message_data in reversed(messages):
                        if not isinstance(message_data, dict):
                            continue
                            
                        # Create synthetic event
                        event = make_synthetic_event(
                            message_id=str(message_data.get("messageId") or ""),
                            author=str(message_data.get("author") or ""),
                            content=message_data.get("content"),
                            meta=message_data.get("meta"),
                            group_id=str(response.get("groupId") or ""),
                            converse_id=panel_id,
                            timestamp=message_data.get("createdAt"),
                            author_info=message_data.get("authorInfo"),
                        )
                        
                        if self._event_processor:
                            await self._event_processor._process_message_event(
                                panel_id, event, TargetKind.PANEL
                            )
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(
                    "Panel fallback worker error for {}: {}",
                    panel_id, e
                )
    
    async def _handle_processed_message(
        self, 
        sender_id: str, 
        chat_id: str, 
        content: str, 
        metadata: Dict[str, Any]
    ) -> None:
        """Handle message processed by EventProcessor."""
        try:
            # Check user permissions
            if not self.is_allowed(sender_id):
                logger.debug("Message from {} blocked by permissions", sender_id)
                return
            
            # Dispatch to message handler
            await self._handle_message(
                sender_id=sender_id,
                chat_id=chat_id,
                content=content,
                metadata=metadata
            )
            
        except Exception as e:
            logger.exception(
                "Error handling processed message from {}: {}", 
                sender_id, e
            )
    
    async def _handle_notify_inbox_append(self, payload: Any) -> None:
        """Handle notify:chat.inbox.append events."""
        try:
            if not isinstance(payload, dict) or payload.get("type") != "message":
                return
                
            detail = payload.get("payload")
            if not isinstance(detail, dict):
                return
                
            # Skip group messages (handled elsewhere)
            if str_field(detail, "groupId"):
                return
                
            converse_id = str_field(detail, "converseId")
            if not converse_id:
                return
            
            # Find session ID from conversation mapping
            session_id = None
            if self._target_manager:
                session_id = self._target_manager.session_by_converse.get(converse_id)
                
            if not session_id:
                # Try refreshing session directory
                if self._target_manager:
                    ws_ready = (
                        self._connection_manager and 
                        self._connection_manager.socket_client and
                        self._connection_manager.connection_state == ConnectionState.READY
                    )
                    await self._target_manager._refresh_sessions(ws_ready)
                    session_id = self._target_manager.session_by_converse.get(converse_id)
                    
            if not session_id:
                logger.debug("Unknown conversation ID: {}", converse_id)
                return
            
            # Create synthetic message event
            event = make_synthetic_event(
                message_id=str(detail.get("messageId") or payload.get("_id") or ""),
                author=str(detail.get("messageAuthor") or ""),
                content=str(
                    detail.get("messagePlainContent") or 
                    detail.get("messageSnippet") or ""
                ),
                meta={
                    "source": "notify:chat.inbox.append",
                    "converseId": converse_id
                },
                group_id="",
                converse_id=converse_id,
                timestamp=payload.get("createdAt"),
            )
            
            if self._event_processor:
                await self._event_processor._process_message_event(
                    session_id, event, TargetKind.SESSION
                )
                
        except Exception as e:
            logger.exception("Error handling inbox append notification: {}", e)
    
    async def _handle_notify_chat_message(self, payload: Any) -> None:
        """Handle notify:chat.message.* events."""
        try:
            if not isinstance(payload, dict):
                return
                
            group_id = str_field(payload, "groupId")
            panel_id = str_field(payload, "converseId", "panelId")
            
            if not group_id or not panel_id:
                return
                
            # Check if we're monitoring this panel
            if self._target_manager and panel_id not in self._target_manager.panel_set:
                return
            
            # Create synthetic event
            event = make_synthetic_event(
                message_id=str(
                    payload.get("_id") or 
                    payload.get("messageId") or ""
                ),
                author=str(payload.get("author") or ""),
                content=payload.get("content"),
                meta=payload.get("meta"),
                group_id=group_id,
                converse_id=panel_id,
                timestamp=payload.get("createdAt"),
                author_info=payload.get("authorInfo"),
            )
            
            if self._event_processor:
                await self._event_processor._process_message_event(
                    panel_id, event, TargetKind.PANEL
                )
                
        except Exception as e:
            logger.exception("Error handling chat message notification: {}", e)
    
    async def send(self, msg: OutboundMessage) -> None:
        """Send outbound message to session or panel."""
        if not self._initialization_complete:
            logger.warning("Cannot send message - channel not initialized")
            return
            
        try:
            # Build content
            parts = []
            if msg.content and msg.content.strip():
                parts.append(msg.content.strip())
            if msg.media:
                parts.extend(
                    media for media in msg.media 
                    if isinstance(media, str) and media.strip()
                )
            
            content = "\n".join(parts).strip()
            if not content:
                logger.debug("Skipping empty message")
                return
            
            # Resolve target
            try:
                target = resolve_mochat_target(msg.chat_id)
            except ValueError as e:
                logger.error("Invalid target '{}': {}", msg.chat_id, e)
                return
            
            # Determine target type
            is_panel = (
                target.is_panel or 
                (self._target_manager and target.id in self._target_manager.panel_set)
            ) and not target.id.startswith("session_")
            
            # Send via appropriate API
            correlation_id = CorrelationId()
            
            try:
                if is_panel:
                    await self._send_panel_message(
                        target.id, content, msg.reply_to, 
                        self._extract_group_id(msg.metadata),
                        correlation_id
                    )
                else:
                    await self._send_session_message(
                        target.id, content, msg.reply_to, correlation_id
                    )
                    
                logger.debug(
                    "Sent message to {} {} [{}]",
                    "panel" if is_panel else "session",
                    target.id,
                    correlation_id
                )
                
            except Exception as e:
                logger.error(
                    "Failed to send message to {}: {} [{}]",
                    msg.chat_id, e, correlation_id
                )
                
        except Exception as e:
            logger.exception("Error in send method: {}", e)
    
    async def _send_session_message(
        self,
        session_id: str,
        content: str,
        reply_to: Optional[str],
        correlation_id: CorrelationId
    ) -> None:
        """Send message to a session."""
        if not self._connection_manager:
            raise ConnectionError("Connection manager not available")
            
        payload = {
            "sessionId": session_id,
            "content": content
        }
        
        if reply_to:
            payload["replyTo"] = reply_to
            
        await self._connection_manager.http_request(
            "POST",
            "/api/claw/sessions/send",
            payload,
            correlation_id
        )
    
    async def _send_panel_message(
        self,
        panel_id: str,
        content: str,
        reply_to: Optional[str],
        group_id: Optional[str],
        correlation_id: CorrelationId
    ) -> None:
        """Send message to a panel."""
        if not self._connection_manager:
            raise ConnectionError("Connection manager not available")
            
        payload = {
            "panelId": panel_id,
            "content": content
        }
        
        if reply_to:
            payload["replyTo"] = reply_to
            
        if group_id:
            payload["groupId"] = group_id
            
        await self._connection_manager.http_request(
            "POST",
            "/api/claw/groups/panels/send",
            payload,
            correlation_id
        )
    
    @staticmethod
    def _extract_group_id(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        """Extract group ID from message metadata."""
        if not isinstance(metadata, dict):
            return None
            
        value = metadata.get("group_id") or metadata.get("groupId")
        return value.strip() if isinstance(value, str) and value.strip() else None
    
    async def stop(self) -> None:
        """Stop all components and clean up resources."""
        logger.info("Stopping Mochat channel...")
        self._running = False
        
        try:
            # Stop background tasks
            if self._refresh_task and not self._refresh_task.done():
                self._refresh_task.cancel()
                try:
                    await self._refresh_task
                except asyncio.CancelledError:
                    pass
                self._refresh_task = None
            
            # Stop fallback workers
            await self._stop_fallback_workers()
            
            # Cleanup components (order matters)
            if self._message_buffer:
                await self._message_buffer.cleanup()
                self._message_buffer = None
            
            if self._state_manager:
                await self._state_manager.save(force=True)
                self._state_manager = None
            
            if self._connection_manager:
                await self._connection_manager.stop()
                self._connection_manager = None
            
            # Clear references
            self._event_processor = None
            self._target_manager = None
            
            logger.info("Mochat channel stopped")
            
        except Exception as e:
            logger.exception("Error during channel shutdown: {}", e)
    
    async def get_health_status(self) -> HealthStatus:
        """Get comprehensive health status."""
        if not self._connection_manager:
            return HealthStatus(
                is_healthy=False,
                connection_state=ConnectionState.DISCONNECTED,
                metrics=ConnectionMetrics(),
                issues=["Not initialized"]
            )
            
        return await self._connection_manager.get_health_status()
    
    @property
    def is_ready(self) -> bool:
        """Check if channel is ready for operations."""
        return (
            self._initialization_complete and
            self._running and
            self._connection_manager is not None and
            self._connection_manager.is_connected
        )
    
    @property 
    def connection_state(self) -> ConnectionState:
        """Get current connection state."""
        if not self._connection_manager:
            return ConnectionState.DISCONNECTED
        return self._connection_manager.connection_state
    
    @property
    def is_websocket_connected(self) -> bool:
        """Check if WebSocket connection is active."""
        return (
            self._connection_manager is not None and
            self._connection_manager.socket_client is not None and
            self._connection_manager.connection_state in {
                ConnectionState.CONNECTED, 
                ConnectionState.READY
            }
        )


# ---------------------------------------------------------------------------
# Backward Compatibility Layer
# ---------------------------------------------------------------------------

# Re-export old function names for backward compatibility
_safe_dict = safe_dict
_str_field = str_field 
_make_synthetic_event = make_synthetic_event

    async def stop(self) -> None:
        """Stop all workers and clean up resources."""
        self._running = False
        if self._refresh_task:
            self._refresh_task.cancel()
            self._refresh_task = None

        await self._stop_fallback_workers()
        await self._cancel_delay_timers()

        if self._socket:
            try:
                await self._socket.disconnect()
            except Exception:
                pass
            self._socket = None

        if self._cursor_save_task:
            self._cursor_save_task.cancel()
            self._cursor_save_task = None
        await self._save_session_cursors()

        if self._http:
            await self._http.aclose()
            self._http = None
        self._ws_connected = self._ws_ready = False

    async def send(self, msg: OutboundMessage) -> None:
        """Send outbound message to session or panel."""
        if not self.config.claw_token:
            logger.warning("Mochat claw_token missing, skip send")
            return

        parts = ([msg.content.strip()] if msg.content and msg.content.strip() else [])
        if msg.media:
            parts.extend(m for m in msg.media if isinstance(m, str) and m.strip())
        content = "\n".join(parts).strip()
        if not content:
            return

        target = resolve_mochat_target(msg.chat_id)
        if not target.id:
            logger.warning("Mochat outbound target is empty")
            return

        is_panel = (target.is_panel or target.id in self._panel_set) and not target.id.startswith("session_")
        try:
            if is_panel:
                await self._api_send("/api/claw/groups/panels/send", "panelId", target.id,
                                     content, msg.reply_to, self._read_group_id(msg.metadata))
            else:
                await self._api_send("/api/claw/sessions/send", "sessionId", target.id,
                                     content, msg.reply_to)
        except Exception as e:
            logger.error("Failed to send Mochat message: {}", e)

    # ---- config / init helpers ---------------------------------------------

    def _seed_targets_from_config(self) -> None:
        sessions, self._auto_discover_sessions = self._normalize_id_list(self.config.sessions)
        panels, self._auto_discover_panels = self._normalize_id_list(self.config.panels)
        self._session_set.update(sessions)
        self._panel_set.update(panels)
        for sid in sessions:
            if sid not in self._session_cursor:
                self._cold_sessions.add(sid)

    @staticmethod
    def _normalize_id_list(values: list[str]) -> tuple[list[str], bool]:
        cleaned = [str(v).strip() for v in values if str(v).strip()]
        return sorted({v for v in cleaned if v != "*"}), "*" in cleaned

    # ---- websocket ---------------------------------------------------------

    async def _start_socket_client(self) -> bool:
        if not SOCKETIO_AVAILABLE:
            logger.warning("python-socketio not installed, Mochat using polling fallback")
            return False

        serializer = "default"
        if not self.config.socket_disable_msgpack:
            if MSGPACK_AVAILABLE:
                serializer = "msgpack"
            else:
                logger.warning("msgpack not installed but socket_disable_msgpack=false; using JSON")

        client = socketio.AsyncClient(
            reconnection=True,
            reconnection_attempts=self.config.max_retry_attempts or None,
            reconnection_delay=max(0.1, self.config.socket_reconnect_delay_ms / 1000.0),
            reconnection_delay_max=max(0.1, self.config.socket_max_reconnect_delay_ms / 1000.0),
            logger=False, engineio_logger=False, serializer=serializer,
        )

        @client.event
        async def connect() -> None:
            self._ws_connected, self._ws_ready = True, False
            logger.info("Mochat websocket connected")
            subscribed = await self._subscribe_all()
            self._ws_ready = subscribed
            await (self._stop_fallback_workers() if subscribed else self._ensure_fallback_workers())

        @client.event
        async def disconnect() -> None:
            if not self._running:
                return
            self._ws_connected = self._ws_ready = False
            logger.warning("Mochat websocket disconnected")
            await self._ensure_fallback_workers()

        @client.event
        async def connect_error(data: Any) -> None:
            logger.error("Mochat websocket connect error: {}", data)

        @client.on("claw.session.events")
        async def on_session_events(payload: dict[str, Any]) -> None:
            await self._handle_watch_payload(payload, "session")

        @client.on("claw.panel.events")
        async def on_panel_events(payload: dict[str, Any]) -> None:
            await self._handle_watch_payload(payload, "panel")

        for ev in ("notify:chat.inbox.append", "notify:chat.message.add",
                    "notify:chat.message.update", "notify:chat.message.recall",
                    "notify:chat.message.delete"):
            client.on(ev, self._build_notify_handler(ev))

        socket_url = (self.config.socket_url or self.config.base_url).strip().rstrip("/")
        socket_path = (self.config.socket_path or "/socket.io").strip().lstrip("/")

        try:
            self._socket = client
            await client.connect(
                socket_url, transports=["websocket"], socketio_path=socket_path,
                auth={"token": self.config.claw_token},
                wait_timeout=max(1.0, self.config.socket_connect_timeout_ms / 1000.0),
            )
            return True
        except Exception as e:
            logger.error("Failed to connect Mochat websocket: {}", e)
            try:
                await client.disconnect()
            except Exception:
                pass
            self._socket = None
            return False

    def _build_notify_handler(self, event_name: str):
        async def handler(payload: Any) -> None:
            if event_name == "notify:chat.inbox.append":
                await self._handle_notify_inbox_append(payload)
            elif event_name.startswith("notify:chat.message."):
                await self._handle_notify_chat_message(payload)
        return handler

    # ---- subscribe ---------------------------------------------------------

    async def _subscribe_all(self) -> bool:
        ok = await self._subscribe_sessions(sorted(self._session_set))
        ok = await self._subscribe_panels(sorted(self._panel_set)) and ok
        if self._auto_discover_sessions or self._auto_discover_panels:
            await self._refresh_targets(subscribe_new=True)
        return ok

    async def _subscribe_sessions(self, session_ids: list[str]) -> bool:
        if not session_ids:
            return True
        for sid in session_ids:
            if sid not in self._session_cursor:
                self._cold_sessions.add(sid)

        ack = await self._socket_call("com.claw.im.subscribeSessions", {
            "sessionIds": session_ids, "cursors": self._session_cursor,
            "limit": self.config.watch_limit,
        })
        if not ack.get("result"):
            logger.error("Mochat subscribeSessions failed: {}", ack.get('message', 'unknown error'))
            return False

        data = ack.get("data")
        items: list[dict[str, Any]] = []
        if isinstance(data, list):
            items = [i for i in data if isinstance(i, dict)]
        elif isinstance(data, dict):
            sessions = data.get("sessions")
            if isinstance(sessions, list):
                items = [i for i in sessions if isinstance(i, dict)]
            elif "sessionId" in data:
                items = [data]
        for p in items:
            await self._handle_watch_payload(p, "session")
        return True

    async def _subscribe_panels(self, panel_ids: list[str]) -> bool:
        if not self._auto_discover_panels and not panel_ids:
            return True
        ack = await self._socket_call("com.claw.im.subscribePanels", {"panelIds": panel_ids})
        if not ack.get("result"):
            logger.error("Mochat subscribePanels failed: {}", ack.get('message', 'unknown error'))
            return False
        return True

    async def _socket_call(self, event_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._socket:
            return {"result": False, "message": "socket not connected"}
        try:
            raw = await self._socket.call(event_name, payload, timeout=10)
        except Exception as e:
            return {"result": False, "message": str(e)}
        return raw if isinstance(raw, dict) else {"result": True, "data": raw}

    # ---- refresh / discovery -----------------------------------------------

    async def _refresh_loop(self) -> None:
        interval_s = max(1.0, self.config.refresh_interval_ms / 1000.0)
        while self._running:
            await asyncio.sleep(interval_s)
            try:
                await self._refresh_targets(subscribe_new=self._ws_ready)
            except Exception as e:
                logger.warning("Mochat refresh failed: {}", e)
            if self._fallback_mode:
                await self._ensure_fallback_workers()

    async def _refresh_targets(self, subscribe_new: bool) -> None:
        if self._auto_discover_sessions:
            await self._refresh_sessions_directory(subscribe_new)
        if self._auto_discover_panels:
            await self._refresh_panels(subscribe_new)

    async def _refresh_sessions_directory(self, subscribe_new: bool) -> None:
        try:
            response = await self._post_json("/api/claw/sessions/list", {})
        except Exception as e:
            logger.warning("Mochat listSessions failed: {}", e)
            return

        sessions = response.get("sessions")
        if not isinstance(sessions, list):
            return

        new_ids: list[str] = []
        for s in sessions:
            if not isinstance(s, dict):
                continue
            sid = _str_field(s, "sessionId")
            if not sid:
                continue
            if sid not in self._session_set:
                self._session_set.add(sid)
                new_ids.append(sid)
                if sid not in self._session_cursor:
                    self._cold_sessions.add(sid)
            cid = _str_field(s, "converseId")
            if cid:
                self._session_by_converse[cid] = sid

        if not new_ids:
            return
        if self._ws_ready and subscribe_new:
            await self._subscribe_sessions(new_ids)
        if self._fallback_mode:
            await self._ensure_fallback_workers()

    async def _refresh_panels(self, subscribe_new: bool) -> None:
        try:
            response = await self._post_json("/api/claw/groups/get", {})
        except Exception as e:
            logger.warning("Mochat getWorkspaceGroup failed: {}", e)
            return

        raw_panels = response.get("panels")
        if not isinstance(raw_panels, list):
            return

        new_ids: list[str] = []
        for p in raw_panels:
            if not isinstance(p, dict):
                continue
            pt = p.get("type")
            if isinstance(pt, int) and pt != 0:
                continue
            pid = _str_field(p, "id", "_id")
            if pid and pid not in self._panel_set:
                self._panel_set.add(pid)
                new_ids.append(pid)

        if not new_ids:
            return
        if self._ws_ready and subscribe_new:
            await self._subscribe_panels(new_ids)
        if self._fallback_mode:
            await self._ensure_fallback_workers()

    # ---- fallback workers --------------------------------------------------

    async def _ensure_fallback_workers(self) -> None:
        if not self._running:
            return
        self._fallback_mode = True
        for sid in sorted(self._session_set):
            t = self._session_fallback_tasks.get(sid)
            if not t or t.done():
                self._session_fallback_tasks[sid] = asyncio.create_task(self._session_watch_worker(sid))
        for pid in sorted(self._panel_set):
            t = self._panel_fallback_tasks.get(pid)
            if not t or t.done():
                self._panel_fallback_tasks[pid] = asyncio.create_task(self._panel_poll_worker(pid))

    async def _stop_fallback_workers(self) -> None:
        self._fallback_mode = False
        tasks = [*self._session_fallback_tasks.values(), *self._panel_fallback_tasks.values()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._session_fallback_tasks.clear()
        self._panel_fallback_tasks.clear()

    async def _session_watch_worker(self, session_id: str) -> None:
        while self._running and self._fallback_mode:
            try:
                payload = await self._post_json("/api/claw/sessions/watch", {
                    "sessionId": session_id, "cursor": self._session_cursor.get(session_id, 0),
                    "timeoutMs": self.config.watch_timeout_ms, "limit": self.config.watch_limit,
                })
                await self._handle_watch_payload(payload, "session")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Mochat watch fallback error ({}): {}", session_id, e)
                await asyncio.sleep(max(0.1, self.config.retry_delay_ms / 1000.0))

    async def _panel_poll_worker(self, panel_id: str) -> None:
        sleep_s = max(1.0, self.config.refresh_interval_ms / 1000.0)
        while self._running and self._fallback_mode:
            try:
                resp = await self._post_json("/api/claw/groups/panels/messages", {
                    "panelId": panel_id, "limit": min(100, max(1, self.config.watch_limit)),
                })
                msgs = resp.get("messages")
                if isinstance(msgs, list):
                    for m in reversed(msgs):
                        if not isinstance(m, dict):
                            continue
                        evt = _make_synthetic_event(
                            message_id=str(m.get("messageId") or ""),
                            author=str(m.get("author") or ""),
                            content=m.get("content"),
                            meta=m.get("meta"), group_id=str(resp.get("groupId") or ""),
                            converse_id=panel_id, timestamp=m.get("createdAt"),
                            author_info=m.get("authorInfo"),
                        )
                        await self._process_inbound_event(panel_id, evt, "panel")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Mochat panel polling error ({}): {}", panel_id, e)
            await asyncio.sleep(sleep_s)

    # ---- inbound event processing ------------------------------------------

    async def _handle_watch_payload(self, payload: dict[str, Any], target_kind: str) -> None:
        if not isinstance(payload, dict):
            return
        target_id = _str_field(payload, "sessionId")
        if not target_id:
            return

        lock = self._target_locks.setdefault(f"{target_kind}:{target_id}", asyncio.Lock())
        async with lock:
            prev = self._session_cursor.get(target_id, 0) if target_kind == "session" else 0
            pc = payload.get("cursor")
            if target_kind == "session" and isinstance(pc, int) and pc >= 0:
                self._mark_session_cursor(target_id, pc)

            raw_events = payload.get("events")
            if not isinstance(raw_events, list):
                return
            if target_kind == "session" and target_id in self._cold_sessions:
                self._cold_sessions.discard(target_id)
                return

            for event in raw_events:
                if not isinstance(event, dict):
                    continue
                seq = event.get("seq")
                if target_kind == "session" and isinstance(seq, int) and seq > self._session_cursor.get(target_id, prev):
                    self._mark_session_cursor(target_id, seq)
                if event.get("type") == "message.add":
                    await self._process_inbound_event(target_id, event, target_kind)

    async def _process_inbound_event(self, target_id: str, event: dict[str, Any], target_kind: str) -> None:
        payload = event.get("payload")
        if not isinstance(payload, dict):
            return

        author = _str_field(payload, "author")
        if not author or (self.config.agent_user_id and author == self.config.agent_user_id):
            return
        if not self.is_allowed(author):
            return

        message_id = _str_field(payload, "messageId")
        seen_key = f"{target_kind}:{target_id}"
        if message_id and self._remember_message_id(seen_key, message_id):
            return

        raw_body = normalize_mochat_content(payload.get("content")) or "[empty message]"
        ai = _safe_dict(payload.get("authorInfo"))
        sender_name = _str_field(ai, "nickname", "email")
        sender_username = _str_field(ai, "agentId")

        group_id = _str_field(payload, "groupId")
        is_group = bool(group_id)
        was_mentioned = resolve_was_mentioned(payload, self.config.agent_user_id)
        require_mention = target_kind == "panel" and is_group and resolve_require_mention(self.config, target_id, group_id)
        use_delay = target_kind == "panel" and self.config.reply_delay_mode == "non-mention"

        if require_mention and not was_mentioned and not use_delay:
            return

        entry = MochatBufferedEntry(
            raw_body=raw_body, author=author, sender_name=sender_name,
            sender_username=sender_username, timestamp=parse_timestamp(event.get("timestamp")),
            message_id=message_id, group_id=group_id,
        )

        if use_delay:
            delay_key = seen_key
            if was_mentioned:
                await self._flush_delayed_entries(delay_key, target_id, target_kind, "mention", entry)
            else:
                await self._enqueue_delayed_entry(delay_key, target_id, target_kind, entry)
            return

        await self._dispatch_entries(target_id, target_kind, [entry], was_mentioned)

    # ---- dedup / buffering -------------------------------------------------

    def _remember_message_id(self, key: str, message_id: str) -> bool:
        seen_set = self._seen_set.setdefault(key, set())
        seen_queue = self._seen_queue.setdefault(key, deque())
        if message_id in seen_set:
            return True
        seen_set.add(message_id)
        seen_queue.append(message_id)
        while len(seen_queue) > MAX_SEEN_MESSAGE_IDS:
            seen_set.discard(seen_queue.popleft())
        return False

    async def _enqueue_delayed_entry(self, key: str, target_id: str, target_kind: str, entry: MochatBufferedEntry) -> None:
        state = self._delay_states.setdefault(key, DelayState())
        async with state.lock:
            state.entries.append(entry)
            if state.timer:
                state.timer.cancel()
            state.timer = asyncio.create_task(self._delay_flush_after(key, target_id, target_kind))

    async def _delay_flush_after(self, key: str, target_id: str, target_kind: str) -> None:
        await asyncio.sleep(max(0, self.config.reply_delay_ms) / 1000.0)
        await self._flush_delayed_entries(key, target_id, target_kind, "timer", None)

    async def _flush_delayed_entries(self, key: str, target_id: str, target_kind: str, reason: str, entry: MochatBufferedEntry | None) -> None:
        state = self._delay_states.setdefault(key, DelayState())
        async with state.lock:
            if entry:
                state.entries.append(entry)
            current = asyncio.current_task()
            if state.timer and state.timer is not current:
                state.timer.cancel()
            state.timer = None
            entries = state.entries[:]
            state.entries.clear()
        if entries:
            await self._dispatch_entries(target_id, target_kind, entries, reason == "mention")

    async def _dispatch_entries(self, target_id: str, target_kind: str, entries: list[MochatBufferedEntry], was_mentioned: bool) -> None:
        if not entries:
            return
        last = entries[-1]
        is_group = bool(last.group_id)
        body = build_buffered_body(entries, is_group) or "[empty message]"
        await self._handle_message(
            sender_id=last.author, chat_id=target_id, content=body,
            metadata={
                "message_id": last.message_id, "timestamp": last.timestamp,
                "is_group": is_group, "group_id": last.group_id,
                "sender_name": last.sender_name, "sender_username": last.sender_username,
                "target_kind": target_kind, "was_mentioned": was_mentioned,
                "buffered_count": len(entries),
            },
        )

    async def _cancel_delay_timers(self) -> None:
        for state in self._delay_states.values():
            if state.timer:
                state.timer.cancel()
        self._delay_states.clear()

    # ---- notify handlers ---------------------------------------------------

    async def _handle_notify_chat_message(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        group_id = _str_field(payload, "groupId")
        panel_id = _str_field(payload, "converseId", "panelId")
        if not group_id or not panel_id:
            return
        if self._panel_set and panel_id not in self._panel_set:
            return

        evt = _make_synthetic_event(
            message_id=str(payload.get("_id") or payload.get("messageId") or ""),
            author=str(payload.get("author") or ""),
            content=payload.get("content"), meta=payload.get("meta"),
            group_id=group_id, converse_id=panel_id,
            timestamp=payload.get("createdAt"), author_info=payload.get("authorInfo"),
        )
        await self._process_inbound_event(panel_id, evt, "panel")

    async def _handle_notify_inbox_append(self, payload: Any) -> None:
        if not isinstance(payload, dict) or payload.get("type") != "message":
            return
        detail = payload.get("payload")
        if not isinstance(detail, dict):
            return
        if _str_field(detail, "groupId"):
            return
        converse_id = _str_field(detail, "converseId")
        if not converse_id:
            return

        session_id = self._session_by_converse.get(converse_id)
        if not session_id:
            await self._refresh_sessions_directory(self._ws_ready)
            session_id = self._session_by_converse.get(converse_id)
        if not session_id:
            return

        evt = _make_synthetic_event(
            message_id=str(detail.get("messageId") or payload.get("_id") or ""),
            author=str(detail.get("messageAuthor") or ""),
            content=str(detail.get("messagePlainContent") or detail.get("messageSnippet") or ""),
            meta={"source": "notify:chat.inbox.append", "converseId": converse_id},
            group_id="", converse_id=converse_id, timestamp=payload.get("createdAt"),
        )
        await self._process_inbound_event(session_id, evt, "session")

    # ---- cursor persistence ------------------------------------------------

    def _mark_session_cursor(self, session_id: str, cursor: int) -> None:
        if cursor < 0 or cursor < self._session_cursor.get(session_id, 0):
            return
        self._session_cursor[session_id] = cursor
        if not self._cursor_save_task or self._cursor_save_task.done():
            self._cursor_save_task = asyncio.create_task(self._save_cursor_debounced())

    async def _save_cursor_debounced(self) -> None:
        await asyncio.sleep(CURSOR_SAVE_DEBOUNCE_S)
        await self._save_session_cursors()

    async def _load_session_cursors(self) -> None:
        if not self._cursor_path.exists():
            return
        try:
            data = json.loads(self._cursor_path.read_text("utf-8"))
        except Exception as e:
            logger.warning("Failed to read Mochat cursor file: {}", e)
            return
        cursors = data.get("cursors") if isinstance(data, dict) else None
        if isinstance(cursors, dict):
            for sid, cur in cursors.items():
                if isinstance(sid, str) and isinstance(cur, int) and cur >= 0:
                    self._session_cursor[sid] = cur

    async def _save_session_cursors(self) -> None:
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            self._cursor_path.write_text(json.dumps({
                "schemaVersion": 1, "updatedAt": datetime.utcnow().isoformat(),
                "cursors": self._session_cursor,
            }, ensure_ascii=False, indent=2) + "\n", "utf-8")
        except Exception as e:
            logger.warning("Failed to save Mochat cursor file: {}", e)

    # ---- HTTP helpers ------------------------------------------------------

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._http:
            raise RuntimeError("Mochat HTTP client not initialized")
        url = f"{self.config.base_url.strip().rstrip('/')}{path}"
        response = await self._http.post(url, headers={
            "Content-Type": "application/json", "X-Claw-Token": self.config.claw_token,
        }, json=payload)
        if not response.is_success:
            raise RuntimeError(f"Mochat HTTP {response.status_code}: {response.text[:200]}")
        try:
            parsed = response.json()
        except Exception:
            parsed = response.text
        if isinstance(parsed, dict) and isinstance(parsed.get("code"), int):
            if parsed["code"] != 200:
                msg = str(parsed.get("message") or parsed.get("name") or "request failed")
                raise RuntimeError(f"Mochat API error: {msg} (code={parsed['code']})")
            data = parsed.get("data")
            return data if isinstance(data, dict) else {}
        return parsed if isinstance(parsed, dict) else {}

    async def _api_send(self, path: str, id_key: str, id_val: str,
                        content: str, reply_to: str | None, group_id: str | None = None) -> dict[str, Any]:
        """Unified send helper for session and panel messages."""
        body: dict[str, Any] = {id_key: id_val, "content": content}
        if reply_to:
            body["replyTo"] = reply_to
        if group_id:
            body["groupId"] = group_id
        return await self._post_json(path, body)

    @staticmethod
    def _read_group_id(metadata: dict[str, Any]) -> str | None:
        if not isinstance(metadata, dict):
            return None
        value = metadata.get("group_id") or metadata.get("groupId")
        return value.strip() if isinstance(value, str) and value.strip() else None
