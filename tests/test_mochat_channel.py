"""Comprehensive unit tests for MochatChannel and supporting classes.

This test suite provides extensive coverage for:
- ConnectionManager with retry logic and health monitoring  
- MessageBuffer with deduplication and delayed processing
- StateManager with persistence
- TargetManager with discovery
- EventProcessor with message handling
- MochatChannel integration

Tests use proper async patterns and mocking for external dependencies.
"""

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import uuid4

import pytest
import httpx

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import MochatConfig
from nanobot.channels.mochat import (
    CircuitBreaker,
    MochatConnectionError,
    ConnectionManager,
    ConnectionState,
    CorrelationId,
    EventProcessor,
    MochatBufferedEntry,
    MochatChannel,
    MochatError,
    MessageBuffer,
    RetryConfig,
    RetryExhaustedError,
    StateManager,
    TargetKind,
    TargetManager,
    build_buffered_body,
    extract_mention_ids,
    make_synthetic_event,
    normalize_mochat_content,
    parse_timestamp,
    resolve_mochat_target,
    resolve_was_mentioned,
    safe_dict,
    str_field,
)


# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_config():
    """Create a mock MochatConfig for testing."""
    config = Mock(spec=MochatConfig)
    config.claw_token = "test_token_123"
    config.base_url = "https://api.example.com"
    config.socket_url = None
    config.socket_path = "/socket.io"
    config.socket_disable_msgpack = False
    config.max_retry_attempts = 3
    config.socket_reconnect_delay_ms = 1000
    config.socket_max_reconnect_delay_ms = 30000
    config.socket_connect_timeout_ms = 5000
    config.watch_timeout_ms = 30000
    config.watch_limit = 50
    config.refresh_interval_ms = 60000
    config.retry_delay_ms = 1000
    config.reply_delay_ms = 2000
    config.reply_delay_mode = "non-mention"
    config.agent_user_id = "agent_123"
    config.sessions = ["session_1", "session_2"]
    config.panels = ["panel_1", "*"]
    config.groups = {}
    config.mention = Mock(require_in_groups=True)
    return config


@pytest.fixture
def mock_bus():
    """Create a mock MessageBus for testing."""
    return Mock(spec=MessageBus)


@pytest.fixture
def temp_state_dir():
    """Create a temporary directory for state management tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def correlation_id():
    """Create a test correlation ID."""
    return CorrelationId()


# ---------------------------------------------------------------------------
# Test utility functions
# ---------------------------------------------------------------------------

class TestUtilityFunctions:
    """Test pure utility functions."""
    
    def test_safe_dict(self):
        """Test safe_dict function."""
        assert safe_dict({"key": "value"}) == {"key": "value"}
        assert safe_dict(None) == {}
        assert safe_dict("string") == {}
        assert safe_dict(123) == {}
        assert safe_dict([]) == {}
    
    def test_str_field(self):
        """Test str_field function."""
        data = {"name": "test", "empty": "", "number": 123, "none": None}
        
        assert str_field(data, "name") == "test"
        assert str_field(data, "empty") == ""
        assert str_field(data, "number") == ""
        assert str_field(data, "none") == ""
        assert str_field(data, "missing") == ""
        assert str_field(data, "missing", "name") == "test"
        assert str_field(data, "missing", "empty", "name") == "test"
    
    def test_normalize_mochat_content(self):
        """Test content normalization."""
        assert normalize_mochat_content("hello") == "hello"
        assert normalize_mochat_content("  test  ") == "test"
        assert normalize_mochat_content(None) == ""
        assert normalize_mochat_content({"msg": "hello"}) == '{"msg": "hello"}'
        assert normalize_mochat_content(123) == "123"
    
    def test_resolve_mochat_target(self):
        """Test target resolution."""
        # Basic targets
        target = resolve_mochat_target("session_123")
        assert target.id == "session_123"
        assert not target.is_panel
        
        target = resolve_mochat_target("panel_456")
        assert target.id == "panel_456"
        assert target.is_panel
        
        # Prefixed targets
        target = resolve_mochat_target("mochat:session_123")
        assert target.id == "session_123"
        assert not target.is_panel
        
        target = resolve_mochat_target("panel:test_panel")
        assert target.id == "test_panel"
        assert target.is_panel
        
        target = resolve_mochat_target("group:test_group")
        assert target.id == "test_group"
        assert target.is_panel
        
        # Error cases
        with pytest.raises(ValueError):
            resolve_mochat_target("")
        with pytest.raises(ValueError):
            resolve_mochat_target("panel:")
    
    def test_extract_mention_ids(self):
        """Test mention ID extraction."""
        assert extract_mention_ids([]) == []
        assert extract_mention_ids("not a list") == []
        
        # String mentions
        assert extract_mention_ids(["user1", "user2"]) == ["user1", "user2"]
        assert extract_mention_ids(["", "  ", "user1"]) == ["user1"]
        
        # Dict mentions
        mentions = [
            {"id": "user1"},
            {"userId": "user2"},
            {"_id": "user3"},
            {"other": "user4"},  # Should be ignored
            "user5"
        ]
        result = extract_mention_ids(mentions)
        assert set(result) == {"user1", "user2", "user3", "user5"}
    
    def test_resolve_was_mentioned(self):
        """Test mention resolution."""
        agent_id = "agent_123"
        
        # Explicit mention flags
        payload = {"meta": {"mentioned": True}}
        assert resolve_was_mentioned(payload, agent_id) is True
        
        payload = {"meta": {"wasMentioned": True}}
        assert resolve_was_mentioned(payload, agent_id) is True
        
        # Mention arrays
        payload = {"meta": {"mentions": [agent_id, "other"]}}
        assert resolve_was_mentioned(payload, agent_id) is True
        
        # Content mentions
        payload = {"content": f"Hello <@{agent_id}>!"}
        assert resolve_was_mentioned(payload, agent_id) is True
        
        payload = {"content": f"Hello @{agent_id}!"}
        assert resolve_was_mentioned(payload, agent_id) is True
        
        # No mention
        payload = {"content": "Hello world"}
        assert resolve_was_mentioned(payload, agent_id) is False
        
        # No agent ID
        payload = {"meta": {"mentioned": True}}
        assert resolve_was_mentioned(payload, None) is False
        assert resolve_was_mentioned(payload, "") is False
    
    def test_build_buffered_body(self):
        """Test buffered body building."""
        # Single entry
        entry = MochatBufferedEntry(raw_body="Hello", author="user1")
        assert build_buffered_body([entry], False) == "Hello"
        
        # Multiple entries, not a group
        entry2 = MochatBufferedEntry(raw_body="World", author="user2")
        assert build_buffered_body([entry, entry2], False) == "Hello\\nWorld"
        
        # Multiple entries, is a group
        entry.sender_name = "John"
        entry2.sender_name = "Jane"
        result = build_buffered_body([entry, entry2], True)
        assert result == "John: Hello\\nJane: World"
        
        # Empty entries
        assert build_buffered_body([], False) == ""
        
        # Entries with empty content
        empty_entry = MochatBufferedEntry(raw_body="", author="user3")
        assert build_buffered_body([empty_entry], False) == ""
    
    def test_parse_timestamp(self):
        """Test timestamp parsing."""
        # Valid ISO format
        iso_time = "2023-01-01T12:00:00Z"
        result = parse_timestamp(iso_time)
        assert isinstance(result, int)
        assert result > 0
        
        # ISO format without Z
        iso_time = "2023-01-01T12:00:00+00:00"
        result = parse_timestamp(iso_time)
        assert isinstance(result, int)
        
        # Invalid formats
        assert parse_timestamp("invalid") is None
        assert parse_timestamp("") is None
        assert parse_timestamp(None) is None
        assert parse_timestamp(123) is None
    
    def test_make_synthetic_event(self):
        """Test synthetic event creation."""
        event = make_synthetic_event(
            message_id="msg_123",
            author="user_456", 
            content="Hello world",
            meta={"test": True},
            group_id="group_789",
            converse_id="converse_abc",
            timestamp="2023-01-01T12:00:00Z",
            author_info={"name": "John"}
        )
        
        assert event["type"] == "message.add"
        assert event["timestamp"] == "2023-01-01T12:00:00Z"
        
        payload = event["payload"]
        assert payload["messageId"] == "msg_123"
        assert payload["author"] == "user_456"
        assert payload["content"] == "Hello world"
        assert payload["meta"] == {"test": True}
        assert payload["groupId"] == "group_789"
        assert payload["converseId"] == "converse_abc"
        assert payload["authorInfo"] == {"name": "John"}
        
        # Test validation
        with pytest.raises(ValueError):
            make_synthetic_event("", "author", "content", {}, "group", "converse")
        with pytest.raises(ValueError):
            make_synthetic_event("msg", "", "content", {}, "group", "converse")
        with pytest.raises(ValueError):
            make_synthetic_event("msg", "author", "content", {}, "group", "")


# ---------------------------------------------------------------------------
# Test data classes and exceptions
# ---------------------------------------------------------------------------

class TestDataClasses:
    """Test data classes and exception types."""
    
    def test_correlation_id(self):
        """Test CorrelationId creation."""
        cid = CorrelationId()
        assert len(str(cid)) == 8
        assert isinstance(str(cid), str)
        
        # Test uniqueness
        cid2 = CorrelationId()
        assert str(cid) != str(cid2)
    
    def test_retry_config(self):
        """Test RetryConfig delay calculation."""
        config = RetryConfig(base_delay_ms=1000, max_delay_ms=10000, jitter=False)
        
        # Test exponential backoff
        assert config.calculate_delay(0) == 1.0  # 1000ms = 1s
        assert config.calculate_delay(1) == 2.0  # 2000ms = 2s
        assert config.calculate_delay(2) == 4.0  # 4000ms = 4s
        
        # Test max delay cap
        assert config.calculate_delay(10) == 10.0  # Capped at 10000ms
        
        # Test jitter (multiple runs should give different results)
        config.jitter = True
        delays = [config.calculate_delay(1) for _ in range(10)]
        assert len(set(delays)) > 1  # Should have variation
        assert all(1.0 <= d <= 2.0 for d in delays)  # Within expected range
    
    def test_mochat_buffered_entry_validation(self):
        """Test MochatBufferedEntry validation."""
        # Valid entry
        entry = MochatBufferedEntry(raw_body="Hello", author="user1")
        assert entry.raw_body == "Hello"
        assert entry.author == "user1"
        
        # Validation errors
        with pytest.raises(ValueError):
            MochatBufferedEntry(raw_body="", author="user1")
        with pytest.raises(ValueError):
            MochatBufferedEntry(raw_body="  ", author="user1")
        with pytest.raises(ValueError):
            MochatBufferedEntry(raw_body="Hello", author="")
        with pytest.raises(ValueError):
            MochatBufferedEntry(raw_body="Hello", author="  ")
    
    def test_mochat_error_types(self):
        """Test exception hierarchy."""
        cid = CorrelationId()
        
        # Base error
        error = MochatError("Base error", cid)
        assert str(error) == "Base error"
        assert error.correlation_id == cid
        
        # Connection error
        conn_error = MochatConnectionError("Connection failed")
        assert isinstance(conn_error, MochatError)
        
        # Retry exhausted error
        retry_error = RetryExhaustedError("Retries exhausted", cid)
        assert isinstance(retry_error, MochatError)
        assert retry_error.correlation_id == cid


# ---------------------------------------------------------------------------
# Test CircuitBreaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_initial_state(self):
        """Test circuit breaker initial state."""
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.can_execute() is True
        assert cb.state == "closed"
        assert cb.failure_count == 0
    
    def test_failure_accumulation(self):
        """Test failure count accumulation."""
        cb = CircuitBreaker(failure_threshold=3)
        
        # Record failures below threshold
        cb.record_failure()
        assert cb.can_execute() is True
        assert cb.failure_count == 1
        assert cb.state == "closed"
        
        cb.record_failure()
        assert cb.can_execute() is True
        assert cb.failure_count == 2
        assert cb.state == "closed"
        
        # Trigger circuit open
        cb.record_failure()
        assert cb.can_execute() is False
        assert cb.failure_count == 3
        assert cb.state == "open"
    
    def test_success_reset(self):
        """Test circuit reset on success."""
        cb = CircuitBreaker(failure_threshold=3)
        
        # Build up failures
        for _ in range(2):
            cb.record_failure()
        
        # Success should reset
        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == "closed"
        assert cb.can_execute() is True
    
    @pytest.mark.asyncio
    async def test_recovery_timeout(self):
        """Test circuit recovery after timeout."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        
        # Trigger circuit open
        cb.record_failure()
        cb.record_failure()
        assert cb.can_execute() is False
        
        # Wait for recovery period
        await asyncio.sleep(0.15)
        
        # Should allow one attempt (half-open)
        assert cb.can_execute() is True
        
        # Success should fully close circuit
        cb.record_success()
        assert cb.state == "closed"
        assert cb.can_execute() is True


# ---------------------------------------------------------------------------
# Test ConnectionManager
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestConnectionManager:
    """Test ConnectionManager functionality."""
    
    async def test_initialization(self, mock_config):
        """Test ConnectionManager initialization."""
        retry_config = RetryConfig(max_attempts=2)
        cm = ConnectionManager(mock_config, retry_config)
        
        assert cm.config == mock_config
        assert cm.retry_config == retry_config
        assert cm.connection_state == ConnectionState.DISCONNECTED
        assert not cm.is_connected
    
    async def test_validation_errors(self):
        """Test configuration validation."""
        config = Mock()
        config.claw_token = ""
        
        cm = ConnectionManager(config)
        
        with pytest.raises(Exception):  # Should raise authentication error
            await cm.start()
    
    @patch('httpx.AsyncClient')
    async def test_http_connectivity_test(self, mock_client_class, mock_config):
        """Test HTTP connectivity testing."""
        # Setup mock HTTP client
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        
        # Successful response
        mock_response = Mock()
        mock_response.is_success = True
        mock_client.get.return_value = mock_response
        
        cm = ConnectionManager(mock_config)
        cm._http_client = mock_client
        
        # Should not raise exception
        await cm._test_http_connectivity()
        
        # Test authentication error
        mock_response.is_success = False
        mock_response.status_code = 401
        
        with pytest.raises(Exception):
            await cm._test_http_connectivity()
    
    async def test_health_status(self, mock_config):
        """Test health status reporting."""
        cm = ConnectionManager(mock_config)
        
        health = await cm.get_health_status()
        assert not health.is_healthy  # Not initialized
        assert health.connection_state == ConnectionState.DISCONNECTED
        assert "HTTP client not available" in health.issues
    
    @patch('httpx.AsyncClient')
    async def test_http_request_retry_logic(self, mock_client_class, mock_config):
        """Test HTTP request retry mechanism."""
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        
        # First attempt fails, second succeeds
        mock_client.post.side_effect = [
            httpx.RequestError("Connection failed"),
            Mock(is_success=True, json=lambda: {"data": "success"})
        ]
        
        retry_config = RetryConfig(max_attempts=2, base_delay_ms=10)
        cm = ConnectionManager(mock_config, retry_config)
        cm._http_client = mock_client
        
        result = await cm.http_request("POST", "/test", {"foo": "bar"})
        assert result == {"data": "success"}
        
        # Verify retry was attempted
        assert mock_client.post.call_count == 2


# ---------------------------------------------------------------------------
# Test StateManager  
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestStateManager:
    """Test StateManager functionality."""
    
    async def test_initialization(self, temp_state_dir):
        """Test StateManager initialization."""
        sm = StateManager(temp_state_dir)
        await sm.load()
        
        assert len(sm.session_cursors) == 0
        assert sm.get_cursor("unknown") == 0
    
    async def test_cursor_management(self, temp_state_dir):
        """Test cursor update and retrieval."""
        sm = StateManager(temp_state_dir)
        await sm.load()
        
        # Update cursors
        sm.update_cursor("session1", 100)
        sm.update_cursor("session2", 200)
        
        assert sm.get_cursor("session1") == 100
        assert sm.get_cursor("session2") == 200
        
        # Test invalid updates (should be ignored)
        sm.update_cursor("session1", 50)  # Lower than current
        assert sm.get_cursor("session1") == 100  # Unchanged
        
        sm.update_cursor("session1", -10)  # Negative
        assert sm.get_cursor("session1") == 100  # Unchanged
    
    async def test_persistence(self, temp_state_dir):
        """Test state persistence across instances."""
        # First instance
        sm1 = StateManager(temp_state_dir)
        await sm1.load()
        sm1.update_cursor("session1", 100)
        sm1.update_cursor("session2", 200)
        await sm1.save(force=True)
        
        # Second instance should load saved state
        sm2 = StateManager(temp_state_dir)
        await sm2.load()
        
        assert sm2.get_cursor("session1") == 100
        assert sm2.get_cursor("session2") == 200
    
    async def test_corrupted_state_file(self, temp_state_dir):
        """Test handling of corrupted state file."""
        # Create invalid JSON file
        cursor_file = temp_state_dir / "session_cursors.json"
        cursor_file.write_text("invalid json", "utf-8")
        
        sm = StateManager(temp_state_dir)
        await sm.load()  # Should not crash
        
        assert len(sm.session_cursors) == 0  # Should be empty


# ---------------------------------------------------------------------------
# Test MessageBuffer
# ---------------------------------------------------------------------------

@pytest.mark.asyncio 
class TestMessageBuffer:
    """Test MessageBuffer functionality."""
    
    async def test_deduplication(self, mock_config):
        """Test message deduplication."""
        mb = MessageBuffer(mock_config)
        
        # First message should not be duplicate
        assert not mb.is_duplicate_message("session:1", "msg1")
        
        # Second time should be duplicate
        assert mb.is_duplicate_message("session:1", "msg1")
        
        # Different message should not be duplicate
        assert not mb.is_duplicate_message("session:1", "msg2")
        
        # Same message, different target should not be duplicate
        assert not mb.is_duplicate_message("session:2", "msg1")
    
    async def test_message_processing_immediate(self, mock_config):
        """Test immediate message processing."""
        mb = MessageBuffer(mock_config)
        
        dispatched_messages = []
        
        async def mock_dispatch(target_id, target_kind, entries, was_mentioned):
            dispatched_messages.append((target_id, target_kind, entries, was_mentioned))
        
        entry = MochatBufferedEntry(raw_body="Hello", author="user1")
        
        # Process without delay
        await mb.process_entry(
            target_key="session:1",
            entry=entry,
            is_group=False,
            was_mentioned=True,
            require_mention=False,
            use_delay=False,
            dispatch_callback=mock_dispatch
        )
        
        # Should be dispatched immediately  
        assert len(dispatched_messages) == 1
        assert dispatched_messages[0][0] == "1"  # target_id
        assert dispatched_messages[0][2][0] == entry  # entries
        assert dispatched_messages[0][3] is True  # was_mentioned
    
    async def test_delayed_processing(self, mock_config):
        """Test delayed message processing."""
        mock_config.reply_delay_ms = 100  # 100ms delay
        mb = MessageBuffer(mock_config)
        
        dispatched_messages = []
        
        async def mock_dispatch(target_id, target_kind, entries, was_mentioned):
            dispatched_messages.append((target_id, target_kind, entries, was_mentioned))
        
        entry = MochatBufferedEntry(raw_body="Hello", author="user1")
        
        # Process with delay (no mention)
        await mb.process_entry(
            target_key="panel:1",
            entry=entry,
            is_group=True,
            was_mentioned=False,
            require_mention=False,
            use_delay=True,
            dispatch_callback=mock_dispatch
        )
        
        # Should not be dispatched yet
        assert len(dispatched_messages) == 0
        
        # Wait for delay to expire
        await asyncio.sleep(0.15)
        
        # Should be dispatched now
        assert len(dispatched_messages) == 1
    
    async def test_mention_flushes_delay(self, mock_config):
        """Test that mentions flush delayed messages."""
        mock_config.reply_delay_ms = 1000  # Long delay
        mb = MessageBuffer(mock_config)
        
        dispatched_messages = []
        
        async def mock_dispatch(target_id, target_kind, entries, was_mentioned):
            dispatched_messages.append((target_id, target_kind, entries, was_mentioned))
        
        # Queue a delayed message
        entry1 = MochatBufferedEntry(raw_body="Hello", author="user1")
        await mb.process_entry(
            target_key="panel:1",
            entry=entry1,
            is_group=True,
            was_mentioned=False,
            require_mention=False,
            use_delay=True,
            dispatch_callback=mock_dispatch
        )
        
        # No dispatch yet
        assert len(dispatched_messages) == 0
        
        # Send mention (should flush queue)
        entry2 = MochatBufferedEntry(raw_body="@agent", author="user2")
        await mb.process_entry(
            target_key="panel:1",
            entry=entry2,
            is_group=True,
            was_mentioned=True,
            require_mention=False,
            use_delay=True,
            dispatch_callback=mock_dispatch
        )
        
        # Should be dispatched immediately with both messages
        assert len(dispatched_messages) == 1
        assert len(dispatched_messages[0][2]) == 2  # Both entries
        assert dispatched_messages[0][3] is True  # was_mentioned=True


# ---------------------------------------------------------------------------
# Test TargetManager
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestTargetManager:
    """Test TargetManager functionality."""
    
    async def test_initialization(self, mock_config):
        """Test TargetManager initialization."""
        mock_cm = Mock()
        tm = TargetManager(mock_config, mock_cm)
        
        # Should parse config targets
        assert "session_1" in tm.session_set
        assert "session_2" in tm.session_set
        assert "panel_1" in tm.panel_set
        assert tm.auto_discover_panels is True  # Due to "*"
        
        # Sessions should start cold
        assert "session_1" in tm.cold_sessions
        assert "session_2" in tm.cold_sessions
    
    async def test_cold_session_management(self, mock_config):
        """Test cold session tracking."""
        mock_cm = Mock()
        tm = TargetManager(mock_config, mock_cm)
        
        assert tm.is_cold_session("session_1")
        
        tm.mark_session_warm("session_1")
        assert not tm.is_cold_session("session_1")
        assert tm.is_cold_session("session_2")  # Still cold
    
    async def test_target_locks(self, mock_config):
        """Test target locking mechanism."""
        mock_cm = Mock()
        tm = TargetManager(mock_config, mock_cm)
        
        lock1 = tm.get_target_lock("session", "test1")
        lock2 = tm.get_target_lock("session", "test1")
        lock3 = tm.get_target_lock("session", "test2")
        
        # Same target should return same lock
        assert lock1 is lock2
        
        # Different target should return different lock
        assert lock1 is not lock3


# ---------------------------------------------------------------------------
# Test EventProcessor
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestEventProcessor:
    """Test EventProcessor functionality."""
    
    async def test_watch_payload_processing(self, mock_config):
        """Test watch payload processing."""
        from unittest.mock import AsyncMock
        
        # Setup mocks
        mock_tm = Mock()
        mock_tm.get_target_lock.return_value = asyncio.Lock()
        mock_tm.is_cold_session.return_value = False
        
        mock_mb = Mock()
        mock_sm = Mock()
        mock_sm.get_cursor.return_value = 0
        
        dispatched = []
        async def mock_dispatch(sender_id, chat_id, content, metadata):
            dispatched.append((sender_id, chat_id, content, metadata))
        
        ep = EventProcessor(
            config=mock_config,
            target_manager=mock_tm,
            message_buffer=mock_mb,
            state_manager=mock_sm,
            dispatch_callback=mock_dispatch
        )
        
        # Mock the internal process method to avoid complex setup
        ep._process_message_event = AsyncMock()
        
        # Test payload processing
        payload = {
            "sessionId": "session_123",
            "cursor": 150,
            "events": [
                {
                    "type": "message.add",
                    "seq": 151,
                    "payload": {
                        "messageId": "msg_123",
                        "author": "user_456",
                        "content": "Hello world"
                    }
                }
            ]
        }
        
        await ep.handle_watch_payload(payload, TargetKind.SESSION)
        
        # Verify cursor was updated
        mock_sm.update_cursor.assert_called_with("session_123", 150)
        
        # Verify message processing was called
        ep._process_message_event.assert_called_once()
    
    async def test_cold_session_skip(self, mock_config):
        """Test that cold sessions skip history processing."""
        mock_tm = Mock()
        mock_tm.get_target_lock.return_value = asyncio.Lock()
        mock_tm.is_cold_session.return_value = True  # Cold session
        
        mock_mb = Mock()
        mock_sm = Mock()
        
        dispatched = []
        async def mock_dispatch(sender_id, chat_id, content, metadata):
            dispatched.append((sender_id, chat_id, content, metadata))
        
        ep = EventProcessor(
            config=mock_config,
            target_manager=mock_tm,
            message_buffer=mock_mb,
            state_manager=mock_sm,
            dispatch_callback=mock_dispatch
        )
        
        payload = {
            "sessionId": "session_123",
            "events": [{"type": "message.add"}]
        }
        
        await ep.handle_watch_payload(payload, TargetKind.SESSION)
        
        # Should mark session warm and skip processing
        mock_tm.mark_session_warm.assert_called_with("session_123")
        assert len(dispatched) == 0
    

# ---------------------------------------------------------------------------
# Test MochatChannel Integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestMochatChannel:
    """Test MochatChannel integration."""
    
    async def test_initialization_validation(self, mock_bus):
        """Test configuration validation during initialization."""
        # Invalid token
        config = Mock()
        config.claw_token = ""
        config.base_url = "https://api.example.com"
        
        channel = MochatChannel(config, mock_bus)
        with pytest.raises(ValueError, match="claw_token is required"):
            channel._validate_config()
        
        # Invalid URL
        config.claw_token = "valid_token"
        config.base_url = "invalid_url"
        
        with pytest.raises(ValueError, match="must start with http"):
            channel._validate_config()
    
    async def test_send_message_validation(self, mock_config, mock_bus):
        """Test outbound message validation."""
        channel = MochatChannel(mock_config, mock_bus)
        
        # Empty content should be skipped
        empty_msg = OutboundMessage(
            channel="mochat",
            chat_id="session_123", 
            content="",
            media=[]
        )
        
        # Should not crash, just skip
        await channel.send(empty_msg)
        
        # Invalid target should be handled gracefully
        invalid_msg = OutboundMessage(
            channel="mochat",
            chat_id="",  # Empty target
            content="Hello"
        )
        
        await channel.send(invalid_msg)  # Should not crash
    
    @patch('nanobot.channels.mochat.ConnectionManager')
    async def test_component_lifecycle(self, mock_cm_class, mock_config, mock_bus):
        """Test component initialization lifecycle."""
        mock_cm = AsyncMock()
        mock_cm_class.return_value = mock_cm
        
        channel = MochatChannel(mock_config, mock_bus)
        
        # Test component initialization
        await channel._initialize_components()
        
        assert channel._state_manager is not None
        assert channel._connection_manager is not None
        assert channel._target_manager is not None
        assert channel._message_buffer is not None
        assert channel._event_processor is not None
    
    async def test_health_status_reporting(self, mock_config, mock_bus):
        """Test health status reporting."""
        channel = MochatChannel(mock_config, mock_bus)
        
        # Before initialization
        health = await channel.get_health_status()
        assert not health.is_healthy
        assert "Not initialized" in health.issues
        
        # Test ready check
        assert not channel.is_ready
        assert channel.connection_state == ConnectionState.DISCONNECTED
    
    async def test_backward_compatibility(self):
        """Test backward compatibility exports."""
        from nanobot.channels.mochat import _safe_dict, _str_field, _make_synthetic_event
        
        # These should work exactly like the new functions
        assert _safe_dict({"key": "value"}) == safe_dict({"key": "value"})
        assert _str_field({"name": "test"}, "name") == str_field({"name": "test"}, "name")
        
        # Synthetic event creation should work
        event = _make_synthetic_event(
            "msg_123", "author", "content", {}, "group", "converse"
        )
        assert event["type"] == "message.add"


# ---------------------------------------------------------------------------
# Performance and stress tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio 
class TestPerformance:
    """Performance and stress tests."""
    
    async def test_large_message_deduplication(self, mock_config):
        """Test deduplication with large number of messages."""
        mb = MessageBuffer(mock_config)
        
        # Add many messages
        for i in range(10000):
            is_duplicate = mb.is_duplicate_message("session:1", f"msg_{i}")
            assert not is_duplicate
        
        # Check queue size limit
        assert len(mb._seen_queues["session:1"]) <= 2000  # MAX_SEEN_MESSAGE_IDS
    
    async def test_concurrent_message_processing(self, mock_config):
        """Test concurrent message processing."""
        mb = MessageBuffer(mock_config)
        
        dispatched = []
        dispatch_lock = asyncio.Lock()
        
        async def mock_dispatch(target_id, target_kind, entries, was_mentioned):
            async with dispatch_lock:
                dispatched.append(len(entries))
        
        # Process many messages concurrently
        tasks = []
        for i in range(100):
            entry = MochatBufferedEntry(raw_body=f"Message {i}", author=f"user_{i}")
            task = mb.process_entry(
                target_key="session:1",
                entry=entry,
                is_group=False,
                was_mentioned=False,
                require_mention=False,
                use_delay=False,
                dispatch_callback=mock_dispatch
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # All messages should be processed
        assert len(dispatched) == 100
        assert all(count == 1 for count in dispatched)  # Each should have 1 entry


if __name__ == "__main__":
    pytest.main([__file__, "-v"])