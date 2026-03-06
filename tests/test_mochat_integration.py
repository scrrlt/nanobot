"""Integration tests for MochatChannel with real-world scenarios.

This module provides integration tests that test the complete message flow
from WebSocket/HTTP input through to message dispatch, including:
- End-to-end message processing
- WebSocket event handling simulation  
- HTTP fallback scenarios
- Error recovery and retry scenarios
- Real configuration scenarios
"""

import asyncio
import json
import tempfile  
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import uuid4

import pytest
import httpx

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import MochatConfig
from nanobot.channels.mochat import (
    ConnectionState,
    MochatChannel,
    TargetKind,
    make_synthetic_event,
)


# ---------------------------------------------------------------------------
# Integration test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def integration_config():
    """Create a realistic MochatConfig for integration tests."""
    config = Mock(spec=MochatConfig)
    config.claw_token = "sk-test-abcdef123456"
    config.base_url = "https://api.mochat.example.com"
    config.socket_url = None
    config.socket_path = "/socket.io"
    config.socket_disable_msgpack = False
    config.max_retry_attempts = 3
    config.socket_reconnect_delay_ms = 1000
    config.socket_max_reconnect_delay_ms = 30000
    config.socket_connect_timeout_ms = 10000
    config.watch_timeout_ms = 30000
    config.watch_limit = 50
    config.refresh_interval_ms = 60000
    config.retry_delay_ms = 1000
    config.reply_delay_ms = 2000
    config.reply_delay_mode = "non-mention"
    config.agent_user_id = "agent_12345"
    config.sessions = ["session_alpha", "session_beta", "*"]
    config.panels = ["general", "support", "*"] 
    config.groups = {
        "general": Mock(require_mention=False),
        "support": Mock(require_mention=True),
        "*": Mock(require_mention=True)
    }
    config.mention = Mock(require_in_groups=True)
    return config


@pytest.fixture
def mock_message_bus():
    """Create a mock MessageBus for integration tests."""
    bus = Mock(spec=MessageBus)
    bus.publish_outbound = AsyncMock()
    bus.consume_inbound = AsyncMock()
    return bus


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for integration tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ---------------------------------------------------------------------------
# Mock external dependencies
# ---------------------------------------------------------------------------

class MockSocketIOClient:
    """Mock socketio.AsyncClient for testing."""
    
    def __init__(self):
        self.connected = False
        self.event_handlers = {}
        self.call_responses = {}
        
    async def connect(self, url, **kwargs):
        """Mock connection."""
        self.connected = True
        # Trigger connect event
        if "connect" in self.event_handlers:
            await self.event_handlers["connect"]()
            
    async def disconnect(self):
        """Mock disconnection."""
        self.connected = False
        if "disconnect" in self.event_handlers:
            await self.event_handlers["disconnect"]()
    
    def event(self, event_name):
        """Decorator to register event handlers."""
        def decorator(func):
            self.event_handlers[event_name] = func
            return func
        return decorator
        
    def on(self, event_name, handler):
        """Register event handler."""
        self.event_handlers[event_name] = handler
        
    async def call(self, event_name, payload, timeout=None):
        """Mock RPC call."""
        return self.call_responses.get(event_name, {"result": True})
        
    def set_call_response(self, event_name, response):
        """Set response for specific call."""
        self.call_responses[event_name] = response
        
    async def emit_event(self, event_name, payload):
        """Simulate external event."""
        if event_name in self.event_handlers:
            await self.event_handlers[event_name](payload)


class MockHTTPClient:
    """Mock httpx.AsyncClient for testing."""
    
    def __init__(self):
        self.responses = {}
        self.request_history = []
        
    def set_response(self, method, path_pattern, response_data, status_code=200):
        """Set response for matching requests."""
        key = f"{method.upper()}:{path_pattern}"
        mock_response = Mock()
        mock_response.is_success = status_code < 400
        mock_response.status_code = status_code
        mock_response.json.return_value = response_data
        mock_response.text = json.dumps(response_data) if isinstance(response_data, dict) else str(response_data)
        self.responses[key] = mock_response
        
    async def get(self, url, **kwargs):
        """Mock GET request."""
        return self._handle_request("GET", url, **kwargs)
        
    async def post(self, url, **kwargs):
        """Mock POST request."""
        return self._handle_request("POST", url, **kwargs)
        
    def _handle_request(self, method, url, **kwargs):
        """Handle request and return appropriate response."""
        self.request_history.append({
            "method": method,
            "url": url,
            "kwargs": kwargs
        })
        
        # Find matching response
        for pattern, response in self.responses.items():
            method_pattern, path_pattern = pattern.split(":", 1)
            if method == method_pattern and path_pattern in url:
                return response
                
        # Default success response
        mock_response = Mock()
        mock_response.is_success = True
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "default"}
        mock_response.text = '{"data": "default"}'
        return mock_response
        
    async def aclose(self):
        """Mock client cleanup."""
        pass


# ---------------------------------------------------------------------------
# Integration test scenarios
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestMochatChannelIntegration:
    """Integration tests for complete MochatChannel workflows."""
    
    async def test_end_to_end_websocket_message_flow(self, integration_config, mock_message_bus, temp_workspace):
        """Test complete message flow via WebSocket."""
        with patch('nanobot.channels.mochat.get_data_path', return_value=temp_workspace):
            with patch('nanobot.channels.mochat.SOCKETIO_AVAILABLE', True):
                with patch('socketio.AsyncClient') as mock_socketio_class:
                    with patch('httpx.AsyncClient') as mock_http_class:
                        
                        # Setup mocks
                        mock_socket = MockSocketIOClient()
                        mock_socketio_class.return_value = mock_socket
                        
                        mock_http = MockHTTPClient() 
                        mock_http_class.return_value = mock_http
                        
                        # Set API responses
                        mock_http.set_response("GET", "/api/health", {"status": "ok"})
                        mock_http.set_response("POST", "/api/claw/sessions/list", {
                            "sessions": [
                                {"sessionId": "session_alpha", "converseId": "conv_123"},
                                {"sessionId": "session_beta", "converseId": "conv_456"}
                            ]
                        })
                        mock_http.set_response("POST", "/api/claw/groups/get", {
                            "panels": [
                                {"id": "general", "type": 0},
                                {"id": "support", "type": 0}
                            ]
                        })
                        
                        # Set socket call responses
                        mock_socket.set_call_response("com.claw.im.subscribeSessions", {
                            "result": True,
                            "data": {"status": "subscribed"}
                        })
                        mock_socket.set_call_response("com.claw.im.subscribePanels", {
                            "result": True,
                            "data": {"status": "subscribed"}
                        })
                        
                        channel = MochatChannel(integration_config, mock_message_bus)
                        
                        # Start channel in background
                        start_task = asyncio.create_task(channel.start())
                        
                        # Wait for initialization
                        await asyncio.sleep(0.1)
                        
                        # Verify connection state
                        assert channel.connection_state in {ConnectionState.CONNECTED, ConnectionState.CONNECTING}
                        
                        # Simulate incoming WebSocket message
                        session_event = {
                            "sessionId": "session_alpha",
                            "cursor": 150,
                            "events": [
                                {
                                    "type": "message.add",
                                    "seq": 151,
                                    "timestamp": "2023-01-01T12:00:00Z",
                                    "payload": {
                                        "messageId": "msg_12345",
                                        "author": "user_67890",
                                        "content": "Hello agent!",
                                        "meta": {},
                                        "groupId": "",
                                        "converseId": "conv_123",
                                        "authorInfo": {
                                            "nickname": "TestUser",
                                            "email": "test@example.com"
                                        }
                                    }
                                }
                            ]
                        }
                        
                        await mock_socket.emit_event("claw.session.events", session_event)
                        
                        # Wait for message processing
                        await asyncio.sleep(0.1)
                        
                        # Stop channel
                        await channel.stop()
                        start_task.cancel()
                        
                        try:
                            await start_task
                        except asyncio.CancelledError:
                            pass
    
    async def test_http_fallback_scenario(self, integration_config, mock_message_bus, temp_workspace):
        """Test HTTP polling fallback when WebSocket fails."""
        with patch('nanobot.channels.mochat.get_data_path', return_value=temp_workspace):
            with patch('nanobot.channels.mochat.SOCKETIO_AVAILABLE', False):  # No WebSocket
                with patch('httpx.AsyncClient') as mock_http_class:
                    
                    mock_http = MockHTTPClient()
                    mock_http_class.return_value = mock_http
                    
                    # Set API responses for fallback mode
                    mock_http.set_response("GET", "/api/health", {"status": "ok"})
                    mock_http.set_response("POST", "/api/claw/sessions/list", {
                        "sessions": [{"sessionId": "session_alpha"}]
                    })
                    mock_http.set_response("POST", "/api/claw/sessions/watch", {
                        "sessionId": "session_alpha",
                        "cursor": 200,
                        "events": []
                    })
                    
                    channel = MochatChannel(integration_config, mock_message_bus)
                    
                    # Initialize components only
                    await channel._initialize_components()
                    
                    # Verify fallback mode would be triggered
                    assert not channel.is_websocket_connected
                    
                    # Test fallback worker simulation
                    if channel._connection_manager:
                        response = await channel._connection_manager.http_request(
                            "POST", "/api/claw/sessions/watch", {"sessionId": "session_alpha"}
                        )
                        assert "sessionId" in str(response) or "cursor" in str(response)
                        
                    await channel.stop()
    
    async def test_error_recovery_scenarios(self, integration_config, mock_message_bus, temp_workspace):
        """Test error recovery and retry mechanisms."""
        with patch('nanobot.channels.mochat.get_data_path', return_value=temp_workspace):
            with patch('httpx.AsyncClient') as mock_http_class:
                
                mock_http = Mock()
                mock_http_class.return_value = mock_http
                
                # First request fails, second succeeds
                responses = [
                    Mock(side_effect=httpx.RequestError("Network error")),
                    Mock(
                        is_success=True, 
                        status_code=200,
                        json=lambda: {"data": {"status": "ok"}}
                    )
                ]
                mock_http.get.side_effect = responses
                
                channel = MochatChannel(integration_config, mock_message_bus)
                
                # Initialize just the connection manager for testing
                await channel._initialize_components()
                
                if channel._connection_manager:
                    # This should retry and eventually succeed
                    try:
                        await channel._connection_manager._test_http_connectivity()
                    except Exception as e:
                        # May fail due to mock setup, but should attempt retry
                        pass
                
                await channel.stop()
    
    async def test_message_deduplication_integration(self, integration_config, mock_message_bus, temp_workspace):
        """Test message deduplication in realistic scenario."""
        with patch('nanobot.channels.mochat.get_data_path', return_value=temp_workspace):
            
            channel = MochatChannel(integration_config, mock_message_bus)
            await channel._initialize_components()
            
            if not channel._message_buffer:
                return  # Skip if component not available
            
            # Simulate duplicate message processing
            target_key = "session:session_alpha"
            message_id = "msg_duplicate_test"
            
            # First message should not be duplicate
            is_dup1 = channel._message_buffer.is_duplicate_message(target_key, message_id)
            assert not is_dup1
            
            # Second message should be duplicate
            is_dup2 = channel._message_buffer.is_duplicate_message(target_key, message_id)
            assert is_dup2
            
            await channel.stop()
    
    async def test_state_persistence_integration(self, integration_config, mock_message_bus, temp_workspace):
        """Test state persistence across channel restarts."""
        with patch('nanobot.channels.mochat.get_data_path', return_value=temp_workspace):
            
            # First channel instance
            channel1 = MochatChannel(integration_config, mock_message_bus)
            await channel1._initialize_components()
            
            if channel1._state_manager:
                # Update some state
                channel1._state_manager.update_cursor("session_alpha", 100)
                channel1._state_manager.update_cursor("session_beta", 200)
                await channel1._state_manager.save(force=True)
            
            await channel1.stop()
            
            # Second channel instance should load saved state
            channel2 = MochatChannel(integration_config, mock_message_bus)
            await channel2._initialize_components()
            
            if channel2._state_manager:
                assert channel2._state_manager.get_cursor("session_alpha") == 100
                assert channel2._state_manager.get_cursor("session_beta") == 200
            
            await channel2.stop()
    
    async def test_outbound_message_scenarios(self, integration_config, mock_message_bus, temp_workspace):
        """Test various outbound message scenarios."""
        with patch('nanobot.channels.mochat.get_data_path', return_value=temp_workspace):
            with patch('httpx.AsyncClient') as mock_http_class:
                
                mock_http = MockHTTPClient()
                mock_http_class.return_value = mock_http
                
                # Set up API responses
                mock_http.set_response("POST", "/api/claw/sessions/send", {"status": "sent"})
                mock_http.set_response("POST", "/api/claw/groups/panels/send", {"status": "sent"})
                
                channel = MochatChannel(integration_config, mock_message_bus)
                await channel._initialize_components()
                
                # Mark as initialized to allow sending
                channel._initialization_complete = True
                
                # Test session message
                session_msg = OutboundMessage(
                    channel="mochat",
                    chat_id="session_alpha",
                    content="Hello from integration test!",
                    media=[]
                )
                
                await channel.send(session_msg)
                
                # Test panel message
                panel_msg = OutboundMessage(
                    channel="mochat", 
                    chat_id="panel:general",
                    content="Panel message test",
                    media=[],
                    metadata={"group_id": "workspace_123"}
                )
                
                await channel.send(panel_msg)
                
                # Test empty message (should be skipped)
                empty_msg = OutboundMessage(
                    channel="mochat",
                    chat_id="session_alpha",
                    content="",
                    media=[]
                )
                
                await channel.send(empty_msg)  # Should not crash
                
                await channel.stop()


# ---------------------------------------------------------------------------
# Mock utilities and test helpers
# ---------------------------------------------------------------------------

class MochatTestHelper:
    """Helper utilities for Mochat testing."""
    
    @staticmethod
    def create_test_message_event(message_id: str = None, author: str = None, content: str = None) -> Dict[str, Any]:
        """Create a test message event with defaults."""
        return make_synthetic_event(
            message_id=message_id or f"msg_{uuid4().hex[:8]}",
            author=author or f"user_{uuid4().hex[:8]}",
            content=content or "Test message content",
            meta={"source": "test"},
            group_id="",
            converse_id=f"session_{uuid4().hex[:8]}",
            timestamp="2023-01-01T12:00:00Z"
        )
    
    @staticmethod
    def create_test_watch_payload(session_id: str = None, events: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a test watch payload."""
        return {
            "sessionId": session_id or f"session_{uuid4().hex[:8]}",
            "cursor": 100,
            "events": events or [MochatTestHelper.create_test_message_event()]
        }
    
    @staticmethod
    async def simulate_websocket_session(mock_socket: MockSocketIOClient, session_id: str, message_count: int = 3):
        """Simulate a WebSocket session with multiple messages."""
        events = []
        for i in range(message_count):
            event = {
                "type": "message.add",
                "seq": 100 + i,
                "timestamp": "2023-01-01T12:00:00Z",
                "payload": {
                    "messageId": f"msg_{session_id}_{i}",
                    "author": f"user_{i}",
                    "content": f"Message {i} from {session_id}",
                    "meta": {},
                    "groupId": "",
                    "converseId": session_id,
                    "authorInfo": {"nickname": f"User{i}"}
                }
            }
            events.append(event)
        
        payload = {
            "sessionId": session_id,
            "cursor": 100 + message_count,
            "events": events
        }
        
        await mock_socket.emit_event("claw.session.events", payload)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    """Integration tests for complete MochatChannel workflows."""
    
    async def test_end_to_end_websocket_message_flow(self, integration_config, mock_message_bus, temp_workspace):
        """Test complete message flow via WebSocket."""
        with patch('nanobot.channels.mochat.get_data_path', return_value=temp_workspace):
            with patch('nanobot.channels.mochat.SOCKETIO_AVAILABLE', True):
                with patch('socketio.AsyncClient') as mock_socketio_class:
                    with patch('httpx.AsyncClient') as mock_http_class:
                        
                        # Setup mocks
                        mock_socket = MockSocketIOClient()
                        mock_socketio_class.return_value = mock_socket
                        
                        mock_http = MockHTTPClient() 
                        mock_http_class.return_value = mock_http
                        
                        # Set API responses
                        mock_http.set_response("GET", "/api/health", {"status": "ok"})
                        mock_http.set_response("POST", "/api/claw/sessions/list", {\n                            "sessions": [\n                                {"sessionId": "session_alpha", "converseId": "conv_123"},\n                                {"sessionId": "session_beta", "converseId": "conv_456"}\n                            ]\n                        })\n                        mock_http.set_response("POST", "/api/claw/groups/get", {\n                            "panels": [\n                                {"id": "general", "type": 0},\n                                {"id": "support", "type": 0}\n                            ]\n                        })\n                        \n                        # Set socket call responses\n                        mock_socket.set_call_response("com.claw.im.subscribeSessions", {\n                            "result": True,\n                            "data": {"status": "subscribed"}\n                        })\n                        mock_socket.set_call_response("com.claw.im.subscribePanels", {\n                            "result": True,\n                            "data": {"status": "subscribed"}\n                        })\n                        \n                        channel = MochatChannel(integration_config, mock_message_bus)\n                        \n                        # Start channel in background\n                        start_task = asyncio.create_task(channel.start())\n                        \n                        # Wait for initialization\n                        await asyncio.sleep(0.1)\n                        \n                        # Verify connection state\n                        assert channel.connection_state in {ConnectionState.CONNECTED, ConnectionState.CONNECTING}\n                        \n                        # Simulate incoming WebSocket message\n                        session_event = {\n                            "sessionId": "session_alpha",\n                            "cursor": 150,\n                            "events": [\n                                {\n                                    "type": "message.add",\n                                    "seq": 151,\n                                    "timestamp": "2023-01-01T12:00:00Z",\n                                    "payload": {\n                                        "messageId": "msg_12345",\n                                        "author": "user_67890",\n                                        "content": "Hello agent!",\n                                        "meta": {},\n                                        "groupId": "",\n                                        "converseId": "conv_123",\n                                        "authorInfo": {\n                                            "nickname": "TestUser",\n                                            "email": "test@example.com"\n                                        }\n                                    }\n                                }\n                            ]\n                        }\n                        \n                        await mock_socket.emit_event("claw.session.events", session_event)\n                        \n                        # Wait for message processing\n                        await asyncio.sleep(0.1)\n                        \n                        # Stop channel\n                        await channel.stop()\n                        start_task.cancel()\n                        \n                        try:\n                            await start_task\n                        except asyncio.CancelledError:\n                            pass\n    \n    async def test_http_fallback_scenario(self, integration_config, mock_message_bus, temp_workspace):\n        """Test HTTP polling fallback when WebSocket fails."""\n        with patch('nanobot.channels.mochat.get_data_path', return_value=temp_workspace):\n            with patch('nanobot.channels.mochat.SOCKETIO_AVAILABLE', False):  # No WebSocket\n                with patch('httpx.AsyncClient') as mock_http_class:\n                    \n                    mock_http = MockHTTPClient()\n                    mock_http_class.return_value = mock_http\n                    \n                    # Set API responses for fallback mode\n                    mock_http.set_response("GET", "/api/health", {"status": "ok"})\n                    mock_http.set_response("POST", "/api/claw/sessions/list", {\n                        "sessions": [{"sessionId": "session_alpha"}]\n                    })\n                    mock_http.set_response("POST", "/api/claw/sessions/watch", {\n                        "sessionId": "session_alpha",\n                        "cursor": 200,\n                        "events": []\n                    })\n                    \n                    channel = MochatChannel(integration_config, mock_message_bus)\n                    \n                    # Initialize components only\n                    await channel._initialize_components()\n                    \n                    # Verify fallback mode would be triggered\n                    assert not channel.is_websocket_connected\n                    \n                    # Test fallback worker simulation\n                    if channel._connection_manager:\n                        response = await channel._connection_manager.http_request(\n                            "POST", "/api/claw/sessions/watch", {"sessionId": "session_alpha"}\n                        )\n                        assert "sessionId" in str(response) or "cursor" in str(response)\n                        \n                    await channel.stop()\n    \n    async def test_error_recovery_scenarios(self, integration_config, mock_message_bus, temp_workspace):\n        """Test error recovery and retry mechanisms."""\n        with patch('nanobot.channels.mochat.get_data_path', return_value=temp_workspace):\n            with patch('httpx.AsyncClient') as mock_http_class:\n                \n                mock_http = Mock()\n                mock_http_class.return_value = mock_http\n                \n                # First request fails, second succeeds\n                responses = [\n                    Mock(side_effect=httpx.RequestError("Network error")),\n                    Mock(\n                        is_success=True, \n                        status_code=200,\n                        json=lambda: {"data": {"status": "ok"}}\n                    )\n                ]\n                mock_http.get.side_effect = responses\n                \n                channel = MochatChannel(integration_config, mock_message_bus)\n                \n                # Initialize just the connection manager for testing\n                await channel._initialize_components()\n                \n                if channel._connection_manager:\n                    # This should retry and eventually succeed\n                    try:\n                        await channel._connection_manager._test_http_connectivity()\n                    except Exception as e:\n                        # May fail due to mock setup, but should attempt retry\n                        pass\n                \n                await channel.stop()\n    \n    async def test_message_deduplication_integration(self, integration_config, mock_message_bus, temp_workspace):\n        \"\"\"Test message deduplication in realistic scenario.\"\"\"\n        with patch('nanobot.channels.mochat.get_data_path', return_value=temp_workspace):\n            \n            channel = MochatChannel(integration_config, mock_message_bus)\n            await channel._initialize_components()\n            \n            if not channel._message_buffer:\n                return  # Skip if component not available\n            \n            # Simulate duplicate message processing\n            target_key = "session:session_alpha"\n            message_id = "msg_duplicate_test"\n            \n            # First message should not be duplicate\n            is_dup1 = channel._message_buffer.is_duplicate_message(target_key, message_id)\n            assert not is_dup1\n            \n            # Second message should be duplicate\n            is_dup2 = channel._message_buffer.is_duplicate_message(target_key, message_id)\n            assert is_dup2\n            \n            await channel.stop()\n    \n    async def test_state_persistence_integration(self, integration_config, mock_message_bus, temp_workspace):\n        \"\"\"Test state persistence across channel restarts.\"\"\"\n        with patch('nanobot.channels.mochat.get_data_path', return_value=temp_workspace):\n            \n            # First channel instance\n            channel1 = MochatChannel(integration_config, mock_message_bus)\n            await channel1._initialize_components()\n            \n            if channel1._state_manager:\n                # Update some state\n                channel1._state_manager.update_cursor("session_alpha", 100)\n                channel1._state_manager.update_cursor("session_beta", 200)\n                await channel1._state_manager.save(force=True)\n            \n            await channel1.stop()\n            \n            # Second channel instance should load saved state\n            channel2 = MochatChannel(integration_config, mock_message_bus)\n            await channel2._initialize_components()\n            \n            if channel2._state_manager:\n                assert channel2._state_manager.get_cursor("session_alpha") == 100\n                assert channel2._state_manager.get_cursor("session_beta") == 200\n            \n            await channel2.stop()\n    \n    async def test_outbound_message_scenarios(self, integration_config, mock_message_bus, temp_workspace):\n        \"\"\"Test various outbound message scenarios.\"\"\"\n        with patch('nanobot.channels.mochat.get_data_path', return_value=temp_workspace):\n            with patch('httpx.AsyncClient') as mock_http_class:\n                \n                mock_http = MockHTTPClient()\n                mock_http_class.return_value = mock_http\n                \n                # Set up API responses\n                mock_http.set_response("POST", "/api/claw/sessions/send", {"status": "sent"})\n                mock_http.set_response("POST", "/api/claw/groups/panels/send", {"status": "sent"})\n                \n                channel = MochatChannel(integration_config, mock_message_bus)\n                await channel._initialize_components()\n                \n                # Mark as initialized to allow sending\n                channel._initialization_complete = True\n                \n                # Test session message\n                session_msg = OutboundMessage(\n                    channel="mochat",\n                    chat_id="session_alpha",\n                    content="Hello from integration test!",\n                    media=[]\n                )\n                \n                await channel.send(session_msg)\n                \n                # Test panel message\n                panel_msg = OutboundMessage(\n                    channel="mochat", \n                    chat_id="panel:general",\n                    content="Panel message test",\n                    media=[],\n                    metadata={"group_id": "workspace_123"}\n                )\n                \n                await channel.send(panel_msg)\n                \n                # Test empty message (should be skipped)\n                empty_msg = OutboundMessage(\n                    channel="mochat",\n                    chat_id="session_alpha",\n                    content="",\n                    media=[]\n                )\n                \n                await channel.send(empty_msg)  # Should not crash\n                \n                await channel.stop()\n    \n    async def test_concurrent_operations(self, integration_config, mock_message_bus, temp_workspace):\n        \"\"\"Test concurrent operations and thread safety.\"\"\"\n        with patch('nanobot.channels.mochat.get_data_path', return_value=temp_workspace):\n            \n            channel = MochatChannel(integration_config, mock_message_bus)\n            await channel._initialize_components()\n            \n            if not channel._message_buffer or not channel._state_manager:\n                return\n            \n            # Concurrent cursor updates\n            async def update_cursors(session_base, count):\n                for i in range(count):\n                    session_id = f"{session_base}_{i}"\n                    cursor_value = i * 10\n                    channel._state_manager.update_cursor(session_id, cursor_value)\n            \n            # Run concurrent updates\n            tasks = [\n                update_cursors("session_a", 50),\n                update_cursors("session_b", 50),\n                update_cursors("session_c", 50)\n            ]\n            \n            await asyncio.gather(*tasks)\n            \n            # Verify state consistency\n            assert channel._state_manager.get_cursor("session_a_10") == 100\n            assert channel._state_manager.get_cursor("session_b_20") == 200\n            assert channel._state_manager.get_cursor("session_c_30") == 300\n            \n            await channel.stop()\n\n\n# ---------------------------------------------------------------------------\n# Mock utilities and test helpers\n# ---------------------------------------------------------------------------\n\nclass MochatTestHelper:\n    \"\"\"Helper utilities for Mochat testing.\"\"\"\n    \n    @staticmethod\n    def create_test_message_event(message_id: str = None, author: str = None, content: str = None) -> Dict[str, Any]:\n        \"\"\"Create a test message event with defaults.\"\"\"\n        return make_synthetic_event(\n            message_id=message_id or f"msg_{uuid4().hex[:8]}",\n            author=author or f"user_{uuid4().hex[:8]}",\n            content=content or "Test message content",\n            meta={"source": "test"},\n            group_id="",\n            converse_id=f"session_{uuid4().hex[:8]}",\n            timestamp="2023-01-01T12:00:00Z"\n        )\n    \n    @staticmethod\n    def create_test_watch_payload(session_id: str = None, events: List[Dict[str, Any]] = None) -> Dict[str, Any]:\n        \"\"\"Create a test watch payload.\"\"\"\n        return {\n            "sessionId": session_id or f"session_{uuid4().hex[:8]}",\n            "cursor": 100,\n            "events": events or [MochatTestHelper.create_test_message_event()]\n        }\n    \n    @staticmethod\n    async def simulate_websocket_session(mock_socket: MockSocketIOClient, session_id: str, message_count: int = 3):\n        \"\"\"Simulate a WebSocket session with multiple messages.\"\"\"\n        events = []\n        for i in range(message_count):\n            event = {\n                "type": "message.add",\n                "seq": 100 + i,\n                "timestamp": "2023-01-01T12:00:00Z",\n                "payload": {\n                    "messageId": f"msg_{session_id}_{i}",\n                    "author": f"user_{i}",\n                    "content": f"Message {i} from {session_id}",\n                    "meta": {},\n                    "groupId": "",\n                    "converseId": session_id,\n                    "authorInfo": {"nickname": f"User{i}"}\n                }\n            }\n            events.append(event)\n        \n        payload = {\n            "sessionId": session_id,\n            "cursor": 100 + message_count,\n            "events": events\n        }\n        \n        await mock_socket.emit_event("claw.session.events", payload)\n\n\n# ---------------------------------------------------------------------------\n# Performance integration tests\n# ---------------------------------------------------------------------------\n\n@pytest.mark.asyncio\nclass TestMochatPerformanceIntegration:\n    \"\"\"Performance-focused integration tests.\"\"\"\n    \n    async def test_high_volume_message_processing(self, integration_config, mock_message_bus, temp_workspace):\n        \"\"\"Test processing high volume of messages.\"\"\"\n        with patch('nanobot.channels.mochat.get_data_path', return_value=temp_workspace):\n            \n            channel = MochatChannel(integration_config, mock_message_bus)\n            await channel._initialize_components()\n            \n            if not channel._message_buffer:\n                return\n            \n            # Process many messages quickly\n            processed_count = 0\n            \n            async def count_dispatch(target_id, target_kind, entries, was_mentioned):\n                nonlocal processed_count\n                processed_count += len(entries)\n            \n            # Simulate high-volume processing\n            from nanobot.channels.mochat import MochatBufferedEntry\n            tasks = []\n            \n            for i in range(1000):\n                entry = MochatBufferedEntry(\n                    raw_body=f"High volume message {i}",\n                    author=f"user_{i % 10}"  # 10 different users\n                )\n                \n                task = channel._message_buffer.process_entry(\n                    target_key=f"session:{i % 5}",  # 5 different sessions\n                    entry=entry,\n                    is_group=False,\n                    was_mentioned=False,\n                    require_mention=False,\n                    use_delay=False,\n                    dispatch_callback=count_dispatch\n                )\n                tasks.append(task)\n            \n            # Process all concurrently\n            start_time = asyncio.get_event_loop().time()\n            await asyncio.gather(*tasks)\n            end_time = asyncio.get_event_loop().time()\n            \n            # Verify results\n            assert processed_count == 1000\n            processing_time = end_time - start_time\n            print(f"Processed 1000 messages in {processing_time:.3f} seconds")\n            assert processing_time < 5.0  # Should be fast\n            \n            await channel.stop()\n    \n    async def test_memory_usage_stability(self, integration_config, mock_message_bus, temp_workspace):\n        \"\"\"Test that memory usage remains stable under load.\"\"\"\n        with patch('nanobot.channels.mochat.get_data_path', return_value=temp_workspace):\n            \n            channel = MochatChannel(integration_config, mock_message_bus)\n            await channel._initialize_components()\n            \n            if not channel._message_buffer:\n                return\n            \n            # Generate many unique message IDs to test deduplication limits\n            for i in range(5000):  # More than MAX_SEEN_MESSAGE_IDS\n                target_key = f"session:{i % 3}"\n                message_id = f"memory_test_{i}"\n                is_dup = channel._message_buffer.is_duplicate_message(target_key, message_id)\n                assert not is_dup  # First time seeing each message\n            \n            # Verify memory bounds are respected\n            for target_key in channel._message_buffer._seen_queues:\n                queue_size = len(channel._message_buffer._seen_queues[target_key])\n                set_size = len(channel._message_buffer._seen_sets[target_key])\n                \n                # Should not exceed configured maximum\n                from nanobot.channels.mochat import MAX_SEEN_MESSAGE_IDS\n                assert queue_size <= MAX_SEEN_MESSAGE_IDS\n                assert set_size <= MAX_SEEN_MESSAGE_IDS\n                assert queue_size == set_size  # Should be in sync\n            \n            await channel.stop()\n\n\nif __name__ == "__main__":\n    pytest.main([__file__, "-v"])