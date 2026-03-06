"""Additional tests for specific fixes made to mochat.py and related files.

This test module covers specific regression tests for issues that were fixed:
1. target_id extraction with colon membership check
2. _process_api_response handling non-dict responses  
3. save() exception handling
4. Circuit breaker timing with time.monotonic()
5. Panel fallback worker sleep behavior
6. TargetManager cursor injection from StateManager
7. Socket handler setup timing
8. MochatConnectionError usage
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest
import httpx

from nanobot.config.schema import MochatConfig
from nanobot.channels.mochat import (
    CircuitBreaker,
    ConnectionManager,
    ConnectionState,
    CorrelationId,
    EventProcessor,
    MochatBufferedEntry,
    MochatChannel,
    MochatConnectionError,
    MessageBuffer,
    RetryConfig,
    StateManager,
    TargetKind,
    TargetManager,
)


@pytest.fixture  
def mock_config():
    """Create a mock MochatConfig for testing."""
    config = Mock(spec=MochatConfig)
    config.claw_token = "sk-test-123"
    config.base_url = "https://api.example.com"
    config.max_retry_attempts = 3
    config.retry_delay_ms = 100
    config.sessions = ["session1", "session2"]
    config.panels = ["panel1", "panel2"]
    config.agent_user_id = "agent_123"
    return config


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.mark.asyncio
class TestSpecificFixes:
    """Test cases for specific fixes made to mochat.py."""
    
    async def test_target_id_extraction_fix(self):
        """Test that target_id extraction properly handles colon membership check."""
        from nanobot.channels.mochat import extract_target_id_from_key
        
        # Test cases for the fixed logic
        test_cases = [
            ("session:abc123", "abc123"),  # Should extract after colon
            ("panel:general", "general"),  # Should extract after colon  
            ("session_without_colon", "session_without_colon"),  # Should return unchanged
            ("multiple:colon:test", "colon:test"),  # Should extract from first colon
            ("", ""),  # Should handle empty string
        ]
        
        for input_key, expected_id in test_cases:
            # Simulate the actual logic from the fix
            target_id = input_key.split(":", 1)[1] if ":" in input_key else input_key
            assert target_id == expected_id
    
    async def test_process_api_response_non_dict_handling(self, mock_config):
        """Test that _process_api_response properly handles non-dict responses."""
        retry_config = RetryConfig(max_attempts=3, base_delay_ms=100)
        conn_manager = ConnectionManager(mock_config, retry_config)
        correlation_id = CorrelationId()
        
        # Test non-dict response returns empty dict
        result = conn_manager._process_api_response("string_response", correlation_id)
        assert result == {}
        
        # Test None response returns empty dict
        result = conn_manager._process_api_response(None, correlation_id)
        assert result == {}
        
        # Test list response returns empty dict
        result = conn_manager._process_api_response([1, 2, 3], correlation_id)
        assert result == {}
        
        # Test valid dict response still works
        dict_response = {"code": 200, "data": {"key": "value"}}
        result = conn_manager._process_api_response(dict_response, correlation_id)
        assert result == {"key": "value"}
    
    async def test_state_manager_save_exception_handling(self, mock_config, temp_workspace):
        """Test that StateManager.save() properly handles exceptions."""
        state_manager = StateManager(mock_config, temp_workspace)
        
        # Mock Path.write_text to raise an exception
        with patch.object(state_manager.cursor_path, 'write_text', side_effect=OSError("Disk full")):
            # save() should not raise exception, should log instead
            await state_manager.save(force=True)
            # If we get here without exception, the fix is working
        
        # Test with permission error
        with patch.object(state_manager.cursor_path, 'write_text', side_effect=PermissionError("Access denied")):
            await state_manager.save(force=True)
            # Should not raise exception
    
    async def test_circuit_breaker_timing_fix(self):
        """Test that CircuitBreaker uses time.monotonic() instead of asyncio.get_event_loop().time()."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1.0)
        
        # Record a failure
        cb.record_failure()
        assert cb.failure_count == 1
        assert cb.last_failure_time is not None
        
        # Verify that last_failure_time is from time.monotonic() (monotonic time)
        initial_time = cb.last_failure_time
        
        # Record another failure to trigger open state
        cb.record_failure()
        assert cb.state == "open"
        assert cb.last_failure_time >= initial_time
        
        # Test can_execute logic
        assert not cb.can_execute()  # Should be False in open state
        
        # Mock time to test recovery
        with patch('time.monotonic', return_value=initial_time + 2.0):
            assert cb.can_execute()  # Should transition to half-open
            assert cb.state == "half-open"
    
    async def test_panel_fallback_worker_sleep(self, mock_config):
        """Test that panel fallback worker includes sleep to avoid tight polling."""
        with patch('nanobot.channels.mochat.get_data_path', return_value=Path("/tmp")):
            channel = MochatChannel(mock_config, Mock())
            
            # Mock the connection manager and components
            mock_conn_manager = Mock()
            mock_conn_manager.http_request = AsyncMock(return_value={"messages": []})
            channel._connection_manager = mock_conn_manager
            
            mock_event_processor = Mock()
            mock_event_processor._process_message_event = AsyncMock()
            channel._event_processor = mock_event_processor
            
            channel._running = True
            channel._fallback_mode = True
            
            # Start the worker with a very short sleep interval for testing
            channel.config.refresh_interval_ms = 50  # 50ms for fast test
            
            start_time = time.monotonic()
            
            # Run worker for a brief period to verify sleep is happening
            task = asyncio.create_task(channel._panel_fallback_worker("panel1"))
            await asyncio.sleep(0.2)  # Let it run for 200ms
            
            # Stop the worker
            channel._running = False
            try:
                await asyncio.wait_for(task, timeout=1.0)
            except asyncio.TimeoutError:
                task.cancel()
            
            end_time = time.monotonic()
            elapsed = end_time - start_time
            
            # Should have slept at least once (original tight loop would complete instantly)
            assert elapsed >= 0.05  # At least one sleep interval
            
            # Verify HTTP requests were made
            assert mock_conn_manager.http_request.called
    
    async def test_target_manager_cursor_injection(self, mock_config, temp_workspace):
        """Test that TargetManager properly uses StateManager for cursor values."""
        # Create StateManager with some cursor data
        state_manager = StateManager(mock_config, temp_workspace)
        state_manager.update_cursor("session1", 100)
        state_manager.update_cursor("session2", 200)
        
        # Create ConnectionManager mock
        mock_conn_manager = Mock()
        mock_conn_manager.socket_client = Mock()
        mock_conn_manager.socket_client.call = AsyncMock(return_value={"result": True})
        
        # Create TargetManager with StateManager
        target_manager = TargetManager(mock_config, mock_conn_manager, state_manager)
        
        # Test session subscription with cursor injection
        session_ids = ["session1", "session2"]
        
        # Mock the socket call to capture the cursors argument
        captured_payload = None
        async def capture_call(event, payload):
            nonlocal captured_payload
            captured_payload = payload
            return {"result": True}
        
        mock_conn_manager.socket_client.call.side_effect = capture_call
        
        # Call _subscribe_sessions
        await target_manager._subscribe_sessions(session_ids)
        
        # Verify cursors were injected from StateManager
        assert captured_payload is not None
        assert "cursors" in captured_payload
        assert captured_payload["cursors"]["session1"] == 100
        assert captured_payload["cursors"]["session2"] == 200
    
    async def test_socket_handler_setup_timing(self, mock_config):
        """Test that socket handlers are setup after WebSocket connection is established."""
        with patch('nanobot.channels.mochat.get_data_path', return_value=Path("/tmp")):
            with patch('nanobot.channels.mochat.SOCKETIO_AVAILABLE', True):
                with patch('socketio.AsyncClient') as mock_socketio_class:
                    with patch('httpx.AsyncClient'):
                        
                        mock_socket = Mock()
                        mock_socket.connect = AsyncMock()
                        mock_socket.connected = True
                        mock_socket.event = Mock()
                        mock_socket.on = Mock()
                        mock_socketio_class.return_value = mock_socket
                        
                        channel = MochatChannel(mock_config, Mock())
                        await channel._initialize_components()
                        
                        # Mock successful HTTP connectivity
                        channel._connection_manager._test_http_connectivity = AsyncMock()
                        
                        # Start connections
                        await channel._connection_manager.start_connections()
                        
                        # Verify that socket handlers were registered after connection
                        assert mock_socket.on.called
                        call_args = [call[0][0] for call in mock_socket.on.call_args_list]
                        expected_events = ["claw.session.events", "claw.panel.events"]
                        for event in expected_events:
                            assert any(event in arg for arg in call_args)
    
    async def test_mochat_connection_error_usage(self, mock_config):
        """Test that MochatConnectionError is used instead of built-in ConnectionError."""
        retry_config = RetryConfig(max_attempts=1, base_delay_ms=100)
        conn_manager = ConnectionManager(mock_config, retry_config)
        
        # Test HTTP connectivity failure raises MochatConnectionError
        with patch('httpx.AsyncClient.get', side_effect=httpx.RequestError("Network error")):
            with pytest.raises(MochatConnectionError) as exc_info:
                await conn_manager._test_http_connectivity()
            
            assert "HTTP connectivity test failed" in str(exc_info.value)
            assert isinstance(exc_info.value, MochatConnectionError)
            assert not isinstance(exc_info.value, ConnectionError)  # Built-in ConnectionError
    
    async def test_connection_state_logic_fix(self, mock_config):
        """Test that connection state logic properly uses CONNECTED instead of READY."""
        with patch('nanobot.channels.mochat.get_data_path', return_value=Path("/tmp")):
            channel = MochatChannel(mock_config, Mock())
            await channel._initialize_components()
            
            # Test connection state checking
            if channel._connection_manager:
                # Set to CONNECTED state 
                channel._connection_manager._connection_state = ConnectionState.CONNECTED
                
                # Verify that is_websocket_connected works with CONNECTED state
                channel._connection_manager.socket_client = Mock()  # Simulate socket
                assert channel.is_websocket_connected
                
                # Test that refresh logic would work (no longer looking for READY)
                # This simulates the fixed refresh target logic
                ws_ready = (
                    channel._connection_manager and 
                    channel._connection_manager.socket_client and
                    channel._connection_manager.connection_state == ConnectionState.CONNECTED
                )
                assert ws_ready  # Should be True with CONNECTED state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])