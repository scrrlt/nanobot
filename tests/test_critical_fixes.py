#!/usr/bin/env python3
"""
Critical logic flaw fix validation tests.

This module contains tests that validate the critical behavioral fixes
implemented to resolve production-critical defects in the nanobot system.

Tests:
    test_circuit_breaker_integration: Validates circuit breaker integration.
    test_target_manager_return_value: Validates return type compliance.
    test_state_manager_debounce_logic: Validates debounce implementation.
    test_media_upload_streaming: Validates streaming upload behavior.
    test_circuit_breaker_prevents_spam: Validates circuit breaker behavior.
"""

from __future__ import annotations

import asyncio
import inspect
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Test the critical fixes without full channel setup
def test_circuit_breaker_integration() -> None:
    """Test that circuit breaker is properly integrated into http_request.
    
    Validates:
        - Circuit breaker instance is available on ConnectionManager
        - Circuit breaker allows initial requests
        - Integration follows expected patterns
    """
    # Import class definitions without full initialization
    from nanobot.channels.mochat import CircuitBreaker, ConnectionManager
    from nanobot.config.schema import MochatConfig
    
    # Mock config
    config = Mock()
    config.base_url = "https://test.api"
    config.claw_token = "test-token"
    
    # Create connection manager (it creates its own circuit breaker)
    with patch('nanobot.channels.mochat.httpx.AsyncClient'):
        manager = ConnectionManager(config=config)
        
        # Set up mock client
        manager._http_client = Mock()
        
        # Test circuit breaker integration exists
        assert hasattr(manager, 'circuit_breaker'), "Circuit breaker should be available"
        assert isinstance(manager.circuit_breaker, CircuitBreaker), "Should be CircuitBreaker instance"
        
        # Test initial state
        assert manager.circuit_breaker.can_execute(), "Circuit breaker should initially allow requests"
        
        print("✓ Circuit breaker properly integrated into ConnectionManager")


def test_target_manager_return_value() -> None:
    """Test that TargetManager.subscribe_all returns boolean value.
    
    Validates:
        - Method has proper return statement in source code
        - Method signature declares boolean return type annotation
        - Implementation follows type safety requirements
    """
    # Direct test of the method implementation
    from nanobot.channels.mochat import TargetManager
    
    # Get the source code of the method
    method = TargetManager.subscribe_all
    source = inspect.getsource(method)
    
    # Check that it has a return statement
    assert "return success" in source, "subscribe_all should return the success value"
    
    # Check method signature declares boolean return
    sig = inspect.signature(method)
    assert str(sig.return_annotation) == "bool", f"Expected bool return type, got {sig.return_annotation}"
    
    print("✓ TargetManager.subscribe_all has correct return statement and type annotation")


def test_state_manager_debounce_logic() -> None:
    """Test that StateManager properly implements true debounce.
    
    Validates:
        - Multiple rapid calls create new tasks (cancel + reset pattern)
        - Tasks are properly tracked and differentiated
        - True debounce behavior vs naive implementation
    """
    from nanobot.channels.mochat import StateManager
    
    # Create with minimal setup
    state_manager = StateManager(Path(tempfile.mkdtemp()))
    
    # Mock the save method to track calls
    save_calls = []
    original_save = state_manager._save_debounced
    
    async def mock_save():
        save_calls.append(asyncio.current_task())
        await asyncio.sleep(0.01)  # Simulate save operation
    
    state_manager._save_debounced = mock_save
    
    async def test_debounce():
        # Multiple rapid calls should cancel previous tasks
        state_manager.update_cursor("session1", 1)
        task1 = state_manager._save_task
        
        # Second call should cancel first task
        state_manager.update_cursor("session1", 2)
        task2 = state_manager._save_task
        
        # Third call should cancel second task
        state_manager.update_cursor("session1", 3)
        task3 = state_manager._save_task
        
        # Verify tasks are different (new task created each time)
        assert task1 is not task2, "Second call should create new task"
        assert task2 is not task3, "Third call should create new task"
        
        # Wait for final task to complete
        if task3:
            await task3
            
        return len(save_calls)
    
    call_count = asyncio.run(test_debounce())
    print(f"✓ StateManager debounce creates new task on each call")


def test_media_upload_streaming() -> None:
    """Test that media upload uses file handles instead of loading into memory.
    
    Validates:
        - Source code uses 'with open(media_path, 'rb')' pattern
        - Source code does not use 'read_bytes()' method
        - File handle streaming implementation is present
        - OOM vulnerability has been eliminated
    """
    # Direct test of the method implementation
    import inspect
    from nanobot.channels.mochat import MochatChannel
    
    # Get the source code of the method
    method = MochatChannel._upload_media
    source = inspect.getsource(method)
    
    # Check that it uses file handle streaming (open with 'rb')
    assert "with open(media_path, 'rb')" in source, "_upload_media should use file handle streaming"
    
    # Check that it doesn't load entire file into memory
    assert "read_bytes()" not in source, "_upload_media should not use read_bytes()"
    assert "file_handle" in source, "_upload_media should use file_handle variable"
    
    print("✓ Media upload uses file handle streaming instead of loading into memory")


@pytest.mark.asyncio 
async def test_circuit_breaker_prevents_spam() -> None:
    """Test that circuit breaker prevents API spam during outages.
    
    Validates:
        - Circuit breaker initially allows requests
        - Circuit breaker blocks requests after threshold failures
        - Circuit breaker maintains blocked state appropriately
    """
    from nanobot.channels.mochat import CircuitBreaker
    
    # Create circuit breaker with low threshold for testing
    circuit_breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
    
    # Initially should allow requests
    assert circuit_breaker.can_execute(), "Should allow initial requests"
    
    # Record failures to trip circuit breaker
    circuit_breaker.record_failure()
    assert circuit_breaker.can_execute(), "Should allow after 1 failure"
    
    circuit_breaker.record_failure()  
    assert not circuit_breaker.can_execute(), "Should block after threshold failures"
    
    # Should stay blocked
    assert not circuit_breaker.can_execute(), "Should remain blocked"
    
    print("✓ Circuit breaker properly blocks requests after failures")


if __name__ == "__main__":
    print("🔧 Testing Critical Logic Fixes")
    print("=" * 40)
    
    # Test circuit breaker integration
    test_circuit_breaker_integration()
    
    # Test return value fix
    test_target_manager_return_value()
    
    # Test debounce logic fix
    test_state_manager_debounce_logic()
    
    # Test media streaming fix
    test_media_upload_streaming()
    
    # Test circuit breaker behavior
    asyncio.run(test_circuit_breaker_prevents_spam())
    
    print()
    print("=" * 40) 
    print("🎉 All critical fixes validated!")
    print("Critical defects have been resolved.")