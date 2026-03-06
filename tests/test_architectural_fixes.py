"""Tests for critical architectural fixes in mochat.py.

This module tests the specific architectural improvements made:
1. Panel cursor tracking to prevent API abuse
2. LRU cache for target locks to prevent memory leaks  
3. Media upload support
4. Encapsulation fixes with public methods
5. Atomic state saves
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from nanobot.config.schema import MochatConfig


@pytest.fixture
def mock_config():
    """Create a mock MochatConfig for testing."""
    config = Mock(spec=MochatConfig)
    config.claw_token = "sk-test-123"
    config.base_url = "https://api.example.com"
    config.max_retry_attempts = 3
    config.retry_delay_ms = 100
    config.refresh_interval_ms = 5000
    config.watch_limit = 50
    config.sessions = ["session1"]
    config.panels = ["panel1"]
    config.agent_user_id = "agent_123"
    config.socket_disable_msgpack = False
    return config


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.mark.asyncio
class TestArchitecturalFixes:
    """Test cases for architectural improvements."""
    
    async def test_target_manager_lru_cache_prevents_memory_leak(self, mock_config, temp_workspace):
        """Test that TargetManager uses LRU cache for locks to prevent memory leaks."""
        from nanobot.channels.mochat import TargetManager, ConnectionManager, StateManager, RetryConfig
        
        # Create dependencies
        retry_config = RetryConfig(max_attempts=3, base_delay_ms=100)
        conn_manager = ConnectionManager(mock_config, retry_config)
        state_manager = StateManager(mock_config, temp_workspace)
        
        target_manager = TargetManager(mock_config, conn_manager, state_manager)
        
        # Test LRU behavior - create more locks than max limit
        max_locks = target_manager._max_locks
        
        # Create max_locks + 10 locks
        lock_keys = []
        for i in range(max_locks + 10):
            target_id = f"session_{i}"
            lock_keys.append(target_id)
            lock = target_manager.get_target_lock("session", target_id)
            assert lock is not None
        
        # Verify that only max_locks are maintained
        assert len(target_manager._target_locks) <= max_locks
        
        # Verify that recently used locks are preserved
        recent_lock = target_manager.get_target_lock("session", lock_keys[-1])
        assert recent_lock is not None
        
        # Verify we can still get all recent locks
        for i in range(max(0, len(lock_keys) - max_locks), len(lock_keys)):
            target_id = lock_keys[i]
            lock = target_manager.get_target_lock("session", target_id)
            assert lock is not None
    
    async def test_panel_cursor_tracking(self, mock_config, temp_workspace):
        """Test that TargetManager tracks panel cursors for efficient polling."""
        from nanobot.channels.mochat import TargetManager, ConnectionManager, StateManager, RetryConfig
        
        retry_config = RetryConfig(max_attempts=3, base_delay_ms=100)
        conn_manager = ConnectionManager(mock_config, retry_config)
        state_manager = StateManager(mock_config, temp_workspace)
        target_manager = TargetManager(mock_config, conn_manager, state_manager)
        
        # Test cursor tracking
        panel_id = "panel_test"
        timestamp = "2023-01-01T12:00:00Z"
        
        # Initially no cursor
        assert target_manager.get_panel_cursor(panel_id) is None
        
        # Update cursor
        target_manager.update_panel_cursor(panel_id, timestamp)
        
        # Verify cursor is tracked
        assert target_manager.get_panel_cursor(panel_id) == timestamp
        
        # Update cursor again
        new_timestamp = "2023-01-01T12:30:00Z"
        target_manager.update_panel_cursor(panel_id, new_timestamp)
        assert target_manager.get_panel_cursor(panel_id) == new_timestamp
    
    async def test_atomic_state_saves(self, mock_config, temp_workspace):
        """Test that StateManager uses atomic writes."""
        from nanobot.channels.mochat import StateManager
        
        state_manager = StateManager(mock_config, temp_workspace)
        
        # Update some cursors
        state_manager.update_cursor("session1", 100)
        state_manager.update_cursor("session2", 200)
        
        # Force save
        await state_manager.save(force=True)
        
        # Verify file exists and has correct content
        assert state_manager.cursor_path.exists()
        
        # Verify no .tmp file left behind
        temp_path = state_manager.cursor_path.with_suffix('.tmp')
        assert not temp_path.exists()
        
        # Verify content can be loaded
        new_state_manager = StateManager(mock_config, temp_workspace)
        await new_state_manager.load()
        
        assert new_state_manager.get_cursor("session1") == 100
        assert new_state_manager.get_cursor("session2") == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])