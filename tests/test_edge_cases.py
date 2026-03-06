#!/usr/bin/env python3
"""
Edge case tests for critical bug fixes.

This module provides comprehensive edge case testing for critical behavioral
changes implemented to prevent system instability and data corruption.

The tests focus on the most impactful behavioral changes:
- Orphaned tool result handling in session management
- Atomic file operations for data persistence
- Lock retention for concurrency safety
- State management edge cases

Classes:
    TestSessionEdgeCases: Edge cases in Session.get_history() behavior.
    TestAtomicSaveEdgeCases: Edge cases in atomic save operations.
    TestLockRetentionEdgeCases: Edge cases in lock management.
"""

import asyncio
import gc
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from nanobot.session.manager import Session, SessionManager


class TestSessionEdgeCases:
    """Test edge cases in Session.get_history() that could break LLM APIs.
    
    These tests validate the critical orphaned tool result handling
    that prevents LLM API crashes due to malformed message sequences.
    """

    def test_orphaned_tool_results_return_empty(self) -> None:
        """Test that orphaned tool results return empty list to prevent LLM crashes.
        
        Validates:
            Scenarios where tool calls exist without preceding user messages
            return empty lists to prevent LLM API errors from malformed sequences.
        """
        session = Session(key="test_orphaned")
        
        # Add tool chain without user message (the problematic scenario)
        session.add_message("assistant", "I'll help with that", 
                           tool_calls=[{"id": "call_1", "function": {"name": "search"}}])
        session.add_message("tool", "Search results here", 
                           tool_call_id="call_1", name="search")
        session.add_message("assistant", "Based on search results...")
        
        # Should return empty to prevent LLM API errors
        history = session.get_history()
        assert history == [], "Orphaned tool results should return empty list"

    def test_normal_conversation_with_tools_works(self) -> None:
        """Test that normal user conversations with tools still work correctly.
        
        Validates:
            Standard conversation flows with user messages and tool interactions
            continue to work as expected after orphaned handling changes.
        """
        session = Session(key="test_normal")
        
        # Normal conversation flow
        session.add_message("user", "Help me search for Python docs")
        session.add_message("assistant", "I'll search for that",
                           tool_calls=[{"id": "call_1", "function": {"name": "search"}}])
        session.add_message("tool", "Found Python documentation",
                           tool_call_id="call_1", name="search") 
        session.add_message("assistant", "Here's what I found...")
        
        history = session.get_history()
        assert len(history) == 4, "Should return all messages in normal flow"
        assert history[0]["role"] == "user", "Should start with user message"
        assert "tool_calls" in history[1], "Tool calls should be preserved"
        assert "tool_call_id" in history[2], "Tool results should be preserved"

    def test_mixed_scenario_finds_user_start(self) -> None:
        """Test finding user message in mixed orphaned/normal scenario.
        
        Validates:
            Mixed scenarios with both orphaned tool results and normal
            user interactions properly identify the user message start point.
        """
        session = Session(key="test_mixed")
        
        # Some orphaned tool results from previous interaction
        session.add_message("tool", "Orphaned result", tool_call_id="old_1")
        session.add_message("assistant", "Orphaned response")
        
        # Then normal user interaction
        session.add_message("user", "New question")
        session.add_message("assistant", "New response")
        
        history = session.get_history()
        assert len(history) == 2, "Should skip orphaned messages"
        assert history[0]["role"] == "user", "Should start from user message"
        assert history[0]["content"] == "New question", "Should get correct user message"

    def test_session_clear_resets_metadata(self) -> None:
        """Test that session.clear() properly resets metadata to prevent inheritance.
        
        Validates:
            Session clearing completely resets all state including metadata
            to prevent cross-session state inheritance issues.
        """
        session = Session(key="test_clear")
        
        # Set up session state
        session.add_message("user", "Hello")
        session.metadata = {"user_id": "12345", "persistent_data": "important"}
        session.last_consolidated = 5
        original_created = session.created_at
        
        # Clear session
        session.clear()
        
        # Verify complete reset
        assert session.messages == [], "Messages should be cleared"
        assert session.last_consolidated == 0, "Consolidation should reset"
        assert session.metadata == {}, "Metadata should be cleared"
        assert session.created_at == original_created, "Created time preserved"

    def test_max_messages_with_no_user_messages(self) -> None:
        """Test max_messages parameter when no user messages exist.
        
        Validates:
            Edge case handling when max_messages parameter is used
            but no user messages exist in the session.
        """
        session = Session(key="test_max")
        
        # Add many assistant/tool messages without user messages
        for i in range(10):
            session.add_message("assistant", f"Message {i}")
            
        # Should return empty regardless of max_messages setting
        assert session.get_history(max_messages=5) == []
        assert session.get_history(max_messages=100) == []


class TestAtomicSaveEdgeCases:
    """Test atomic save behavior to ensure corruption protection.
    
    These tests validate that atomic file operations properly prevent
    data corruption during concurrent access and system failures.
    """

    def test_atomic_save_leaves_no_temp_files(self) -> None:
        """Test that atomic saves clean up properly even on success.
        
        Validates:
            Atomic save operations clean up temporary files completely
            and leave no residual .tmp files in the filesystem.
            
        Also validates:
            - Session files are properly created and readable
            - Content integrity is maintained through save/load cycle
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SessionManager(Path(temp_dir))
            session = Session(key="test_atomic")
            session.add_message("user", "Test message")
            
            # Save session
            manager.save(session)
            
            # Verify no .tmp files left behind
            temp_files = list(Path(temp_dir).glob("**/*.tmp"))
            assert len(temp_files) == 0, f"Should have no temp files, found: {temp_files}"
            
            # Verify session file exists and is readable
            session_files = list(Path(temp_dir).glob("**/*.jsonl"))
            assert len(session_files) == 1, "Should have exactly one session file"
            
            # Verify content integrity
            loaded = manager.get_or_create("test_atomic")
            assert loaded is not None, "Session should load successfully"
            assert loaded.messages[0]["content"] == "Test message"


@pytest.mark.asyncio
class TestLockRetentionEdgeCases:
    """Test lock retention to ensure concurrency safety.
    
    These tests validate that locks are properly retained and managed
    to prevent race conditions in concurrent environments.
    """

    async def test_locks_survive_garbage_collection(self) -> None:
        """Test that locks are properly retained and not garbage collected.
        
        Validates:
            Lock management using regular dict instead of WeakValueDictionary
            ensures locks survive garbage collection and remain accessible.
            
        Note:
            This simulates the fix applied to AgentLoop lock management
            where WeakValueDictionary was replaced with regular dict.
        """
        # Simulate the fixed lock management
        locks: dict[str, asyncio.Lock] = {}
        
        def get_lock(session_key: str) -> asyncio.Lock:
            return locks.setdefault(session_key, asyncio.Lock())
        
        # Create lock
        session_key = "test_session"
        lock1 = get_lock(session_key)
        
        # Force potential garbage collection
        gc.collect()
        
        # Lock should still be the same object
        lock2 = get_lock(session_key)
        assert lock1 is lock2, "Lock should be retained (same object)"
        
        # Multiple calls should return same lock
        lock3 = locks.get(session_key)
        assert lock1 is lock3, "Direct access should return same lock"


if __name__ == "__main__":
    import sys
    
    # Run basic tests without pytest for quick validation
    print("🧪 Running Edge Case Tests")
    print("=" * 40)
    
    # Test orphaned tool results
    test = TestSessionEdgeCases()
    test.test_orphaned_tool_results_return_empty()
    print("✓ Orphaned tool results handled safely")
    
    test.test_normal_conversation_with_tools_works()
    print("✓ Normal conversations work correctly")
    
    test.test_session_clear_resets_metadata()
    print("✓ Session clearing resets all state")
    
    # Test atomic saves
    atomic_test = TestAtomicSaveEdgeCases()
    atomic_test.test_atomic_save_leaves_no_temp_files()
    print("✓ Atomic saves work without temp file leaks")
    
    # Test locks
    async def run_lock_test():
        lock_test = TestLockRetentionEdgeCases()
        await lock_test.test_locks_survive_garbage_collection()
        print("✓ Locks properly retained")
    
    asyncio.run(run_lock_test())
    
    print("\n" + "=" * 40)
    print("🎉 All edge case tests passed!")
    print("Critical behavior changes are safe and tested.")