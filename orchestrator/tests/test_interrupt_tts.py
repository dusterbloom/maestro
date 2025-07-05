import pytest
import json
import time
import threading
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
from src.main import app, orchestrator, VoiceOrchestrator


@pytest.fixture
def client():
    """Create a test client for the FastAPI app"""
    return TestClient(app)


@pytest.fixture
def mock_orchestrator():
    """Create a mock orchestrator with test session data"""
    orchestrator_mock = Mock(spec=VoiceOrchestrator)
    orchestrator_mock.active_tts_sessions = {}
    return orchestrator_mock


class TestInterruptTtsEndpoint:
    """Test cases for the /interrupt-tts endpoint"""
    
    def test_interrupt_active_session_success(self, client):
        """Test successful interruption of an active TTS session"""
        session_id = "test_session_123"
        
        # Set up an active session in the orchestrator
        abort_flag = threading.Event()
        orchestrator.active_tts_sessions[session_id] = {
            "abort_flag": abort_flag,
            "start_time": time.time()
        }
        
        # Send interrupt request
        response = client.post("/interrupt-tts", json={"session_id": session_id})
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "interrupted"
        assert data["session_id"] == session_id
        assert "interrupt_time_ms" in data
        assert data["message"] == "TTS generation interrupted successfully"
        
        # Verify session was cleaned up
        assert session_id not in orchestrator.active_tts_sessions
        
        # Verify abort flag was set
        assert abort_flag.is_set()
    
    def test_interrupt_nonexistent_session(self, client):
        """Test interruption request for non-existent session"""
        session_id = "nonexistent_session"
        
        # Ensure session doesn't exist
        if session_id in orchestrator.active_tts_sessions:
            del orchestrator.active_tts_sessions[session_id]
        
        # Send interrupt request
        response = client.post("/interrupt-tts", json={"session_id": session_id})
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "no_active_session"
        assert data["session_id"] == session_id
        assert "interrupt_time_ms" in data
        assert data["message"] == "No active TTS session to interrupt"
    
    def test_interrupt_missing_session_id(self, client):
        """Test interrupt request with missing session_id"""
        response = client.post("/interrupt-tts", json={})
        
        # Should return validation error
        assert response.status_code == 422  # FastAPI validation error
    
    def test_interrupt_invalid_request_body(self, client):
        """Test interrupt request with invalid JSON"""
        response = client.post("/interrupt-tts", 
                             headers={"Content-Type": "application/json"},
                             content="invalid json")
        
        # Should return validation error
        assert response.status_code == 422
    
    def test_interrupt_multiple_sessions(self, client):
        """Test interrupting multiple sessions"""
        session_ids = ["session_1", "session_2", "session_3"]
        abort_flags = []
        
        # Set up multiple active sessions
        for session_id in session_ids:
            abort_flag = threading.Event()
            abort_flags.append(abort_flag)
            orchestrator.active_tts_sessions[session_id] = {
                "abort_flag": abort_flag,
                "start_time": time.time()
            }
        
        # Interrupt each session
        for i, session_id in enumerate(session_ids):
            response = client.post("/interrupt-tts", json={"session_id": session_id})
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "interrupted"
            assert data["session_id"] == session_id
            
            # Verify session was cleaned up
            assert session_id not in orchestrator.active_tts_sessions
            
            # Verify abort flag was set
            assert abort_flags[i].is_set()
    
    def test_interrupt_timing_measurement(self, client):
        """Test that interrupt timing is measured accurately"""
        session_id = "timing_test_session"
        
        # Set up active session
        abort_flag = threading.Event()
        orchestrator.active_tts_sessions[session_id] = {
            "abort_flag": abort_flag,
            "start_time": time.time()
        }
        
        start_time = time.time()
        response = client.post("/interrupt-tts", json={"session_id": session_id})
        end_time = time.time()
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify timing is reasonable (should be very fast, < 100ms)
        interrupt_time_ms = data["interrupt_time_ms"]
        actual_time_ms = (end_time - start_time) * 1000
        
        assert interrupt_time_ms >= 0
        assert interrupt_time_ms <= actual_time_ms + 10  # Allow small margin for measurement differences


class TestVoiceOrchestratorInterruptMethods:
    """Test cases for VoiceOrchestrator interrupt-related methods"""
    
    def test_interrupt_tts_session_success(self):
        """Test successful TTS session interruption"""
        session_id = "test_session"
        abort_flag = threading.Event()
        
        # Set up session
        orchestrator.active_tts_sessions[session_id] = {
            "abort_flag": abort_flag,
            "start_time": time.time()
        }
        
        # Interrupt session
        result = orchestrator.interrupt_tts_session(session_id)
        
        assert result is True
        assert session_id not in orchestrator.active_tts_sessions
        assert abort_flag.is_set()
    
    def test_interrupt_tts_session_not_found(self):
        """Test interrupting non-existent session"""
        session_id = "nonexistent"
        
        # Ensure session doesn't exist
        if session_id in orchestrator.active_tts_sessions:
            del orchestrator.active_tts_sessions[session_id]
        
        result = orchestrator.interrupt_tts_session(session_id)
        
        assert result is False
    
    def test_cleanup_completed_session(self):
        """Test cleaning up a completed session"""
        session_id = "completed_session"
        
        # Set up session
        orchestrator.active_tts_sessions[session_id] = {
            "abort_flag": threading.Event(),
            "start_time": time.time()
        }
        
        # Clean up session
        orchestrator.cleanup_completed_session(session_id)
        
        assert session_id not in orchestrator.active_tts_sessions
    
    def test_cleanup_nonexistent_session(self):
        """Test cleaning up non-existent session (should not error)"""
        session_id = "nonexistent"
        
        # Should not raise exception
        orchestrator.cleanup_completed_session(session_id)
    
    def test_session_tracking_isolation(self):
        """Test that sessions are isolated from each other"""
        session1 = "session_1"
        session2 = "session_2"
        
        abort_flag1 = threading.Event()
        abort_flag2 = threading.Event()
        
        # Set up two sessions
        orchestrator.active_tts_sessions[session1] = {
            "abort_flag": abort_flag1,
            "start_time": time.time()
        }
        orchestrator.active_tts_sessions[session2] = {
            "abort_flag": abort_flag2,
            "start_time": time.time()
        }
        
        # Interrupt only session1
        result = orchestrator.interrupt_tts_session(session1)
        
        assert result is True
        assert session1 not in orchestrator.active_tts_sessions
        assert session2 in orchestrator.active_tts_sessions
        assert abort_flag1.is_set()
        assert not abort_flag2.is_set()
        
        # Clean up remaining session
        orchestrator.cleanup_completed_session(session2)


@pytest.fixture(autouse=True)
def cleanup_sessions():
    """Clean up any test sessions after each test"""
    yield
    # Clear all test sessions
    orchestrator.active_tts_sessions.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])