import pytest
import asyncio
import time
import json
import threading
from fastapi.testclient import TestClient
import websockets
from websockets.client import WebSocketClientProtocol
import requests
from typing import Optional

from orchestrator.src.main import app, orchestrator


class IntegrationTestClient:
    """Helper class for end-to-end testing of voice interruption flow"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = TestClient(app)
        self.session_id = f"integration_test_{int(time.time())}"
        
    def interrupt_tts(self) -> dict:
        """Send TTS interruption request"""
        response = self.client.post("/interrupt-tts", json={"session_id": self.session_id})
        return response.json(), response.status_code
    
    def start_tts_session(self) -> None:
        """Start a mock TTS session for testing"""
        abort_flag = threading.Event()
        orchestrator.active_tts_sessions[self.session_id] = {
            "abort_flag": abort_flag,
            "start_time": time.time()
        }
        return abort_flag
    
    def is_session_active(self) -> bool:
        """Check if session is still active"""
        return self.session_id in orchestrator.active_tts_sessions


@pytest.fixture
def integration_client():
    """Create integration test client"""
    client = IntegrationTestClient()
    yield client
    # Cleanup
    if client.session_id in orchestrator.active_tts_sessions:
        del orchestrator.active_tts_sessions[client.session_id]


class TestEndToEndInterruption:
    """End-to-end integration tests for voice interruption"""
    
    def test_full_interruption_flow(self, integration_client):
        """Test complete interruption flow from start to cleanup"""
        # 1. Start TTS session
        abort_flag = integration_client.start_tts_session()
        assert integration_client.is_session_active()
        assert not abort_flag.is_set()
        
        # 2. Send interruption request
        start_time = time.time()
        result, status_code = integration_client.interrupt_tts()
        interruption_time = (time.time() - start_time) * 1000
        
        # 3. Verify immediate response
        assert status_code == 200
        assert result["status"] == "interrupted"
        assert result["session_id"] == integration_client.session_id
        assert result["interrupt_time_ms"] < 50  # Should be very fast
        
        # 4. Verify session cleanup
        assert not integration_client.is_session_active()
        assert abort_flag.is_set()
        
        # 5. Verify subsequent interruption attempts fail gracefully
        result2, status_code2 = integration_client.interrupt_tts()
        assert status_code2 == 200
        assert result2["status"] == "no_active_session"
    
    def test_interruption_latency_requirements(self, integration_client):
        """Test that interruption meets latency requirements (< 50ms)"""
        # Start multiple sessions to test under load
        session_count = 10
        sessions = []
        
        for i in range(session_count):
            session_id = f"{integration_client.session_id}_{i}"
            abort_flag = threading.Event()
            orchestrator.active_tts_sessions[session_id] = {
                "abort_flag": abort_flag,
                "start_time": time.time()
            }
            sessions.append((session_id, abort_flag))
        
        # Measure interruption latency for each session
        latencies = []
        for session_id, abort_flag in sessions:
            start_time = time.time()
            response = integration_client.client.post("/interrupt-tts", 
                                                    json={"session_id": session_id})
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            # Verify successful interruption
            assert response.status_code == 200
            result = response.json()
            assert result["status"] == "interrupted"
            assert abort_flag.is_set()
        
        # Verify all latencies meet requirements
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        print(f"Interruption latencies - Avg: {avg_latency:.2f}ms, Max: {max_latency:.2f}ms")
        
        assert avg_latency < 25  # Average should be very fast
        assert max_latency < 50  # Even worst case should be under 50ms
        assert all(l < 100 for l in latencies)  # All should be reasonable
    
    def test_concurrent_interruptions(self, integration_client):
        """Test handling of concurrent interruption requests"""
        session_count = 5
        sessions = []
        
        # Set up multiple concurrent sessions
        for i in range(session_count):
            session_id = f"{integration_client.session_id}_concurrent_{i}"
            abort_flag = threading.Event()
            orchestrator.active_tts_sessions[session_id] = {
                "abort_flag": abort_flag,
                "start_time": time.time()
            }
            sessions.append((session_id, abort_flag))
        
        # Send concurrent interruption requests
        def interrupt_session(session_id):
            response = integration_client.client.post("/interrupt-tts", 
                                                    json={"session_id": session_id})
            return response.json(), response.status_code, session_id
        
        # Use threading to send concurrent requests
        threads = []
        results = []
        
        def run_interrupt(session_id):
            result = interrupt_session(session_id)
            results.append(result)
        
        # Start all threads simultaneously
        for session_id, _ in sessions:
            thread = threading.Thread(target=run_interrupt, args=(session_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all to complete
        for thread in threads:
            thread.join(timeout=5)  # 5 second timeout
        
        # Verify all interruptions succeeded
        assert len(results) == session_count
        
        for result, status_code, session_id in results:
            assert status_code == 200
            assert result["status"] == "interrupted"
            assert result["session_id"] == session_id
        
        # Verify all sessions were cleaned up
        for session_id, abort_flag in sessions:
            assert session_id not in orchestrator.active_tts_sessions
            assert abort_flag.is_set()
    
    def test_interruption_error_handling(self, integration_client):
        """Test error handling in interruption flow"""
        # Test 1: Interrupt non-existent session
        fake_session = "non_existent_session_12345"
        result, status_code = integration_client.client.post("/interrupt-tts", 
                                                           json={"session_id": fake_session})
        
        assert status_code == 200  # Should not error, just report no session
        result_data = result.json() if hasattr(result, 'json') else result
        assert result_data["status"] == "no_active_session"
        
        # Test 2: Invalid request body
        response = integration_client.client.post("/interrupt-tts", json={})
        assert response.status_code == 422  # Validation error
        
        # Test 3: Malformed JSON (if possible with TestClient)
        response = integration_client.client.post("/interrupt-tts", 
                                                content="invalid json",
                                                headers={"Content-Type": "application/json"})
        assert response.status_code == 422  # Should handle gracefully
    
    def test_session_isolation(self, integration_client):
        """Test that session interruptions are properly isolated"""
        # Create multiple sessions
        session_a = f"{integration_client.session_id}_a"
        session_b = f"{integration_client.session_id}_b"
        session_c = f"{integration_client.session_id}_c"
        
        abort_flag_a = threading.Event()
        abort_flag_b = threading.Event()
        abort_flag_c = threading.Event()
        
        orchestrator.active_tts_sessions[session_a] = {
            "abort_flag": abort_flag_a,
            "start_time": time.time()
        }
        orchestrator.active_tts_sessions[session_b] = {
            "abort_flag": abort_flag_b,
            "start_time": time.time()
        }
        orchestrator.active_tts_sessions[session_c] = {
            "abort_flag": abort_flag_c,
            "start_time": time.time()
        }
        
        # Interrupt only session B
        response = integration_client.client.post("/interrupt-tts", 
                                                json={"session_id": session_b})
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "interrupted"
        
        # Verify only session B was affected
        assert session_a in orchestrator.active_tts_sessions
        assert session_b not in orchestrator.active_tts_sessions
        assert session_c in orchestrator.active_tts_sessions
        
        assert not abort_flag_a.is_set()
        assert abort_flag_b.is_set()
        assert not abort_flag_c.is_set()
        
        # Clean up remaining sessions
        orchestrator.cleanup_completed_session(session_a)
        orchestrator.cleanup_completed_session(session_c)


class TestInterruptionMetrics:
    """Test interruption performance metrics and monitoring"""
    
    def test_interruption_timing_accuracy(self, integration_client):
        """Test that reported interruption times are accurate"""
        abort_flag = integration_client.start_tts_session()
        
        # Measure actual time
        start_time = time.time()
        result, status_code = integration_client.interrupt_tts()
        actual_time = (time.time() - start_time) * 1000
        
        assert status_code == 200
        reported_time = result["interrupt_time_ms"]
        
        # Reported time should be close to actual time (within 10ms tolerance)
        time_diff = abs(reported_time - actual_time)
        assert time_diff < 10, f"Time difference too large: {time_diff}ms"
        
        # Reported time should be reasonable
        assert 0 <= reported_time <= 100
    
    def test_interruption_success_rate(self, integration_client):
        """Test interruption success rate under various conditions"""
        success_count = 0
        total_attempts = 50
        
        for i in range(total_attempts):
            # Create session
            session_id = f"{integration_client.session_id}_rate_test_{i}"
            abort_flag = threading.Event()
            orchestrator.active_tts_sessions[session_id] = {
                "abort_flag": abort_flag,
                "start_time": time.time()
            }
            
            # Attempt interruption
            response = integration_client.client.post("/interrupt-tts", 
                                                    json={"session_id": session_id})
            
            if response.status_code == 200:
                result = response.json()
                if result["status"] == "interrupted":
                    success_count += 1
        
        success_rate = success_count / total_attempts
        print(f"Interruption success rate: {success_rate:.2%}")
        
        # Should have very high success rate (>95%)
        assert success_rate >= 0.95


@pytest.fixture(autouse=True)
def cleanup_test_sessions():
    """Clean up any test sessions before and after each test"""
    # Clean up before test
    sessions_to_remove = [
        session_id for session_id in orchestrator.active_tts_sessions.keys()
        if "integration_test" in session_id or "test_" in session_id
    ]
    for session_id in sessions_to_remove:
        del orchestrator.active_tts_sessions[session_id]
    
    yield
    
    # Clean up after test
    sessions_to_remove = [
        session_id for session_id in orchestrator.active_tts_sessions.keys()
        if "integration_test" in session_id or "test_" in session_id
    ]
    for session_id in sessions_to_remove:
        del orchestrator.active_tts_sessions[session_id]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])