#!/usr/bin/env python3
"""
Comprehensive Test Suite for Speaker Embedding Pipeline
Tests the complete voice assistant with magical speaker identification feature.

Usage:
    python test_speaker_embedding_pipeline.py

Requirements:
    pip install requests websockets asyncio pytest numpy
"""

import asyncio
import base64
import json
import logging
import time
import requests
import websockets
import numpy as np
from typing import Dict, List, Optional, Tuple
import sys
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    name: str
    success: bool
    message: str
    duration: float = 0.0

class SpeakerEmbeddingPipelineTest:
    """Comprehensive test suite for the speaker embedding pipeline"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.base_urls = {
            'orchestrator': 'http://localhost:8000',
            'whisper_live': 'ws://localhost:9090',
            'kokoro': 'http://localhost:8880', 
            'diglett': 'http://localhost:3210',
            'chromadb': 'http://localhost:8002',
            'redis': 'redis://localhost:6379',
            'ui': 'http://localhost:3001'
        }
        self.websocket_urls = {
            'orchestrator_ws': 'ws://localhost:8000/ws/v1/voice',
            'whisper_live_ws': 'ws://localhost:9090'
        }
    
    def log_result(self, name: str, success: bool, message: str, duration: float = 0.0):
        """Log test result"""
        result = TestResult(name, success, message, duration)
        self.results.append(result)
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} {name}: {message} ({duration:.2f}s)")
    
    async def test_basic_connectivity(self) -> bool:
        """Test 1: Basic connectivity to all services"""
        logger.info("🔌 Testing basic connectivity to all services...")
        all_connected = True
        
        # Test HTTP services
        http_services = {
            'Orchestrator': f"{self.base_urls['orchestrator']}/health",
            'Kokoro TTS': f"{self.base_urls['kokoro']}/health", 
            'ChromaDB': f"{self.base_urls['chromadb']}/api/v2/heartbeat",
            'UI': self.base_urls['ui']
        }
        
        # Test Diglett separately (no health endpoint)
        diglett_services = {
            'Diglett': f"{self.base_urls['diglett']}/docs"
        }
        
        for service_name, url in http_services.items():
            start_time = time.time()
            try:
                response = requests.get(url, timeout=5)
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    self.log_result(f"Connect to {service_name}", True, 
                                  f"HTTP {response.status_code}", duration)
                else:
                    self.log_result(f"Connect to {service_name}", False, 
                                  f"HTTP {response.status_code}", duration)
                    all_connected = False
            except Exception as e:
                duration = time.time() - start_time
                self.log_result(f"Connect to {service_name}", False, str(e), duration)
                all_connected = False
        
        # Test Diglett docs endpoint  
        for service_name, url in diglett_services.items():
            start_time = time.time()
            try:
                response = requests.get(url, timeout=5)
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    self.log_result(f"Connect to {service_name}", True, 
                                  f"HTTP {response.status_code}", duration)
                else:
                    self.log_result(f"Connect to {service_name}", False, 
                                  f"HTTP {response.status_code}", duration)
                    all_connected = False
            except Exception as e:
                duration = time.time() - start_time
                self.log_result(f"Connect to {service_name}", False, str(e), duration)
                all_connected = False
        
        # Test Redis connectivity (basic ping)
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0)
            start_time = time.time()
            r.ping()
            duration = time.time() - start_time
            self.log_result("Connect to Redis", True, "PING successful", duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Connect to Redis", False, str(e), duration)
            all_connected = False
        
        return all_connected
    
    async def test_websocket_connectivity(self) -> bool:
        """Test 2: WebSocket connectivity"""
        logger.info("🌐 Testing WebSocket connectivity...")
        all_connected = True
        
        # Test Orchestrator WebSocket
        try:
            start_time = time.time()
            session_id = f"test_session_{int(time.time())}"
            uri = f"{self.websocket_urls['orchestrator_ws']}/{session_id}"
            
            async with websockets.connect(uri, timeout=5) as websocket:
                duration = time.time() - start_time
                self.log_result("Orchestrator WebSocket", True, "Connection established", duration)
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Orchestrator WebSocket", False, str(e), duration)
            all_connected = False
        
        # Test WhisperLive WebSocket
        try:
            start_time = time.time()
            async with websockets.connect(self.websocket_urls['whisper_live_ws'], timeout=5) as websocket:
                # Send required config as first message
                config = {
                    "uid": f"test_{int(time.time())}",
                    "language": "en",
                    "task": "transcribe",
                    "model": "tiny",
                    "use_vad": True
                }
                await websocket.send(json.dumps(config))
                
                # Wait for SERVER_READY response
                response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                duration = time.time() - start_time
                
                if "SERVER_READY" in response:
                    self.log_result("WhisperLive WebSocket", True, "SERVER_READY received", duration)
                else:
                    self.log_result("WhisperLive WebSocket", True, "Connected (no SERVER_READY)", duration)
                    
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("WhisperLive WebSocket", False, str(e), duration)
            all_connected = False
        
        return all_connected
    
    async def test_service_apis(self) -> bool:
        """Test 3: Individual service API functionality"""
        logger.info("🔧 Testing individual service APIs...")
        all_apis_working = True
        
        # Test ChromaDB collection creation
        try:
            start_time = time.time()
            import chromadb
            client = chromadb.HttpClient(host="localhost", port=8002)
            collection = client.get_or_create_collection("test_collection")
            duration = time.time() - start_time
            self.log_result("ChromaDB Collection", True, f"Created/accessed collection", duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("ChromaDB Collection", False, str(e), duration)
            all_apis_working = False
        
        # Test Diglett embedding endpoint
        try:
            start_time = time.time()
            # Create dummy audio data (WAV format simulation)
            dummy_audio = self.generate_dummy_audio()
            
            files = {"file": ("test_audio.wav", dummy_audio, "audio/wav")}
            response = requests.post(f"{self.base_urls['diglett']}/embed", files=files, timeout=10)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                if "speaker_embedding" in result:
                    embedding_size = len(result["speaker_embedding"])
                    self.log_result("Diglett Embedding", True, f"Generated {embedding_size}D embedding", duration)
                else:
                    self.log_result("Diglett Embedding", False, "No embedding in response", duration)
                    all_apis_working = False
            else:
                self.log_result("Diglett Embedding", False, f"HTTP {response.status_code}", duration)
                all_apis_working = False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Diglett Embedding", False, str(e), duration)
            all_apis_working = False
        
        # Test Kokoro TTS
        try:
            start_time = time.time()
            tts_payload = {
                "model": "kokoro",
                "input": "Hello, this is a test.",
                "voice": "af_bella",
                "response_format": "wav"
            }
            
            response = requests.post(f"{self.base_urls['kokoro']}/v1/audio/speech", 
                                   json=tts_payload, timeout=10)
            duration = time.time() - start_time
            
            if response.status_code == 200 and len(response.content) > 0:
                audio_size = len(response.content)
                self.log_result("Kokoro TTS", True, f"Generated {audio_size} bytes audio", duration)
            else:
                self.log_result("Kokoro TTS", False, f"HTTP {response.status_code}", duration)
                all_apis_working = False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Kokoro TTS", False, str(e), duration)
            all_apis_working = False
        
        return all_apis_working
    
    async def test_speaker_identification_workflow(self) -> bool:
        """Test 4: Complete speaker identification workflow"""
        logger.info("🎤 Testing speaker identification workflow...")
        
        session_id = f"test_session_{int(time.time())}"
        uri = f"{self.websocket_urls['orchestrator_ws']}/{session_id}"
        
        try:
            start_time = time.time()
            async with websockets.connect(uri, timeout=10) as websocket:
                
                # Step 1: Send audio data
                dummy_audio = self.generate_dummy_audio()
                audio_base64 = base64.b64encode(dummy_audio).decode()
                
                audio_message = {
                    "event": "audio_stream",
                    "audio_data": audio_base64
                }
                
                await websocket.send(json.dumps(audio_message))
                logger.info("📤 Sent audio data to orchestrator")
                
                # Step 2: Listen for responses
                responses = []
                transcript_received = False
                speaker_identified = False
                assistant_response = False
                
                timeout_duration = 30  # 30 seconds timeout
                end_time = time.time() + timeout_duration
                
                while time.time() < end_time:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        message = json.loads(response)
                        responses.append(message)
                        
                        event = message.get("event")
                        logger.info(f"📥 Received event: {event}")
                        
                        if event == "transcript.update":
                            transcript_received = True
                            logger.info(f"🗣️ Transcript: {message.get('data', {}).get('text', '')}")
                            
                        elif event == "speaker.identified":
                            speaker_identified = True
                            data = message.get("data", {})
                            logger.info(f"👤 Speaker identified: {data.get('name')} (status: {data.get('status')})")
                            
                        elif event == "assistant.speak":
                            assistant_response = True
                            data = message.get("data", {})
                            logger.info(f"🤖 Assistant response: {data.get('text', '')}")
                            
                        elif event == "error":
                            error_msg = message.get("data", {}).get("message", "Unknown error")
                            logger.error(f"❌ Error received: {error_msg}")
                            
                    except asyncio.TimeoutError:
                        # Continue listening, but check if we've received what we need
                        if transcript_received or speaker_identified:
                            break
                        continue
                
                duration = time.time() - start_time
                
                # Evaluate results
                if transcript_received:
                    self.log_result("Transcript Generation", True, "Received transcript from audio", duration)
                else:
                    self.log_result("Transcript Generation", False, "No transcript received", duration)
                
                if speaker_identified:
                    self.log_result("Speaker Identification", True, "Speaker identified/created", duration)
                else:
                    self.log_result("Speaker Identification", False, "No speaker identification", duration)
                
                if assistant_response:
                    self.log_result("Assistant Response", True, "Received assistant response", duration)
                else:
                    self.log_result("Assistant Response", False, "No assistant response", duration)
                
                return transcript_received and speaker_identified
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Speaker Workflow", False, str(e), duration)
            return False
    
    async def test_speaker_name_claiming(self) -> bool:
        """Test 5: Speaker name claiming functionality"""
        logger.info("📝 Testing speaker name claiming...")
        
        session_id = f"test_claim_{int(time.time())}"
        uri = f"{self.websocket_urls['orchestrator_ws']}/{session_id}"
        
        try:
            start_time = time.time()
            async with websockets.connect(uri, timeout=10) as websocket:
                
                # First, create a new speaker by sending audio
                dummy_audio = self.generate_dummy_audio()
                audio_base64 = base64.b64encode(dummy_audio).decode()
                
                audio_message = {
                    "event": "audio_stream", 
                    "audio_data": audio_base64
                }
                await websocket.send(json.dumps(audio_message))
                
                # Wait for speaker identification
                user_id = None
                for _ in range(10):  # Wait up to 10 responses
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                        message = json.loads(response)
                        
                        if message.get("event") == "speaker.identified":
                            user_id = message.get("data", {}).get("user_id")
                            break
                    except asyncio.TimeoutError:
                        continue
                
                if not user_id:
                    self.log_result("Speaker Name Claiming", False, "No user_id received", time.time() - start_time)
                    return False
                
                # Now test name claiming
                claim_message = {
                    "event": "speaker.claim",
                    "user_id": user_id,
                    "new_name": "TestUser123"
                }
                await websocket.send(json.dumps(claim_message))
                
                # Wait for name update confirmation
                name_updated = False
                for _ in range(5):
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                        message = json.loads(response)
                        
                        if message.get("event") == "speaker.renamed":
                            data = message.get("data", {})
                            if data.get("new_name") == "TestUser123":
                                name_updated = True
                                break
                    except asyncio.TimeoutError:
                        continue
                
                duration = time.time() - start_time
                
                if name_updated:
                    self.log_result("Speaker Name Claiming", True, "Name successfully updated", duration)
                    return True
                else:
                    self.log_result("Speaker Name Claiming", False, "Name update not confirmed", duration)
                    return False
                    
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Speaker Name Claiming", False, str(e), duration)
            return False
    
    async def test_memory_persistence(self) -> bool:
        """Test 6: Memory persistence across sessions"""
        logger.info("💾 Testing memory persistence...")
        
        # This test would require creating a speaker, disconnecting, 
        # then reconnecting with the same voice to verify recognition
        # For now, we'll test the storage components
        
        try:
            start_time = time.time()
            
            # Test ChromaDB persistence
            import chromadb
            client = chromadb.HttpClient(host="localhost", port=8002)
            collection = client.get_or_create_collection("speaker_embeddings")
            
            # Add a test embedding
            test_embedding = np.random.rand(128).tolist()
            test_id = f"test_user_{int(time.time())}"
            
            collection.add(
                embeddings=[test_embedding],
                ids=[test_id]
            )
            
            # Query it back
            results = collection.get(ids=[test_id])
            
            duration = time.time() - start_time
            
            if results and results["ids"]:
                self.log_result("ChromaDB Persistence", True, "Embedding stored and retrieved", duration)
                
                # Test Redis storage
                import redis
                r = redis.Redis(host='localhost', port=6379, db=0)
                
                # Store speaker profile
                profile_key = f"speaker:{test_id}"
                r.hset(profile_key, mapping={
                    "name": "TestUser",
                    "status": "active"
                })
                
                # Retrieve it
                profile = r.hgetall(profile_key)
                
                if profile and profile.get(b"name") == b"TestUser":
                    self.log_result("Redis Persistence", True, "Profile stored and retrieved", duration)
                    return True
                else:
                    self.log_result("Redis Persistence", False, "Profile retrieval failed", duration)
                    return False
            else:
                self.log_result("ChromaDB Persistence", False, "Embedding retrieval failed", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Memory Persistence", False, str(e), duration)
            return False
    
    def generate_dummy_audio(self) -> bytes:
        """Generate dummy WAV audio data for testing"""
        # Simple WAV header + some random audio data
        # This is a minimal WAV file structure for testing
        sample_rate = 16000
        duration = 1.0  # 1 second
        samples = int(sample_rate * duration)
        
        # WAV header (44 bytes)
        wav_header = bytearray()
        wav_header.extend(b'RIFF')
        wav_header.extend((samples * 2 + 36).to_bytes(4, 'little'))  # File size
        wav_header.extend(b'WAVE')
        wav_header.extend(b'fmt ')
        wav_header.extend((16).to_bytes(4, 'little'))  # Format chunk size
        wav_header.extend((1).to_bytes(2, 'little'))   # Audio format (PCM)
        wav_header.extend((1).to_bytes(2, 'little'))   # Number of channels
        wav_header.extend(sample_rate.to_bytes(4, 'little'))  # Sample rate
        wav_header.extend((sample_rate * 2).to_bytes(4, 'little'))  # Byte rate
        wav_header.extend((2).to_bytes(2, 'little'))   # Block align
        wav_header.extend((16).to_bytes(2, 'little'))  # Bits per sample
        wav_header.extend(b'data')
        wav_header.extend((samples * 2).to_bytes(4, 'little'))  # Data chunk size
        
        # Generate some simple audio data (sine wave)
        audio_data = bytearray()
        for i in range(samples):
            # Simple sine wave at 440 Hz
            value = int(16000 * np.sin(2 * np.pi * 440 * i / sample_rate))
            audio_data.extend(value.to_bytes(2, 'little', signed=True))
        
        return bytes(wav_header + audio_data)
    
    def print_summary(self):
        """Print test summary"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        
        print("\n" + "="*80)
        print("🧪 SPEAKER EMBEDDING PIPELINE TEST SUMMARY")
        print("="*80)
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"📊 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print(f"\n❌ FAILED TESTS:")
            for result in self.results:
                if not result.success:
                    print(f"   • {result.name}: {result.message}")
        
        print("\n📋 DETAILED RESULTS:")
        for result in self.results:
            status = "✅" if result.success else "❌"
            print(f"   {status} {result.name:<30} {result.message:<40} ({result.duration:.2f}s)")
        
        print("="*80)
        
        return failed_tests == 0

async def main():
    """Run the complete test suite"""
    print("🚀 Starting Speaker Embedding Pipeline Test Suite...")
    print("This will test the complete voice assistant with magical speaker identification.")
    print("="*80)
    
    tester = SpeakerEmbeddingPipelineTest()
    
    # Run all tests in sequence
    tests = [
        ("Basic Connectivity", tester.test_basic_connectivity),
        ("WebSocket Connectivity", tester.test_websocket_connectivity), 
        ("Service APIs", tester.test_service_apis),
        ("Speaker Identification Workflow", tester.test_speaker_identification_workflow),
        ("Speaker Name Claiming", tester.test_speaker_name_claiming),
        ("Memory Persistence", tester.test_memory_persistence)
    ]
    
    overall_success = True
    for test_name, test_func in tests:
        print(f"\n🔄 Running {test_name}...")
        try:
            success = await test_func()
            if not success:
                overall_success = False
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            overall_success = False
    
    # Print final summary
    success = tester.print_summary()
    
    if success:
        print("🎉 All tests passed! Speaker embedding pipeline is working correctly.")
        sys.exit(0)
    else:
        print("⚠️ Some tests failed. Check the results above for details.")
        sys.exit(1)

if __name__ == "__main__":
    # Check if running in async context
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)