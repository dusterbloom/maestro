# Testing Guide - Voice Orchestrator

## TEST-001: E2E Latency Test Implementation

### Objective
Create automated end-to-end latency measurement that validates <500ms requirement.

### Test Script: scripts/latency-test.py

```python
#!/usr/bin/env python3
import asyncio
import time
import json
import websockets
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class LatencyMeasurement:
    test_name: str
    audio_size_ms: int
    total_latency_ms: float
    components: Dict[str, float]

class LatencyTester:
    def __init__(self, ws_url: str = "ws://localhost:8000/ws"):
        self.ws_url = ws_url
        self.results: List[LatencyMeasurement] = []
        
    async def generate_audio(self, duration_ms: int) -> bytes:
        """Generate synthetic audio data"""
        sample_rate = 16000
        samples = int(sample_rate * duration_ms / 1000)
        # Generate silence (real audio not needed for latency test)
        audio_data = np.zeros(samples, dtype=np.int16)
        return audio_data.tobytes()
    
    async def measure_single_request(self, audio_data: bytes, test_name: str) -> LatencyMeasurement:
        """Measure latency for a single request"""
        async with websockets.connect(self.ws_url) as websocket:
            start_time = time.time()
            
            # Send audio
            await websocket.send(audio_data)
            
            # Wait for response
            response = await websocket.recv()
            
            end_time = time.time()
            total_latency = (end_time - start_time) * 1000  # Convert to ms
            
            return LatencyMeasurement(
                test_name=test_name,
                audio_size_ms=len(audio_data) * 1000 // (16000 * 2),  # Rough estimate
                total_latency_ms=total_latency,
                components={}  # Would need server-side instrumentation for component breakdown
            )
    
    async def run_test_suite(self):
        """Run all latency tests"""
        test_cases = [
            ("Simple greeting", 1000),    # 1 second audio
            ("Medium query", 3000),        # 3 seconds
            ("Complex query", 5000),       # 5 seconds
        ]
        
        for test_name, duration_ms in test_cases:
            audio_data = await self.generate_audio(duration_ms)
            
            # Run 5 iterations per test
            for i in range(5):
                try:
                    measurement = await self.measure_single_request(
                        audio_data, 
                        f"{test_name} (iteration {i+1})"
                    )
                    self.results.append(measurement)
                    await asyncio.sleep(1)  # Avoid overwhelming the server
                except Exception as e:
                    print(f"Error in {test_name}: {e}")
    
    async def run_concurrent_test(self, num_users: int = 10):
        """Test concurrent users"""
        audio_data = await self.generate_audio(1000)
        
        async def single_user(user_id: int):
            return await self.measure_single_request(
                audio_data, 
                f"Concurrent user {user_id}"
            )
        
        tasks = [single_user(i) for i in range(num_users)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, LatencyMeasurement):
                self.results.append(result)
    
    def generate_report(self) -> Dict:
        """Generate JSON report"""
        if not self.results:
            return {"error": "No results collected"}
        
        latencies = [r.total_latency_ms for r in self.results]
        
        return {
            "summary": {
                "total_tests": len(self.results),
                "avg_latency_ms": np.mean(latencies),
                "p50_latency_ms": np.percentile(latencies, 50),
                "p95_latency_ms": np.percentile(latencies, 95),
                "p99_latency_ms": np.percentile(latencies, 99),
                "min_latency_ms": np.min(latencies),
                "max_latency_ms": np.max(latencies),
                "target_met": np.percentile(latencies, 95) <= 500
            },
            "details": [
                {
                    "test": r.test_name,
                    "latency_ms": r.total_latency_ms,
                    "audio_size_ms": r.audio_size_ms
                }
                for r in self.results
            ]
        }

async def main():
    print("🧪 Voice Orchestrator Latency Test")
    print("=" * 50)
    
    tester = LatencyTester()
    
    # Check if service is running
    try:
        async with websockets.connect(tester.ws_url):
            print("✅ Connected to orchestrator")
    except:
        print("❌ Cannot connect to orchestrator. Is it running?")
        return
    
    # Run tests
    print("\nRunning single-user tests...")
    await tester.run_test_suite()
    
    