#!/usr/bin/env python3
"""
Voice Orchestrator Latency Test

Tests end-to-end latency of the voice pipeline to ensure <500ms target is met.
"""

import asyncio
import json
import time
import numpy as np
import websockets
from dataclasses import dataclass
from typing import List, Dict, Optional
import argparse
import statistics

@dataclass
class LatencyMeasurement:
    test_name: str
    audio_duration_ms: int
    total_latency_ms: float
    success: bool
    error: Optional[str] = None

class VoiceLatencyTester:
    def __init__(self, ws_url: str = "ws://localhost:8000/ws"):
        self.ws_url = ws_url
        self.results: List[LatencyMeasurement] = []
        
    def generate_test_audio(self, duration_ms: int) -> bytes:
        """Generate synthetic audio data for testing"""
        sample_rate = 16000
        samples = int(sample_rate * duration_ms / 1000)
        
        # Generate a simple sine wave as test audio
        frequency = 440  # A note
        t = np.linspace(0, duration_ms / 1000, samples, False)
        audio_data = np.sin(2 * np.pi * frequency * t)
        
        # Convert to float32 and then to bytes
        audio_float32 = audio_data.astype(np.float32)
        return audio_float32.tobytes()
    
    async def measure_single_request(
        self, 
        audio_data: bytes, 
        test_name: str,
        timeout: float = 30.0
    ) -> LatencyMeasurement:
        """Measure latency for a single voice request"""
        
        audio_duration = len(audio_data) // 4 // 16  # Rough estimate: bytes -> samples -> ms
        
        try:
            async with websockets.connect(self.ws_url) as websocket:
                start_time = time.time()
                
                # Send audio data
                await websocket.send(audio_data)
                
                # Wait for audio response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                    end_time = time.time()
                    
                    total_latency = (end_time - start_time) * 1000  # Convert to ms
                    
                    # Check if we received audio data back
                    success = isinstance(response, bytes) and len(response) > 0
                    
                    return LatencyMeasurement(
                        test_name=test_name,
                        audio_duration_ms=audio_duration,
                        total_latency_ms=total_latency,
                        success=success,
                        error=None if success else "No audio response received"
                    )
                    
                except asyncio.TimeoutError:
                    return LatencyMeasurement(
                        test_name=test_name,
                        audio_duration_ms=audio_duration,
                        total_latency_ms=timeout * 1000,
                        success=False,
                        error=f"Timeout after {timeout}s"
                    )
                    
        except Exception as e:
            return LatencyMeasurement(
                test_name=test_name,
                audio_duration_ms=audio_duration,
                total_latency_ms=0,
                success=False,
                error=str(e)
            )
    
    async def run_latency_test_suite(self, iterations: int = 5):
        """Run comprehensive latency tests"""
        test_cases = [
            ("Short phrase (1s)", 1000),
            ("Medium phrase (2s)", 2000),
            ("Long phrase (3s)", 3000),
            ("Very short (500ms)", 500),
        ]
        
        print("🧪 Starting Voice Orchestrator Latency Tests")
        print("=" * 60)
        
        for test_name, duration_ms in test_cases:
            print(f"\n📝 {test_name}")
            print("-" * 40)
            
            audio_data = self.generate_test_audio(duration_ms)
            
            for i in range(iterations):
                print(f"  Iteration {i+1}/{iterations}... ", end="", flush=True)
                
                measurement = await self.measure_single_request(
                    audio_data, 
                    f"{test_name} (iter {i+1})"
                )
                
                self.results.append(measurement)
                
                if measurement.success:
                    print(f"✅ {measurement.total_latency_ms:.1f}ms")
                else:
                    print(f"❌ {measurement.error}")
                
                # Brief pause between tests
                await asyncio.sleep(0.5)
    
    async def run_concurrent_test(self, num_concurrent: int = 3, iterations: int = 2):
        """Test concurrent users"""
        print(f"\n🔄 Concurrent User Test ({num_concurrent} users)")
        print("-" * 40)
        
        audio_data = self.generate_test_audio(1500)  # 1.5s audio
        
        for iteration in range(iterations):
            print(f"  Concurrent iteration {iteration+1}/{iterations}...")
            
            # Launch concurrent requests
            tasks = []
            for user_id in range(num_concurrent):
                task = self.measure_single_request(
                    audio_data,
                    f"Concurrent user {user_id+1} (iter {iteration+1})"
                )
                tasks.append(task)
            
            # Wait for all to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, LatencyMeasurement):
                    self.results.append(result)
                    status = "✅" if result.success else "❌"
                    latency = f"{result.total_latency_ms:.1f}ms" if result.success else result.error
                    print(f"    User {i+1}: {status} {latency}")
                else:
                    print(f"    User {i+1}: ❌ Exception: {result}")
            
            await asyncio.sleep(1)  # Pause between concurrent iterations
    
    def generate_report(self) -> Dict:
        """Generate comprehensive test report"""
        if not self.results:
            return {"error": "No test results available"}
        
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        if not successful_results:
            return {
                "summary": {
                    "total_tests": len(self.results),
                    "successful_tests": 0,
                    "failed_tests": len(failed_results),
                    "success_rate": 0.0,
                    "target_met": False
                },
                "failures": [{"test": r.test_name, "error": r.error} for r in failed_results]
            }
        
        latencies = [r.total_latency_ms for r in successful_results]
        
        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        # Check if target is met (95th percentile under 500ms)
        target_met = p95_latency <= 500
        
        return {
            "summary": {
                "total_tests": len(self.results),
                "successful_tests": len(successful_results),
                "failed_tests": len(failed_results),
                "success_rate": len(successful_results) / len(self.results) * 100,
                "avg_latency_ms": round(avg_latency, 2),
                "median_latency_ms": round(median_latency, 2),
                "p95_latency_ms": round(p95_latency, 2),
                "p99_latency_ms": round(p99_latency, 2),
                "min_latency_ms": round(min_latency, 2),
                "max_latency_ms": round(max_latency, 2),
                "target_met": target_met,
                "target_threshold_ms": 500
            },
            "details": [
                {
                    "test": r.test_name,
                    "latency_ms": round(r.total_latency_ms, 2),
                    "audio_duration_ms": r.audio_duration_ms,
                    "success": r.success,
                    "error": r.error
                }
                for r in self.results
            ]
        }
    
    def print_report(self):
        """Print formatted test report"""
        report = self.generate_report()
        
        if "error" in report:
            print(f"❌ {report['error']}")
            return
        
        summary = report["summary"]
        
        print("\n" + "="*60)
        print("📊 LATENCY TEST RESULTS")
        print("="*60)
        
        # Summary statistics
        print(f"📈 Tests: {summary['successful_tests']}/{summary['total_tests']} successful ({summary['success_rate']:.1f}%)")
        print(f"⏱️  Average latency: {summary['avg_latency_ms']}ms")
        print(f"📊 Median latency: {summary['median_latency_ms']}ms")
        print(f"🎯 95th percentile: {summary['p95_latency_ms']}ms")
        print(f"📈 99th percentile: {summary['p99_latency_ms']}ms")
        print(f"⚡ Fastest: {summary['min_latency_ms']}ms")
        print(f"🐌 Slowest: {summary['max_latency_ms']}ms")
        
        # Target assessment
        target_met = summary['target_met']
        print(f"\n🎯 TARGET ASSESSMENT")
        print(f"Target: <{summary['target_threshold_ms']}ms (95th percentile)")
        
        if target_met:
            print(f"✅ TARGET MET! (P95: {summary['p95_latency_ms']}ms)")
        else:
            print(f"❌ TARGET MISSED (P95: {summary['p95_latency_ms']}ms)")
            print(f"   Exceeds target by {summary['p95_latency_ms'] - summary['target_threshold_ms']}ms")
        
        # Failure analysis
        if summary['failed_tests'] > 0:
            print(f"\n⚠️  FAILURES ({summary['failed_tests']} tests)")
            for detail in report['details']:
                if not detail['success']:
                    print(f"   ❌ {detail['test']}: {detail['error']}")

async def main():
    parser = argparse.ArgumentParser(description='Voice Orchestrator Latency Test')
    parser.add_argument('--url', default='ws://localhost:8000/ws', help='WebSocket URL')
    parser.add_argument('--iterations', type=int, default=5, help='Iterations per test')
    parser.add_argument('--concurrent', type=int, default=3, help='Concurrent users for stress test')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    
    args = parser.parse_args()
    
    tester = VoiceLatencyTester(args.url)
    
    # Check if service is available
    try:
        async with websockets.connect(args.url, timeout=5):
            print("✅ Connected to Voice Orchestrator")
    except Exception as e:
        print(f"❌ Cannot connect to Voice Orchestrator at {args.url}")
        print(f"Error: {e}")
        print("\n💡 Make sure the service is running:")
        print("   docker-compose up -d")
        return 1
    
    try:
        # Run test suites
        await tester.run_latency_test_suite(args.iterations)
        await tester.run_concurrent_test(args.concurrent, 2)
        
        # Output results
        if args.json:
            report = tester.generate_report()
            print(json.dumps(report, indent=2))
        else:
            tester.print_report()
        
        # Return appropriate exit code
        report = tester.generate_report()
        if report.get("summary", {}).get("target_met", False):
            return 0
        else:
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)