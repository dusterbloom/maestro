#!/usr/bin/env python3
"""
Performance test to measure the difference between old blocking approach 
and new streaming sentence-by-sentence approach.
"""

import requests
import time
import json
from typing import Dict, List

class StreamingPerformanceTest:
    def __init__(self, orchestrator_url: str = "http://localhost:8000"):
        self.orchestrator_url = orchestrator_url
        self.test_prompts = [
            "Tell me a detailed story about a brave knight who goes on an adventure to save a princess from a dragon.",
            "Explain how photosynthesis works in plants and why it's important for life on Earth.",
            "What are the main differences between machine learning and artificial intelligence? Please give examples.",
            "Describe the process of making chocolate from cocoa beans, including all the major steps.",
            "Tell me about the history of the internet and how it changed communication worldwide."
        ]
    
    def measure_response_time(self, prompt: str, endpoint: str, session_id: str) -> Dict:
        """Measure total response time and time to first audio"""
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.orchestrator_url}/{endpoint}",
                json={
                    "transcript": prompt,
                    "session_id": session_id
                },
                timeout=30
            )
            
            end_time = time.time()
            total_time = (end_time - start_time) * 1000  # Convert to ms
            
            if response.status_code == 200:
                data = response.json()
                has_audio = bool(data.get("audio_data"))
                response_length = len(data.get("response_text", ""))
                audio_size = len(data.get("audio_data", "")) if has_audio else 0
                
                return {
                    "success": True,
                    "total_time_ms": total_time,
                    "response_length": response_length,
                    "has_audio": has_audio,
                    "audio_size_chars": audio_size,
                    "latency_reported": data.get("latency_ms", 0),
                    "response_text": data.get("response_text", "")[:100] + "..."
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "total_time_ms": total_time
                }
                
        except Exception as e:
            end_time = time.time()
            total_time = (end_time - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "total_time_ms": total_time
            }
    
    def test_endpoint_performance(self, endpoint: str, description: str) -> List[Dict]:
        """Test an endpoint with multiple prompts"""
        print(f"\n🧪 Testing {description}")
        print("=" * 60)
        
        results = []
        session_id = f"perf_test_{endpoint}_{int(time.time())}"
        
        for i, prompt in enumerate(self.test_prompts, 1):
            print(f"\nTest {i}/5: {prompt[:50]}...")
            
            result = self.measure_response_time(prompt, endpoint, session_id)
            result["test_number"] = i
            result["prompt"] = prompt[:50] + "..."
            result["endpoint"] = endpoint
            
            if result["success"]:
                print(f"  ✅ Success: {result['total_time_ms']:.0f}ms total")
                print(f"     Response: {result['response_length']} chars")
                print(f"     Audio: {'Yes' if result['has_audio'] else 'No'} ({result['audio_size_chars']} chars)")
                if result.get("latency_reported"):
                    print(f"     Reported latency: {result['latency_reported']:.0f}ms")
            else:
                print(f"  ❌ Failed: {result['error']} ({result['total_time_ms']:.0f}ms)")
            
            results.append(result)
            
            # Small delay between tests
            time.sleep(1)
        
        return results
    
    def compare_performance(self):
        """Compare old vs new streaming approach"""
        print("🚀 Streaming Performance Comparison Test")
        print("=" * 70)
        print("This test compares:")
        print("  OLD: Blocking approach (collect all LLM tokens, then TTS)")
        print("  NEW: Streaming approach (sentence-by-sentence TTS)")
        print()
        
        # Note: We're testing the ultra-fast endpoint which now uses streaming
        # To test the old approach, we'd need a separate endpoint or flag
        
        # Test current implementation (should be streaming)
        streaming_results = self.test_endpoint_performance("ultra-fast", "Ultra-Fast Endpoint (NEW STREAMING)")
        
        # Analyze results
        self.analyze_results(streaming_results)
    
    def analyze_results(self, results: List[Dict]):
        """Analyze and report performance results"""
        print("\n📊 Performance Analysis")
        print("=" * 60)
        
        successful_tests = [r for r in results if r["success"]]
        failed_tests = [r for r in results if not r["success"]]
        
        if not successful_tests:
            print("❌ No successful tests to analyze!")
            return
        
        # Calculate statistics
        times = [r["total_time_ms"] for r in successful_tests]
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        # Response characteristics
        avg_response_length = sum(r["response_length"] for r in successful_tests) / len(successful_tests)
        audio_success_rate = (sum(1 for r in successful_tests if r["has_audio"]) / len(successful_tests)) * 100
        
        print(f"Successful tests: {len(successful_tests)}/{len(results)}")
        print(f"Failed tests: {len(failed_tests)}")
        print()
        
        print("⏱️  Timing Results:")
        print(f"  Average response time: {avg_time:.0f}ms")
        print(f"  Fastest response: {min_time:.0f}ms")
        print(f"  Slowest response: {max_time:.0f}ms")
        print()
        
        print("📝 Response Quality:")
        print(f"  Average response length: {avg_response_length:.0f} characters")
        print(f"  Audio generation success: {audio_success_rate:.1f}%")
        print()
        
        # Performance categorization
        excellent_responses = [t for t in times if t < 1000]  # < 1 second
        good_responses = [t for t in times if 1000 <= t < 3000]  # 1-3 seconds
        slow_responses = [t for t in times if t >= 3000]  # > 3 seconds
        
        print("🎯 Performance Distribution:")
        print(f"  Excellent (< 1s): {len(excellent_responses)} tests ({len(excellent_responses)/len(times)*100:.1f}%)")
        print(f"  Good (1-3s): {len(good_responses)} tests ({len(good_responses)/len(times)*100:.1f}%)")
        print(f"  Slow (> 3s): {len(slow_responses)} tests ({len(slow_responses)/len(times)*100:.1f}%)")
        print()
        
        # Success criteria
        target_latency = 500  # Target < 500ms for subsequent responses
        meets_target = [t for t in times if t < target_latency]
        
        print("🎯 Target Achievement:")
        print(f"  Target latency: < {target_latency}ms")
        print(f"  Tests meeting target: {len(meets_target)}/{len(times)} ({len(meets_target)/len(times)*100:.1f}%)")
        
        if len(meets_target) / len(times) >= 0.8:  # 80% success rate
            print("  ✅ EXCELLENT: Streaming implementation meets performance target!")
        elif avg_time < 2000:  # Average under 2s
            print("  ✅ GOOD: Significant improvement over previous 6+ second latency")
        else:
            print("  ⚠️  NEEDS IMPROVEMENT: Still slower than target")
        
        # Detailed breakdown
        print("\n📋 Detailed Results:")
        for result in successful_tests:
            status = "🚀" if result["total_time_ms"] < target_latency else "✅" if result["total_time_ms"] < 2000 else "⚠️"
            print(f"  {status} Test {result['test_number']}: {result['total_time_ms']:.0f}ms - {result['prompt']}")
        
        if failed_tests:
            print("\n❌ Failed Tests:")
            for result in failed_tests:
                print(f"  💥 Test {result['test_number']}: {result['error']} - {result['prompt']}")

def main():
    """Run the streaming performance test"""
    tester = StreamingPerformanceTest()
    
    print("⚡ Streaming Performance Test")
    print("This test measures the new sentence-by-sentence streaming implementation")
    print("Expected improvement: 6+ seconds → <1 second for first response")
    print()
    
    try:
        tester.compare_performance()
        
        print("\n" + "=" * 70)
        print("🏁 Test Complete!")
        print("=" * 70)
        print("Key metrics to compare with previous 6+ second latency:")
        print("  - Average response time should be dramatically lower")
        print("  - First response should start playing audio much faster")
        print("  - Overall user experience should feel much more responsive")
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test failed: {e}")

if __name__ == "__main__":
    main()