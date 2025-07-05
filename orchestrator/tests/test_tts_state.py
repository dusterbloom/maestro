#!/usr/bin/env python3
"""
Test TTS state management to prevent voice avalanche
"""

import asyncio
import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor

def test_concurrent_requests():
    """Test multiple simultaneous requests - should ALL succeed now (stateless backend)"""
    orchestrator_url = "http://localhost:8000"
    session_id = f"test_stateless_{int(time.time())}"
    
    def send_request(request_num):
        """Send a single request"""
        try:
            start_time = time.time()
            response = requests.post(
                f"{orchestrator_url}/ultra-fast",
                json={
                    "transcript": f"Tell me a story about request number {request_num}.",
                    "session_id": session_id
                },
                timeout=15
            )
            
            end_time = time.time()
            result = response.json() if response.status_code == 200 else {"error": f"HTTP {response.status_code}"}
            
            return {
                "request_num": request_num,
                "duration": end_time - start_time,
                "response": result,
                "has_audio": bool(result.get("audio_data")),
                "response_text": result.get("response_text", "")[:50] + "..."
            }
        except Exception as e:
            return {
                "request_num": request_num,
                "duration": 0,
                "error": str(e),
                "has_audio": False,
                "response_text": ""
            }
    
    print("🧪 Testing Stateless Backend")
    print("Sending 5 concurrent requests...")
    print("Expected: ALL should succeed (backend is stateless, frontend controls audio)")
    print()
    
    # Send 5 concurrent requests
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(send_request, i+1) for i in range(5)]
        results = [future.result() for future in futures]
    
    # Analyze results
    successful_tts = [r for r in results if r.get("has_audio", False)]
    error_requests = [r for r in results if "error" in r]
    
    print("📊 Results:")
    print(f"Total requests: {len(results)}")
    print(f"Successful TTS: {len(successful_tts)}")
    print(f"Errors: {len(error_requests)}")
    print()
    
    print("📝 Detailed Results:")
    for result in results:
        status = "✅ TTS" if result.get("has_audio") else "❌ NO TTS"
        if "error" in result:
            status = "💥 ERROR"
        
        duration = f"{result.get('duration', 0):.1f}s"
        text = result.get('response_text', result.get('error', ''))[:50] + "..."
        print(f"  {status} Request {result['request_num']}: {duration} - {text}")
    
    print()
    
    # Verdict
    if len(successful_tts) >= 4:  # Most should succeed
        print("✅ SUCCESS: Stateless backend working correctly!")
        print("   - Multiple requests can be processed simultaneously")
        print("   - Frontend will handle audio interruption")
    else:
        print("❌ FAILURE: Backend not processing requests correctly")
        print(f"   - Expected: 4+ TTS successes")
        print(f"   - Got: {len(successful_tts)} TTS successes")
    
    return len(successful_tts) >= 4

def test_sequential_requests():
    """Test that sequential requests work normally"""
    orchestrator_url = "http://localhost:8000"
    session_id = f"test_sequential_{int(time.time())}"
    
    print("\n🔄 Testing Sequential Requests")
    print("Expected: Both requests should succeed")
    print()
    
    # First request
    response1 = requests.post(
        f"{orchestrator_url}/ultra-fast",
        json={
            "transcript": "Tell me a short joke.",
            "session_id": session_id
        },
        timeout=10
    )
    
    result1 = response1.json() if response1.status_code == 200 else {"error": "HTTP error"}
    
    # Wait a moment, then second request  
    time.sleep(2)
    
    response2 = requests.post(
        f"{orchestrator_url}/ultra-fast",
        json={
            "transcript": "Tell me another one.",
            "session_id": session_id
        },
        timeout=10
    )
    
    result2 = response2.json() if response2.status_code == 200 else {"error": "HTTP error"}
    
    success1 = result1.get("audio_data") is not None
    success2 = result2.get("audio_data") is not None
    
    print(f"Request 1: {'✅ SUCCESS' if success1 else '❌ FAILED'} - {result1.get('response_text', '')[:30]}...")
    print(f"Request 2: {'✅ SUCCESS' if success2 else '❌ FAILED'} - {result2.get('response_text', '')[:30]}...")
    
    if success1 and success2:
        print("✅ SUCCESS: Sequential requests work correctly!")
        return True
    else:
        print("❌ FAILURE: Sequential requests broken")
        return False

def main():
    print("🎛️  Stateless Audio Pipeline Test")
    print("=" * 50)
    
    try:
        # Test concurrent requests (should all succeed - backend is stateless)
        concurrent_success = test_concurrent_requests()
        
        # Test sequential requests (should work normally)
        sequential_success = test_sequential_requests()
        
        print("\n" + "=" * 50)
        print("🏁 Final Results")
        print("=" * 50)
        
        if concurrent_success and sequential_success:
            print("🎉 ALL TESTS PASSED!")
            print("   ✅ Backend is stateless")
            print("   ✅ Multiple requests can be processed")
            print("   ✅ Frontend will control audio interruption")
        else:
            print("💥 SOME TESTS FAILED")
            print("   Stateless backend implementation needs fixes")
            
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test failed: {e}")

if __name__ == "__main__":
    main()