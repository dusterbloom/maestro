#!/usr/bin/env python3
"""
Comprehensive Voice Interruption Debug Test
This script tests every component of the interruption system to find failures.
"""

import requests
import time
import json
import threading
import websocket
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InterruptionDebugTest:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session_id = f"debug_test_{int(time.time())}"
        self.test_results = {}
        
    def test_health_check(self) -> bool:
        """Test 1: Basic health check"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            success = response.status_code == 200
            self.test_results["health_check"] = {
                "success": success,
                "status_code": response.status_code,
                "response": response.json() if success else None
            }
            logger.info(f"✅ Health check: {success}")
            return success
        except Exception as e:
            self.test_results["health_check"] = {"success": False, "error": str(e)}
            logger.error(f"❌ Health check failed: {e}")
            return False
    
    def test_interrupt_endpoint_direct(self) -> bool:
        """Test 2: Direct interrupt endpoint test"""
        try:
            # Test with non-existent session (should return no_active_session)
            response = requests.post(
                f"{self.base_url}/interrupt-tts",
                json={"session_id": "non_existent_session"},
                timeout=5
            )
            
            success = response.status_code == 200
            result = response.json() if success else None
            
            self.test_results["interrupt_endpoint_direct"] = {
                "success": success,
                "status_code": response.status_code,
                "response": result,
                "expected_status": "no_active_session"
            }
            
            if success and result:
                expected = result.get("status") == "no_active_session"
                logger.info(f"✅ Interrupt endpoint direct test: {expected}")
                return expected
            else:
                logger.error(f"❌ Interrupt endpoint failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.test_results["interrupt_endpoint_direct"] = {"success": False, "error": str(e)}
            logger.error(f"❌ Interrupt endpoint direct test failed: {e}")
            return False
    
    def test_create_tts_session(self) -> bool:
        """Test 3: Create a TTS session and verify it's tracked"""
        try:
            # Send a request to create a TTS session
            response = requests.post(
                f"{self.base_url}/ultra-fast-stream",
                json={
                    "transcript": "This is a test sentence for debugging interruption.",
                    "session_id": self.session_id
                },
                stream=True,
                timeout=10
            )
            
            success = response.status_code == 200
            
            # Give it a moment to start processing
            time.sleep(0.5)
            
            # Check if session was created by testing debug endpoint
            session_check = self.test_session_tracking()
            
            self.test_results["create_tts_session"] = {
                "success": success,
                "status_code": response.status_code,
                "session_created": session_check,
                "session_id": self.session_id
            }
            
            logger.info(f"✅ TTS session creation: {success}, session tracked: {session_check}")
            return success and session_check
            
        except Exception as e:
            self.test_results["create_tts_session"] = {"success": False, "error": str(e)}
            logger.error(f"❌ TTS session creation failed: {e}")
            return False
    
    def test_session_tracking(self) -> bool:
        """Test 4: Check if sessions are being tracked"""
        try:
            # First, add the debug endpoint if it doesn't exist
            self.add_debug_endpoint()
            
            response = requests.get(f"{self.base_url}/debug/sessions", timeout=5)
            
            if response.status_code == 404:
                # Debug endpoint not found, try to infer from interrupt response
                interrupt_response = requests.post(
                    f"{self.base_url}/interrupt-tts",
                    json={"session_id": self.session_id},
                    timeout=5
                )
                
                success = interrupt_response.status_code == 200
                result = interrupt_response.json() if success else None
                
                # If session exists, interrupt should succeed; if not, should get no_active_session
                session_exists = result and result.get("status") == "interrupted"
                
                self.test_results["session_tracking"] = {
                    "success": success,
                    "method": "interrupt_inference",
                    "session_exists": session_exists,
                    "interrupt_response": result
                }
                
                logger.info(f"✅ Session tracking (via interrupt): {session_exists}")
                return session_exists
            else:
                success = response.status_code == 200
                result = response.json() if success else None
                
                session_exists = False
                if result and "active_sessions" in result:
                    session_exists = self.session_id in result["active_sessions"]
                
                self.test_results["session_tracking"] = {
                    "success": success,
                    "method": "debug_endpoint",
                    "session_exists": session_exists,
                    "active_sessions": result.get("active_sessions", []) if result else []
                }
                
                logger.info(f"✅ Session tracking (via debug): {session_exists}")
                return session_exists
                
        except Exception as e:
            self.test_results["session_tracking"] = {"success": False, "error": str(e)}
            logger.error(f"❌ Session tracking test failed: {e}")
            return False
    
    def test_interrupt_active_session(self) -> bool:
        """Test 5: Interrupt an active session"""
        try:
            # First create a session by starting TTS
            logger.info("Creating active TTS session...")
            
            def start_tts():
                try:
                    response = requests.post(
                        f"{self.base_url}/ultra-fast-stream",
                        json={
                            "transcript": "This is a very long test sentence that should take several seconds to generate and stream, giving us time to interrupt it while it's actively processing and generating audio content.",
                            "session_id": self.session_id
                        },
                        stream=True,
                        timeout=30
                    )
                    
                    # Consume the stream slowly to keep session active
                    for chunk in response.iter_content(chunk_size=1024):
                        time.sleep(0.1)  # Slow consumption
                        if chunk:
                            pass  # Process chunk
                            
                except Exception as e:
                    logger.error(f"TTS streaming error: {e}")
            
            # Start TTS in background
            tts_thread = threading.Thread(target=start_tts, daemon=True)
            tts_thread.start()
            
            # Give it time to start
            time.sleep(1.0)
            
            # Now try to interrupt
            logger.info("Attempting to interrupt active session...")
            interrupt_response = requests.post(
                f"{self.base_url}/interrupt-tts",
                json={"session_id": self.session_id},
                timeout=5
            )
            
            success = interrupt_response.status_code == 200
            result = interrupt_response.json() if success else None
            
            interrupted = result and result.get("status") == "interrupted"
            
            self.test_results["interrupt_active_session"] = {
                "success": success,
                "interrupted": interrupted,
                "interrupt_response": result,
                "session_id": self.session_id
            }
            
            logger.info(f"✅ Interrupt active session: {interrupted}")
            return interrupted
            
        except Exception as e:
            self.test_results["interrupt_active_session"] = {"success": False, "error": str(e)}
            logger.error(f"❌ Interrupt active session test failed: {e}")
            return False
    
    def test_frontend_api_integration(self) -> bool:
        """Test 6: Test the frontend API proxy"""
        try:
            # Test the Next.js API route
            response = requests.post(
                "http://localhost:3000/api/interrupt-tts",
                json={"session_id": self.session_id},
                timeout=10
            )
            
            success = response.status_code == 200
            result = response.json() if success else None
            
            self.test_results["frontend_api_integration"] = {
                "success": success,
                "status_code": response.status_code,
                "response": result
            }
            
            logger.info(f"✅ Frontend API integration: {success}")
            return success
            
        except Exception as e:
            self.test_results["frontend_api_integration"] = {"success": False, "error": str(e)}
            logger.error(f"❌ Frontend API integration test failed: {e}")
            return False
    
    def add_debug_endpoint(self):
        """Add debug endpoint to the orchestrator if possible"""
        # This would require modifying the running service
        # For now, we'll rely on the interrupt endpoint for session checking
        pass
    
    def test_timing_performance(self) -> bool:
        """Test 7: Measure interrupt timing performance"""
        try:
            times = []
            
            for i in range(5):
                start_time = time.time()
                
                response = requests.post(
                    f"{self.base_url}/interrupt-tts",
                    json={"session_id": f"timing_test_{i}"},
                    timeout=5
                )
                
                end_time = time.time()
                elapsed_ms = (end_time - start_time) * 1000
                times.append(elapsed_ms)
                
                time.sleep(0.1)  # Brief pause between tests
            
            avg_time = sum(times) / len(times)
            max_time = max(times)
            
            performance_good = avg_time < 50 and max_time < 100
            
            self.test_results["timing_performance"] = {
                "success": True,
                "avg_time_ms": avg_time,
                "max_time_ms": max_time,
                "all_times_ms": times,
                "performance_good": performance_good
            }
            
            logger.info(f"✅ Timing performance: avg={avg_time:.2f}ms, max={max_time:.2f}ms, good={performance_good}")
            return performance_good
            
        except Exception as e:
            self.test_results["timing_performance"] = {"success": False, "error": str(e)}
            logger.error(f"❌ Timing performance test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results"""
        logger.info("🚀 Starting comprehensive interruption debug tests...")
        
        tests = [
            ("Health Check", self.test_health_check),
            ("Interrupt Endpoint Direct", self.test_interrupt_endpoint_direct),
            ("Create TTS Session", self.test_create_tts_session),
            ("Session Tracking", self.test_session_tracking),
            ("Interrupt Active Session", self.test_interrupt_active_session),
            ("Frontend API Integration", self.test_frontend_api_integration),
            ("Timing Performance", self.test_timing_performance),
        ]
        
        results_summary = {}
        
        for test_name, test_func in tests:
            logger.info(f"\n--- Running {test_name} ---")
            try:
                result = test_func()
                results_summary[test_name] = result
                status = "✅ PASS" if result else "❌ FAIL"
                logger.info(f"{status}: {test_name}")
            except Exception as e:
                results_summary[test_name] = False
                logger.error(f"❌ ERROR: {test_name} - {e}")
        
        # Generate summary report
        total_tests = len(tests)
        passed_tests = sum(1 for result in results_summary.values() if result)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🏁 TEST SUMMARY: {passed_tests}/{total_tests} tests passed")
        logger.info(f"{'='*60}")
        
        for test_name, passed in results_summary.items():
            status = "✅" if passed else "❌"
            logger.info(f"{status} {test_name}")
        
        # Detailed results
        self.test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "pass_rate": passed_tests / total_tests,
            "individual_results": results_summary
        }
        
        return self.test_results
    
    def generate_report(self) -> str:
        """Generate a detailed test report"""
        report = []
        report.append("# Voice Interruption Debug Test Report")
        report.append(f"Test Session ID: {self.session_id}")
        report.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        if "summary" in self.test_results:
            summary = self.test_results["summary"]
            report.append(f"## Summary")
            report.append(f"- Total Tests: {summary['total_tests']}")
            report.append(f"- Passed: {summary['passed_tests']}")
            report.append(f"- Pass Rate: {summary['pass_rate']:.1%}")
            report.append("")
        
        report.append("## Detailed Results")
        for test_name, test_data in self.test_results.items():
            if test_name == "summary":
                continue
                
            report.append(f"### {test_name}")
            if isinstance(test_data, dict):
                for key, value in test_data.items():
                    report.append(f"- {key}: {value}")
            else:
                report.append(f"- Result: {test_data}")
            report.append("")
        
        return "\n".join(report)


if __name__ == "__main__":
    # Run the comprehensive test
    tester = InterruptionDebugTest()
    results = tester.run_all_tests()
    
    # Save detailed report
    report = tester.generate_report()
    with open("interruption_debug_report.txt", "w") as f:
        f.write(report)
    
    # Save raw results as JSON
    with open("interruption_debug_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("📄 Reports saved:")
    print("- interruption_debug_report.txt")
    print("- interruption_debug_results.json")
    print("="*60)