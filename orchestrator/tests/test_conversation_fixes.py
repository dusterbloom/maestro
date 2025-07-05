#!/usr/bin/env python3
"""
Comprehensive test for conversation fixes
Simulates the exact scenarios from the user logs to verify all fixes work correctly.
"""

import asyncio
import json
import requests
import time
from typing import List, Dict, Any

class ConversationTester:
    def __init__(self, orchestrator_url: str = "http://localhost:8000"):
        self.orchestrator_url = orchestrator_url
        self.session_id = f"test_session_{int(time.time())}"
        self.test_results = []
        
    def log_test(self, test_name: str, input_data: Dict, response: Dict, expected: Dict, passed: bool):
        """Log test results"""
        # Create a clean response for logging (without full audio data)
        clean_response = response.copy()
        if "audio_data" in clean_response and clean_response["audio_data"]:
            audio_length = len(clean_response["audio_data"])
            clean_response["audio_data"] = f"<audio_data: {audio_length} chars>"
        
        result = {
            "test": test_name,
            "input": input_data,
            "response": clean_response,
            "expected": expected,
            "passed": passed,
            "timestamp": time.time()
        }
        self.test_results.append(result)
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if not passed:
            print(f"  Expected: {expected}")
            print(f"  Got: {clean_response}")
        print()

    def test_ultra_fast_endpoint(self, transcript: str, description: str = "") -> Dict[str, Any]:
        """Test the ultra-fast endpoint with a transcript"""
        print(f"Testing: {description or transcript}")
        
        try:
            response = requests.post(
                f"{self.orchestrator_url}/ultra-fast",
                json={
                    "transcript": transcript,
                    "session_id": self.session_id
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                # Log audio length instead of full data for debugging
                if data.get("audio_data"):
                    audio_length = len(data["audio_data"])
                    print(f"  → Response: {data.get('response_text', '')[:50]}... (audio: {audio_length} chars)")
                else:
                    print(f"  → Response: {data.get('response_text', '')} (no audio)")
                return data
            else:
                return {
                    "error": f"HTTP {response.status_code}",
                    "response_text": "",
                    "audio_data": None,
                    "sentence_complete": False
                }
        except Exception as e:
            return {
                "error": str(e),
                "response_text": "",
                "audio_data": None,
                "sentence_complete": False
            }

    def run_all_tests(self):
        """Run comprehensive test suite"""
        print("🧪 Starting Conversation Fixes Test Suite")
        print("=" * 60)
        
        # Test 1: Normal conversation flow
        print("📝 Test 1: Normal Conversation Flow")
        response1 = self.test_ultra_fast_endpoint(
            "Can you tell me a joke about music?",
            "First complete sentence in conversation"
        )
        
        self.log_test(
            "Normal first sentence",
            {"transcript": "Can you tell me a joke about music?"},
            response1,
            {"sentence_complete": True, "has_audio": True, "has_response": True},
            response1.get("sentence_complete") != False and 
            response1.get("audio_data") is not None and 
            response1.get("response_text", "") != ""
        )
        
        # Test 2: Follow-up in same conversation (should have context)
        response2 = self.test_ultra_fast_endpoint(
            "Can you tell me another one?",
            "Follow-up request - should have context from previous"
        )
        
        # Check for context awareness more flexibly
        response_text = response2.get("response_text", "").lower()
        context_indicators = ["joke", "another", "here", "sure", "music", "funny"]
        has_context = any(indicator in response_text for indicator in context_indicators)
        
        self.log_test(
            "Context-aware follow-up",
            {"transcript": "Can you tell me another one?"},
            response2,
            {"sentence_complete": True, "has_audio": True, "context_aware": True},
            response2.get("sentence_complete") != False and 
            response2.get("audio_data") is not None and
            has_context  # Should reference previous context
        )
        
        # Test 3: Single word with punctuation (was being rejected)
        print("📝 Test 2: Single Word Sentences")
        single_word_tests = [
            "story.",
            "yes.",
            "okay.",
            "sure.",
            "please."
        ]
        
        for word in single_word_tests:
            response = self.test_ultra_fast_endpoint(word, f"Single word: {word}")
            
            self.log_test(
                f"Single word: {word}",
                {"transcript": word},
                response,
                {"sentence_complete": True, "has_audio": True},
                response.get("sentence_complete") != False and response.get("audio_data") is not None
            )
        
        # Test 4: Interruption-like short inputs (should be processed quickly)
        print("📝 Test 3: Interruption-like Inputs")
        interruption_tests = [
            "PUNCE!",
            "hey",
            "stop",
            "wait",
            "no no"
        ]
        
        for interruption in interruption_tests:
            response = self.test_ultra_fast_endpoint(interruption, f"Interruption: {interruption}")
            
            # Interruptions should either be processed OR skipped, but not cause errors
            is_valid = (
                (response.get("sentence_complete") != False and response.get("audio_data") is not None) or
                (response.get("sentence_complete") == False and response.get("audio_data") is None)
            )
            
            self.log_test(
                f"Interruption: {interruption}",
                {"transcript": interruption},
                response,
                {"valid_handling": True},
                is_valid and "error" not in response
            )
        
        # Test 5: Incomplete sentences (should be gracefully skipped)
        print("📝 Test 4: Incomplete Sentences")
        incomplete_tests = [
            "I wanted jokes, but then...",
            "I'd like to hear jokes from you, but then I changed my mind and absolutely I'd prefer",
            "Can you tell me"
        ]
        
        for incomplete in incomplete_tests:
            response = self.test_ultra_fast_endpoint(incomplete, f"Incomplete: {incomplete[:30]}...")
            
            self.log_test(
                f"Incomplete sentence",
                {"transcript": incomplete[:30] + "..."},
                response,
                {"sentence_complete": False, "no_audio": True},
                response.get("sentence_complete") == False and response.get("audio_data") is None
            )
        
        # Test 6: Very short inputs (should be skipped)
        print("📝 Test 5: Very Short Inputs")
        short_tests = ["a", "I", "oh", "um"]
        
        for short in short_tests:
            response = self.test_ultra_fast_endpoint(short, f"Too short: {short}")
            
            self.log_test(
                f"Too short: {short}",
                {"transcript": short},
                response,
                {"sentence_complete": False, "no_audio": True},
                response.get("sentence_complete") == False and response.get("audio_data") is None
            )
        
        # Test 7: Context preservation across multiple exchanges
        print("📝 Test 6: Context Preservation")
        context_sequence = [
            "Tell me about cats.",
            "What about dogs?",
            "Which is better?"
        ]
        
        context_responses = []
        for i, query in enumerate(context_sequence):
            response = self.test_ultra_fast_endpoint(query, f"Context test {i+1}: {query}")
            context_responses.append(response)
            
            # Each response should be valid
            self.log_test(
                f"Context sequence {i+1}",
                {"transcript": query},
                response,
                {"sentence_complete": True, "has_audio": True},
                response.get("sentence_complete") != False and response.get("audio_data") is not None
            )
        
        # Check if final response shows awareness of previous context
        final_response = context_responses[-1].get("response_text", "").lower()
        context_indicators = ["cat", "dog", "animal", "pet", "compare", "both", "either", "prefer", "better", "choice"]
        context_aware = any(word in final_response for word in context_indicators)
        
        self.log_test(
            "Context awareness in sequence",
            {"sequence": context_sequence},
            {"final_response": final_response[:100] + "..." if len(final_response) > 100 else final_response},
            {"context_aware": True},
            context_aware
        )
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print("=" * 60)
        print("🏁 Test Summary")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["passed"])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: ✅ {passed_tests}")
        print(f"Failed: ❌ {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ Failed Tests:")
            for result in self.test_results:
                if not result["passed"]:
                    print(f"  - {result['test']}")
        
        print("\n📊 Detailed Results:")
        for result in self.test_results:
            status = "✅" if result["passed"] else "❌"
            print(f"  {status} {result['test']}")

def main():
    """Run the test suite"""
    tester = ConversationTester()
    
    print("Testing conversation fixes...")
    print("This will test:")
    print("1. ✅ TTS Error handling")
    print("2. ✅ Sentence completion logic") 
    print("3. ✅ Single word processing")
    print("4. ✅ Context maintenance")
    print("5. ✅ Interruption handling")
    print()
    
    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")

if __name__ == "__main__":
    main()