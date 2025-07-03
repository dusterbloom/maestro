#!/usr/bin/env python3
"""
VAD Performance Testing Suite
Tests Voice Activity Detection latency and accuracy with WhisperLive
"""

import asyncio
import json
import time
import numpy as np
import soundfile as sf
import websockets
from typing import List, Dict, Optional
import argparse
import os

class VADPerformanceTester:
    def __init__(self, whisper_url: str = "ws://localhost:9090"):
        self.whisper_url = whisper_url
        self.test_results: List[Dict] = []
        
    async def test_silence_detection(self, duration_seconds: float = 3.0) -> Dict:
        """Test VAD accuracy on silence"""
        print(f"Testing silence detection for {duration_seconds}s...")
        
        # Generate silence audio (16kHz mono)
        sample_rate = 16000
        samples = int(sample_rate * duration_seconds)
        silence_audio = np.zeros(samples, dtype=np.float32)
        
        start_time = time.time()
        result = await self._test_audio_with_vad(silence_audio, sample_rate)
        end_time = time.time()
        
        test_result = {
            "test_type": "silence_detection",
            "duration_seconds": duration_seconds,
            "processing_time_ms": (end_time - start_time) * 1000,
            "vad_triggered": result["vad_triggered"],
            "transcription_count": len(result["transcriptions"]),
            "expected_vad": False,
            "accuracy": "pass" if not result["vad_triggered"] else "fail"
        }
        
        self.test_results.append(test_result)
        return test_result
    
    async def test_voice_detection(self, audio_file: str) -> Dict:
        """Test VAD accuracy on speech audio"""
        print(f"Testing voice detection with {audio_file}...")
        
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        # Load audio file
        audio_data, sample_rate = sf.read(audio_file)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            # Simple resampling
            target_length = int(len(audio_data) * 16000 / sample_rate)
            audio_data = np.interp(
                np.linspace(0, len(audio_data), target_length),
                np.arange(len(audio_data)),
                audio_data
            )
            sample_rate = 16000
        
        # Ensure mono and float32
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]
        audio_data = audio_data.astype(np.float32)
        
        start_time = time.time()
        result = await self._test_audio_with_vad(audio_data, sample_rate)
        end_time = time.time()
        
        test_result = {
            "test_type": "voice_detection",
            "audio_file": audio_file,
            "duration_seconds": len(audio_data) / sample_rate,
            "processing_time_ms": (end_time - start_time) * 1000,
            "vad_triggered": result["vad_triggered"],
            "transcription_count": len(result["transcriptions"]),
            "expected_vad": True,
            "accuracy": "pass" if result["vad_triggered"] else "fail",
            "transcriptions": result["transcriptions"]
        }
        
        self.test_results.append(test_result)
        return test_result
    
    async def test_vad_sensitivity(self, thresholds: List[float] = [0.3, 0.5, 0.7]) -> Dict:
        """Test VAD sensitivity at different thresholds"""
        print("Testing VAD sensitivity at different thresholds...")
        
        # Generate low-volume speech simulation
        sample_rate = 16000
        duration = 2.0
        samples = int(sample_rate * duration)
        
        # Create varying amplitude sine wave (simulates low-level speech)
        t = np.linspace(0, duration, samples)
        frequencies = [200, 400, 800]  # Simulate speech harmonics
        audio_data = np.zeros(samples)
        
        for freq in frequencies:
            audio_data += 0.1 * np.sin(2 * np.pi * freq * t)
        
        # Add some noise
        audio_data += 0.02 * np.random.randn(samples)
        audio_data = audio_data.astype(np.float32)
        
        sensitivity_results = []
        for threshold in thresholds:
            print(f"  Testing threshold: {threshold}")
            
            start_time = time.time()
            result = await self._test_audio_with_vad(
                audio_data, sample_rate, vad_threshold=threshold
            )
            end_time = time.time()
            
            sensitivity_results.append({
                "threshold": threshold,
                "vad_triggered": result["vad_triggered"],
                "transcription_count": len(result["transcriptions"]),
                "processing_time_ms": (end_time - start_time) * 1000
            })
        
        test_result = {
            "test_type": "vad_sensitivity",
            "thresholds_tested": thresholds,
            "results": sensitivity_results
        }
        
        self.test_results.append(test_result)
        return test_result
    
    async def _test_audio_with_vad(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int,
        vad_threshold: float = 0.5
    ) -> Dict:
        """Test audio with WhisperLive VAD"""
        
        transcriptions = []
        vad_triggered = False
        
        try:
            async with websockets.connect(self.whisper_url) as websocket:
                # Send WhisperLive configuration with VAD parameters
                config = {
                    "uid": f"vad_test_{int(time.time())}",
                    "language": "en",
                    "task": "transcribe",
                    "model": "tiny",
                    "use_vad": True,
                    "vad_parameters": {
                        "threshold": vad_threshold,
                        "min_silence_duration_ms": 300,
                        "speech_pad_ms": 400,
                    },
                    "max_clients": 4,
                    "max_connection_time": 600,
                    "send_last_n_segments": 10,
                    "no_speech_thresh": 0.3,
                    "clip_audio": False,
                    "same_output_threshold": 8
                }
                
                await websocket.send(json.dumps(config))
                
                # Wait for server ready
                ready_message = await websocket.recv()
                ready_data = json.loads(ready_message)
                if ready_data.get("message") != "SERVER_READY":
                    raise Exception(f"Unexpected response: {ready_data}")
                
                # Send audio in chunks (similar to real-time streaming)
                chunk_size = 4096  # 16kHz @ 256ms chunks
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    
                    # Convert to bytes (16-bit PCM)
                    chunk_int16 = (chunk * 32767).astype(np.int16)
                    await websocket.send(chunk_int16.tobytes())
                    
                    # Small delay to simulate real-time
                    await asyncio.sleep(0.1)
                
                # Send end of audio
                await websocket.send("END_OF_AUDIO")
                
                # Collect responses for up to 5 seconds
                timeout_time = time.time() + 5.0
                while time.time() < timeout_time:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        
                        if isinstance(response, str):
                            try:
                                data = json.loads(response)
                                if "segments" in data and data["segments"]:
                                    vad_triggered = True
                                    for segment in data["segments"]:
                                        if segment.get("text", "").strip():
                                            transcriptions.append({
                                                "text": segment["text"],
                                                "start": segment.get("start", 0),
                                                "end": segment.get("end", 0)
                                            })
                            except json.JSONDecodeError:
                                pass  # Non-JSON message
                    except asyncio.TimeoutError:
                        break  # No more messages
                        
        except Exception as e:
            print(f"Error during VAD test: {e}")
            return {"vad_triggered": False, "transcriptions": [], "error": str(e)}
        
        return {"vad_triggered": vad_triggered, "transcriptions": transcriptions}
    
    async def run_comprehensive_test(self, audio_files: Optional[List[str]] = None):
        """Run comprehensive VAD performance test suite"""
        print("=== Starting VAD Performance Test Suite ===")
        
        # Test 1: Silence detection
        await self.test_silence_detection(duration_seconds=2.0)
        await self.test_silence_detection(duration_seconds=5.0)
        
        # Test 2: VAD sensitivity
        await self.test_vad_sensitivity()
        
        # Test 3: Voice detection (if audio files provided)
        if audio_files:
            for audio_file in audio_files:
                try:
                    await self.test_voice_detection(audio_file)
                except Exception as e:
                    print(f"Failed to test {audio_file}: {e}")
        
        # Generate summary report
        self.generate_report()
    
    def generate_report(self):
        """Generate performance test report"""
        print("\n=== VAD Performance Test Report ===")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.get("accuracy") == "pass")
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests / total_tests * 100):.1f}%")
        
        print("\nDetailed Results:")
        for i, result in enumerate(self.test_results, 1):
            print(f"\n{i}. {result['test_type'].replace('_', ' ').title()}")
            
            if result["test_type"] == "vad_sensitivity":
                for threshold_result in result["results"]:
                    print(f"   Threshold {threshold_result['threshold']}: "
                          f"VAD={'✓' if threshold_result['vad_triggered'] else '✗'} "
                          f"({threshold_result['processing_time_ms']:.1f}ms)")
            else:
                accuracy_symbol = "✓" if result.get("accuracy") == "pass" else "✗"
                print(f"   {accuracy_symbol} VAD Triggered: {result['vad_triggered']}")
                print(f"   Processing Time: {result['processing_time_ms']:.1f}ms")
                
                if "transcriptions" in result and result["transcriptions"]:
                    print(f"   Transcriptions: {len(result['transcriptions'])}")
                    for trans in result["transcriptions"]:
                        print(f"     - '{trans['text']}'")
        
        # Save detailed results to JSON
        with open("vad_test_results.json", "w") as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"\nDetailed results saved to: vad_test_results.json")

async def main():
    parser = argparse.ArgumentParser(description="VAD Performance Testing Suite")
    parser.add_argument("--whisper-url", default="ws://localhost:9090", 
                       help="WhisperLive WebSocket URL")
    parser.add_argument("--audio-files", nargs="*", 
                       help="Audio files to test voice detection")
    
    args = parser.parse_args()
    
    tester = VADPerformanceTester(args.whisper_url)
    await tester.run_comprehensive_test(args.audio_files)

if __name__ == "__main__":
    asyncio.run(main())