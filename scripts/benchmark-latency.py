#!/usr/bin/env python3
"""
Comprehensive Latency Benchmarking Suite
Tests end-to-end latency across the voice pipeline
"""

import asyncio
import json
import time
import aiohttp
import websockets
import numpy as np
import soundfile as sf
import argparse
from typing import Dict, List, Optional, Tuple
import statistics
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os

class LatencyBenchmark:
    def __init__(self, 
                 orchestrator_url: str = "http://localhost:8000",
                 whisper_url: str = "ws://localhost:9090",
                 tts_url: str = "http://localhost:8880"):
        self.orchestrator_url = orchestrator_url
        self.whisper_url = whisper_url
        self.tts_url = tts_url
        self.results: List[Dict] = []
        
    async def benchmark_component_latency(self, num_tests: int = 10) -> Dict:
        """Benchmark individual component latencies"""
        print("=== Component Latency Benchmark ===")
        
        # Test data
        test_text = "Hello, this is a test sentence for latency measurement."
        test_audio = self._generate_test_audio()
        
        component_results = {
            "stt_latency": [],
            "llm_latency": [],
            "tts_latency": [],
            "orchestrator_latency": []
        }
        
        for i in range(num_tests):
            print(f"Running test {i+1}/{num_tests}")
            
            # Test STT latency (WhisperLive)
            stt_latency = await self._test_stt_latency(test_audio)
            component_results["stt_latency"].append(stt_latency)
            
            # Test LLM latency (via orchestrator)
            llm_latency = await self._test_llm_latency(test_text)
            component_results["llm_latency"].append(llm_latency)
            
            # Test TTS latency (Kokoro)
            tts_latency = await self._test_tts_latency(test_text)
            component_results["tts_latency"].append(tts_latency)
            
            # Test full orchestrator pipeline
            orch_latency = await self._test_orchestrator_latency(test_text)
            component_results["orchestrator_latency"].append(orch_latency)
            
            # Small delay between tests
            await asyncio.sleep(1)
        
        # Calculate statistics
        stats = {}
        for component, latencies in component_results.items():
            valid_latencies = [l for l in latencies if l is not None]
            if valid_latencies:
                stats[component] = {
                    "mean": statistics.mean(valid_latencies),
                    "median": statistics.median(valid_latencies),
                    "min": min(valid_latencies),
                    "max": max(valid_latencies),
                    "std": statistics.stdev(valid_latencies) if len(valid_latencies) > 1 else 0,
                    "p95": np.percentile(valid_latencies, 95),
                    "p99": np.percentile(valid_latencies, 99),
                    "success_rate": len(valid_latencies) / num_tests * 100
                }
            else:
                stats[component] = {"error": "No successful measurements"}
        
        return {
            "test_type": "component_latency",
            "num_tests": num_tests,
            "timestamp": datetime.now().isoformat(),
            "raw_data": component_results,
            "statistics": stats
        }
    
    async def benchmark_end_to_end_latency(self, num_tests: int = 5) -> Dict:
        """Benchmark complete end-to-end pipeline latency"""
        print("=== End-to-End Latency Benchmark ===")
        
        e2e_results = {
            "total_latency": [],
            "stt_component": [],
            "llm_component": [],
            "tts_component": [],
            "network_overhead": []
        }
        
        test_sentences = [
            "Hello, how are you today?",
            "What's the weather like?",
            "Can you help me with this question?",
            "Tell me a short story.",
            "What time is it right now?"
        ]
        
        for i in range(num_tests):
            sentence = test_sentences[i % len(test_sentences)]
            print(f"E2E test {i+1}/{num_tests}: '{sentence}'")
            
            start_time = time.time()
            
            try:
                # Measure full pipeline through orchestrator streaming
                timing_data = await self._test_streaming_pipeline(sentence)
                
                total_time = time.time() - start_time
                
                e2e_results["total_latency"].append(total_time * 1000)  # Convert to ms
                e2e_results["stt_component"].append(timing_data.get("stt_time", 0))
                e2e_results["llm_component"].append(timing_data.get("llm_time", 0))
                e2e_results["tts_component"].append(timing_data.get("tts_time", 0))
                e2e_results["network_overhead"].append(timing_data.get("network_time", 0))
                
            except Exception as e:
                print(f"E2E test {i+1} failed: {e}")
                # Add None values to maintain list lengths
                for key in e2e_results:
                    e2e_results[key].append(None)
            
            await asyncio.sleep(2)  # Longer delay for complex tests
        
        # Calculate statistics
        stats = {}
        for component, latencies in e2e_results.items():
            valid_latencies = [l for l in latencies if l is not None]
            if valid_latencies:
                stats[component] = {
                    "mean": statistics.mean(valid_latencies),
                    "median": statistics.median(valid_latencies),
                    "min": min(valid_latencies),
                    "max": max(valid_latencies),
                    "std": statistics.stdev(valid_latencies) if len(valid_latencies) > 1 else 0,
                    "p95": np.percentile(valid_latencies, 95),
                    "success_rate": len(valid_latencies) / num_tests * 100
                }
        
        return {
            "test_type": "end_to_end_latency",
            "num_tests": num_tests,
            "timestamp": datetime.now().isoformat(),
            "raw_data": e2e_results,
            "statistics": stats
        }
    
    async def benchmark_concurrent_users(self, user_counts: List[int] = [1, 2, 4]) -> Dict:
        """Benchmark performance under concurrent load"""
        print("=== Concurrent Users Benchmark ===")
        
        concurrent_results = {}
        
        for user_count in user_counts:
            print(f"Testing {user_count} concurrent users...")
            
            # Create concurrent tasks
            tasks = []
            for i in range(user_count):
                task = self._simulate_user_session(f"user_{i}", duration_seconds=30)
                tasks.append(task)
            
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            # Process results
            successful_sessions = [r for r in results if not isinstance(r, Exception)]
            failed_sessions = [r for r in results if isinstance(r, Exception)]
            
            if successful_sessions:
                all_latencies = []
                for session in successful_sessions:
                    all_latencies.extend(session.get("latencies", []))
                
                concurrent_results[f"{user_count}_users"] = {
                    "successful_sessions": len(successful_sessions),
                    "failed_sessions": len(failed_sessions),
                    "total_duration": total_time,
                    "avg_latency": statistics.mean(all_latencies) if all_latencies else 0,
                    "median_latency": statistics.median(all_latencies) if all_latencies else 0,
                    "p95_latency": np.percentile(all_latencies, 95) if all_latencies else 0,
                    "throughput_rps": len(all_latencies) / total_time if total_time > 0 else 0
                }
            else:
                concurrent_results[f"{user_count}_users"] = {
                    "error": "All sessions failed",
                    "failed_sessions": len(failed_sessions)
                }
        
        return {
            "test_type": "concurrent_users",
            "user_counts": user_counts,
            "timestamp": datetime.now().isoformat(),
            "results": concurrent_results
        }
    
    async def _test_stt_latency(self, audio_data: np.ndarray) -> Optional[float]:
        """Test STT latency with WhisperLive"""
        try:
            start_time = time.time()
            
            async with websockets.connect(self.whisper_url) as websocket:
                # Send config
                config = {
                    "uid": f"latency_test_{int(time.time())}",
                    "language": "en",
                    "task": "transcribe",
                    "model": "tiny",
                    "use_vad": True,
                    "vad_parameters": {"threshold": 0.5, "min_silence_duration_ms": 300},
                    "max_clients": 4,
                    "max_connection_time": 600,
                    "send_last_n_segments": 10,
                    "no_speech_thresh": 0.3,
                    "clip_audio": False,
                    "same_output_threshold": 8
                }
                
                await websocket.send(json.dumps(config))
                await websocket.recv()  # Wait for SERVER_READY
                
                # Send audio
                chunk_size = 4096
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    chunk_int16 = (chunk * 32767).astype(np.int16)
                    await websocket.send(chunk_int16.tobytes())
                
                await websocket.send("END_OF_AUDIO")
                
                # Wait for transcription
                while True:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        if isinstance(response, str):
                            data = json.loads(response)
                            if "segments" in data and data["segments"]:
                                return (time.time() - start_time) * 1000
                    except asyncio.TimeoutError:
                        break
            
            return None
            
        except Exception as e:
            print(f"STT test failed: {e}")
            return None
    
    async def _test_llm_latency(self, text: str) -> Optional[float]:
        """Test LLM latency via orchestrator"""
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.orchestrator_url}/process-transcript",
                    json={"transcript": text, "session_id": "latency_test"}
                ) as response:
                    if response.status == 200:
                        await response.json()
                        return (time.time() - start_time) * 1000
            
            return None
            
        except Exception as e:
            print(f"LLM test failed: {e}")
            return None
    
    async def _test_tts_latency(self, text: str) -> Optional[float]:
        """Test TTS latency with Kokoro"""
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.tts_url}/v1/audio/speech",
                    json={
                        "input": text,
                        "voice": "af_bella",
                        "response_format": "wav",
                        "stream": False
                    }
                ) as response:
                    if response.status == 200:
                        await response.read()  # Read the audio data
                        return (time.time() - start_time) * 1000
            
            return None
            
        except Exception as e:
            print(f"TTS test failed: {e}")
            return None
    
    async def _test_orchestrator_latency(self, text: str) -> Optional[float]:
        """Test full orchestrator pipeline latency"""
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.orchestrator_url}/process-transcript-stream",
                    json={"transcript": text, "session_id": "latency_test"}
                ) as response:
                    if response.status == 200:
                        # Read streaming response until complete
                        async for line in response.content:
                            if b'"type": "complete"' in line:
                                return (time.time() - start_time) * 1000
            
            return None
            
        except Exception as e:
            print(f"Orchestrator test failed: {e}")
            return None
    
    async def _test_streaming_pipeline(self, text: str) -> Dict:
        """Test streaming pipeline with detailed timing"""
        timing_data = {
            "stt_time": 0,
            "llm_time": 0,
            "tts_time": 0,
            "network_time": 0
        }
        
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.orchestrator_url}/process-transcript-stream",
                    json={"transcript": text, "session_id": "e2e_test"}
                ) as response:
                    
                    first_chunk_time = None
                    audio_start_time = None
                    
                    async for line in response.content:
                        current_time = time.time()
                        
                        if first_chunk_time is None:
                            first_chunk_time = current_time
                            timing_data["network_time"] = (current_time - start_time) * 1000
                        
                        if b'"type": "text"' in line and timing_data["llm_time"] == 0:
                            timing_data["llm_time"] = (current_time - start_time) * 1000
                        
                        if b'"type": "audio"' in line:
                            if audio_start_time is None:
                                audio_start_time = current_time
                                timing_data["tts_time"] = (current_time - start_time) * 1000
                        
                        if b'"type": "complete"' in line:
                            break
            
            return timing_data
            
        except Exception as e:
            print(f"Streaming pipeline test failed: {e}")
            return timing_data
    
    async def _simulate_user_session(self, user_id: str, duration_seconds: int = 30) -> Dict:
        """Simulate a user session with multiple interactions"""
        session_results = {
            "user_id": user_id,
            "latencies": [],
            "errors": []
        }
        
        end_time = time.time() + duration_seconds
        interaction_count = 0
        
        test_phrases = [
            "Hello there",
            "How are you?",
            "What time is it?",
            "Tell me a joke",
            "What's the weather like?"
        ]
        
        while time.time() < end_time:
            phrase = test_phrases[interaction_count % len(test_phrases)]
            
            try:
                latency = await self._test_orchestrator_latency(phrase)
                if latency is not None:
                    session_results["latencies"].append(latency)
                else:
                    session_results["errors"].append(f"Interaction {interaction_count}: No response")
                
            except Exception as e:
                session_results["errors"].append(f"Interaction {interaction_count}: {str(e)}")
            
            interaction_count += 1
            await asyncio.sleep(2)  # Wait between interactions
        
        return session_results
    
    def _generate_test_audio(self, duration: float = 2.0, sample_rate: int = 16000) -> np.ndarray:
        """Generate test audio for STT testing"""
        # Generate a simple spoken-like audio pattern
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Simulate speech with multiple harmonics
        audio = np.zeros_like(t)
        fundamental_freq = 200  # Hz
        
        for harmonic in range(1, 5):
            frequency = fundamental_freq * harmonic
            amplitude = 0.3 / harmonic
            audio += amplitude * np.sin(2 * np.pi * frequency * t)
        
        # Add some noise and envelope
        noise = 0.05 * np.random.randn(len(t))
        envelope = np.exp(-0.5 * (t - duration/2)**2 / (duration/4)**2)
        
        audio = (audio + noise) * envelope
        
        return audio.astype(np.float32)
    
    def generate_report(self, save_plots: bool = True):
        """Generate comprehensive benchmark report"""
        if not self.results:
            print("No benchmark results to report")
            return
        
        print("\n=== LATENCY BENCHMARK REPORT ===")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        for result in self.results:
            test_type = result["test_type"]
            print(f"\n--- {test_type.replace('_', ' ').title()} ---")
            
            if test_type in ["component_latency", "end_to_end_latency"]:
                stats = result["statistics"]
                
                print(f"{'Component':<20} {'Mean (ms)':<12} {'Median (ms)':<12} {'P95 (ms)':<12} {'Success %':<12}")
                print("-" * 80)
                
                for component, data in stats.items():
                    if "error" in data:
                        print(f"{component:<20} ERROR: {data['error']}")
                    else:
                        print(f"{component:<20} {data['mean']:<12.1f} {data['median']:<12.1f} "
                              f"{data['p95']:<12.1f} {data['success_rate']:<12.1f}")
            
            elif test_type == "concurrent_users":
                print(f"{'Users':<10} {'Success Rate':<15} {'Avg Latency (ms)':<18} {'P95 (ms)':<12} {'Throughput (RPS)':<18}")
                print("-" * 90)
                
                for user_count, data in result["results"].items():
                    if "error" in data:
                        print(f"{user_count:<10} ERROR: {data['error']}")
                    else:
                        success_rate = data['successful_sessions'] / (data['successful_sessions'] + data['failed_sessions']) * 100
                        print(f"{user_count:<10} {success_rate:<15.1f} {data['avg_latency']:<18.1f} "
                              f"{data['p95_latency']:<12.1f} {data['throughput_rps']:<18.2f}")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"latency_benchmark_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nDetailed results saved to: {filename}")
        
        # Generate plots if requested
        if save_plots:
            self._generate_plots(timestamp)
    
    def _generate_plots(self, timestamp: str):
        """Generate visualization plots"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Voice Pipeline Latency Benchmark Results', fontsize=16)
            
            # Plot 1: Component latencies
            component_data = None
            for result in self.results:
                if result["test_type"] == "component_latency":
                    component_data = result
                    break
            
            if component_data:
                components = []
                means = []
                p95s = []
                
                for comp, stats in component_data["statistics"].items():
                    if "error" not in stats:
                        components.append(comp.replace('_', ' ').title())
                        means.append(stats["mean"])
                        p95s.append(stats["p95"])
                
                x = np.arange(len(components))
                width = 0.35
                
                axes[0, 0].bar(x - width/2, means, width, label='Mean', alpha=0.8)
                axes[0, 0].bar(x + width/2, p95s, width, label='P95', alpha=0.8)
                axes[0, 0].set_xlabel('Component')
                axes[0, 0].set_ylabel('Latency (ms)')
                axes[0, 0].set_title('Component Latencies')
                axes[0, 0].set_xticks(x)
                axes[0, 0].set_xticklabels(components, rotation=45)
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: E2E latency distribution
            e2e_data = None
            for result in self.results:
                if result["test_type"] == "end_to_end_latency":
                    e2e_data = result
                    break
            
            if e2e_data:
                total_latencies = [l for l in e2e_data["raw_data"]["total_latency"] if l is not None]
                if total_latencies:
                    axes[0, 1].hist(total_latencies, bins=10, alpha=0.7, edgecolor='black')
                    axes[0, 1].set_xlabel('Total Latency (ms)')
                    axes[0, 1].set_ylabel('Frequency')
                    axes[0, 1].set_title('End-to-End Latency Distribution')
                    axes[0, 1].grid(True, alpha=0.3)
                    
                    # Add statistics text
                    mean_lat = np.mean(total_latencies)
                    p95_lat = np.percentile(total_latencies, 95)
                    axes[0, 1].axvline(mean_lat, color='red', linestyle='--', label=f'Mean: {mean_lat:.1f}ms')
                    axes[0, 1].axvline(p95_lat, color='orange', linestyle='--', label=f'P95: {p95_lat:.1f}ms')
                    axes[0, 1].legend()
            
            # Plot 3: Concurrent users performance
            concurrent_data = None
            for result in self.results:
                if result["test_type"] == "concurrent_users":
                    concurrent_data = result
                    break
            
            if concurrent_data:
                user_counts = []
                avg_latencies = []
                throughputs = []
                
                for user_count, data in concurrent_data["results"].items():
                    if "error" not in data:
                        user_counts.append(int(user_count.split('_')[0]))
                        avg_latencies.append(data["avg_latency"])
                        throughputs.append(data["throughput_rps"])
                
                if user_counts:
                    axes[1, 0].plot(user_counts, avg_latencies, 'o-', label='Avg Latency')
                    axes[1, 0].set_xlabel('Concurrent Users')
                    axes[1, 0].set_ylabel('Average Latency (ms)')
                    axes[1, 0].set_title('Latency vs Concurrent Users')
                    axes[1, 0].grid(True, alpha=0.3)
                    
                    # Secondary y-axis for throughput
                    ax2 = axes[1, 0].twinx()
                    ax2.plot(user_counts, throughputs, 's-', color='orange', label='Throughput')
                    ax2.set_ylabel('Throughput (RPS)', color='orange')
                    
                    # Combine legends
                    lines1, labels1 = axes[1, 0].get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    axes[1, 0].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # Plot 4: Target vs Actual Performance
            target_latency = 500  # ms from requirements
            
            if component_data:
                actual_total = 0
                component_contributions = []
                component_names = []
                
                for comp, stats in component_data["statistics"].items():
                    if "error" not in stats and comp != "orchestrator_latency":
                        actual_total += stats["mean"]
                        component_contributions.append(stats["mean"])
                        component_names.append(comp.replace('_', ' ').title())
                
                # Pie chart of latency breakdown
                if component_contributions:
                    colors = plt.cm.Set3(np.linspace(0, 1, len(component_contributions)))
                    wedges, texts, autotexts = axes[1, 1].pie(component_contributions, 
                                                            labels=component_names, 
                                                            autopct='%1.1f%%',
                                                            colors=colors,
                                                            startangle=90)
                    axes[1, 1].set_title(f'Latency Breakdown\n(Total: {actual_total:.1f}ms vs Target: {target_latency}ms)')
                    
                    # Add target line reference
                    if actual_total > target_latency:
                        axes[1, 1].text(0, -1.3, f'⚠️ Exceeds target by {actual_total - target_latency:.1f}ms', 
                                       ha='center', color='red', fontweight='bold')
                    else:
                        axes[1, 1].text(0, -1.3, f'✅ Under target by {target_latency - actual_total:.1f}ms', 
                                       ha='center', color='green', fontweight='bold')
            
            plt.tight_layout()
            plot_filename = f"latency_benchmark_plots_{timestamp}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Plots saved to: {plot_filename}")
            
        except ImportError:
            print("Matplotlib not available, skipping plot generation")
        except Exception as e:
            print(f"Error generating plots: {e}")
    
    async def run_full_benchmark(self, 
                                component_tests: int = 10,
                                e2e_tests: int = 5,
                                concurrent_users: List[int] = [1, 2, 4]):
        """Run complete benchmark suite"""
        print("🚀 Starting Comprehensive Latency Benchmark Suite")
        print("=" * 60)
        
        # Health check
        print("Performing health checks...")
        healthy = await self._health_check()
        if not healthy:
            print("❌ Health check failed. Please ensure all services are running.")
            return
        
        print("✅ All services healthy. Starting benchmark...\n")
        
        # Run benchmarks
        try:
            # Component latency test
            component_result = await self.benchmark_component_latency(component_tests)
            self.results.append(component_result)
            
            # End-to-end latency test
            e2e_result = await self.benchmark_end_to_end_latency(e2e_tests)
            self.results.append(e2e_result)
            
            # Concurrent users test
            concurrent_result = await self.benchmark_concurrent_users(concurrent_users)
            self.results.append(concurrent_result)
            
            # Generate report
            self.generate_report()
            
        except KeyboardInterrupt:
            print("\n⚠️ Benchmark interrupted by user")
            if self.results:
                print("Generating partial report...")
                self.generate_report()
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
    
    async def _health_check(self) -> bool:
        """Check if all services are healthy"""
        services = [
            ("Orchestrator", f"{self.orchestrator_url}/health"),
            ("Kokoro TTS", f"{self.tts_url}/health")
        ]
        
        all_healthy = True
        
        async with aiohttp.ClientSession() as session:
            for service_name, url in services:
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            print(f"  ✅ {service_name}: OK")
                        else:
                            print(f"  ❌ {service_name}: HTTP {response.status}")
                            all_healthy = False
                except Exception as e:
                    print(f"  ❌ {service_name}: {e}")
                    all_healthy = False
        
        # Check WhisperLive WebSocket
        try:
            async with websockets.connect(self.whisper_url) as ws:
                await ws.send(json.dumps({"uid": "health_check", "language": "en", "task": "transcribe", "model": "tiny"}))
                response = await asyncio.wait_for(ws.recv(), timeout=5)
                print(f"  ✅ WhisperLive: OK")
        except Exception as e:
            print(f"  ❌ WhisperLive: {e}")
            all_healthy = False
        
        return all_healthy

async def main():
    parser = argparse.ArgumentParser(description="Comprehensive Latency Benchmark Suite")
    parser.add_argument("--orchestrator-url", default="http://localhost:8000",
                       help="Orchestrator service URL")
    parser.add_argument("--whisper-url", default="ws://localhost:9090",
                       help="WhisperLive WebSocket URL")
    parser.add_argument("--tts-url", default="http://localhost:8880",
                       help="Kokoro TTS service URL")
    parser.add_argument("--component-tests", type=int, default=10,
                       help="Number of component tests to run")
    parser.add_argument("--e2e-tests", type=int, default=5,
                       help="Number of end-to-end tests to run")
    parser.add_argument("--concurrent-users", type=int, nargs="+", default=[1, 2, 4],
                       help="Concurrent user counts to test")
    
    args = parser.parse_args()
    
    benchmark = LatencyBenchmark(
        args.orchestrator_url,
        args.whisper_url,
        args.tts_url
    )
    
    await benchmark.run_full_benchmark(
        args.component_tests,
        args.e2e_tests,
        args.concurrent_users
    )

if __name__ == "__main__":
    asyncio.run(main())