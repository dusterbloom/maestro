#!/usr/bin/env python3
"""
Terminal audio client for testing the voice orchestrator
Records real audio from microphone and sends to orchestrator WebSocket
"""

import asyncio
import websockets
import pyaudio
import numpy as np
import json
import sys
import threading
import queue
import base64
import wave
import io
from datetime import datetime
import concurrent.futures

# Audio configuration
SAMPLE_RATE = 16000
CHUNK_SIZE = 256  # Match orchestrator chunk size
FORMAT = pyaudio.paFloat32
CHANNELS = 1

class TerminalVoiceClient:
    def __init__(self, session_id="terminal-test"):
        self.session_id = session_id
        self.ws = None
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.ws_url = f"ws://localhost:8000/ws/voice"
        self.audio_queue = queue.Queue()
        self.loop = None
        self.output_stream = None
        self.is_tts_playing = False
        self.tts_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
    async def connect(self):
        """Connect to orchestrator WebSocket"""
        print(f"🔌 Connecting to {self.ws_url}...")
        self.ws = await websockets.connect(self.ws_url)
        print("✅ Connected to orchestrator")
        self.loop = asyncio.get_event_loop()
        
        # Start message receiver
        asyncio.create_task(self.receive_messages())
        # Start audio sender
        asyncio.create_task(self.audio_sender())
        
    async def receive_messages(self):
        """Receive and display messages from orchestrator"""
        try:
            async for message in self.ws:
                data = json.loads(message)
                msg_type = data.get('type', 'unknown')
                
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                
                if msg_type == 'ready':
                    print(f"\n[{timestamp}] ✅ Session ready: {data.get('session_id')}")
                elif msg_type == 'live_transcript':
                    text = data.get('text', '').strip()
                    if text:
                        print(f"\n[{timestamp}] 📝 Live: {text}")
                elif msg_type == 'processing_started':
                    text = data.get('text', '').strip()
                    print(f"\n[{timestamp}] 🔄 Processing: {text}")
                elif msg_type == 'sentence_audio':
                    text = data.get('text', '')
                    print(f"\n[{timestamp}] 🔊 TTS: {text[:50]}{'...' if len(text) > 50 else ''}")
                    self.is_tts_playing = True
                    # Play the TTS audio (keep audio queue active for interrupts)
                    await self.play_tts_audio(data)
                    self.is_tts_playing = False
                elif msg_type == 'interrupted':
                    print(f"\n[{timestamp}] 🛑 INTERRUPTED!")
                    # Stop TTS immediately when interrupted
                    if self.is_tts_playing and self.output_stream:
                        self.output_stream.stop_stream()
                        self.is_tts_playing = False
                elif msg_type == 'processing_complete':
                    print(f"\n[{timestamp}] ✅ Processing complete")
                elif msg_type == 'error':
                    print(f"\n[{timestamp}] ❌ Error: {data.get('message', 'Unknown error')}")
                elif msg_type == 'segments':
                    # Only show if it's a completed segment
                    segments = data.get('segments', [])
                    completed_segments = [s for s in segments if s.get('completed', False)]
                    if completed_segments:
                        for seg in completed_segments:
                            print(f"\n[{timestamp}] 🎯 Completed: {seg.get('text', '').strip()}")
                else:
                    # Don't spam with raw segment data unless it's important
                    if msg_type not in ['segments']:
                        print(f"\n[{timestamp}] 📨 {msg_type}: {data}")
                    
        except websockets.exceptions.ConnectionClosed:
            print("❌ WebSocket connection closed")
        except Exception as e:
            print(f"❌ Error receiving messages: {e}")
            
    def start_recording(self):
        """Start recording audio from microphone"""
        if self.is_recording:
            return
            
        print("🎤 Starting audio recording...")
        self.is_recording = True
        
        try:
            # Try to use the pulse device (index 0) for WSL/Linux
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=0,  # Use pulse device
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=self.audio_callback
            )
            
            self.stream.start_stream()
            print(f"🎙️ Recording active - speak to test! (16000 Hz, Float32, device_index=0)")
            print(f"🔥 [TERMINAL_DEBUG] Audio config: rate={SAMPLE_RATE}, format={FORMAT}, channels={CHANNELS}, chunk={CHUNK_SIZE}")
        except Exception as e:
            print(f"❌ Failed to start audio recording: {e}")
            print("💡 Trying alternative audio settings...")
            try:
                # Fallback: Try default device but KEEP EXACT SAME SETTINGS as main
                self.stream = self.audio.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,  # CRITICAL: Must match orchestrator (16000 Hz)
                    input=True,
                    input_device_index=1,  # Try default device
                    frames_per_buffer=CHUNK_SIZE,  # CRITICAL: Must match main settings (256)
                    stream_callback=self.audio_callback
                )
                self.stream.start_stream()
                print(f"🎙️ Recording active with fallback settings! (16000 Hz, Float32, device_index=1)")
                print(f"🔥 [TERMINAL_DEBUG] Fallback audio config: rate={SAMPLE_RATE}, format={FORMAT}, channels={CHANNELS}, chunk={CHUNK_SIZE}")
            except Exception as e2:
                print(f"❌ Could not initialize audio: {e2}")
                self.is_recording = False
        
    def stop_recording(self):
        """Stop recording audio"""
        if not self.is_recording:
            return
            
        print("🛑 Stopping audio recording...")
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            
    def audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback - queues audio data"""
        if status:
            print(f"⚠️ Audio callback status: {status}")
            
        if self.is_recording:
            # CRITICAL FIX: Terminal client must send exact same chunk size as orchestrator expects
            # PyAudio gives us 256 samples (1024 bytes) but orchestrator expects 64 samples (256 bytes)
            # Split the 1024-byte chunk into 4x 256-byte chunks to match orchestrator format
            
            # Convert to numpy array for level monitoring
            audio_array = np.frombuffer(in_data, dtype=np.float32)
            audio_level = np.abs(audio_array).mean()
            
            # Show audio level bar in terminal
            level_bar = "█" * int(audio_level * 200)
            if audio_level > 0.01:
                sys.stdout.write(f"\r🎤 Level: {level_bar:<20} {audio_level:.4f}")
                sys.stdout.flush()
            
            # CRITICAL FIX: Split 1024-byte chunks into 256-byte chunks (orchestrator format)
            chunk_size = 256  # 64 samples * 4 bytes = 256 bytes (orchestrator expects this)
            for i in range(0, len(in_data), chunk_size):
                chunk = in_data[i:i + chunk_size]
                if len(chunk) == chunk_size:  # Only send complete chunks
                    self.audio_queue.put(chunk)
            
            # Debug: Log when significant audio is queued during TTS
            if hasattr(self, 'is_tts_playing') and self.is_tts_playing and audio_level > 0.02:
                num_chunks = len(in_data) // chunk_size
                print(f"\r🎤 [INTERRUPT!] Voice detected during TTS: level={audio_level:.6f}, split into {num_chunks} chunks of {chunk_size} bytes each")
                print(f"\r🔥 [TERMINAL_FIX] Original: {len(in_data)} bytes → Split into {num_chunks}x {chunk_size}-byte chunks for orchestrator")
            
        return (None, pyaudio.paContinue)
        
    async def audio_sender(self):
        """Async task to send queued audio data"""
        while True:
            try:
                # Get audio data from queue (non-blocking)
                audio_data = self.audio_queue.get_nowait()
                
                # CRITICAL FIX: Keep queue small but don't discard voice audio
                while self.audio_queue.qsize() > 3:
                    # Remove oldest audio to prevent WebSocket overflow, but keep sending current audio
                    try:
                        discarded = self.audio_queue.get_nowait()
                    except queue.Empty:
                        break
                
                if self.ws and hasattr(self.ws, 'closed') and not self.ws.closed:
                    # CRITICAL: Always log what we're actually sending vs detecting
                    actual_level = np.abs(np.frombuffer(audio_data, dtype=np.float32)).mean()
                    print(f"\r🔥 [CRITICAL] SENDING chunk: {len(audio_data)} bytes, level={actual_level:.6f}")
                    await self.ws.send(audio_data)
                    # Debug: Log when audio is sent during TTS
                    if hasattr(self, 'is_tts_playing') and self.is_tts_playing:
                        print(f"\r🔥 [WEBSOCKET_FIX] Audio SENT during TTS: {len(audio_data)} bytes, queue={self.audio_queue.qsize()}")
                elif self.ws:
                    await self.ws.send(audio_data)
                    if hasattr(self, 'is_tts_playing') and self.is_tts_playing:
                        print(f"\r🔥 [WEBSOCKET_FIX] Audio SENT during TTS (fallback): {len(audio_data)} bytes")
                self.audio_queue.task_done()
            except queue.Empty:
                # No audio data available, sleep briefly
                # Reduce sleep during TTS for faster interrupt response
                sleep_time = 0.0001 if hasattr(self, 'is_tts_playing') and self.is_tts_playing else 0.001
                await asyncio.sleep(sleep_time)
            except Exception as e:
                print(f"\n❌ Error sending audio: {e}")
                await asyncio.sleep(0.1)
    
    async def send_audio(self, audio_data):
        """Send audio data to orchestrator"""
        try:
            if self.ws and not self.ws.closed:
                await self.ws.send(audio_data)
        except Exception as e:
            print(f"\n❌ Error sending audio: {e}")
    
    def _play_tts_audio_sync(self, audio_bytes):
        """Synchronous TTS audio playback - runs in thread pool"""
        try:
            # Create output stream if needed
            if not self.output_stream:
                self.output_stream = self.audio.open(
                    format=pyaudio.paInt16,  # TTS usually outputs 16-bit PCM
                    channels=1,
                    rate=22050,  # Common TTS sample rate
                    output=True,
                    frames_per_buffer=1024
                )
            
            # Play the audio directly - this runs in a separate thread
            self.output_stream.write(audio_bytes)
            
        except Exception as e:
            print(f"\n❌ Error playing TTS audio: {e}")

    async def play_tts_audio(self, tts_data):
        """Play TTS audio from base64 encoded data - NON-BLOCKING"""
        try:
            audio_b64 = tts_data.get('audio_data', '')
            if not audio_b64:
                return
                
            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_b64)
            
            # Play audio in thread pool to avoid blocking event loop and audio sender
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.tts_thread_pool, self._play_tts_audio_sync, audio_bytes)
            
        except Exception as e:
            print(f"\n❌ Error playing TTS audio: {e}")
    
    def clear_audio_queue(self):
        """Clear the audio queue to prevent WebSocket frame overflow"""
        try:
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
        except queue.Empty:
            pass
            
    async def send_interrupt(self):
        """Send interrupt signal"""
        print("\n🛑 Sending interrupt...")
        if self.ws and not self.ws.closed:
            await self.ws.send(json.dumps({"type": "interrupt"}))
            
    async def run_interactive(self):
        """Run interactive terminal interface"""
        await self.connect()
        
        print("\n" + "="*50)
        print("Voice Orchestrator Terminal Client")
        print("="*50)
        print("\nCommands:")
        print("  [SPACE] - Toggle recording on/off")
        print("  [i]     - Send interrupt signal")
        print("  [q]     - Quit")
        print("\nPress SPACE to start recording...")
        print("="*50 + "\n")
        
        # Start recording automatically
        self.start_recording()
        
        # Handle keyboard input in separate thread
        def keyboard_handler():
            while True:
                try:
                    # Cross-platform keyboard input
                    try:
                        # Try Unix/Linux/macOS approach
                        import termios, tty
                        old_settings = termios.tcgetattr(sys.stdin)
                        try:
                            tty.setraw(sys.stdin.fileno())
                            key = sys.stdin.read(1)
                        finally:
                            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                    except ImportError:
                        # Windows fallback
                        import msvcrt
                        key = msvcrt.getch().decode('utf-8')
                    
                    if key == ' ':
                        if self.is_recording:
                            print("\n⏸️  Pausing recording...")
                            self.stop_recording()
                        else:
                            print("\n▶️  Resuming recording...")
                            self.start_recording()
                    elif key == 'i':
                        if self.loop:
                            asyncio.run_coroutine_threadsafe(self.send_interrupt(), self.loop)
                    elif key == 'q':
                        print("\n👋 Quitting...")
                        # Stop TTS immediately
                        if self.is_tts_playing and self.output_stream:
                            self.output_stream.stop_stream()
                            self.is_tts_playing = False
                        self.stop_recording()
                        if self.loop:
                            self.loop.call_soon_threadsafe(self.loop.stop)
                        break
                except Exception as e:
                    print(f"\n❌ Keyboard error: {e}")
                    break
                    
        keyboard_thread = threading.Thread(target=keyboard_handler, daemon=True)
        keyboard_thread.start()
        
        # Keep running
        try:
            await asyncio.Future()  # Run forever
        except asyncio.CancelledError:
            pass
        finally:
            self.stop_recording()
            if self.output_stream:
                self.output_stream.stop_stream()
                self.output_stream.close()
            if self.ws:
                await self.ws.close()
            self.audio.terminate()

async def main():
    client = TerminalVoiceClient()
    try:
        await client.run_interactive()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting Voice Orchestrator Terminal Client...")
    asyncio.run(main())