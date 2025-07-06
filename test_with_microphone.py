#!/usr/bin/env python3
"""
Interactive microphone test for speaker embedding pipeline
"""

import asyncio
import base64
import json
import logging
import time
import websockets
import pyaudio
import wave
import tempfile
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MicrophoneTest:
    def __init__(self):
        self.websocket_url = "ws://localhost:8000/ws/v1/voice"
        
    def record_audio(self, duration=3):
        """Record audio from microphone for specified duration"""
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        
        logger.info(f"🎤 Recording for {duration} seconds... Speak now!")
        
        p = pyaudio.PyAudio()
        
        stream = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK)
        
        frames = []
        
        for i in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        logger.info("🎤 Recording finished!")
        
        # Save to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            wf = wave.open(tmp_file.name, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            # Read back as bytes
            with open(tmp_file.name, 'rb') as f:
                audio_data = f.read()
            
            os.unlink(tmp_file.name)
            return audio_data
    
    async def test_pipeline_with_mic(self):
        """Test the complete pipeline with real microphone input"""
        logger.info("🚀 Starting microphone test...")
        
        # Record audio
        audio_data = self.record_audio(duration=3)
        logger.info(f"📁 Recorded {len(audio_data)} bytes of audio")
        
        # Connect to orchestrator
        session_id = f"mic_test_{int(time.time())}"
        uri = f"{self.websocket_url}/{session_id}"
        
        try:
            async with websockets.connect(uri, timeout=10) as websocket:
                logger.info("🔗 Connected to orchestrator WebSocket")
                
                # Send audio data
                audio_base64 = base64.b64encode(audio_data).decode()
                message = {
                    "event": "audio_stream",
                    "audio_data": audio_base64
                }
                
                await websocket.send(json.dumps(message))
                logger.info("📤 Sent audio to orchestrator")
                
                # Listen for responses
                timeout_time = time.time() + 30  # 30 second timeout
                responses_received = []
                
                while time.time() < timeout_time:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        message = json.loads(response)
                        event = message.get("event")
                        data = message.get("data", {})
                        
                        responses_received.append(event)
                        
                        if event == "transcript.update":
                            logger.info(f"🗣️  TRANSCRIPT: {data.get('text', '')}")
                            
                        elif event == "speaker.identified":
                            logger.info(f"👤 SPEAKER IDENTIFIED: {data.get('name')} (status: {data.get('status')})")
                            
                        elif event == "assistant.speak":
                            logger.info(f"🤖 ASSISTANT: {data.get('text', '')}")
                            
                        elif event == "error":
                            logger.error(f"❌ ERROR: {data.get('message', '')}")
                            
                        # Stop if we get a complete workflow
                        if len(responses_received) >= 2:  # Got some responses
                            break
                            
                    except asyncio.TimeoutError:
                        continue
                        
                # Summary
                logger.info("📊 PIPELINE TEST SUMMARY:")
                logger.info(f"   Responses received: {responses_received}")
                
                if "transcript.update" in responses_received:
                    logger.info("   ✅ Transcription: Working")
                else:
                    logger.info("   ❌ Transcription: Failed")
                    
                if "speaker.identified" in responses_received:
                    logger.info("   ✅ Speaker ID: Working")
                else:
                    logger.info("   ❌ Speaker ID: Failed")
                    
                if "assistant.speak" in responses_received:
                    logger.info("   ✅ Assistant: Working")
                else:
                    logger.info("   ❌ Assistant: Failed")
                
        except Exception as e:
            logger.error(f"❌ Pipeline test failed: {e}")

async def main():
    """Run the microphone test"""
    print("🎤 MICROPHONE TEST for Speaker Embedding Pipeline")
    print("=" * 60)
    print("This will record 3 seconds from your microphone and test the complete pipeline.")
    print("Make sure your microphone is working and speak clearly when prompted.")
    print()
    
    input("Press Enter when ready to start recording...")
    
    tester = MicrophoneTest()
    await tester.test_pipeline_with_mic()

if __name__ == "__main__":
    try:
        import pyaudio
    except ImportError:
        print("❌ pyaudio not installed. Install with: pip install pyaudio")
        exit(1)
        
    asyncio.run(main())