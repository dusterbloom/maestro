import asyncio
import time
import json
import numpy as np
import pyaudio
from typing import AsyncIterator, Optional
from genai_processors import Processor, ProcessorPart, streams
from genai_processors.content_api import AudioPart

class ContinuousAudioInput(Processor):
    def __init__(self, websocket, context=None):
        self.websocket = websocket
        self.context = context or {}
        self.sample_rate = 16000
        
    async def __call__(self, input_stream: AsyncIterator[ProcessorPart]) -> AsyncIterator[ProcessorPart]:
        try:
            while True:
                message = await self.websocket.receive()
                
                if message["type"] == "bytes":
                    audio_data = np.frombuffer(message["bytes"], dtype=np.int16).astype(np.float32) / 32768.0
                    
                    yield AudioPart(
                        audio_data=audio_data,
                        sample_rate=self.sample_rate,
                        metadata={"timestamp": time.time(), "source": "websocket"}
                    )
                    
                elif message["type"] == "text":
                    data = json.loads(message["text"])
                    
                    if data.get("type") == "vad_start":
                        self.context["is_user_speaking"] = True
                        
                        if self.context.get("is_assistant_speaking"):
                            await self._handle_interruption()
                            
                    elif data.get("type") == "vad_stop":
                        self.context["is_user_speaking"] = False
                        
        except Exception as e:
            print(f"Audio input error: {e}")
    
    async def _handle_interruption(self):
        self.context["state"] = "interrupted"
        
        if "current_tasks" in self.context:
            for task in self.context["current_tasks"]:
                if not task.done():
                    task.cancel()
        
        await self.websocket.send_json({"type": "interruption"})
        self.context["is_assistant_speaking"] = False

class MicrophoneInput(Processor):
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio = None
        self.stream = None
        
    async def __aenter__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        return self
        
    async def __aexit__(self, *args):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()
            
    async def __call__(self, input_stream: AsyncIterator[ProcessorPart]) -> AsyncIterator[ProcessorPart]:
        while True:
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.float32)
                
                yield AudioPart(
                    audio_data=audio_data,
                    sample_rate=self.sample_rate,
                    metadata={"source": "microphone"}
                )
                
                await asyncio.sleep(0.001)
                
            except Exception as e:
                print(f"Microphone error: {e}")
                break