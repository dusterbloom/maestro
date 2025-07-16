import numpy as np
import pyaudio
import asyncio
from typing import AsyncIterator
from genai_processors import Processor, ProcessorPart
from genai_processors.content_api import AudioPart, TextPart

class WebSocketAudioOutput(Processor):
    def __init__(self, websocket, context=None):
        self.websocket = websocket
        self.context = context or {}
        
    async def __call__(self, input_stream: AsyncIterator[ProcessorPart]) -> AsyncIterator[ProcessorPart]:
        async for part in input_stream:
            try:
                if isinstance(part, AudioPart):
                    await self.websocket.send_json({
                        "type": "audio",
                        "data": part.audio_data.tolist() if isinstance(part.audio_data, np.ndarray) else list(part.audio_data),
                        "sample_rate": part.sample_rate,
                        "format": part.metadata.get("format", "raw"),
                        "text": part.metadata.get("text", "")
                    })
                    
                elif isinstance(part, TextPart):
                    if part.text:
                        await self.websocket.send_json({
                            "type": "transcript",
                            "text": part.text,
                            "is_final": part.metadata.get("is_final", False),
                            "source": part.metadata.get("source", "unknown")
                        })
                
                yield part
                
            except Exception as e:
                print(f"Output error: {e}")

class SpeakerOutput(Processor):
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.audio = None
        self.stream = None
        
    async def __aenter__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            output=True
        )
        return self
        
    async def __aexit__(self, *args):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()
            
    async def __call__(self, input_stream: AsyncIterator[ProcessorPart]) -> AsyncIterator[ProcessorPart]:
        async for part in input_stream:
            if isinstance(part, AudioPart):
                try:
                    if part.metadata.get("format") == "mp3":
                        import io
                        from pydub import AudioSegment
                        
                        audio_segment = AudioSegment.from_mp3(io.BytesIO(part.audio_data))
                        audio_data = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / 32768.0
                        
                        if audio_segment.frame_rate != self.sample_rate:
                            import librosa
                            audio_data = librosa.resample(audio_data, orig_sr=audio_segment.frame_rate, target_sr=self.sample_rate)
                    else:
                        audio_data = part.audio_data
                        
                    self.stream.write(audio_data.astype(np.float32).tobytes())
                    
                except Exception as e:
                    print(f"Speaker output error: {e}")
                    
            yield part