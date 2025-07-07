
import numpy as np
import time
import io
import wave
from collections import deque
import logging
from orchestrator.src.config import config

logger = logging.getLogger(__name__)

class AudioBufferManager:
    """Manages audio buffer accumulation for speaker recognition."""
    
    def __init__(self, buffer_duration_ms: int = 10000, sample_rate: int = 16000):
        self.buffer_duration_ms = buffer_duration_ms
        self.sample_rate = sample_rate
        self.max_samples = int((buffer_duration_ms / 1000) * sample_rate)
        self.audio_buffer = deque(maxlen=self.max_samples)
        
    def add_audio_chunk(self, audio_chunk: np.ndarray) -> bool:
        """Add audio chunk to buffer. Returns True if buffer is full."""
        self.audio_buffer.extend(audio_chunk)
        return len(self.audio_buffer) >= self.max_samples
    
    def get_buffer_as_wav(self, apply_vad: bool = True) -> bytes:
        """Get current buffer as WAV file bytes with optional VAD filtering."""
        if not self.audio_buffer:
            return b""
            
        float_array = np.array(list(self.audio_buffer), dtype=np.float32)
        
        # Simple energy-based VAD
        if apply_vad:
            frame_length = int(self.sample_rate * 0.02)
            hop_length = int(self.sample_rate * 0.01)
            rms_threshold = 0.01 
            
            rms = np.array([
                np.sqrt(np.mean(np.square(float_array[i:i+frame_length])))
                for i in range(0, len(float_array) - frame_length, hop_length)
            ])
            
            silent_frames = np.where(rms < rms_threshold)[0]
            
            # A more sophisticated VAD would be better here
            if len(silent_frames) > len(rms) * 0.8: # If >80% silent, probably not speech
                return b""

        int16_array = np.clip(float_array * 32767, -32767, 32767).astype(np.int16)
        
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(int16_array.tobytes())
        
        wav_buffer.seek(0)
        return wav_buffer.read()

    def clear_buffer(self):
        self.audio_buffer.clear()
