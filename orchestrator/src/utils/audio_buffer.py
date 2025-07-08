
import numpy as np
import time
import io
import wave
from collections import deque
import logging
from config import config

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
        
        # Simple energy-based VAD with debugging
        if apply_vad:
            frame_length = int(self.sample_rate * 0.02)
            hop_length = int(self.sample_rate * 0.01)
            rms_threshold = 0.01 
            
            rms = np.array([
                np.sqrt(np.mean(np.square(float_array[i:i+frame_length])))
                for i in range(0, len(float_array) - frame_length, hop_length)
            ])
            
            silent_frames = np.where(rms < rms_threshold)[0]
            silence_ratio = len(silent_frames) / len(rms) if len(rms) > 0 else 1.0
            
            logger.info(f"VAD Analysis: {len(rms)} frames, {len(silent_frames)} silent, ratio={silence_ratio:.2f}, max_rms={np.max(rms):.4f}, avg_rms={np.mean(rms):.4f}")
            
            # Instead of hard rejection, use a more permissive approach
            # Allow processing if there's any meaningful audio content
            if silence_ratio > 0.95 or np.max(rms) < 0.001:  # Only reject if almost completely silent
                logger.warning(f"Rejecting audio: too silent (silence_ratio={silence_ratio:.2f}, max_rms={np.max(rms):.4f})")
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
