import httpx
import asyncio
import numpy as np
import time
import hashlib
import io
import wave
from collections import deque
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict
from datetime import datetime
from config import config

@dataclass
class SpeakerEvent:
    """Event for agentic speaker recognition responses"""
    event_type: str  # 'speaker_identified', 'speaker_registered', 'confidence_low'
    user_id: Optional[str]
    confidence: float
    timestamp: str
    audio_hash: str
    session_id: str
    context: Dict

class AudioBufferManager:
    """Manages 5-second audio buffer accumulation for accurate speaker recognition"""
    
    def __init__(self, buffer_duration_ms: int = 5000, sample_rate: int = 16000):
        self.buffer_duration_ms = buffer_duration_ms
        self.sample_rate = sample_rate
        self.max_samples = int((buffer_duration_ms / 1000) * sample_rate)
        
        # Audio buffer (deque for efficient append/pop)
        self.audio_buffer = deque(maxlen=self.max_samples)
        self.buffer_start_time = None
        
    def add_audio_chunk(self, audio_chunk: np.ndarray) -> bool:
        """Add audio chunk to buffer. Returns True if buffer is ready (5+ seconds)"""
        if self.buffer_start_time is None:
            self.buffer_start_time = time.time()
            
        # Add samples to buffer
        for sample in audio_chunk:
            self.audio_buffer.append(sample)
            
        # Check if we have 5 seconds of audio
        current_samples = len(self.audio_buffer)
        duration_seconds = current_samples / self.sample_rate
        
        return duration_seconds >= (self.buffer_duration_ms / 1000)
    
    def get_buffer_as_bytes(self) -> bytes:
        """Get current buffer as int16 PCM bytes for Diglett"""
        if not self.audio_buffer:
            return b""
            
        # Convert float32 samples to int16 PCM
        float_array = np.array(list(self.audio_buffer), dtype=np.float32)
        int16_array = (float_array * 32767).astype(np.int16)
        return int16_array.tobytes()
    
    def clear_buffer(self):
        """Clear buffer and reset timing"""
        self.audio_buffer.clear()
        self.buffer_start_time = None
    
    def get_buffer_duration_seconds(self) -> float:
        """Get current buffer duration in seconds"""
        return len(self.audio_buffer) / self.sample_rate

class VoiceService:
    """Enhanced VoiceService with 5-second buffering and agentic responses"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(base_url=config.DIGLETT_URL, timeout=10.0)
        
        # Audio buffer for accumulating 5-second samples
        self.audio_buffer_manager = AudioBufferManager()
        
        # Speaker registry for single-speaker magical recognition
        self.registered_speaker = None  # {"user_id": str, "name": str, "embedding": List[float], "confidence_threshold": float}
        
        # Event handlers for agentic responses
        self.event_handlers = {}
        
        # Confidence settings
        self.confidence_threshold = 0.8  # High threshold for magical recognition
        self.registration_confidence = 0.9  # Very high threshold for registration
    
    def on_speaker_event(self, event_type: str):
        """Decorator to register event handlers"""
        def decorator(func):
            self.event_handlers[event_type] = func
            return func
        return decorator
    
    async def emit_speaker_event(self, event: SpeakerEvent):
        """Emit speaker event and trigger agentic response"""
        handler = self.event_handlers.get(event.event_type)
        if handler:
            try:
                await handler(event)
            except Exception as e:
                print(f"Event handler error for {event.event_type}: {e}")
    
    async def accumulate_audio_chunk(self, audio_chunk: bytes, session_id: str) -> Optional[Dict]:
        """Accumulate audio chunk and check for speaker when 5 seconds reached"""
        try:
            # Convert bytes to float32 array
            float_array = np.frombuffer(audio_chunk, dtype=np.float32)
            
            # Add to buffer
            buffer_ready = self.audio_buffer_manager.add_audio_chunk(float_array)
            
            if buffer_ready:
                # We have 5 seconds - perform speaker recognition
                buffer_bytes = self.audio_buffer_manager.get_buffer_as_bytes()
                
                # Get embedding
                embedding = await self.get_embedding(buffer_bytes)
                if not embedding:
                    self.audio_buffer_manager.clear_buffer()
                    return None
                
                # Check for speaker recognition
                result = await self._identify_or_register_speaker(embedding, session_id)
                
                # Clear buffer for next accumulation
                self.audio_buffer_manager.clear_buffer()
                
                return result
            
            # Not ready yet - return buffer status
            duration = self.audio_buffer_manager.get_buffer_duration_seconds()
            return {
                "status": "accumulating",
                "buffer_duration": duration,
                "target_duration": 5.0,
                "progress": duration / 5.0
            }
            
        except Exception as e:
            print(f"Error accumulating audio: {e}")
            return None
    
    async def _identify_or_register_speaker(self, embedding: List[float], session_id: str) -> Dict:
        """Identify existing speaker or register new one based on confidence"""
        try:
            audio_hash = hashlib.md5(str(embedding).encode()).hexdigest()[:8]
            
            if self.registered_speaker:
                # Check similarity with registered speaker
                stored_embedding = self.registered_speaker["embedding"]
                confidence = self._calculate_similarity(embedding, stored_embedding)
                
                if confidence >= self.confidence_threshold:
                    # Magical recognition!
                    event = SpeakerEvent(
                        event_type="speaker_identified",
                        user_id=self.registered_speaker["user_id"],
                        confidence=confidence,
                        timestamp=datetime.utcnow().isoformat(),
                        audio_hash=audio_hash,
                        session_id=session_id,
                        context={"name": self.registered_speaker["name"], "recognition_type": "magical"}
                    )
                    
                    await self.emit_speaker_event(event)
                    
                    return {
                        "status": "identified",
                        "user_id": self.registered_speaker["user_id"],
                        "name": self.registered_speaker["name"],
                        "confidence": confidence,
                        "is_magical": True,
                        "greeting": f"Hello {self.registered_speaker['name']}! "
                    }
                else:
                    # Low confidence
                    event = SpeakerEvent(
                        event_type="confidence_low",
                        user_id=self.registered_speaker["user_id"],
                        confidence=confidence,
                        timestamp=datetime.utcnow().isoformat(),
                        audio_hash=audio_hash,
                        session_id=session_id,
                        context={"threshold": self.confidence_threshold, "recognition_type": "uncertain"}
                    )
                    
                    await self.emit_speaker_event(event)
                    
                    return {
                        "status": "uncertain",
                        "confidence": confidence,
                        "threshold": self.confidence_threshold,
                        "greeting": "I'm not sure if I recognize your voice. "
                    }
            else:
                # No registered speaker - register this one
                user_id = f"speaker_{int(time.time())}"
                self.registered_speaker = {
                    "user_id": user_id,
                    "name": "Friend",  # Default name until they tell us
                    "embedding": embedding,
                    "confidence_threshold": self.confidence_threshold
                }
                
                event = SpeakerEvent(
                    event_type="speaker_registered",
                    user_id=user_id,
                    confidence=1.0,
                    timestamp=datetime.utcnow().isoformat(),
                    audio_hash=audio_hash,
                    session_id=session_id,
                    context={"registration_type": "automatic", "is_first_speaker": True}
                )
                
                await self.emit_speaker_event(event)
                
                return {
                    "status": "registered",
                    "user_id": user_id,
                    "name": "Friend",
                    "confidence": 1.0,
                    "is_new": True,
                    "greeting": "Hello! I don't think we've met before. What should I call you? "
                }
                
        except Exception as e:
            print(f"Error in speaker identification: {e}")
            return {"status": "error", "message": str(e)}
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings"""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            similarity = dot_product / (norm1 * norm2)
            
            # Convert to 0-1 range (cosine similarity is -1 to 1)
            return (similarity + 1) / 2
            
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
    
    async def update_speaker_name(self, name: str) -> bool:
        """Update the registered speaker's name"""
        if self.registered_speaker:
            self.registered_speaker["name"] = name
            print(f"Updated speaker name to: {name}")
            return True
        return False
    
    def get_registered_speaker(self) -> Optional[Dict]:
        """Get current registered speaker info"""
        if self.registered_speaker:
            return {
                "user_id": self.registered_speaker["user_id"],
                "name": self.registered_speaker["name"],
                "confidence_threshold": self.registered_speaker["confidence_threshold"]
            }
        return None
    
    def clear_registered_speaker(self):
        """Clear registered speaker (for testing/reset)"""
        self.registered_speaker = None
        print("Cleared registered speaker")

    async def get_embedding(self, audio_data: bytes) -> list[float] | None:
        """Get embedding from Diglett service (unchanged core functionality)"""
        try:
            # Handle different input formats
            if isinstance(audio_data, np.ndarray):
                # Already int16 PCM
                pcm_bytes = audio_data.tobytes()
            else:
                # Convert Float32Array bytes to int16 PCM format that Diglett expects
                try:
                    # Try to interpret as float32 first
                    float_array = np.frombuffer(audio_data, dtype=np.float32)
                    int16_array = (float_array * 32767).astype(np.int16)
                    pcm_bytes = int16_array.tobytes()
                except ValueError:
                    # Maybe it's already int16 PCM
                    pcm_bytes = audio_data
            
            # Send as file upload (Diglett expects File() upload, not raw bytes)
            files = {"file": ("audio.pcm", pcm_bytes, "audio/pcm")}
            response = await self.client.post("/embed", files=files)
            response.raise_for_status()
            result = response.json()
            
            # Diglett returns direct dict format: {"speaker_embedding": [...], "avg_db": float, "speaker_name": str}
            if isinstance(result, dict):
                return result.get("speaker_embedding")
                
            return None
        except Exception as e:
            print(f"Error getting embedding from Diglett: {e}")
            return None
