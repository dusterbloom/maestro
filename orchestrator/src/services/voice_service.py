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
    """Manages 10-second audio buffer accumulation for definitive speaker recognition"""
    
    def __init__(self, buffer_duration_ms: int = 10000, sample_rate: int = 16000, timeout_seconds: int = 30):
        self.buffer_duration_ms = buffer_duration_ms
        self.sample_rate = sample_rate
        self.max_samples = int((buffer_duration_ms / 1000) * sample_rate)
        self.timeout_seconds = timeout_seconds
        
        # Audio buffer (deque for efficient append/pop)
        self.audio_buffer = deque(maxlen=self.max_samples)
        self.buffer_start_time = None
        self.last_audio_time = None
        
    def add_audio_chunk(self, audio_chunk: np.ndarray) -> bool:
        """Add audio chunk to buffer. Returns True if buffer is ready (10+ seconds)"""
        current_time = time.time()
        
        if self.buffer_start_time is None:
            self.buffer_start_time = current_time
        
        self.last_audio_time = current_time
            
        # Add samples to buffer
        for sample in audio_chunk:
            self.audio_buffer.append(sample)
            
        # Check if we have 10 seconds of audio
        current_samples = len(self.audio_buffer)
        duration_seconds = current_samples / self.sample_rate
        
        return duration_seconds >= (self.buffer_duration_ms / 1000)
    
    def is_timed_out(self) -> bool:
        """Check if buffer has timed out (no audio for timeout_seconds)"""
        if self.last_audio_time is None:
            return False
        return (time.time() - self.last_audio_time) > self.timeout_seconds
    
    def get_buffer_as_bytes(self) -> bytes:
        """Get current buffer as int16 PCM bytes for Diglett"""
        if not self.audio_buffer:
            return b""
            
        # Convert float32 samples to int16 PCM
        float_array = np.array(list(self.audio_buffer), dtype=np.float32)
        int16_array = (float_array * 32767).astype(np.int16)
        return int16_array.tobytes()
    
    def get_buffer_as_wav(self, apply_vad: bool = True, vad_threshold: float = None) -> bytes:
        """Get current buffer as WAV file bytes for Diglett with optional VAD filtering"""
        if not self.audio_buffer:
            return b""
            
        # Convert float32 samples to numpy array
        float_array = np.array(list(self.audio_buffer), dtype=np.float32)
        
        if apply_vad:
            # Apply Voice Activity Detection to filter out silent parts
            voice_active_samples = self._extract_voice_segments(float_array, threshold=vad_threshold)
            
            if len(voice_active_samples) == 0:
                print("⚠️ No voice activity detected in audio buffer")
                return b""
            
            print(f"🎤 VAD filtering: {len(float_array)} -> {len(voice_active_samples)} samples ({len(voice_active_samples)/len(float_array)*100:.1f}% retained)")
            float_array = voice_active_samples
        
        # Convert to int16 PCM
        int16_array = (float_array * 32767).astype(np.int16)
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)  # 16kHz
            wav_file.writeframes(int16_array.tobytes())
        
        wav_buffer.seek(0)
        return wav_buffer.read()
    
    def _extract_voice_segments(self, audio_array: np.ndarray, 
                               threshold: float = None, 
                               min_segment_duration_ms: int = None, 
                               pad_duration_ms: int = None) -> np.ndarray:
        """Extract voice-active segments from audio using simple energy-based VAD"""
        
        # Use config defaults if not specified
        if threshold is None:
            threshold = config.SPEAKER_VAD_THRESHOLD
        if min_segment_duration_ms is None:
            min_segment_duration_ms = config.SPEAKER_VAD_MIN_SEGMENT_MS
        if pad_duration_ms is None:
            pad_duration_ms = config.SPEAKER_VAD_PAD_MS
        
        # Calculate energy in sliding window
        window_size = int(self.sample_rate * 0.025)  # 25ms window
        hop_size = int(self.sample_rate * 0.010)     # 10ms hop
        
        # Calculate RMS energy for each frame
        frame_energies = []
        for i in range(0, len(audio_array) - window_size, hop_size):
            frame = audio_array[i:i + window_size]
            energy = np.sqrt(np.mean(frame ** 2))
            frame_energies.append(energy)
        
        frame_energies = np.array(frame_energies)
        
        # Apply voice activity detection
        voice_active = frame_energies > threshold
        
        # Smooth VAD decisions (remove short gaps and spurious detections)
        min_segment_frames = int(min_segment_duration_ms / 10)  # Convert ms to frames
        
        # Fill short gaps (< 200ms)
        for i in range(1, len(voice_active) - 1):
            if not voice_active[i] and voice_active[i-1] and voice_active[i+1]:
                gap_size = 1
                j = i + 1
                while j < len(voice_active) and not voice_active[j]:
                    gap_size += 1
                    j += 1
                if gap_size < min_segment_frames // 2:  # Fill gaps shorter than 100ms
                    voice_active[i:j] = True
        
        # Remove short voice segments (< 200ms) 
        i = 0
        while i < len(voice_active):
            if voice_active[i]:
                segment_start = i
                while i < len(voice_active) and voice_active[i]:
                    i += 1
                segment_length = i - segment_start
                
                if segment_length < min_segment_frames:
                    voice_active[segment_start:i] = False
            else:
                i += 1
        
        # Extract voice segments with padding
        voice_segments = []
        pad_samples = int(self.sample_rate * pad_duration_ms / 1000)
        
        i = 0
        while i < len(voice_active):
            if voice_active[i]:
                segment_start = i
                while i < len(voice_active) and voice_active[i]:
                    i += 1
                segment_end = i
                
                # Convert frame indices to sample indices
                start_sample = max(0, segment_start * hop_size - pad_samples)
                end_sample = min(len(audio_array), segment_end * hop_size + pad_samples)
                
                # Extract segment
                segment = audio_array[start_sample:end_sample]
                voice_segments.append(segment)
            else:
                i += 1
        
        if not voice_segments:
            return np.array([], dtype=np.float32)
        
        # Concatenate all voice segments
        combined_voice = np.concatenate(voice_segments)
        
        # Ensure we don't exceed the original duration too much
        max_samples = len(audio_array)
        if len(combined_voice) > max_samples:
            combined_voice = combined_voice[:max_samples]
        
        return combined_voice
    
    def clear_buffer(self):
        """Clear buffer and reset timing"""
        self.audio_buffer.clear()
        self.buffer_start_time = None
        self.last_audio_time = None
    
    def get_buffer_duration_seconds(self) -> float:
        """Get current buffer duration in seconds"""
        return len(self.audio_buffer) / self.sample_rate

class VoiceService:
    """Enhanced VoiceService with 10-second buffering for definitive speaker recognition"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(base_url=config.DIGLETT_URL, timeout=15.0)
        
        # Audio buffer for accumulating 10-second samples
        self.audio_buffer_manager = AudioBufferManager()
        
        # Speaker registry for single-speaker magical recognition
        self.registered_speaker = None  # {"user_id": str, "name": str, "embedding": List[float], "confidence_threshold": float}
        
        # Event handlers for agentic responses
        self.event_handlers = {}
        
        # Confidence settings for definitive recognition
        self.confidence_threshold = 0.7  # Lower threshold for better recognition
        self.registration_confidence = 0.8  # Confidence for auto-registration
    
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
        """Accumulate audio chunk and check for speaker when 10 seconds reached"""
        try:
            # Convert bytes to float32 array
            float_array = np.frombuffer(audio_chunk, dtype=np.float32)
            
            # Add to buffer
            buffer_ready = self.audio_buffer_manager.add_audio_chunk(float_array)
            
            if buffer_ready:
                # We have 10 seconds - perform definitive speaker recognition
                wav_bytes = self.audio_buffer_manager.get_buffer_as_wav()
                
                # Get embedding using WAV format
                embedding = await self.get_embedding(wav_bytes)
                if not embedding:
                    self.audio_buffer_manager.clear_buffer()
                    return None
                
                # Check for speaker recognition with definitive result
                result = await self._identify_or_register_speaker_definitively(embedding, session_id)
                
                # Clear buffer for next accumulation
                self.audio_buffer_manager.clear_buffer()
                
                return result
            
            # Not ready yet - return buffer status  
            duration = self.audio_buffer_manager.get_buffer_duration_seconds()
            return {
                "status": "accumulating",
                "buffer_duration": duration,
                "target_duration": 10.0,
                "progress": duration / 10.0
            }
            
        except Exception as e:
            print(f"Error accumulating audio: {e}")
            return None
    
    async def _identify_or_register_speaker_definitively(self, embedding: List[float], session_id: str) -> Dict:
        """Definitively identify existing speaker or register new one with ChromaDB storage"""
        try:
            audio_hash = hashlib.md5(str(embedding).encode()).hexdigest()[:8]
            
            if self.registered_speaker:
                # Check similarity with registered speaker
                stored_embedding = self.registered_speaker["embedding"]
                confidence = self._calculate_similarity(embedding, stored_embedding)
                
                if confidence >= self.confidence_threshold:
                    # DEFINITIVE RECOGNITION - high confidence match
                    event = SpeakerEvent(
                        event_type="speaker_identified",
                        user_id=self.registered_speaker["user_id"],
                        confidence=confidence,
                        timestamp=datetime.utcnow().isoformat(),
                        audio_hash=audio_hash,
                        session_id=session_id,
                        context={"name": self.registered_speaker["name"], "recognition_type": "definitive"}
                    )
                    
                    await self.emit_speaker_event(event)
                    
                    return {
                        "status": "identified",
                        "user_id": self.registered_speaker["user_id"],
                        "name": self.registered_speaker["name"],
                        "confidence": confidence,
                        "is_definitive": True,
                        "greeting": f"Welcome back, {self.registered_speaker['name']}! "
                    }
                else:
                    # Different speaker - register as new user
                    await self._register_new_speaker_to_storage(embedding, session_id)
                    return {
                        "status": "registered",
                        "user_id": "new_speaker",
                        "name": "Friend",
                        "confidence": 1.0,
                        "is_new": True,
                        "is_definitive": True,
                        "greeting": "Hello! I don't recognize your voice. What should I call you? "
                    }
            else:
                # No registered speaker - register this one definitively
                result = await self._register_new_speaker_to_storage(embedding, session_id)
                return result
                
        except Exception as e:
            print(f"Error in definitive speaker identification: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _register_new_speaker_to_storage(self, embedding: List[float], session_id: str) -> Dict:
        """Register new speaker to both local registry and ChromaDB/Redis storage"""
        try:
            user_id = f"speaker_{int(time.time())}"
            
            # Store locally for immediate recognition
            self.registered_speaker = {
                "user_id": user_id,
                "name": "Friend",  # Default name until they tell us
                "embedding": embedding,
                "confidence_threshold": self.confidence_threshold
            }
            
            # TODO: Store in ChromaDB and Redis via memory service
            # This will be handled by the agentic system when name is learned
            
            event = SpeakerEvent(
                event_type="speaker_registered",
                user_id=user_id,
                confidence=1.0,
                timestamp=datetime.utcnow().isoformat(),
                audio_hash=hashlib.md5(str(embedding).encode()).hexdigest()[:8],
                session_id=session_id,
                context={"registration_type": "definitive", "is_first_speaker": True}
            )
            
            await self.emit_speaker_event(event)
            
            return {
                "status": "registered",
                "user_id": user_id,
                "name": "Friend",
                "confidence": 1.0,
                "is_new": True,
                "is_definitive": True,
                "greeting": "Hello! I don't think we've met before. What should I call you? "
            }
            
        except Exception as e:
            print(f"Error registering new speaker: {e}")
            return {"status": "error", "message": str(e)}

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
        """Get embedding from Diglett service using WAV format"""
        try:
            print(f"🎤 Sending {len(audio_data)} bytes of WAV audio to Diglett for embedding...")
            
            # audio_data should now be WAV format from get_buffer_as_wav()
            # Send as file upload (Diglett expects WAV file upload)
            files = {"file": ("audio.wav", audio_data, "audio/wav")}
            response = await self.client.post("/embed", files=files)
            
            print(f"🔍 Diglett response status: {response.status_code}")
            print(f"🔍 Diglett response headers: {dict(response.headers)}")
            
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if 'application/json' not in content_type:
                print(f"❌ Diglett returned non-JSON content-type: {content_type}")
                response_text = response.text[:200] + "..." if len(response.text) > 200 else response.text
                print(f"❌ Response content preview: {response_text}")
                return None
            
            result = response.json()
            print(f"✅ Diglett JSON response: {result}")
            
            # Diglett returns: {"speaker_name": str, "speaker_embedding": [float array], "avg_db": float}
            if isinstance(result, dict) and "speaker_embedding" in result:
                embedding = result.get("speaker_embedding")
                speaker_name = result.get("speaker_name", "Unknown")
                avg_db = result.get("avg_db", 0.0)
                
                print(f"✅ Successfully got embedding from Diglett:")
                print(f"   → Speaker: {speaker_name}")
                print(f"   → Embedding length: {len(embedding) if embedding else 0}")
                print(f"   → Average dB: {avg_db}")
                
                return embedding
                
            print(f"❌ Unexpected Diglett response format: {result}")
            return None
            
        except Exception as e:
            print(f"❌ Error getting embedding from Diglett: {e}")
            print(f"❌ Exception type: {type(e).__name__}")
            return None
