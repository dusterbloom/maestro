import asyncio
import numpy as np
import time
import hashlib
import io
import wave
import tempfile
import logging
import os
from collections import deque
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from config import config
import soundfile as sf

# Resemblyzer imports
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

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
        """Get current buffer as WAV file bytes for Resemblyzer with optional VAD filtering"""
        if not self.audio_buffer:
            return b""
            
        # Convert float32 samples to numpy array
        float_array = np.array(list(self.audio_buffer), dtype=np.float32)
        
        if apply_vad:
            # Apply Voice Activity Detection to filter out silent parts
            voice_active_samples = self._extract_voice_segments(float_array, threshold=vad_threshold)
            
            if len(voice_active_samples) == 0:
                logger.warning("⚠️ No voice activity detected in audio buffer")
                return b""
            
            logger.info(f"🎤 VAD filtering: {len(float_array)} -> {len(voice_active_samples)} samples ({len(voice_active_samples)/len(float_array)*100:.1f}% retained)")
            float_array = voice_active_samples
        
        # Validate audio data
        if len(float_array) == 0:
            logger.error("❌ Empty audio array after processing")
            return b""
        
        # Normalize audio to prevent clipping
        max_val = np.max(np.abs(float_array))
        if max_val > 1.0:
            float_array = float_array / max_val
            logger.info(f"🔧 Normalized audio: max was {max_val:.4f}")
        
        # Convert to int16 PCM with proper scaling
        int16_array = np.clip(float_array * 32767, -32767, 32767).astype(np.int16)
        
        # Create WAV file in memory using proper wave library
        wav_buffer = io.BytesIO()
        try:
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)  # 16kHz
                wav_file.writeframes(int16_array.tobytes())
            
            wav_buffer.seek(0)
            wav_bytes = wav_buffer.read()
            
            # Validate WAV file size
            if len(wav_bytes) < 44:  # WAV header is 44 bytes minimum
                logger.error(f"❌ Generated WAV too small: {len(wav_bytes)} bytes")
                return b""
            
            logger.info(f"✅ Generated WAV: {len(wav_bytes)} bytes ({len(int16_array)} samples)")
            return wav_bytes
            
        except Exception as wav_error:
            logger.error(f"❌ WAV creation failed: {wav_error}")
            return b""
    
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
    """Enhanced VoiceService with 10-second buffering for definitive speaker recognition using Resemblyzer"""
    
    def __init__(self, memory_service=None):
        # Initialize Resemblyzer voice encoder - this could be blocking, but only happens once at startup
        logger.info(f"Initializing Resemblyzer VoiceEncoder on device: {config.RESEMBLYZER_DEVICE}")
        self.voice_encoder = VoiceEncoder(device=config.RESEMBLYZER_DEVICE)
        logger.info(f"✅ VoiceEncoder initialized successfully")
        
        # Thread pool executor for CPU-intensive operations
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="resemblyzer")
        
        # Audio buffer for accumulating 10-second samples
        self.audio_buffer_manager = AudioBufferManager()
        
        # Memory service for speaker storage/retrieval
        self.memory_service = memory_service
        
        # Speaker registry for single-speaker magical recognition
        self.registered_speaker = None  # {"user_id": str, "name": str, "embedding": List[float], "confidence_threshold": float}
        
        # Event handlers for agentic responses
        self.event_handlers = {}
        
        # Confidence settings for definitive recognition (using cosine similarity)
        self.confidence_threshold = config.SPEAKER_SIMILARITY_THRESHOLD  # Cosine similarity threshold
        self.registration_confidence = 0.8  # Confidence for auto-registration
        
        # Timeout for embedding operations (configurable)
        self.embedding_timeout = config.SPEAKER_EMBEDDING_TIMEOUT
    
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
                logger.error(f"Event handler error for {event.event_type}: {e}")
    
    async def accumulate_audio_chunk(self, audio_chunk: bytes, session_id: str) -> Optional[Dict]:
        """Accumulate audio chunk and check for speaker when 10 seconds reached (NON-BLOCKING)"""
        try:
            # Convert bytes to float32 array
            float_array = np.frombuffer(audio_chunk, dtype=np.float32)
            
            # Add to buffer
            buffer_ready = self.audio_buffer_manager.add_audio_chunk(float_array)
            
            if buffer_ready:
                # We have 10 seconds - start NON-BLOCKING speaker recognition
                logger.info(f"🚀 Starting NON-BLOCKING speaker recognition for session {session_id}")
                
                # Start background task without awaiting (non-blocking)
                asyncio.create_task(self._perform_nonblocking_speaker_identification(session_id))
                
                # Return immediately with buffering status
                return {
                    "status": "processing_started",
                    "message": "Speaker identification started in background",
                    "session_id": session_id,
                    "buffer_duration": 10.0
                }
            
            # Not ready yet - return buffer status  
            duration = self.audio_buffer_manager.get_buffer_duration_seconds()
            return {
                "status": "accumulating",
                "buffer_duration": duration,
                "target_duration": 10.0,
                "progress": duration / 10.0
            }
            
        except Exception as e:
            logger.error(f"Error accumulating audio: {e}")
            return None
    
    async def _perform_nonblocking_speaker_identification(self, session_id: str):
        """🚀 NON-BLOCKING: Perform speaker identification without blocking conversation"""
        try:
            logger.info(f"🔄 Background speaker identification starting for session {session_id}")
            
            # Get audio data from buffer without VAD preprocessing to keep full 10 seconds
            wav_bytes = self.audio_buffer_manager.get_buffer_as_wav(apply_vad=False)  # Keep full duration
            
            if not wav_bytes:
                logger.error(f"❌ No audio data available for session {session_id}")
                return
            
            # Get embedding using full audio duration
            embedding = await self.get_embedding(wav_bytes)
            if not embedding:
                logger.error(f"❌ Failed to generate embedding for session {session_id}")
                # Clear buffer and return
                self.audio_buffer_manager.clear_buffer()
                return
            
            # Check if this speaker was already identified to avoid duplicates
            if self.registered_speaker:
                stored_embedding = self.registered_speaker["embedding"]
                similarity = self._calculate_similarity(embedding, stored_embedding)
                
                if similarity >= self.confidence_threshold:
                    logger.info(f"✅ Session {session_id} - SAME SPEAKER recognized with confidence {similarity:.4f}")
                    # Don't create new registration, just update session state
                    self.audio_buffer_manager.clear_buffer()
                    return
            
            # Perform identification or registration
            result = await self._identify_or_register_speaker_definitively(embedding, session_id)
            
            # Clear buffer after processing
            self.audio_buffer_manager.clear_buffer()
            
            logger.info(f"✅ Background speaker identification completed for session {session_id}: {result}")
            
        except Exception as e:
            logger.error(f"❌ Background speaker identification failed for session {session_id}: {e}")
            # Always clear buffer on error
            self.audio_buffer_manager.clear_buffer()
    
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
                    # Low confidence - query all known speakers from storage
                    return await self._query_and_identify_speaker(embedding, session_id)
            else:
                # No speaker registered locally, try to find one in storage
                return await self._query_and_identify_speaker(embedding, session_id)
                
        except Exception as e:
            logger.error(f"Error in definitive speaker identification: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _query_and_identify_speaker(self, embedding: List[float], session_id: str) -> Dict:
        """Query ChromaDB for all known speakers and identify the best match"""
        try:
            # This requires memory service to be available
            if not self.memory_service:
                # Fallback to registering if no memory service
                return await self._register_new_speaker_to_storage(embedding, session_id)

            all_speakers = await self.memory_service.get_all_speaker_profiles()
            
            if not all_speakers:
                # No speakers in DB, register this one
                return await self._register_new_speaker_to_storage(embedding, session_id)

            best_match = None
            highest_confidence = 0.0

            for speaker in all_speakers:
                confidence = self._calculate_similarity(embedding, speaker["embedding"])
                if confidence > highest_confidence:
                    highest_confidence = confidence
                    best_match = speaker

            if highest_confidence >= self.confidence_threshold:
                # Found a definitive match in the database
                user_id = best_match["user_id"]
                name = best_match["name"]
                
                # Update local registry for faster subsequent checks
                self.registered_speaker = {
                    "user_id": user_id,
                    "name": name,
                    "embedding": best_match["embedding"],
                    "confidence_threshold": self.confidence_threshold
                }

                event = SpeakerEvent(
                    event_type="speaker_identified",
                    user_id=user_id,
                    confidence=highest_confidence,
                    timestamp=datetime.utcnow().isoformat(),
                    audio_hash=hashlib.md5(str(embedding).encode()).hexdigest()[:8],
                    session_id=session_id,
                    context={"name": name, "recognition_type": "definitive_db_match"}
                )
                await self.emit_speaker_event(event)

                return {
                    "status": "identified",
                    "user_id": user_id,
                    "name": name,
                    "confidence": highest_confidence,
                    "is_definitive": True
                }
            else:
                # No match found, register as a new speaker
                return await self._register_new_speaker_to_storage(embedding, session_id)

        except Exception as e:
            logger.error(f"Error querying and identifying speaker: {e}")
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
            logger.error(f"Error registering new speaker: {e}")
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
            logger.error(f"Error in speaker identification: {e}")
            return {"status": "error", "message": str(e)}
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between Resemblyzer embeddings (already L2-normalized)"""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Resemblyzer embeddings are already L2-normalized, so dot product = cosine similarity
            similarity = np.dot(vec1, vec2)
            
            # Resemblyzer cosine similarity is already in 0-1 range (since embeddings are normalized)
            # Clamp to ensure it's in valid range
            similarity = np.clip(similarity, 0.0, 1.0)
            
            logger.debug(f"Cosine similarity between embeddings: {similarity:.4f}")
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    async def update_speaker_name(self, name: str) -> bool:
        """Update the registered speaker's name"""
        if self.registered_speaker:
            self.registered_speaker["name"] = name
            logger.info(f"Updated speaker name to: {name}")
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
        logger.info("Cleared registered speaker")
    
    def cleanup(self):
        """Clean up resources (thread pool executor)"""
        if hasattr(self, 'executor') and self.executor:
            self.executor.shutdown(wait=False)
            logger.info("Thread pool executor shutdown")

    async def get_embedding(self, audio_data: bytes) -> list[float] | None:
        """Get embedding using Resemblyzer from audio data (supports both WAV and raw formats) - NON-BLOCKING"""
        try:
            logger.info(f"🎤 Generating speaker embedding from {len(audio_data)} bytes of audio using Resemblyzer...")
            
            # Validate input audio data
            if len(audio_data) < 1000:  # At least 1KB for valid audio
                logger.error("❌ Audio data too small to be valid")
                return None
            
            # Run the CPU-intensive embedding generation in thread pool with timeout
            loop = asyncio.get_event_loop()
            
            logger.info(f"🔄 Starting embedding generation in thread pool...")
            start_time = time.time()
            
            try:
                # Create a task for embedding generation
                task = asyncio.create_task(
                    loop.run_in_executor(
                        self.executor,
                        self._generate_embedding_sync,
                        audio_data
                    )
                )
                
                # Use asyncio.wait_for to add timeout protection
                result = await asyncio.wait_for(task, timeout=self.embedding_timeout)
                
                elapsed_time = time.time() - start_time
                logger.info(f"✅ Embedding generation completed in {elapsed_time:.2f}s")
                return result
                
            except asyncio.TimeoutError:
                elapsed_time = time.time() - start_time
                logger.error(f"❌ Embedding generation timed out after {elapsed_time:.2f}s (limit: {self.embedding_timeout}s)")
                # Cancel the task to stop the background operation
                if 'task' in locals() and not task.done():
                    task.cancel()
                    logger.info(f"🚫 Cancelled timed-out embedding task")
                return None
                
        except Exception as e:
            elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
            logger.error(f"❌ Error generating Resemblyzer embedding after {elapsed_time:.2f}s: {e}")
            logger.error(f"❌ Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return None
    
    def _generate_embedding_sync(self, audio_data: bytes) -> list[float] | None:
        """Synchronous embedding generation - runs in thread pool"""
        try:
            # Check if this is actually WAV data or raw audio samples
            is_wav_format = self._detect_wav_format(audio_data)
            
            if is_wav_format:
                logger.info("📁 Detected WAV format, using WAV parsing")
                return self._process_wav_audio_sync(audio_data)
            else:
                logger.info("🎵 Detected raw audio samples, using raw processing")
                return self._process_raw_audio_sync(audio_data)
                
        except Exception as e:
            logger.error(f"❌ Synchronous embedding generation failed: {e}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return None
    
    def _detect_wav_format(self, audio_data: bytes) -> bool:
        """Detect if audio data is in WAV format or raw samples"""
        if len(audio_data) < 12:
            return False
        
        # Check for WAV header: 'RIFF' + size + 'WAVE'
        header = audio_data[:12]
        is_wav = (
            header[:4] == b'RIFF' and 
            header[8:12] == b'WAVE'
        )
        
        if is_wav:
            logger.info("✅ Valid WAV header detected")
        else:
            logger.info(f"🔍 Not WAV format - header: {header[:16]}")
        
        return is_wav
    
    def _process_wav_audio_sync(self, wav_data: bytes) -> list[float] | None:
        """Process audio data that's already in WAV format - synchronous version"""
        try:
            # Create temporary file for WAV data
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(wav_data)
                temp_file_path = temp_file.name
            
            try:
                # Read WAV file using soundfile
                audio_array, sample_rate = sf.read(temp_file_path, dtype='float32')
                logger.info(f"📊 Read WAV: shape={audio_array.shape}, sr={sample_rate}")
                
                return self._generate_embedding_from_array_sync(audio_array, sample_rate)
                
            finally:
                Path(temp_file_path).unlink()
                
        except Exception as e:
            logger.error(f"❌ WAV processing failed: {e}")
            return None
    
    async def _process_wav_audio(self, wav_data: bytes) -> list[float] | None:
        """Process audio data that's already in WAV format"""
        try:
            # Create temporary file for WAV data
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(wav_data)
                temp_file_path = temp_file.name
            
            try:
                # Read WAV file using soundfile
                audio_array, sample_rate = sf.read(temp_file_path, dtype='float32')
                logger.info(f"📊 Read WAV: shape={audio_array.shape}, sr={sample_rate}")
                
                return await self._generate_embedding_from_array(audio_array, sample_rate)
                
            finally:
                Path(temp_file_path).unlink()
                
        except Exception as e:
            logger.error(f"❌ WAV processing failed: {e}")
            return None
    
    def _process_raw_audio_sync(self, raw_data: bytes) -> list[float] | None:
        """Process raw audio samples (float32 or int16) - synchronous version"""
        try:
            # Try to interpret as float32 samples (most likely from browser)
            sample_rate = 16000  # Default sample rate
            
            # Check if data length is consistent with float32
            if len(raw_data) % 4 == 0:  # float32 = 4 bytes per sample
                logger.info("🔄 Attempting float32 interpretation...")
                try:
                    audio_array = np.frombuffer(raw_data, dtype=np.float32)
                    logger.info(f"📊 Float32 interpretation: {len(audio_array)} samples")
                    
                    # Validate the audio data
                    if self._validate_audio_array(audio_array):
                        return self._generate_embedding_from_array_sync(audio_array, sample_rate)
                        
                except Exception as float32_error:
                    logger.warning(f"⚠️ Float32 interpretation failed: {float32_error}")
            
            # Try to interpret as int16 samples
            if len(raw_data) % 2 == 0:  # int16 = 2 bytes per sample
                logger.info("🔄 Attempting int16 interpretation...")
                try:
                    int16_array = np.frombuffer(raw_data, dtype=np.int16)
                    # Convert to float32 and normalize
                    audio_array = int16_array.astype(np.float32) / 32767.0
                    logger.info(f"📊 Int16 interpretation: {len(audio_array)} samples")
                    
                    # Validate the audio data
                    if self._validate_audio_array(audio_array):
                        return self._generate_embedding_from_array_sync(audio_array, sample_rate)
                        
                except Exception as int16_error:
                    logger.warning(f"⚠️ Int16 interpretation failed: {int16_error}")
            
            logger.error("❌ Could not interpret raw audio data as float32 or int16")
            return None
            
        except Exception as e:
            logger.error(f"❌ Raw audio processing failed: {e}")
            return None
    
    async def _process_raw_audio(self, raw_data: bytes) -> list[float] | None:
        """Process raw audio samples (float32 or int16)"""
        try:
            # Try to interpret as float32 samples (most likely from browser)
            sample_rate = 16000  # Default sample rate
            
            # Check if data length is consistent with float32
            if len(raw_data) % 4 == 0:  # float32 = 4 bytes per sample
                logger.info("🔄 Attempting float32 interpretation...")
                try:
                    audio_array = np.frombuffer(raw_data, dtype=np.float32)
                    logger.info(f"📊 Float32 interpretation: {len(audio_array)} samples")
                    
                    # Validate the audio data
                    if self._validate_audio_array(audio_array):
                        return await self._generate_embedding_from_array(audio_array, sample_rate)
                        
                except Exception as float32_error:
                    logger.warning(f"⚠️ Float32 interpretation failed: {float32_error}")
            
            # Try to interpret as int16 samples
            if len(raw_data) % 2 == 0:  # int16 = 2 bytes per sample
                logger.info("🔄 Attempting int16 interpretation...")
                try:
                    int16_array = np.frombuffer(raw_data, dtype=np.int16)
                    # Convert to float32 and normalize
                    audio_array = int16_array.astype(np.float32) / 32767.0
                    logger.info(f"📊 Int16 interpretation: {len(audio_array)} samples")
                    
                    # Validate the audio data
                    if self._validate_audio_array(audio_array):
                        return await self._generate_embedding_from_array(audio_array, sample_rate)
                        
                except Exception as int16_error:
                    logger.warning(f"⚠️ Int16 interpretation failed: {int16_error}")
            
            logger.error("❌ Could not interpret raw audio data as float32 or int16")
            return None
            
        except Exception as e:
            logger.error(f"❌ Raw audio processing failed: {e}")
            return None
    
    def _validate_audio_array(self, audio_array: np.ndarray) -> bool:
        """Validate that audio array looks reasonable"""
        if len(audio_array) == 0:
            logger.error("❌ Empty audio array")
            return False
        
        # Check for reasonable amplitude range
        max_val = np.max(np.abs(audio_array))
        if max_val == 0:
            logger.error("❌ Silent audio (all zeros)")
            return False
        
        if max_val > 100:  # Likely not normalized properly
            logger.warning(f"⚠️ Unusually high amplitude: {max_val}")
        
        # Check for reasonable duration (at least 0.1 seconds)
        duration = len(audio_array) / 16000  # Assume 16kHz
        if duration < 0.1:
            logger.warning(f"⚠️ Very short audio: {duration:.3f}s")
        
        logger.info(f"✅ Audio validation passed: {len(audio_array)} samples, max_amp={max_val:.4f}, duration={duration:.2f}s")
        return True
    
    def _generate_embedding_from_array_sync(self, audio_array: np.ndarray, sample_rate: int) -> list[float] | None:
        """Generate Resemblyzer embedding from numpy audio array - synchronous version"""
        try:
            # Handle multi-channel audio
            if audio_array.ndim > 1:
                if audio_array.shape[1] > 1:  # Stereo/multichannel
                    audio_array = audio_array.mean(axis=1)  # Average channels
                    logger.info(f"🔄 Converted multi-channel to mono")
                else:
                    audio_array = audio_array.ravel()  # Remove extra dimensions
            
            # Calculate original duration
            duration_seconds = len(audio_array) / sample_rate
            logger.info(f"📊 Original audio: {len(audio_array)} samples, {duration_seconds:.2f}s")
            
            # Check audio energy level before preprocessing
            audio_rms = np.sqrt(np.mean(audio_array ** 2))
            audio_db = 20 * np.log10(audio_rms + 1e-8)
            logger.info(f"🔊 Original audio energy: RMS={audio_rms:.6f}, dB={audio_db:.2f}")
            
            # If audio is very quiet, boost it before preprocessing
            if audio_rms < 0.001:  # Very quiet audio
                boost_factor = 0.01 / (audio_rms + 1e-8)
                boost_factor = min(boost_factor, 50)  # Cap boost at 50x
                audio_array = audio_array * boost_factor
                new_rms = np.sqrt(np.mean(audio_array ** 2))
                logger.info(f"🔊 Boosted quiet audio: {audio_rms:.6f} -> {new_rms:.6f} (factor: {boost_factor:.1f}x)")
            
            # Use CONSERVATIVE preprocessing to preserve speaker characteristics
            try:
                # Try Resemblyzer preprocessing but check if it removes too much
                preprocessed_test = preprocess_wav(audio_array, sample_rate)
                retention_ratio = len(preprocessed_test) / len(audio_array) if len(audio_array) > 0 else 0
                
                if retention_ratio < 0.3:  # If preprocessing removes more than 70% of audio
                    logger.warning(f"⚠️ Resemblyzer preprocessing too aggressive ({retention_ratio*100:.1f}% retained), using manual preprocessing")
                    
                    # Manual conservative preprocessing
                    # Just normalize and ensure reasonable amplitude
                    max_val = np.max(np.abs(audio_array))
                    if max_val > 0:
                        preprocessed_wav = audio_array / max_val * 0.5  # Normalize to 50% to avoid clipping
                    else:
                        preprocessed_wav = audio_array
                    
                    logger.info(f"🔧 Manual preprocessing: {len(preprocessed_wav)} samples retained (100.0%)")
                else:
                    preprocessed_wav = preprocessed_test
                    logger.info(f"🔧 Resemblyzer preprocessing: {len(preprocessed_wav)} samples retained ({retention_ratio*100:.1f}%)")
                    
            except Exception as preprocess_error:
                logger.warning(f"⚠️ Preprocessing failed: {preprocess_error}, using manual normalization")
                # Fallback to simple normalization
                max_val = np.max(np.abs(audio_array))
                if max_val > 0:
                    preprocessed_wav = audio_array / max_val * 0.5
                else:
                    preprocessed_wav = audio_array
            
            # Ensure minimum duration for Resemblyzer (at least 1.6 seconds)
            min_samples = int(1.6 * sample_rate)  # Resemblyzer minimum
            if len(preprocessed_wav) < min_samples:
                logger.info(f"📏 Extending audio from {len(preprocessed_wav)} to {min_samples} samples")
                
                if len(preprocessed_wav) > 0:
                    # Repeat the audio instead of padding with zeros
                    repetitions = (min_samples // len(preprocessed_wav)) + 1
                    extended_audio = np.tile(preprocessed_wav, repetitions)[:min_samples]
                    preprocessed_wav = extended_audio
                    logger.info(f"🔄 Extended by repeating audio pattern")
                else:
                    # Last resort: create minimal test tone
                    t = np.linspace(0, 1.6, min_samples)
                    preprocessed_wav = 0.01 * np.sin(2 * np.pi * 440 * t)  # 440Hz tone
                    logger.warning(f"⚠️ Created fallback test tone")
            
            # Final validation of preprocessed audio
            final_rms = np.sqrt(np.mean(preprocessed_wav ** 2))
            final_db = 20 * np.log10(final_rms + 1e-8)
            
            if final_rms < 1e-6:  # Still too quiet after all processing
                logger.error(f"❌ Final audio still too quiet: RMS={final_rms:.8f}, dB={final_db:.2f}")
                return None
            
            # Generate embedding using Resemblyzer - THE BLOCKING OPERATION
            embedding = self.voice_encoder.embed_utterance(preprocessed_wav)
            
            # Convert numpy array to list for JSON serialization
            embedding_list = embedding.tolist()
            
            logger.info(f"✅ Successfully generated Resemblyzer embedding:")
            logger.info(f"   → Original duration: {duration_seconds:.2f}s")
            logger.info(f"   → Processed duration: {len(preprocessed_wav)/sample_rate:.2f}s")
            logger.info(f"   → Sample rate: {sample_rate}Hz")
            logger.info(f"   → Final audio dB: {final_db:.2f}")
            logger.info(f"   → Embedding dimensions: {len(embedding_list)}")
            logger.info(f"   → Embedding vector norm: {np.linalg.norm(embedding):.4f}")
            
            return embedding_list
            
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return None
    
    async def _generate_embedding_from_array(self, audio_array: np.ndarray, sample_rate: int) -> list[float] | None:
        """Generate Resemblyzer embedding from numpy audio array with robust preprocessing"""
        try:
            # Handle multi-channel audio
            if audio_array.ndim > 1:
                if audio_array.shape[1] > 1:  # Stereo/multichannel
                    audio_array = audio_array.mean(axis=1)  # Average channels
                    logger.info(f"🔄 Converted multi-channel to mono")
                else:
                    audio_array = audio_array.ravel()  # Remove extra dimensions
            
            # Calculate original duration
            duration_seconds = len(audio_array) / sample_rate
            logger.info(f"📊 Original audio: {len(audio_array)} samples, {duration_seconds:.2f}s")
            
            # Check audio energy level before preprocessing
            audio_rms = np.sqrt(np.mean(audio_array ** 2))
            audio_db = 20 * np.log10(audio_rms + 1e-8)
            logger.info(f"🔊 Original audio energy: RMS={audio_rms:.6f}, dB={audio_db:.2f}")
            
            # If audio is very quiet, boost it before preprocessing
            if audio_rms < 0.001:  # Very quiet audio
                boost_factor = 0.01 / (audio_rms + 1e-8)
                boost_factor = min(boost_factor, 50)  # Cap boost at 50x
                audio_array = audio_array * boost_factor
                new_rms = np.sqrt(np.mean(audio_array ** 2))
                logger.info(f"🔊 Boosted quiet audio: {audio_rms:.6f} -> {new_rms:.6f} (factor: {boost_factor:.1f}x)")
            
            # Use CONSERVATIVE preprocessing to preserve speaker characteristics
            try:
                # Try Resemblyzer preprocessing but check if it removes too much
                preprocessed_test = preprocess_wav(audio_array, sample_rate)
                retention_ratio = len(preprocessed_test) / len(audio_array) if len(audio_array) > 0 else 0
                
                if retention_ratio < 0.3:  # If preprocessing removes more than 70% of audio
                    logger.warning(f"⚠️ Resemblyzer preprocessing too aggressive ({retention_ratio*100:.1f}% retained), using manual preprocessing")
                    
                    # Manual conservative preprocessing
                    # Just normalize and ensure reasonable amplitude
                    max_val = np.max(np.abs(audio_array))
                    if max_val > 0:
                        preprocessed_wav = audio_array / max_val * 0.5  # Normalize to 50% to avoid clipping
                    else:
                        preprocessed_wav = audio_array
                    
                    logger.info(f"🔧 Manual preprocessing: {len(preprocessed_wav)} samples retained (100.0%)")
                else:
                    preprocessed_wav = preprocessed_test
                    logger.info(f"🔧 Resemblyzer preprocessing: {len(preprocessed_wav)} samples retained ({retention_ratio*100:.1f}%)")
                    
            except Exception as preprocess_error:
                logger.warning(f"⚠️ Preprocessing failed: {preprocess_error}, using manual normalization")
                # Fallback to simple normalization
                max_val = np.max(np.abs(audio_array))
                if max_val > 0:
                    preprocessed_wav = audio_array / max_val * 0.5
                else:
                    preprocessed_wav = audio_array
            
            # Ensure minimum duration for Resemblyzer (at least 1.6 seconds)
            min_samples = int(1.6 * sample_rate)  # Resemblyzer minimum
            if len(preprocessed_wav) < min_samples:
                logger.info(f"📏 Extending audio from {len(preprocessed_wav)} to {min_samples} samples")
                
                if len(preprocessed_wav) > 0:
                    # Repeat the audio instead of padding with zeros
                    repetitions = (min_samples // len(preprocessed_wav)) + 1
                    extended_audio = np.tile(preprocessed_wav, repetitions)[:min_samples]
                    preprocessed_wav = extended_audio
                    logger.info(f"🔄 Extended by repeating audio pattern")
                else:
                    # Last resort: create minimal test tone
                    t = np.linspace(0, 1.6, min_samples)
                    preprocessed_wav = 0.01 * np.sin(2 * np.pi * 440 * t)  # 440Hz tone
                    logger.warning(f"⚠️ Created fallback test tone")
            
            # Final validation of preprocessed audio
            final_rms = np.sqrt(np.mean(preprocessed_wav ** 2))
            final_db = 20 * np.log10(final_rms + 1e-8)
            
            if final_rms < 1e-6:  # Still too quiet after all processing
                logger.error(f"❌ Final audio still too quiet: RMS={final_rms:.8f}, dB={final_db:.2f}")
                return None
            
            # Generate embedding using Resemblyzer
            embedding = self.voice_encoder.embed_utterance(preprocessed_wav)
            
            # Convert numpy array to list for JSON serialization
            embedding_list = embedding.tolist()
            
            logger.info(f"✅ Successfully generated Resemblyzer embedding:")
            logger.info(f"   → Original duration: {duration_seconds:.2f}s")
            logger.info(f"   → Processed duration: {len(preprocessed_wav)/sample_rate:.2f}s")
            logger.info(f"   → Sample rate: {sample_rate}Hz")
            logger.info(f"   → Final audio dB: {final_db:.2f}")
            logger.info(f"   → Embedding dimensions: {len(embedding_list)}")
            logger.info(f"   → Embedding vector norm: {np.linalg.norm(embedding):.4f}")
            
            return embedding_list
            
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return None
            logger.info(f"   → Embedding dimensions: {len(embedding_list)}")
            logger.info(f"   → Average dB: {avg_db:.2f}")
            logger.info(f"   → Embedding vector norm: {np.linalg.norm(embedding):.4f}")
            
            return embedding_list
            
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return None
