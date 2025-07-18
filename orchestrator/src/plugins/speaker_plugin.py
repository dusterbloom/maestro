"""Speaker recognition plugin for parallel audio processing."""

import asyncio
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import time
import json
from plugins.base_plugin import BasePlugin, Event, PluginConfig

logger = logging.getLogger(__name__)


class SpeakerPlugin(BasePlugin):
    """Plugin for speaker identification and voice analysis."""
    
    def __init__(self, config: Optional[PluginConfig] = None):
        super().__init__(config or PluginConfig(
            enabled=True,
            priority=2,  # Lower priority than memory
            max_workers=3,
            timeout=10.0  # Moderate timeout for audio processing
        ))
        self._speaker_profiles: Dict[str, Dict[str, Any]] = {}
        self._audio_buffer: Dict[str, List[np.ndarray]] = {}
        self._processing_queue: asyncio.Queue = asyncio.Queue()
        self._processor_task: Optional[asyncio.Task] = None
        self._model_loaded = False
    
    async def initialize(self) -> None:
        """Initialize the speaker plugin."""
        logger.info("Initializing speaker recognition plugin")
        
        # Load speaker profiles (placeholder for ML models)
        await self._load_speaker_models()
        
        # Start audio processing worker
        self._processor_task = asyncio.create_task(self._audio_processor())
        
        self._model_loaded = True
        logger.info("Speaker plugin initialized")
    
    async def process_event(self, event: Event) -> None:
        """Process audio events for speaker recognition."""
        try:
            if event.event_type == "audio_data":
                await self._process_audio_chunk(event.data)
            elif event.event_type == "voice_activity_start":
                await self._start_speaker_detection(event.data)
            elif event.event_type == "voice_activity_end":
                await self._finalize_speaker_analysis(event.data)
                
        except Exception as e:
            logger.error(f"Error processing speaker event: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown the speaker plugin."""
        logger.info("Shutting down speaker plugin")
        
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        
        await self._save_speaker_profiles()
    
    async def _load_speaker_models(self) -> None:
        """Load speaker recognition models (placeholder)."""
        # In real implementation, load pre-trained models
        self._speaker_profiles = {
            "user_001": {
                "voice_features": [0.1, 0.2, 0.3, 0.4, 0.5],
                "confidence": 0.95,
                "last_seen": time.time()
            }
        }
        logger.debug("Speaker models loaded")
    
    async def _process_audio_chunk(self, data: Dict[str, Any]) -> None:
        """Process audio chunk for speaker identification."""
        user_id = data.get("user_id", "default")
        audio_data = data.get("audio", [])
        timestamp = data.get("timestamp", time.time())
        
        if not audio_data:
            return
        
        # Convert to numpy array
        try:
            audio_array = np.array(audio_data, dtype=np.float32)
            
            # Add to buffer
            if user_id not in self._audio_buffer:
                self._audio_buffer[user_id] = []
            
            self._audio_buffer[user_id].append(audio_array)
            
            # Limit buffer size
            if len(self._audio_buffer[user_id]) > 100:  # ~3 seconds at 30fps
                self._audio_buffer[user_id] = self._audio_buffer[user_id][-50:]
            
            # Queue for processing if enough data
            if len(self._audio_buffer[user_id]) >= 10:
                await self._processing_queue.put({
                    "user_id": user_id,
                    "audio": np.concatenate(self._audio_buffer[user_id]),
                    "timestamp": timestamp
                })
                
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
    
    async def _start_speaker_detection(self, data: Dict[str, Any]) -> None:
        """Start speaker detection for new voice activity."""
        user_id = data.get("user_id", "default")
        
        # Initialize speaker detection
        self._audio_buffer[user_id] = []
        
        logger.debug(f"Started speaker detection for user {user_id}")
    
    async def _finalize_speaker_analysis(self, data: Dict[str, Any]) -> None:
        """Finalize speaker analysis when voice activity ends."""
        user_id = data.get("user_id", "default")
        
        if user_id in self._audio_buffer and self._audio_buffer[user_id]:
            # Process remaining audio
            audio_data = np.concatenate(self._audio_buffer[user_id])
            
            # Perform final speaker identification
            speaker_id, confidence = await self._identify_speaker(audio_data)
            
            if speaker_id and confidence > 0.7:
                # Emit speaker identified event
                await self._emit_speaker_identified(user_id, speaker_id, confidence)
            
            # Clear buffer
            del self._audio_buffer[user_id]
    
    async def _audio_processor(self) -> None:
        """Background audio processing worker."""
        while True:
            try:
                # Get audio data from queue
                audio_data = await self._processing_queue.get()
                
                # Process in background without blocking
                asyncio.create_task(self._process_audio_async(audio_data))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in audio processor: {e}")
    
    async def _process_audio_async(self, data: Dict[str, Any]) -> None:
        """Process audio data asynchronously."""
        try:
            user_id = data["user_id"]
            audio = data["audio"]
            
            # Extract audio features (simplified)
            features = await self._extract_audio_features(audio)
            
            # Identify speaker
            speaker_id, confidence = await self._identify_speaker(features)
            
            # Cache result
            if speaker_id and confidence > 0.5:
                cache_key = f"speaker:{user_id}:{int(time.time())}"
                self._speaker_profiles[cache_key] = {
                    "speaker_id": speaker_id,
                    "confidence": confidence,
                    "features": features.tolist() if isinstance(features, np.ndarray) else features,
                    "timestamp": time.time()
                }
                
                # Emit event if high confidence
                if confidence > 0.8:
                    await self._emit_speaker_identified(user_id, speaker_id, confidence)
                    
        except Exception as e:
            logger.error(f"Error in async audio processing: {e}")
    
    async def _extract_audio_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract audio features for speaker identification."""
        # Simplified feature extraction
        # In real implementation, use MFCC, spectral features, etc.
        
        # Basic features: RMS energy, zero crossing rate, spectral centroid
        features = []
        
        # RMS energy
        rms = np.sqrt(np.mean(audio**2))
        features.append(rms)
        
        # Zero crossing rate
        zcr = np.sum(np.diff(np.sign(audio)) != 0) / len(audio)
        features.append(zcr)
        
        # Spectral centroid (simplified)
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft)
        freqs = np.fft.fftfreq(len(audio), d=1.0/16000)  # Assuming 16kHz
        
        # Weighted average frequency
        if np.sum(magnitude) > 0:
            centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            features.append(abs(centroid))
        else:
            features.append(0)
        
        # Add more features for better identification
        features.extend([
            np.std(audio),
            np.max(audio),
            np.min(audio),
            np.mean(np.abs(audio))
        ])
        
        return np.array(features)
    
    async def _identify_speaker(self, features: np.ndarray) -> Tuple[Optional[str], float]:
        """Identify speaker from audio features."""
        # Simplified speaker identification
        # In real implementation, use ML models
        
        best_match = None
        best_confidence = 0.0
        
        # Compare with known profiles
        for speaker_id, profile in self._speaker_profiles.items():
            if isinstance(profile, dict) and "voice_features" in profile:
                stored_features = np.array(profile["voice_features"])
                
                # Simple cosine similarity
                if len(features) == len(stored_features):
                    similarity = self._cosine_similarity(features, stored_features)
                    if similarity > best_confidence:
                        best_confidence = similarity
                        best_match = speaker_id
        
        # If no good match, create new speaker ID
        if best_confidence < 0.6:
            new_speaker_id = f"speaker_{int(time.time())}"
            self._speaker_profiles[new_speaker_id] = {
                "voice_features": features.tolist(),
                "confidence": 0.5,
                "last_seen": time.time()
            }
            return new_speaker_id, 0.5
        
        return best_match, best_confidence
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def _emit_speaker_identified(self, user_id: str, speaker_id: str, confidence: float) -> None:
        """Emit speaker identified event."""
        from plugins.base_plugin import Event
        
        # This would be called from the plugin manager
        # For now, we'll log it
        logger.info(f"Speaker identified: {speaker_id} (confidence: {confidence:.2f}) for user {user_id}")
    
    async def _save_speaker_profiles(self) -> None:
        """Save speaker profiles to persistent storage."""
        # Placeholder for persistent storage
        logger.debug("Speaker profiles saved")
    
    def get_speaker_info(self, speaker_id: str) -> Optional[Dict[str, Any]]:
        """Get speaker information (synchronous access)."""
        return self._speaker_profiles.get(speaker_id)
    
    def get_recent_speakers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recently identified speakers."""
        speakers = []
        for speaker_id, profile in self._speaker_profiles.items():
            if isinstance(profile, dict) and "last_seen" in profile:
                speakers.append({
                    "speaker_id": speaker_id,
                    "last_seen": profile["last_seen"],
                    "confidence": profile.get("confidence", 0)
                })
        
        return sorted(speakers, key=lambda x: x["last_seen"], reverse=True)[:limit]