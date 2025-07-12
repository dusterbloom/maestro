"""
VoiceActivityProcessor - GenAI Processors integration for real-time barge-in detection

This processor handles voice activity detection during TTS playback to enable
real-time interruption capabilities. It monitors audio input and signals
the processor chain to halt and restart when user speech is detected.
"""

import asyncio
import logging
import numpy as np
import time
from typing import AsyncIterable, Callable, Dict, Optional, Set

from genai_processors import content_api
from genai_processors import processor

from .config import config, VoiceMetadata


logger = logging.getLogger(__name__)


class VoiceActivityProcessor(processor.Processor):
    """
    Monitors audio input for voice activity to enable real-time barge-in.
    
    Features:
    - Real-time voice activity detection during TTS playback
    - Energy-based VAD with dynamic threshold adjustment
    - Configurable sensitivity and timing parameters
    - Integration with existing processor chain for interruption
    - Minimal latency impact on audio processing
    - Robust false positive filtering
    """
    
    def __init__(
        self,
        session_id: str = "default",
        energy_threshold: float = None,
        dynamic_threshold: bool = None,
        pause_threshold: float = None,
        min_speech_duration: float = 0.3,
        min_silence_duration: float = 0.5,
        sample_rate: int = 16000,
        frame_duration: float = 0.02,  # 20ms frames
        interrupt_callback: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Configuration
        self.session_id = session_id
        self.energy_threshold = energy_threshold or config.VAD_ENERGY_THRESHOLD
        self.dynamic_threshold = dynamic_threshold if dynamic_threshold is not None else config.VAD_DYNAMIC_THRESHOLD
        self.pause_threshold = pause_threshold or config.VAD_PAUSE_THRESHOLD
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        
        # Audio configuration
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration)
        
        # VAD state
        self.is_speech_active = False
        self.speech_start_time: Optional[float] = None
        self.last_speech_time: Optional[float] = None
        self.background_energy = 0.0
        self.adaptive_threshold = self.energy_threshold
        
        # Processing state
        self.monitoring_active = False
        self.tts_playing = False
        self.interrupt_callback = interrupt_callback
        self.interruption_count = 0
        
        # Audio buffer for analysis
        self.audio_buffer = np.array([], dtype=np.float32)
        self.energy_history = []
        self.max_history_length = 50  # ~1 second of energy history
        
        # Performance metrics
        self.frames_processed = 0
        self.false_positives = 0
        self.successful_interruptions = 0
        
        logger.info(f"VoiceActivityProcessor initialized for session {session_id}")
        logger.info(f"Energy threshold: {self.energy_threshold}, Dynamic: {self.dynamic_threshold}")
    
    async def call(self, content: AsyncIterable[content_api.ProcessorPart]) -> AsyncIterable[content_api.ProcessorPart]:
        """
        Monitor audio input stream for voice activity and trigger interruptions.
        
        Args:
            content: Stream of audio ProcessorParts to monitor
            
        Yields:
            Original ProcessorParts (pass-through) with VAD metadata added
        """
        try:
            async for part in content:
                # Process audio parts for VAD
                if self._is_audio_part(part):
                    vad_result = await self._process_audio_for_vad(part)
                    
                    # Add VAD metadata to the part
                    if part.metadata is None:
                        part.metadata = {}
                    
                    part.metadata.update({
                        "vad_energy": vad_result["energy"],
                        "vad_speech_detected": vad_result["speech_detected"],
                        "vad_threshold": vad_result["threshold"],
                        "vad_monitoring": self.monitoring_active
                    })
                    
                    # Check for interruption condition
                    if (self.monitoring_active and 
                        self.tts_playing and 
                        vad_result["speech_detected"] and
                        await self._validate_interruption()):
                        
                        await self._trigger_interruption()
                
                # Monitor TTS state from pipeline
                elif self._is_tts_audio_part(part):
                    self.tts_playing = True
                    self.monitoring_active = config.ENABLE_INTERRUPTION
                
                # Pass through the part unchanged
                yield part
                
        except Exception as e:
            logger.error(f"Error in VoiceActivityProcessor: {e}")
    
    def _is_audio_part(self, part: content_api.ProcessorPart) -> bool:
        """Check if ProcessorPart contains audio input data."""
        return (part.mimetype and 
                part.mimetype.startswith("audio/") and
                part.metadata and
                part.metadata.get("content_type") == VoiceMetadata.AUDIO_INPUT)
    
    def _is_tts_audio_part(self, part: content_api.ProcessorPart) -> bool:
        """Check if ProcessorPart contains TTS audio output."""
        return (part.metadata and
                part.metadata.get("content_type") == VoiceMetadata.TTS_AUDIO)
    
    async def _process_audio_for_vad(self, part: content_api.ProcessorPart) -> Dict:
        """Process audio data for voice activity detection."""
        try:
            # Extract audio data
            audio_data = self._extract_audio_data(part)
            if audio_data is None:
                return {"energy": 0.0, "speech_detected": False, "threshold": self.adaptive_threshold}
            
            # Convert to numpy array if needed
            if isinstance(audio_data, bytes):
                # Assuming 16-bit PCM
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio_array = np.array(audio_data, dtype=np.float32)
            
            # Add to buffer
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_array])
            
            # Process frames
            vad_results = []
            while len(self.audio_buffer) >= self.frame_size:
                frame = self.audio_buffer[:self.frame_size]
                self.audio_buffer = self.audio_buffer[self.frame_size:]
                
                frame_result = self._process_frame(frame)
                vad_results.append(frame_result)
                self.frames_processed += 1
            
            # Return aggregated result
            if vad_results:
                avg_energy = np.mean([r["energy"] for r in vad_results])
                speech_detected = any(r["speech_detected"] for r in vad_results)
                return {
                    "energy": avg_energy,
                    "speech_detected": speech_detected,
                    "threshold": self.adaptive_threshold
                }
            else:
                return {"energy": 0.0, "speech_detected": False, "threshold": self.adaptive_threshold}
                
        except Exception as e:
            logger.error(f"Error processing audio for VAD: {e}")
            return {"energy": 0.0, "speech_detected": False, "threshold": self.adaptive_threshold}
    
    def _extract_audio_data(self, part: content_api.ProcessorPart):
        """Extract audio data from ProcessorPart."""
        try:
            if hasattr(part, 'data') and isinstance(part.data, (bytes, np.ndarray)):
                return part.data
            elif hasattr(part, 'content') and isinstance(part.content, (bytes, np.ndarray)):
                return part.content
            elif hasattr(part, 'audio_bytes'):
                return part.audio_bytes
            else:
                logger.warning(f"Could not extract audio data from ProcessorPart: {type(part)}")
                return None
        except Exception as e:
            logger.error(f"Error extracting audio data: {e}")
            return None
    
    def _process_frame(self, frame: np.ndarray) -> Dict:
        """Process a single audio frame for VAD."""
        # Calculate frame energy (RMS)
        energy = np.sqrt(np.mean(frame ** 2))
        
        # Update energy history
        self.energy_history.append(energy)
        if len(self.energy_history) > self.max_history_length:
            self.energy_history.pop(0)
        
        # Update background energy and adaptive threshold
        if self.dynamic_threshold:
            self._update_adaptive_threshold(energy)
        
        # Determine if speech is present
        speech_detected = energy > self.adaptive_threshold
        
        # Update speech state
        current_time = time.time()
        if speech_detected:
            if not self.is_speech_active:
                self.speech_start_time = current_time
                self.is_speech_active = True
            self.last_speech_time = current_time
        else:
            # Check for end of speech
            if (self.is_speech_active and 
                self.last_speech_time and
                current_time - self.last_speech_time > self.min_silence_duration):
                self.is_speech_active = False
                self.speech_start_time = None
        
        return {
            "energy": energy,
            "speech_detected": speech_detected and self.is_speech_active,
            "threshold": self.adaptive_threshold,
            "background_energy": self.background_energy
        }
    
    def _update_adaptive_threshold(self, current_energy: float):
        """Update adaptive threshold based on background noise."""
        if len(self.energy_history) < 10:
            return  # Need some history first
        
        # Calculate background energy as median of recent low-energy frames
        sorted_energies = sorted(self.energy_history)
        low_energy_frames = sorted_energies[:len(sorted_energies)//3]  # Bottom third
        
        if low_energy_frames:
            self.background_energy = np.median(low_energy_frames)
            
            # Set adaptive threshold as multiple of background energy
            noise_factor = 3.0  # Adjust based on sensitivity needs
            self.adaptive_threshold = max(
                self.background_energy * noise_factor,
                self.energy_threshold * 0.5  # Minimum threshold
            )
    
    async def _validate_interruption(self) -> bool:
        """Validate that detected speech justifies interruption."""
        current_time = time.time()
        
        # Check minimum speech duration
        if (self.speech_start_time and 
            current_time - self.speech_start_time < self.min_speech_duration):
            return False
        
        # Additional validation: ensure energy is significantly above background
        if len(self.energy_history) >= 3:
            recent_energy = np.mean(self.energy_history[-3:])
            if recent_energy < self.background_energy * 4.0:  # Conservative threshold
                self.false_positives += 1
                return False
        
        return True
    
    async def _trigger_interruption(self):
        """Trigger interruption of the processor chain."""
        try:
            self.interruption_count += 1
            self.successful_interruptions += 1
            
            logger.info(f"Session {self.session_id}: Voice activity detected - triggering interruption #{self.interruption_count}")
            
            # Reset TTS state
            self.tts_playing = False
            self.monitoring_active = False
            
            # Call interrupt callback if provided
            if self.interrupt_callback:
                try:
                    if asyncio.iscoroutinefunction(self.interrupt_callback):
                        await self.interrupt_callback()
                    else:
                        self.interrupt_callback()
                except Exception as e:
                    logger.error(f"Error calling interrupt callback: {e}")
            
            # Reset speech state
            self.is_speech_active = False
            self.speech_start_time = None
            
            logger.debug(f"Session {self.session_id}: Interruption processing complete")
            
        except Exception as e:
            logger.error(f"Error triggering interruption: {e}")
    
    def set_monitoring_state(self, active: bool, tts_playing: bool = False):
        """Manually set monitoring state (called by orchestrator)."""
        self.monitoring_active = active
        self.tts_playing = tts_playing
        
        if not active:
            # Reset speech state when monitoring stops
            self.is_speech_active = False
            self.speech_start_time = None
        
        logger.debug(f"Session {self.session_id}: VAD monitoring={active}, TTS playing={tts_playing}")
    
    def update_sensitivity(self, energy_threshold: float = None, pause_threshold: float = None):
        """Update VAD sensitivity parameters."""
        if energy_threshold is not None:
            self.energy_threshold = energy_threshold
            logger.info(f"Session {self.session_id}: Energy threshold updated to {energy_threshold}")
        
        if pause_threshold is not None:
            self.pause_threshold = pause_threshold
            logger.info(f"Session {self.session_id}: Pause threshold updated to {pause_threshold}")
    
    def reset_state(self):
        """Reset VAD state (useful for session cleanup)."""
        self.is_speech_active = False
        self.speech_start_time = None
        self.last_speech_time = None
        self.monitoring_active = False
        self.tts_playing = False
        self.audio_buffer = np.array([], dtype=np.float32)
        self.energy_history.clear()
        
        logger.info(f"Session {self.session_id}: VAD state reset")
    
    def get_metrics(self) -> Dict:
        """Get VAD performance metrics."""
        current_time = time.time()
        
        return {
            "session_id": self.session_id,
            "monitoring_active": self.monitoring_active,
            "tts_playing": self.tts_playing,
            "is_speech_active": self.is_speech_active,
            
            # Timing metrics
            "speech_duration": (current_time - self.speech_start_time) if self.speech_start_time else 0,
            "time_since_last_speech": (current_time - self.last_speech_time) if self.last_speech_time else None,
            
            # Processing metrics
            "frames_processed": self.frames_processed,
            "interruption_count": self.interruption_count,
            "successful_interruptions": self.successful_interruptions,
            "false_positives": self.false_positives,
            
            # VAD configuration
            "energy_threshold": self.energy_threshold,
            "adaptive_threshold": self.adaptive_threshold,
            "background_energy": self.background_energy,
            "dynamic_threshold": self.dynamic_threshold,
            
            # Audio configuration
            "sample_rate": self.sample_rate,
            "frame_size": self.frame_size,
            "frame_duration": self.frame_duration,
            
            # Buffer state
            "audio_buffer_size": len(self.audio_buffer),
            "energy_history_length": len(self.energy_history),
            "recent_energy": np.mean(self.energy_history[-5:]) if len(self.energy_history) >= 5 else 0
        }
    
    def get_current_state(self) -> Dict:
        """Get current VAD state for debugging."""
        return {
            "speech_active": self.is_speech_active,
            "monitoring": self.monitoring_active,
            "tts_playing": self.tts_playing,
            "energy_threshold": self.adaptive_threshold,
            "background_energy": self.background_energy,
            "recent_energies": self.energy_history[-10:] if len(self.energy_history) >= 10 else self.energy_history
        }
