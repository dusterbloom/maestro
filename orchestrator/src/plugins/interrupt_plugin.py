"""
InterruptPlugin for voice activity detection and automatic interruption during TTS playback.
This plugin implements event-driven interrupt detection for ultra-low latency response.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional
from plugins.base_plugin import BasePlugin, PluginConfig, Event

logger = logging.getLogger(__name__)

class InterruptPlugin(BasePlugin):
    """Plugin for detecting voice activity and triggering interrupts during TTS playback"""
    
    def __init__(self, config: PluginConfig):
        super().__init__(config)
        self.voice_activity_threshold = 0.015  # Minimum audio level for voice detection
        self.interrupt_debounce_time = 0.5  # Seconds to wait before allowing another interrupt
        self.voice_duration_threshold = 0.2  # Minimum duration of voice activity to trigger interrupt
        
        # Session-specific interrupt state
        self.session_states: Dict[str, Dict[str, Any]] = {}
        
        # Reference to orchestrator for direct interrupt calls
        self.orchestrator = None
        
    async def initialize(self):
        """Initialize the interrupt plugin"""
        logger.info("🛑 InterruptPlugin initialized - ready for voice activity detection")
        
        # Note: Event handlers are registered directly with session event bus in main.py
        # when a new session is created
        
    async def cleanup(self):
        """Clean up plugin resources"""
        logger.info("🛑 InterruptPlugin cleanup")
        self.session_states.clear()
        
    def _get_session_state(self, session_id: str) -> Dict[str, Any]:
        """Get or create session state for interrupt tracking"""
        if session_id not in self.session_states:
            self.session_states[session_id] = {
                "last_interrupt_time": 0,
                "voice_activity_start": None,
                "consecutive_voice_chunks": 0,
                "interrupt_count": 0
            }
        return self.session_states[session_id]
    
    async def _handle_audio_monitor(self, event_data: Dict[str, Any]):
        """Handle continuous audio monitoring events"""
        session_id = event_data.get("session_id")
        audio_level = event_data.get("audio_level", 0)
        voice_detected = event_data.get("voice_detected", False)
        is_processing = event_data.get("is_processing", False)
        tts_active = event_data.get("tts_active", False)
        timestamp = event_data.get("timestamp", time.time())
        
        # DEBUG: Log every audio_monitor event to verify it's working
        logger.debug(f"🔍 [INTERRUPT_DEBUG] audio_monitor event: session={session_id}, level={audio_level:.4f}, voice={voice_detected}, tts_active={tts_active}, processing={is_processing}")
        
        if not session_id:
            logger.warning(f"🔍 [INTERRUPT_DEBUG] No session_id in audio_monitor event")
            return
            
        session_state = self._get_session_state(session_id)
        
        # Only monitor for interrupts during TTS playback
        if not tts_active:
            # Reset voice activity tracking when not in TTS
            if session_state["voice_activity_start"] is not None:
                logger.debug(f"🔍 [INTERRUPT_DEBUG] Resetting voice tracking - TTS not active for session {session_id}")
            session_state["voice_activity_start"] = None
            session_state["consecutive_voice_chunks"] = 0
            return
            
        # DEBUG: Log when TTS is active to verify we're monitoring
        logger.info(f"🔍 [INTERRUPT_DEBUG] TTS ACTIVE - monitoring for interrupts: session={session_id}, level={audio_level:.4f}, voice={voice_detected}, threshold={self.voice_activity_threshold}")
            
        # TEMPORARY DEBUG: Trigger interrupt on ANY audio above noise floor during TTS
        # This will help us test if the interrupt system works at all
        if audio_level > 0.000010:  # Any audio above noise floor
            logger.info(f"🚨 [INTERRUPT_TEST] FORCING INTERRUPT! level={audio_level:.6f} > 0.000010")
            # Force trigger interrupt to test the system
            await self._trigger_interrupt(session_id, audio_level, timestamp)
        
        # Track voice activity during TTS
        if voice_detected and audio_level > self.voice_activity_threshold:
            # Start tracking voice activity
            if session_state["voice_activity_start"] is None:
                session_state["voice_activity_start"] = timestamp
                session_state["consecutive_voice_chunks"] = 1
                logger.info(f"🔍 [INTERRUPT_DEBUG] 🎤 Voice activity STARTED for session {session_id}, level={audio_level:.4f} > threshold={self.voice_activity_threshold}")
            else:
                session_state["consecutive_voice_chunks"] += 1
                
            # Check if voice activity has lasted long enough to trigger interrupt
            voice_duration = timestamp - session_state["voice_activity_start"]
            logger.info(f"🔍 [INTERRUPT_DEBUG] Voice duration: {voice_duration:.3f}s, threshold: {self.voice_duration_threshold}s")
            if voice_duration >= self.voice_duration_threshold:
                logger.info(f"🔍 [INTERRUPT_DEBUG] Voice duration reached threshold - TRIGGERING INTERRUPT!")
                await self._trigger_interrupt(session_id, audio_level, timestamp)
                
        else:
            # Reset voice activity tracking when voice stops
            if session_state["voice_activity_start"] is not None:
                logger.debug(f"🔍 [INTERRUPT_DEBUG] 🎤 Voice activity ENDED for session {session_id} (level={audio_level:.4f}, voice_detected={voice_detected})")
                session_state["voice_activity_start"] = None
                session_state["consecutive_voice_chunks"] = 0
            elif tts_active and audio_level > 0.005:  # Log when we have audio but no voice detection during TTS
                logger.debug(f"🔍 [INTERRUPT_DEBUG] Audio detected during TTS but no voice: level={audio_level:.4f}, voice={voice_detected}")
    
    async def _handle_voice_during_tts(self, event_data: Dict[str, Any]):
        """Handle voice detection during TTS events"""
        session_id = event_data.get("session_id")
        audio_level = event_data.get("audio_level", 0)
        timestamp = event_data.get("timestamp", time.time())
        
        logger.info(f"🔍 [INTERRUPT_DEBUG] 🗣️ voice_during_tts event received: session={session_id}, level={audio_level:.4f}")
        
        if not session_id:
            logger.warning(f"🔍 [INTERRUPT_DEBUG] No session_id in voice_during_tts event")
            return
            
        # This is a higher-level event that indicates definite voice activity during TTS
        # Use this for more aggressive interrupt detection
        logger.info(f"🔍 [INTERRUPT_DEBUG] Triggering interrupt from voice_during_tts event")
        await self._trigger_interrupt(session_id, audio_level, timestamp)
    
    async def _trigger_interrupt(self, session_id: str, audio_level: float, timestamp: float):
        """Trigger an interrupt if conditions are met"""
        session_state = self._get_session_state(session_id)
        
        # Check debounce time
        time_since_last_interrupt = timestamp - session_state["last_interrupt_time"]
        if time_since_last_interrupt < self.interrupt_debounce_time:
            # This is now a debug message because it's a low-level detail
            logger.debug(f"⏰ Interrupt debounced for session {session_id}")
            return
            
        # Update interrupt state
        session_state["last_interrupt_time"] = timestamp
        session_state["interrupt_count"] += 1
        
        logger.info(f"🛑 INTERRUPT TRIGGERED for session {session_id} due to voice activity.")
        
        # --- START OF CORRECTION ---
        # Directly call the orchestrator's main interrupt method for an immediate stop.
        if self.orchestrator:
            try:
                logger.info(f"🚨 [INTERRUPT_DEBUG] CALLING orchestrator.interrupt_session({session_id}) NOW!")
                # This is the "big red button". It kills the backend pipeline instantly.
                await self.orchestrator.interrupt_session(session_id)
                logger.info(f"✅ [INTERRUPT_DEBUG] orchestrator.interrupt_session({session_id}) COMPLETED!")
            except Exception as e:
                logger.error(f"❌ [INTERRUPT_DEBUG] Error triggering interrupt: {e}")
                import traceback
                logger.error(f"❌ [INTERRUPT_DEBUG] Traceback: {traceback.format_exc()}")
        else:
            logger.error(f"❌ [INTERRUPT_DEBUG] NO ORCHESTRATOR REFERENCE! self.orchestrator={self.orchestrator}")
            logger.error(f"❌ [INTERRUPT_DEBUG] This is why interrupts don't work!")
        # --- END OF CORRECTION ---
        
        # Reset voice activity tracking
        session_state["voice_activity_start"] = None
        session_state["consecutive_voice_chunks"] = 0


    async def _handle_interrupted(self, event_data: Dict[str, Any]):
        """Handle interrupt acknowledgment"""
        session_id = event_data.get("session_id")
        if session_id and session_id in self.session_states:
            # Reset voice activity state after successful interrupt
            session_state = self.session_states[session_id]
            session_state["voice_activity_start"] = None
            session_state["consecutive_voice_chunks"] = 0
            logger.info(f"✅ Interrupt acknowledged for session {session_id}")
    
    async def _handle_processing_complete(self, event_data: Dict[str, Any]):
        """Handle processing completion - reset interrupt state"""
        session_id = event_data.get("session_id")
        if session_id and session_id in self.session_states:
            # Reset interrupt state when processing completes
            session_state = self.session_states[session_id]
            session_state["voice_activity_start"] = None
            session_state["consecutive_voice_chunks"] = 0
            logger.debug(f"🔄 Reset interrupt state for session {session_id}")
    
    async def process_event(self, event: Event) -> None:
        """Process an event. Must be non-blocking."""
        # The event processing is handled by the event_bus system
        # This method is required by BasePlugin but not used directly
        pass
    
    async def shutdown(self) -> None:
        """Shutdown the plugin gracefully."""
        await self.cleanup()
    
    async def handle_event(self, event_type: str, data: Dict[str, Any]):
        """Handle plugin events"""
        # This is already handled by the registered event handlers
        pass