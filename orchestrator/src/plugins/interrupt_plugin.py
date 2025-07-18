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
        
        # Register event handlers
        self.register_event_handler("audio_monitor", self._handle_audio_monitor)
        self.register_event_handler("voice_during_tts", self._handle_voice_during_tts)
        self.register_event_handler("interrupted", self._handle_interrupted)
        self.register_event_handler("processing_complete", self._handle_processing_complete)
        
        self.is_running = True
        
    async def cleanup(self):
        """Clean up plugin resources"""
        logger.info("🛑 InterruptPlugin cleanup")
        self.session_states.clear()
        self.is_running = False
        
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
        
        if not session_id:
            return
            
        session_state = self._get_session_state(session_id)
        
        # Only monitor for interrupts during TTS playback
        if not tts_active:
            # Reset voice activity tracking when not in TTS
            session_state["voice_activity_start"] = None
            session_state["consecutive_voice_chunks"] = 0
            return
            
        # Track voice activity during TTS
        if voice_detected and audio_level > self.voice_activity_threshold:
            # Start tracking voice activity
            if session_state["voice_activity_start"] is None:
                session_state["voice_activity_start"] = timestamp
                session_state["consecutive_voice_chunks"] = 1
                logger.debug(f"🎤 Voice activity started for session {session_id}")
            else:
                session_state["consecutive_voice_chunks"] += 1
                
            # Check if voice activity has lasted long enough to trigger interrupt
            voice_duration = timestamp - session_state["voice_activity_start"]
            if voice_duration >= self.voice_duration_threshold:
                await self._trigger_interrupt(session_id, audio_level, timestamp)
                
        else:
            # Reset voice activity tracking when voice stops
            if session_state["voice_activity_start"] is not None:
                logger.debug(f"🎤 Voice activity ended for session {session_id}")
                session_state["voice_activity_start"] = None
                session_state["consecutive_voice_chunks"] = 0
    
    async def _handle_voice_during_tts(self, event_data: Dict[str, Any]):
        """Handle voice detection during TTS events"""
        session_id = event_data.get("session_id")
        audio_level = event_data.get("audio_level", 0)
        timestamp = event_data.get("timestamp", time.time())
        
        if not session_id:
            return
            
        # This is a higher-level event that indicates definite voice activity during TTS
        # Use this for more aggressive interrupt detection
        await self._trigger_interrupt(session_id, audio_level, timestamp)
    
    async def _trigger_interrupt(self, session_id: str, audio_level: float, timestamp: float):
        """Trigger an interrupt if conditions are met"""
        session_state = self._get_session_state(session_id)
        
        # Check debounce time
        time_since_last_interrupt = timestamp - session_state["last_interrupt_time"]
        if time_since_last_interrupt < self.interrupt_debounce_time:
            logger.debug(f"⏰ Interrupt debounced for session {session_id}")
            return
            
        # Update interrupt state
        session_state["last_interrupt_time"] = timestamp
        session_state["interrupt_count"] += 1
        
        logger.info(f"🛑 Triggering interrupt for session {session_id} (audio_level: {audio_level})")
        
        # Directly call orchestrator interrupt method for ultra-low latency
        if self.orchestrator:
            try:
                await self.orchestrator.interrupt_session(session_id)
            except Exception as e:
                logger.error(f"Error triggering interrupt: {e}")
        else:
            logger.warning("No orchestrator reference available for interrupt")
        
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
    
    async def process_event(self, event: "Event") -> None:
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