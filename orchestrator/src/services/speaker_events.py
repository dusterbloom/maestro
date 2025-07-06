"""
Agentic Speaker Event System
Handles magical speaker recognition events and triggers appropriate responses
"""
import asyncio
import logging
from typing import Dict, Callable, Any
from services.voice_service import SpeakerEvent

logger = logging.getLogger(__name__)

class AgenticSpeakerSystem:
    """Manages agentic responses to speaker recognition events"""
    
    def __init__(self, voice_service, memory_service=None):
        self.voice_service = voice_service
        self.memory_service = memory_service
        
        # Register event handlers
        self._register_event_handlers()
        
    def _register_event_handlers(self):
        """Register all agentic event handlers"""
        
        @self.voice_service.on_speaker_event("speaker_identified")
        async def on_speaker_identified(event: SpeakerEvent):
            """Handle magical speaker recognition"""
            logger.info(f"🎭 MAGICAL RECOGNITION: {event.context.get('name')} identified with {event.confidence:.2f} confidence")
            
            # Store recognition event in memory if available
            if self.memory_service:
                await self._store_recognition_event(event)
                
            # Trigger personalized response
            await self._trigger_personalized_greeting(event)
        
        @self.voice_service.on_speaker_event("speaker_registered")
        async def on_speaker_registered(event: SpeakerEvent):
            """Handle new speaker registration"""
            logger.info(f"🆕 NEW SPEAKER REGISTERED: {event.user_id}")
            
            # Store registration in memory
            if self.memory_service:
                await self._store_registration_event(event)
                
            # Trigger welcome sequence
            await self._trigger_welcome_sequence(event)
        
        @self.voice_service.on_speaker_event("confidence_low")
        async def on_confidence_low(event: SpeakerEvent):
            """Handle uncertain speaker recognition"""
            logger.warning(f"⚠️ LOW CONFIDENCE: {event.confidence:.2f} (threshold: {event.context.get('threshold')})")
            
            # Trigger uncertainty response
            await self._trigger_uncertainty_response(event)
    
    async def _store_recognition_event(self, event: SpeakerEvent):
        """Store recognition event in memory system"""
        try:
            if self.memory_service:
                # Create memory entry for recognition
                memory_data = {
                    "event_type": "speaker_recognition",
                    "user_id": event.user_id,
                    "confidence": event.confidence,
                    "timestamp": event.timestamp,
                    "session_id": event.session_id,
                    "recognition_type": "magical"
                }
                
                # Store in user's profile
                await self.memory_service.add_user_memory(event.user_id, memory_data)
                logger.info(f"Stored recognition event for {event.user_id}")
                
        except Exception as e:
            logger.error(f"Error storing recognition event: {e}")
    
    async def _store_registration_event(self, event: SpeakerEvent):
        """Store registration event in memory system"""
        try:
            if self.memory_service:
                # Create memory entry for registration
                memory_data = {
                    "event_type": "speaker_registration",
                    "user_id": event.user_id,
                    "timestamp": event.timestamp,
                    "session_id": event.session_id,
                    "is_first_speaker": event.context.get("is_first_speaker", False)
                }
                
                # Store in user's profile
                await self.memory_service.add_user_memory(event.user_id, memory_data)
                logger.info(f"Stored registration event for {event.user_id}")
                
        except Exception as e:
            logger.error(f"Error storing registration event: {e}")
    
    async def _trigger_personalized_greeting(self, event: SpeakerEvent):
        """Trigger personalized greeting for recognized speaker"""
        try:
            name = event.context.get("name", "Friend")
            
            # Enhanced personalized context based on confidence
            if event.confidence > 0.95:
                greeting_context = f"(Speaking with absolute certainty) Welcome back {name}! "
            elif event.confidence > 0.9:
                greeting_context = f"Hello {name}! Great to see you again. "
            else:
                greeting_context = f"Hi {name}! "
            
            # Store context for next response
            self._current_speaker_context = greeting_context
            
            logger.info(f"Set personalized greeting context: {greeting_context}")
            
        except Exception as e:
            logger.error(f"Error triggering personalized greeting: {e}")
    
    async def _trigger_welcome_sequence(self, event: SpeakerEvent):
        """Trigger welcome sequence for new speaker"""
        try:
            if event.context.get("is_first_speaker"):
                welcome_context = "Hello! I don't think we've met before. What should I call you? "
            else:
                welcome_context = "Welcome! "
            
            # Store context for next response
            self._current_speaker_context = welcome_context
            
            logger.info(f"Set welcome context: {welcome_context}")
            
        except Exception as e:
            logger.error(f"Error triggering welcome sequence: {e}")
    
    async def _trigger_uncertainty_response(self, event: SpeakerEvent):
        """Trigger response for uncertain recognition"""
        try:
            uncertainty_context = "I'm not completely sure I recognize your voice. "
            
            # Store context for next response
            self._current_speaker_context = uncertainty_context
            
            logger.info(f"Set uncertainty context: {uncertainty_context}")
            
        except Exception as e:
            logger.error(f"Error triggering uncertainty response: {e}")
    
    def get_current_speaker_context(self) -> str:
        """Get current speaker context for response generation"""
        context = getattr(self, '_current_speaker_context', "")
        # Clear after use
        self._current_speaker_context = ""
        return context
    
    async def handle_name_learning(self, name: str, session_id: str):
        """Handle when speaker tells us their name"""
        try:
            # Update speaker name
            success = await self.voice_service.update_speaker_name(name)
            
            if success:
                logger.info(f"Learned speaker name: {name}")
                
                # Store name learning event
                if self.memory_service:
                    registered_speaker = self.voice_service.get_registered_speaker()
                    if registered_speaker:
                        memory_data = {
                            "event_type": "name_learned",
                            "user_id": registered_speaker["user_id"],
                            "name": name,
                            "timestamp": asyncio.get_event_loop().time(),
                            "session_id": session_id
                        }
                        await self.memory_service.add_user_memory(registered_speaker["user_id"], memory_data)
                
                return f"Nice to meet you, {name}! I'll remember your voice. "
            else:
                return "I'm sorry, I couldn't update your name right now. "
                
        except Exception as e:
            logger.error(f"Error handling name learning: {e}")
            return "I'm sorry, there was an error learning your name. "