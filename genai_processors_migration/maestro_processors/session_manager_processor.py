"""
SessionManagerProcessor - GenAI Processors integration for session and conversation management

This processor maintains conversation context, session lifecycle, and provides
coordination between other processors in the chain. It preserves the existing
session management capabilities while adapting to the GenAI Processors framework.
"""

import asyncio
import logging
import time
from typing import AsyncIterable, Dict, List, Optional, Set

from genai_processors import content_api
from genai_processors import processor

from .config import config, VoiceMetadata


logger = logging.getLogger(__name__)


class SessionManagerProcessor(processor.Processor):
    """
    Manages conversation context and session state for voice interactions.
    
    Features:
    - Conversation history tracking and management
    - Session lifecycle management
    - Cross-processor coordination and state sharing
    - Performance metrics collection
    - Resource cleanup and optimization
    - Support for multiple concurrent sessions
    """
    
    def __init__(
        self,
        session_id: str = "default",
        max_history_length: int = 20,
        session_timeout: float = 3600,  # 1 hour
        enable_memory: bool = None,
        memory_service_url: str = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Session configuration
        self.session_id = session_id
        self.max_history_length = max_history_length
        self.session_timeout = session_timeout
        self.enable_memory = enable_memory if enable_memory is not None else config.MEMORY_ENABLED
        self.memory_service_url = memory_service_url or config.AMEM_URL
        
        # Session state
        self.created_at = time.time()
        self.last_activity = time.time()
        self.is_active = True
        
        # Conversation tracking
        self.conversation_history: List[Dict] = []
        self.current_exchange: Dict = {}
        self.exchange_count = 0
        
        # Processing state
        self.current_transcript = ""
        self.current_llm_response = ""
        self.processing_stage = "idle"  # idle, stt, llm, tts
        self.last_interrupted_text: Optional[str] = None
        
        # Performance metrics
        self.total_exchanges = 0
        self.total_processing_time = 0.0
        self.average_latency = 0.0
        self.pipeline_metrics: Dict = {}
        
        # Processor coordination
        self.processor_states: Dict[str, Dict] = {}
        self.interrupt_event = asyncio.Event()
        
        logger.info(f"SessionManagerProcessor initialized for session {session_id}")
        logger.info(f"Memory enabled: {self.enable_memory}, Max history: {max_history_length}")
    
    async def call(self, content: AsyncIterable[content_api.ProcessorPart]) -> AsyncIterable[content_api.ProcessorPart]:
        """
        Process and coordinate content flow between processors.
        
        Args:
            content: Stream of ProcessorParts from various pipeline stages
            
        Yields:
            Enhanced ProcessorParts with session context and coordination metadata
        """
        try:
            async for part in content:
                # Update activity timestamp
                self.last_activity = time.time()
                
                # Process based on content type and stage
                enhanced_part = await self._process_pipeline_part(part)
                
                # Update session state based on part
                await self._update_session_state(enhanced_part)
                
                # Check for session timeout
                if self._is_session_expired():
                    logger.warning(f"Session {self.session_id} expired, marking for cleanup")
                    self.is_active = False
                
                # Yield the enhanced part
                yield enhanced_part
                
        except Exception as e:
            logger.error(f"Error in SessionManagerProcessor: {e}")
    
    async def _process_pipeline_part(self, part: content_api.ProcessorPart) -> content_api.ProcessorPart:
        """Process and enhance a ProcessorPart with session context."""
        try:
            # Ensure metadata exists
            if part.metadata is None:
                part.metadata = {}
            
            # Add session context to metadata
            part.metadata.update({
                "session_id": self.session_id,
                "session_created_at": self.created_at,
                "session_last_activity": self.last_activity,
                "session_exchange_count": self.exchange_count,
                "session_total_exchanges": self.total_exchanges,
                "session_processing_stage": self.processing_stage
            })
            
            # Process based on content type
            content_type = part.metadata.get("content_type")
            stage = part.metadata.get("stage")
            
            if content_type == VoiceMetadata.TRANSCRIPT and stage == VoiceMetadata.STAGE_STT:
                await self._handle_transcript_part(part)
            elif content_type == VoiceMetadata.LLM_TEXT and stage == VoiceMetadata.STAGE_LLM:
                await self._handle_llm_part(part)
            elif content_type == VoiceMetadata.TTS_AUDIO and stage == VoiceMetadata.STAGE_TTS:
                await self._handle_tts_part(part)
            
            # Add pipeline performance metrics
            if self.pipeline_metrics:
                part.metadata["pipeline_metrics"] = self.pipeline_metrics.copy()
            
            return part
            
        except Exception as e:
            logger.error(f"Error processing pipeline part: {e}")
            return part
    
    async def _handle_transcript_part(self, part: content_api.ProcessorPart):
        """Handle transcript ProcessorPart."""
        try:
            if part.metadata.get("is_complete", False):
                transcript_text = str(part.content)
                
                # Check for duplicate or interrupted text
                if (self.last_interrupted_text and 
                    transcript_text.strip() == self.last_interrupted_text.strip()):
                    logger.info(f"Session {self.session_id}: Ignoring previously interrupted text")
                    self.last_interrupted_text = None
                    part.metadata["ignored"] = True
                    return
                
                # Start new exchange
                self.current_exchange = {
                    "exchange_id": self.exchange_count,
                    "user_input": transcript_text,
                    "timestamp": time.time(),
                    "stt_latency": part.metadata.get("stt_latency_ms", 0),
                    "stage": "transcript_complete"
                }
                
                self.current_transcript = transcript_text
                self.processing_stage = "llm"
                
                logger.info(f"Session {self.session_id}: Transcript complete - {transcript_text}")
                
            else:
                # Live transcript update
                self.current_transcript = str(part.content)
                part.metadata["live_transcript"] = True
                
        except Exception as e:
            logger.error(f"Error handling transcript part: {e}")
    
    async def _handle_llm_part(self, part: content_api.ProcessorPart):
        """Handle LLM text ProcessorPart."""
        try:
            if part.metadata.get("is_complete", False):
                llm_text = str(part.content)
                
                # Add to current response
                if "llm_response" not in self.current_exchange:
                    self.current_exchange["llm_response"] = ""
                
                self.current_exchange["llm_response"] += llm_text
                self.current_llm_response += llm_text
                
                # Update exchange with LLM metrics
                self.current_exchange.update({
                    "llm_sentence_latency": part.metadata.get("llm_sentence_latency_ms", 0),
                    "llm_ttft": part.metadata.get("llm_ttft_ms", 0),
                    "stage": "llm_sentence_complete"
                })
                
                self.processing_stage = "tts"
                
                logger.debug(f"Session {self.session_id}: LLM sentence complete - {llm_text}")
                
                # Check if this is the final sentence
                if part.metadata.get("is_final", False):
                    self.current_exchange["stage"] = "llm_complete"
                    logger.info(f"Session {self.session_id}: LLM response complete")
                
        except Exception as e:
            logger.error(f"Error handling LLM part: {e}")
    
    async def _handle_tts_part(self, part: content_api.ProcessorPart):
        """Handle TTS audio ProcessorPart."""
        try:
            if part.metadata.get("is_complete", False):
                # Update exchange with TTS metrics
                self.current_exchange.update({
                    "tts_latency": part.metadata.get("tts_latency_ms", 0),
                    "stage": "tts_complete"
                })
                
                self.processing_stage = "output"
                
                # Check if this completes the exchange
                sequence = part.metadata.get("sequence_number", 0)
                logger.debug(f"Session {self.session_id}: TTS audio complete for sequence {sequence}")
                
                # If this is the final audio part, complete the exchange
                if self._is_exchange_complete():
                    await self._complete_exchange()
                
        except Exception as e:
            logger.error(f"Error handling TTS part: {e}")
    
    def _is_exchange_complete(self) -> bool:
        """Check if the current exchange is complete."""
        return (self.current_exchange.get("stage") == "tts_complete" and
                "llm_response" in self.current_exchange and
                "user_input" in self.current_exchange)
    
    async def _complete_exchange(self):
        """Complete the current conversation exchange."""
        try:
            # Calculate total exchange latency
            if "timestamp" in self.current_exchange:
                total_latency = (time.time() - self.current_exchange["timestamp"]) * 1000
                self.current_exchange["total_latency_ms"] = total_latency
                
                # Update performance metrics
                self.total_processing_time += total_latency / 1000
                self.total_exchanges += 1
                self.average_latency = self.total_processing_time / self.total_exchanges
            
            # Store in conversation history
            self.conversation_history.append(self.current_exchange.copy())
            
            # Trim history if needed
            if len(self.conversation_history) > self.max_history_length:
                self.conversation_history = self.conversation_history[-self.max_history_length:]
            
            # Store in external memory if enabled
            if self.enable_memory:
                await self._store_in_memory(self.current_exchange)
            
            # Log completion
            logger.info(f"Session {self.session_id}: Exchange {self.exchange_count} complete")
            logger.info(f"Total latency: {self.current_exchange.get('total_latency_ms', 0):.2f}ms")
            
            # Reset for next exchange
            self.exchange_count += 1
            self.current_exchange = {}
            self.current_transcript = ""
            self.current_llm_response = ""
            self.processing_stage = "idle"
            
        except Exception as e:
            logger.error(f"Error completing exchange: {e}")
    
    async def _store_in_memory(self, exchange: Dict):
        """Store exchange in external memory service (A-MEM)."""
        try:
            if not self.memory_service_url:
                return
            
            # Prepare memory payload
            memory_data = {
                "session_id": self.session_id,
                "user_input": exchange.get("user_input", ""),
                "ai_response": exchange.get("llm_response", ""),
                "timestamp": exchange.get("timestamp", time.time()),
                "exchange_id": exchange.get("exchange_id", 0),
                "metadata": {
                    "total_latency_ms": exchange.get("total_latency_ms", 0),
                    "stt_latency": exchange.get("stt_latency", 0),
                    "llm_ttft": exchange.get("llm_ttft", 0),
                    "tts_latency": exchange.get("tts_latency", 0)
                }
            }
            
            # Note: Actual HTTP call to memory service would go here
            # For now, just log the intent
            logger.debug(f"Session {self.session_id}: Would store exchange in memory: {memory_data}")
            
        except Exception as e:
            logger.error(f"Error storing in memory: {e}")
    
    async def _update_session_state(self, part: content_api.ProcessorPart):
        """Update session state based on ProcessorPart."""
        try:
            # Update pipeline metrics
            stage = part.metadata.get("stage")
            if stage and stage in [VoiceMetadata.STAGE_STT, VoiceMetadata.STAGE_LLM, VoiceMetadata.STAGE_TTS]:
                latency_key = f"{stage}_latency_ms"
                if latency_key in part.metadata:
                    self.pipeline_metrics[stage] = part.metadata[latency_key]
            
            # Track processor states
            content_type = part.metadata.get("content_type")
            if content_type:
                processor_name = self._get_processor_name_from_content_type(content_type)
                if processor_name:
                    self.processor_states[processor_name] = {
                        "last_update": time.time(),
                        "status": part.metadata.get("status", "unknown"),
                        "sequence": part.metadata.get("sequence_number", 0)
                    }
            
        except Exception as e:
            logger.error(f"Error updating session state: {e}")
    
    def _get_processor_name_from_content_type(self, content_type: str) -> Optional[str]:
        """Map content type to processor name."""
        mapping = {
            VoiceMetadata.TRANSCRIPT: "whisper_live",
            VoiceMetadata.LLM_TEXT: "ollama_stream",
            VoiceMetadata.TTS_AUDIO: "kokoro_tts",
            VoiceMetadata.AUDIO_INPUT: "audio_input"
        }
        return mapping.get(content_type)
    
    def _is_session_expired(self) -> bool:
        """Check if session has expired."""
        return (time.time() - self.last_activity) > self.session_timeout
    
    async def interrupt_session(self):
        """Handle session interruption (barge-in)."""
        try:
            logger.info(f"Session {self.session_id}: Interruption triggered")
            
            # Store current text as interrupted for filtering
            if self.current_transcript:
                self.last_interrupted_text = self.current_transcript
            
            # Reset current exchange state
            self.current_exchange = {}
            self.current_transcript = ""
            self.current_llm_response = ""
            self.processing_stage = "idle"
            
            # Signal interrupt event
            self.interrupt_event.set()
            
            # Reset interrupt event after short delay
            await asyncio.sleep(0.1)
            self.interrupt_event.clear()
            
            logger.info(f"Session {self.session_id}: Interruption handling complete")
            
        except Exception as e:
            logger.error(f"Error handling session interruption: {e}")
    
    def get_conversation_context(self, max_exchanges: int = 3) -> str:
        """Get conversation context for LLM prompting."""
        try:
            if not self.conversation_history:
                return ""
            
            context_parts = []
            recent_history = self.conversation_history[-max_exchanges:]
            
            for exchange in recent_history:
                user_input = exchange.get("user_input", "")
                ai_response = exchange.get("llm_response", "")
                
                if user_input and ai_response:
                    context_parts.append(f"User: {user_input}")
                    context_parts.append(f"Assistant: {ai_response}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error building conversation context: {e}")
            return ""
    
    def get_session_summary(self) -> Dict:
        """Get comprehensive session summary."""
        current_time = time.time()
        session_duration = current_time - self.created_at
        
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "session_duration": session_duration,
            "is_active": self.is_active,
            "is_expired": self._is_session_expired(),
            
            # Conversation metrics
            "total_exchanges": self.total_exchanges,
            "current_exchange_count": self.exchange_count,
            "conversation_length": len(self.conversation_history),
            "average_latency_ms": self.average_latency * 1000,
            
            # Current state
            "processing_stage": self.processing_stage,
            "current_transcript": self.current_transcript,
            "current_response_length": len(self.current_llm_response),
            
            # Configuration
            "max_history_length": self.max_history_length,
            "memory_enabled": self.enable_memory,
            "session_timeout": self.session_timeout
        }
    
    def get_metrics(self) -> Dict:
        """Get detailed session metrics."""
        metrics = self.get_session_summary()
        
        # Add processor states
        metrics["processor_states"] = self.processor_states.copy()
        
        # Add pipeline metrics
        metrics["pipeline_metrics"] = self.pipeline_metrics.copy()
        
        # Add current exchange state
        if self.current_exchange:
            metrics["current_exchange"] = self.current_exchange.copy()
        
        return metrics
    
    async def cleanup_session(self):
        """Clean up session resources."""
        try:
            logger.info(f"Cleaning up session {self.session_id}")
            
            # Mark as inactive
            self.is_active = False
            
            # Store final state if memory is enabled
            if self.enable_memory and self.conversation_history:
                # Store session summary
                session_summary = {
                    "session_id": self.session_id,
                    "total_exchanges": self.total_exchanges,
                    "session_duration": time.time() - self.created_at,
                    "final_conversation_length": len(self.conversation_history),
                    "cleanup_timestamp": time.time()
                }
                logger.debug(f"Session summary: {session_summary}")
            
            # Clear conversation history to free memory
            self.conversation_history.clear()
            self.current_exchange.clear()
            self.processor_states.clear()
            self.pipeline_metrics.clear()
            
            logger.info(f"Session {self.session_id} cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during session cleanup: {e}")
