"""
OllamaStreamProcessor - GenAI Processors integration for Ollama LLM streaming

This processor integrates with the existing Ollama API while providing
sentence-level streaming for immediate TTS processing. It maintains
conversation context and implements real-time token processing.
"""

import asyncio
import json
import logging
import ollama
import re
import time
from typing import AsyncIterable, Dict, List, Optional, Set

from genai_processors import content_api
from genai_processors import processor

from .config import config, VoiceMetadata


logger = logging.getLogger(__name__)


class OllamaStreamProcessor(processor.Processor):
    """
    Streams LLM responses from Ollama with sentence boundary detection.
    
    Features:
    - Integrates with existing Ollama client and API
    - Real-time token streaming with sentence detection
    - Maintains conversation history and context
    - Configurable models and parameters
    - Supports interruption for barge-in functionality
    - Performance metrics and latency tracking
    """
    
    def __init__(
        self,
        model: str = None,
        host: str = None,
        session_id: str = "default",
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = 0.8,
        context_window: int = 3,
        sentence_chars: str = None,
        min_sentence_length: int = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Ollama configuration
        self.model = model or config.LLM_MODEL
        self.host = host or config.OLLAMA_URL
        self.session_id = session_id
        
        # LLM parameters
        self.max_tokens = max_tokens or config.LLM_MAX_TOKENS
        self.temperature = temperature or config.LLM_TEMPERATURE
        self.top_p = top_p
        
        # Conversation management
        self.context_window = context_window
        self.conversation_history: List[Dict[str, str]] = []
        
        # Sentence processing
        self.sentence_chars = sentence_chars or config.SENTENCE_BOUNDARY_CHARS
        self.min_sentence_length = min_sentence_length or config.SENTENCE_MIN_LENGTH
        self.sentence_pattern = re.compile(f'[{re.escape(self.sentence_chars)}]')
        
        # Processing state
        self.current_response = ""
        self.sentence_buffer = ""
        self.sentence_count = 0
        self.interrupted = False
        
        # Performance tracking
        self.llm_start_time: Optional[float] = None
        self.first_token_time: Optional[float] = None
        self.processing_count = 0
        
        # Ollama client
        self.client = ollama.AsyncClient(host=self.host)
        
        logger.info(f"OllamaStreamProcessor initialized for session {session_id}")
        logger.info(f"Model: {self.model}, Host: {self.host}")
    
    async def call(self, content: AsyncIterable[content_api.ProcessorPart]) -> AsyncIterable[content_api.ProcessorPart]:
        """
        Process transcript input and stream LLM response with sentence detection.
        
        Args:
            content: Stream of transcript ProcessorParts
            
        Yields:
            ProcessorParts containing LLM text responses (sentence-level)
        """
        try:
            async for part in content:
                # Only process completed transcripts
                if (part.metadata and 
                    part.metadata.get("content_type") == VoiceMetadata.TRANSCRIPT and
                    part.metadata.get("is_complete", False)):
                    
                    transcript_text = part.content if isinstance(part.content, str) else str(part.content)
                    
                    # Check for interruption before processing
                    if self.interrupted:
                        logger.info(f"Session {self.session_id}: LLM processing interrupted")
                        self.interrupted = False
                        continue
                    
                    # Process the transcript through LLM
                    async for response_part in self._process_transcript(transcript_text, part.metadata):
                        yield response_part
                        
                        # Check for interruption during streaming
                        if self.interrupted:
                            logger.info(f"Session {self.session_id}: LLM streaming interrupted")
                            self.interrupted = False
                            break
                            
        except Exception as e:
            logger.error(f"Error in OllamaStreamProcessor: {e}")
    
    async def _process_transcript(
        self, 
        transcript: str, 
        source_metadata: Dict
    ) -> AsyncIterable[content_api.ProcessorPart]:
        """Process a transcript through Ollama LLM with streaming."""
        try:
            # Build conversation context
            context_prompt = self._build_context_prompt(transcript)
            
            # Start LLM processing
            self.llm_start_time = time.time()
            self.first_token_time = None
            self.current_response = ""
            self.sentence_buffer = ""
            self.sentence_count = 0
            
            logger.info(f"Session {self.session_id}: Starting LLM processing for: {transcript}")
            
            # Stream tokens from Ollama
            stream = await self.client.generate(
                model=self.model,
                prompt=context_prompt,
                stream=True,
                options={
                    "num_predict": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "stop": ["User:", "Human:", "Assistant:"],
                    "num_ctx": 4096,
                    "repeat_penalty": 1.1
                }
            )
            
            # Process streaming response
            async for chunk in stream:
                if self.interrupted:
                    break
                
                token = chunk.get("response", "")
                if token:
                    # Track first token time for TTFT metrics
                    if self.first_token_time is None:
                        self.first_token_time = time.time()
                        ttft_ms = (self.first_token_time - self.llm_start_time) * 1000
                        logger.info(f"Session {self.session_id}: LLM TTFT: {ttft_ms:.2f}ms")
                    
                    # Add token to buffers
                    self.current_response += token
                    self.sentence_buffer += token
                    
                    # Check for sentence completion
                    if self._is_sentence_complete(self.sentence_buffer):
                        sentence = self.sentence_buffer.strip()
                        if len(sentence) >= self.min_sentence_length:
                            # Yield complete sentence immediately for TTS
                            yield await self._create_sentence_part(
                                sentence, 
                                source_metadata,
                                is_complete=True
                            )
                            
                            self.sentence_count += 1
                            logger.debug(f"Session {self.session_id}: Completed sentence {self.sentence_count}: {sentence}")
                        
                        # Reset sentence buffer
                        self.sentence_buffer = ""
                
                # Check if generation is complete
                if chunk.get("done", False):
                    # Handle any remaining text in buffer
                    if self.sentence_buffer.strip():
                        remaining_text = self.sentence_buffer.strip()
                        if len(remaining_text) >= self.min_sentence_length:
                            yield await self._create_sentence_part(
                                remaining_text,
                                source_metadata, 
                                is_complete=True,
                                is_final=True
                            )
                    
                    # Store conversation history
                    self._update_conversation_history(transcript, self.current_response)
                    
                    # Calculate total processing time
                    total_time = time.time() - self.llm_start_time
                    self.processing_count += 1
                    
                    logger.info(f"Session {self.session_id}: LLM processing complete")
                    logger.info(f"Session {self.session_id}: Total response time: {total_time:.3f}s")
                    logger.info(f"Session {self.session_id}: Generated {self.sentence_count} sentences")
                    
                    break
                    
        except Exception as e:
            logger.error(f"Error processing transcript through Ollama: {e}")
            
            # Yield error response part
            error_part = content_api.ProcessorPart(
                content="I'm sorry, I couldn't process your request right now.",
                mime_type="text/plain",
                metadata={
                    "session_id": self.session_id,
                    "content_type": VoiceMetadata.LLM_TEXT,
                    "stage": VoiceMetadata.STAGE_LLM,
                    "status": VoiceMetadata.STATUS_ERROR,
                    "timestamp": time.time(),
                    "error": str(e)
                }
            )
            yield error_part
    
    def _build_context_prompt(self, current_input: str) -> str:
        """Build conversation prompt with context from history."""
        context_parts = []
        
        # Add recent conversation history
        for exchange in self.conversation_history[-self.context_window:]:
            context_parts.append(f"User: {exchange['user']}")
            context_parts.append(f"Assistant: {exchange['assistant']}")
        
        # Add current input
        context_parts.append(f"User: {current_input}")
        context_parts.append("Assistant:")
        
        return "\n".join(context_parts)
    
    def _is_sentence_complete(self, text: str) -> bool:
        """Check if text contains a complete sentence."""
        # Look for sentence ending punctuation
        if self.sentence_pattern.search(text):
            # Additional checks for common abbreviations and edge cases
            stripped = text.strip()
            
            # Avoid breaking on common abbreviations
            if stripped.endswith(('.', '?', '!')):
                # Check for common abbreviations (simple heuristic)
                words = stripped.split()
                if len(words) > 0:
                    last_word = words[-1].lower()
                    # Common abbreviations that shouldn't end sentences
                    if last_word in ['mr.', 'mrs.', 'dr.', 'vs.', 'etc.', 'i.e.', 'e.g.']:
                        return False
                
                return len(stripped) >= self.min_sentence_length
        
        return False
    
    async def _create_sentence_part(
        self, 
        sentence: str, 
        source_metadata: Dict,
        is_complete: bool = True,
        is_final: bool = False
    ) -> content_api.ProcessorPart:
        """Create a ProcessorPart for a completed sentence."""
        current_time = time.time()
        
        # Calculate latency metrics
        sentence_latency = current_time - self.llm_start_time if self.llm_start_time else 0
        ttft = (self.first_token_time - self.llm_start_time) if self.first_token_time and self.llm_start_time else 0
        
        metadata = {
            "session_id": self.session_id,
            "content_type": VoiceMetadata.LLM_TEXT,
            "stage": VoiceMetadata.STAGE_LLM,
            "status": VoiceMetadata.STATUS_COMPLETE,
            "timestamp": current_time,
            "sequence_number": self.sentence_count,
            "is_complete": is_complete,
            "is_final": is_final,
            "sentence_length": len(sentence),
            "total_response_length": len(self.current_response),
            
            # Latency metrics
            "llm_sentence_latency_ms": sentence_latency * 1000,
            "llm_ttft_ms": ttft * 1000,
            
            # Model information
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            
            # Source information
            "source_transcript": source_metadata.get("content", ""),
            "source_timestamp": source_metadata.get("timestamp"),
            "source_stt_latency": source_metadata.get("stt_latency_ms")
        }
        
        return content_api.ProcessorPart(
            content=sentence,
            mime_type="text/plain",
            metadata=metadata
        )
    
    def _update_conversation_history(self, user_input: str, assistant_response: str):
        """Update conversation history with the latest exchange."""
        self.conversation_history.append({
            "user": user_input,
            "assistant": assistant_response,
            "timestamp": time.time()
        })
        
        # Keep only recent history to manage memory
        max_history = 10  # Keep last 10 exchanges
        if len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]
    
    async def interrupt(self):
        """Interrupt current LLM processing for barge-in functionality."""
        logger.info(f"OllamaStreamProcessor interrupted for session {self.session_id}")
        self.interrupted = True
        
        # Reset processing state
        self.current_response = ""
        self.sentence_buffer = ""
        self.sentence_count = 0
        self.llm_start_time = None
        self.first_token_time = None
    
    def clear_history(self):
        """Clear conversation history for the session."""
        logger.info(f"Clearing conversation history for session {self.session_id}")
        self.conversation_history.clear()
    
    def get_conversation_summary(self) -> Dict:
        """Get a summary of the current conversation."""
        return {
            "session_id": self.session_id,
            "history_length": len(self.conversation_history),
            "last_exchange": self.conversation_history[-1] if self.conversation_history else None,
            "total_exchanges": len(self.conversation_history)
        }
    
    def get_metrics(self) -> Dict:
        """Get performance metrics for monitoring."""
        current_time = time.time()
        
        return {
            "session_id": self.session_id,
            "model": self.model,
            "processing_count": self.processing_count,
            "conversation_length": len(self.conversation_history),
            "current_response_length": len(self.current_response),
            "sentence_count": self.sentence_count,
            "interrupted": self.interrupted,
            
            # Current processing state
            "is_processing": self.llm_start_time is not None,
            "processing_time": (current_time - self.llm_start_time) if self.llm_start_time else 0,
            
            # Configuration
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "context_window": self.context_window,
            "min_sentence_length": self.min_sentence_length,
            
            # Last processing metrics
            "last_ttft_ms": (self.first_token_time - self.llm_start_time) * 1000 if self.first_token_time and self.llm_start_time else None
        }
