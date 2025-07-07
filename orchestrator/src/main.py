import asyncio
import base64
import json
import logging
import os
import io
import time
import re
import numpy as np
from typing import Optional
import uuid
import httpx
import ollama
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from config import config
from services.voice_service import VoiceService, AudioBufferManager
from services.memory_service import MemoryService
from services.speaker_events import AgenticSpeakerSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Orchestrator", version="1.0.0")

class SentenceCompletionDetector:
    """Lightweight sentence completion detection using pattern analysis"""
    
    def __init__(self):
        # Patterns that indicate sentence completion
        self.sentence_endings = re.compile(r'[.!?]+\s*$')
        self.ellipsis_pattern = re.compile(r'\.{3,}\s*$')  # ... indicates continuation
        
        # Words that typically don't end sentences
        self.continuation_words = {
            'and', 'but', 'or', 'so', 'then', 'when', 'where', 'how', 'why', 'what', 'who',
            'the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her',
            'with', 'without', 'for', 'from', 'to', 'of', 'in', 'on', 'at', 'by'
        }
        
        # Track conversation state per session
        self.session_states = {}  # session_id -> {processed_sentences: set, last_processed: str}
    
    def is_sentence_complete(self, text: str, session_id: str = "default") -> tuple[bool, str]:
        """
        Determine if a sentence is complete and ready for processing
        Returns: (is_complete, cleaned_sentence)
        """
        if not text or not text.strip():
            return False, ""
            
        cleaned = text.strip()
        
        # Initialize session state if not exists
        if session_id not in self.session_states:
            self.session_states[session_id] = {
                'processed_sentences': set(),
                'last_processed': ""
            }
        
        session_state = self.session_states[session_id]
        
        # Extract the latest sentence from the text (handles multi-sentence concatenated inputs)
        sentences = re.split(r'[.!?]+', cleaned)
        if not sentences:
            return False, ""
            
        # Get the last non-empty sentence
        last_sentence = ""
        for sentence in reversed(sentences):
            sentence = sentence.strip()
            if sentence and len(sentence.split()) >= 2:  # Must have at least 2 words
                last_sentence = sentence
                break
        
        if not last_sentence:
            return False, ""
        
        # Check if this specific sentence was already processed
        if last_sentence in session_state['processed_sentences']:
            logger.info(f"Sentence already processed in session {session_id}: {last_sentence}")
            return False, ""
        
        # Reconstruct sentence with punctuation for validation
        full_sentence = last_sentence
        if not self.sentence_endings.search(cleaned):
            # If the original text ends with punctuation, add it to the sentence
            for ending in ['.', '!', '?']:
                if cleaned.endswith(ending):
                    full_sentence = last_sentence + ending
                    break
            else:
                # No sentence ending found
                return False, ""
        else:
            # Find the punctuation that belongs to this sentence
            remaining_text = cleaned[cleaned.rfind(last_sentence):]
            punctuation_match = self.sentence_endings.search(remaining_text)
            if punctuation_match:
                full_sentence = last_sentence + punctuation_match.group()
        
        # Check for ellipsis (indicates incomplete thought)
        if self.ellipsis_pattern.search(full_sentence):
            return False, ""
        
        # Check if last word suggests continuation
        words = last_sentence.split()
        if words:
            last_word = words[-1].lower().rstrip('.,!?')
            if last_word in self.continuation_words:
                return False, ""
        
        # Additional heuristics for completeness
        # Allow single meaningful words that end with punctuation (story., yes., okay., etc.)
        if len(words) < config.MIN_WORD_COUNT:
            if len(words) == 1 and len(words[0]) >= 3 and self.sentence_endings.search(full_sentence):
                # Single word with punctuation and at least 3 characters is valid
                pass  
            else:
                return False, ""
        
        # Mark as processed and return complete
        session_state['processed_sentences'].add(last_sentence)
        session_state['last_processed'] = last_sentence
        logger.info(f"Processing new sentence in session {session_id}: {last_sentence}")
        return True, full_sentence
    
    def reset_session(self, session_id: str = "default"):
        """Reset state for specific conversation session"""
        if session_id in self.session_states:
            del self.session_states[session_id]
            logger.info(f"Reset session state for {session_id}")

class RealtimeTokenBuffer:
    """
    Token streaming buffer with sentence boundary detection (RealtimeTTS pattern)
    Buffers LLM tokens and yields sentence fragments when ready for TTS
    """
    
    def __init__(self, min_chars: int = 15, max_buffer_time: float = 2.0):
        # Buffering thresholds
        self.min_chars = min_chars  # Minimum chars before considering sentence
        self.max_buffer_time = max_buffer_time  # Force output after this many seconds
        
        # Sentence boundary patterns (real sentence endings only)
        self.sentence_endings = re.compile(r'[.!?](?![.])\s*')  # Don't match if followed by more dots (ellipsis)
        self.ellipsis_pattern = re.compile(r'\.{2,}')  # Match ellipsis (2 or more dots)
        self.continuation_words = {
            'and', 'but', 'or', 'so', 'then', 'when', 'where', 'how', 'why', 
            'what', 'who', 'the', 'a', 'an', 'this', 'that', 'these', 'those',
            'with', 'without', 'for', 'from', 'to', 'of', 'in', 'on', 'at', 'by',
            'however', 'although', 'because', 'since', 'while', 'if', 'unless'
        }
        
        # Buffer state
        self.buffer = ""
        self.last_fragment_time = time.time()
        
    def feed(self, token: str) -> str | None:
        """
        Feed a token into the buffer (like RealtimeTTS .feed() method)
        Returns sentence fragment if ready for TTS, None if still buffering
        """
        # Replace ellipsis with natural pauses as tokens arrive
        if "..." in token:
            token = token.replace("...", ", ")
        elif ".." in token:
            token = token.replace("..", ", ")
            
        self.buffer += token
        
        # Check if we should yield a sentence fragment
        fragment = self._check_for_fragment()
        if fragment:
            self.buffer = self.buffer[len(fragment):].strip()
            self.last_fragment_time = time.time()
            return fragment.strip()
            
        # Force output if buffer timeout exceeded
        current_time = time.time()
        if (current_time - self.last_fragment_time > self.max_buffer_time and 
            len(self.buffer.strip()) >= self.min_chars):
            fragment = self.buffer.strip()
            # Also clean any remaining ellipsis in forced output
            fragment = self.ellipsis_pattern.sub(', ', fragment)
            self.buffer = ""
            self.last_fragment_time = current_time
            return fragment
            
        return None
    
    def flush(self) -> str | None:
        """Get remaining buffer content as final fragment"""
        if self.buffer.strip():
            fragment = self.buffer.strip()
            # Clean any remaining ellipsis in final fragment
            fragment = self.ellipsis_pattern.sub(', ', fragment)
            self.buffer = ""
            return fragment
        return None
    
    def _check_for_fragment(self) -> str | None:
        """Check if current buffer contains a ready sentence fragment"""
        if len(self.buffer.strip()) < self.min_chars:
            return None
        
        # First, handle ellipsis as pauses - replace with comma for natural speech
        if self.ellipsis_pattern.search(self.buffer):
            # Replace ellipsis with comma and space for natural pause in TTS
            cleaned_buffer = self.ellipsis_pattern.sub(', ', self.buffer)
            # Update the buffer with cleaned version
            self.buffer = cleaned_buffer
            
        # Look for real sentence endings (not ellipsis)
        match = self.sentence_endings.search(self.buffer)
        if match:
            # Found sentence ending, check if it's a natural break
            end_pos = match.end()
            potential_fragment = self.buffer[:end_pos].strip()
            remaining = self.buffer[end_pos:].strip()
            
            # Don't break if next word suggests continuation
            if remaining:
                next_words = remaining.split()
                if next_words and next_words[0].lower() in self.continuation_words:
                    return None
                    
            # Also check if the sentence ends with a continuation word
            words = potential_fragment.split()
            if words and words[-1].lower().rstrip('.,!?') in self.continuation_words:
                return None
                
            return potential_fragment
            
        return None

from enum import Enum

class ConversationState(Enum):
    GREETING = "GREETING"
    ENROLLING = "ENROLLING"
    CONFIRMING_ENROLLMENT = "CONFIRMING_ENROLLMENT"
    RECOGNIZED = "RECOGNIZED"
    INCOGNITO = "INCOGNITO"

class ConversationManager:
    def __init__(self):
        self.sessions = {}

    def get_state(self, session_id: str) -> ConversationState:
        return self.sessions.get(session_id, {"state": ConversationState.GREETING})["state"]

    def set_state(self, session_id: str, state: ConversationState):
        if session_id not in self.sessions:
            self.sessions[session_id] = {}
        self.sessions[session_id]["state"] = state

    def get_session_data(self, session_id: str) -> dict:
        return self.sessions.get(session_id, {})

    def set_session_data(self, session_id: str, data: dict):
        if session_id not in self.sessions:
            self.sessions[session_id] = {}
        self.sessions[session_id].update(data)

conversation_manager = ConversationManager()

class VoiceOrchestrator:
    def __init__(self):
        self.whisper_host = config.WHISPER_URL.split("://")[1].split(":")[0]
        self.whisper_port = int(config.WHISPER_URL.split(":")[-1])
        self.ollama_url = config.OLLAMA_URL
        self.tts_url = config.TTS_URL
        self.memory_enabled = config.MEMORY_ENABLED
        
        # Initialize sentence completion detector
        self.sentence_detector = SentenceCompletionDetector()
        
        # Simple session memory for conversation continuity
        self.session_history = {}  # session_id -> list of {"user": str, "assistant": str}
        
        # Track active TTS sessions for interruption capability
        self.active_tts_sessions = {}  # session_id -> {"thread": Thread, "queue": Queue, "abort_flag": Event}
        
        # Speaker embedding services with agentic system (ENHANCED)
        if self.memory_enabled:
            self.memory_service = MemoryService()
            self.voice_service = VoiceService(memory_service=self.memory_service)
            # Initialize agentic speaker system
            self.agentic_speaker_system = AgenticSpeakerSystem(self.voice_service, self.memory_service)
        else:
            self.voice_service = None
            self.memory_service = None
            self.agentic_speaker_system = None
            
        # Track session-persistent audio accumulation for speaker recognition
        self.session_audio_buffers = {}  # session_id -> AudioBufferManager
        self.session_speaker_states = {}  # session_id -> {"status": str, "speaker_info": dict}
        
        # Session speaker recognition states:
        # - "not_started": No audio accumulated yet
        # - "accumulating": Collecting 10 seconds of audio
        # - "identified": Speaker identified, no more accumulation needed
        # - "completed": Session complete

    async def generate_response(self, text: str, context: str = "") -> str:
        """Generate response using Ollama with streaming"""
        prompt = f"{context}\n\nUser: {text}\nAssistant:" if context else f"User: {text}\nAssistant:"
        
        try:
            async with httpx.AsyncClient(timeout=config.OLLAMA_TIMEOUT) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": config.LLM_MODEL,
                        "prompt": prompt,
                        "stream": True  # Enable real-time streaming
                    }
                )
                response.raise_for_status()
                
                # Handle streaming NDJSON response
                full_response = ""
                async for line in response.aiter_lines():
                    if line:
                        chunk = json.loads(line)
                        if chunk.get("response"):
                            full_response += chunk["response"]
                            # Optionally yield partial responses for real-time streaming
                        if chunk.get("done"):
                            break
                return full_response.strip()
                
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return "I'm sorry, I couldn't process your request right now."
    
    async def synthesize(self, text: str) -> bytes:
        """Convert text to speech using Kokoro TTS with optimized parameters"""
        if not text or not text.strip():
            logger.warning("Empty text provided to TTS")
            return b""
            
        try:
            start_time = time.time()
            
            # Optimize TTS request with connection pooling and reduced timeout
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=2.0, read=10.0, write=5.0, pool=None),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            ) as client:
                response = await client.post(
                    f"{self.tts_url}/v1/audio/speech",  # Correct Kokoro endpoint
                    json={
                        "model": "kokoro",  # Required parameter from source
                        "input": text[:500],  # Limit text length for faster processing
                        "voice": config.TTS_VOICE,
                        "response_format": "wav",  # Keep WAV for compatibility
                        "stream": False,
                        "speed": config.TTS_SPEED,
                        "volume_multiplier": config.TTS_VOLUME
                    }
                )
                response.raise_for_status()
                
                elapsed = time.time() - start_time
                audio_size = len(response.content)
                logger.info(f"✅ TTS completed in {elapsed:.2f}s - {audio_size} bytes")
                
                return response.content
                
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error(f"❌ TTS timeout after {elapsed:.2f}s")
            return b""
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ TTS synthesis error after {elapsed:.2f}s: {e}")
            return b""

    async def stream_llm_tokens(self, text: str, context: str = ""):
        """Stream LLM tokens using native ollama client for maximum performance"""
        
        # Enhanced system context for speaker awareness
        system_context = """You are a helpful voice assistant. You have speaker recognition capabilities and can remember users by their voice.

Key behaviors:
- When you recognize a returning user, greet them warmly by name
- When you meet someone new, ask for their name and remember it
- Be conversational, helpful, and personable
- Keep responses concise for voice interaction
- If unsure about voice recognition, politely ask for clarification"""
        
        # Build enhanced prompt with speaker awareness
        if context:
            prompt = f"{system_context}\n\n{context}\n\nUser: {text}\nAssistant:"
        else:
            prompt = f"{system_context}\n\nUser: {text}\nAssistant:"
        
        try:
            # Use native ollama streaming for best performance
            client = ollama.AsyncClient(host=self.ollama_url)
            stream = await client.generate(
                model=config.LLM_MODEL,
                prompt=prompt,
                stream=True,
                options={
                    "num_predict": config.LLM_MAX_TOKENS,  # Use configurable token limit
                    "temperature": config.LLM_TEMPERATURE,
                    "top_p": 0.8,
                    "num_ctx": 2048,        # Increased context for better responses
                }
            )
            
            async for chunk in stream:
                if chunk.get('response'):
                    yield chunk['response']
                if chunk.get('done'):
                    break
                    
        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            yield ""

    def interrupt_tts(self, session_id: str) -> bool:
        """Interrupt active TTS streaming for a session"""
        if session_id in self.active_tts_sessions:
            session_info = self.active_tts_sessions[session_id]
            if "abort_flag" in session_info and not session_info["abort_flag"].is_set():
                session_info["abort_flag"].set()
                logger.info(f"🛑 TTS for session {session_id} interrupted.")
                return True
        logger.warning(f"No active TTS session found to interrupt for session_id: {session_id}")
        return False

    def cleanup_completed_session(self, session_id: str):
        """Clean up a completed session including speaker recognition data"""
        try:
            # Clean up TTS sessions
            if session_id in self.active_tts_sessions:
                del self.active_tts_sessions[session_id]
                logger.info(f"Cleaned up completed TTS session {session_id}")
                
            # Clean up speaker recognition data to prevent memory leaks
            if session_id in self.session_audio_buffers:
                self.session_audio_buffers[session_id].clear_buffer()
                del self.session_audio_buffers[session_id]
                logger.info(f"Cleared audio buffer for session {session_id}")
                
            if session_id in self.session_speaker_states:
                del self.session_speaker_states[session_id]
                logger.info(f"Cleared speaker state for session {session_id}")
                
        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {e}")

    # ENHANCED: Session-persistent definitive speaker identification with 10-second buffering
    async def accumulate_speaker_audio(self, audio_data: bytes, session_id: str) -> dict:
        """
        Session-persistent audio accumulation for definitive speaker recognition.
        Accumulates across multiple utterances until 10 seconds reached, then identifies once.
        """
        if not self.memory_enabled or not self.voice_service:
            return {"user_id": "guest", "name": "Guest", "is_new": False, "greeting": ""}
            
        try:
            # Check current session state
            current_state = self.session_speaker_states.get(session_id, {"status": "not_started"})
            
            # If already identified, return cached result and skip accumulation
            if current_state["status"] == "identified":
                logger.debug(f"Speaker already identified for session {session_id}")
                return current_state.get("speaker_info", {})
            
            # Get or create session audio buffer
            if session_id not in self.session_audio_buffers:
                self.session_audio_buffers[session_id] = AudioBufferManager()
                logger.info(f"🎤 Started new audio accumulation for session {session_id}")
            
            buffer_manager = self.session_audio_buffers[session_id]
            
            # Convert audio bytes to float32 array and add to session buffer
            float_array = np.frombuffer(audio_data, dtype=np.float32)
            buffer_ready = buffer_manager.add_audio_chunk(float_array)
            
            # Update session state to accumulating
            if current_state["status"] == "not_started":
                self.session_speaker_states[session_id] = {"status": "accumulating"}
                logger.info(f"🎤 Session {session_id} started accumulating audio")
            
            if buffer_ready:
                # We have 10 seconds! Start non-blocking speaker identification
                logger.info(f"🎯 Session {session_id} reached 10 seconds - starting background identification")
                
                # Mark as processing to avoid duplicate runs
                self.session_speaker_states[session_id] = {"status": "processing"}
                
                # Start background task for speaker identification (non-blocking)
                asyncio.create_task(self._perform_background_speaker_identification(session_id, buffer_manager))
                
                # Return immediately - don't block the conversation
                return {
                    "status": "processing",
                    "message": "Speaker identification running in background",
                    "user_id": "guest", 
                    "name": "Friend", 
                    "is_new": False, 
                    "greeting": ""
                }
            
            else:
                # Still accumulating - return progress
                duration = buffer_manager.get_buffer_duration_seconds()
                progress = duration / 10.0
                
                return {
                    "status": "accumulating",
                    "buffer_duration": duration,
                    "target_duration": 10.0,
                    "progress": progress,
                    "session_id": session_id
                }
            
        except Exception as e:
            logger.error(f"Session-persistent speaker accumulation error for {session_id}: {e}")
            return {"user_id": "guest", "name": "Guest", "is_new": False, "greeting": ""}
    
    def get_session_speaker_info(self, session_id: str) -> dict:
        """Get cached speaker info for session if already identified"""
        session_state = self.session_speaker_states.get(session_id, {})
        if session_state.get("status") == "identified":
            return session_state.get("speaker_info", {})
        return {}
    
    async def handle_name_learning(self, transcript: str, session_id: str) -> Optional[str]:
        """
        Check if transcript contains name introduction and learn it
        Returns personalized response if name was learned
        """
        if not self.agentic_speaker_system:
            return None
            
        # Simple name detection patterns
        name_patterns = [
            r"my name is (\w+)",
            r"i'm (\w+)",
            r"call me (\w+)",
            r"i am (\w+)"
        ]
        
        import re
        transcript_lower = transcript.lower()
        
        for pattern in name_patterns:
            match = re.search(pattern, transcript_lower)
            if match:
                name = match.group(1).capitalize()
                response = await self.agentic_speaker_system.handle_name_learning(name, session_id)
                logger.info(f"Learned name from transcript: {name}")
                return response
                
        return None
    
    async def _perform_background_speaker_identification(self, session_id: str, buffer_manager):
        """🚀 NON-BLOCKING: Perform speaker identification in background with duplicate prevention"""
        try:
            logger.info(f"🔄 Background speaker identification starting for session {session_id}")
            
            # Check if already processing/completed to avoid duplicates
            current_state = self.session_speaker_states.get(session_id, {})
            if current_state.get("status") in ["identified", "completed"]:
                logger.info(f"⏭️ Session {session_id} already processed, skipping duplicate")
                return
            
            # Get audio data without VAD to preserve full 10 seconds
            wav_bytes = buffer_manager.get_buffer_as_wav(apply_vad=False)
            
            if not wav_bytes:
                logger.error(f"❌ No audio data available for session {session_id}")
                self.session_speaker_states[session_id] = {
                    "status": "completed", 
                    "speaker_info": {"user_id": "guest", "name": "Friend", "is_new": False}
                }
                return
            
            # Check if we already have a registered speaker to avoid multiple registrations
            if hasattr(self.voice_service, 'registered_speaker') and self.voice_service.registered_speaker:
                logger.info(f"🔍 Checking against existing registered speaker")
                
                # Get embedding for comparison
                embedding = await self.voice_service.get_embedding(wav_bytes)
                if embedding:
                    stored_embedding = self.voice_service.registered_speaker["embedding"]
                    similarity = self.voice_service._calculate_similarity(embedding, stored_embedding)
                    
                    if similarity >= self.voice_service.confidence_threshold:
                        # Same speaker - no need to register again
                        existing_speaker = self.voice_service.registered_speaker
                        result = {
                            "status": "identified",
                            "user_id": existing_speaker["user_id"],
                            "name": existing_speaker["name"],
                            "confidence": similarity,
                            "is_definitive": True,
                            "greeting": f"Welcome back, {existing_speaker['name']}!"
                        }
                        
                        self.session_speaker_states[session_id] = {
                            "status": "identified",
                            "speaker_info": result
                        }
                        
                        logger.info(f"✅ Session {session_id} - SAME SPEAKER identified (confidence: {similarity:.4f})")
                        return
            
            # New speaker or no existing registration - proceed with full identification
            embedding = await self.voice_service.get_embedding(wav_bytes)
            if not embedding:
                logger.error(f"❌ Background identification failed - no embedding for session {session_id}")
                self.session_speaker_states[session_id] = {
                    "status": "completed", 
                    "speaker_info": {"user_id": "guest", "name": "Friend", "is_new": False}
                }
                return
            
            # Perform definitive identification or registration
            result = await self.voice_service._identify_or_register_speaker_definitively(embedding, session_id)
            
            # Store result in session state
            self.session_speaker_states[session_id] = {
                "status": "identified",
                "speaker_info": result
            }
            
            # Get agentic context if available
            if self.agentic_speaker_system and result.get("status") in ["identified", "registered"]:
                agentic_context = self.agentic_speaker_system.get_current_speaker_context()
                if agentic_context:
                    result["greeting"] = agentic_context
                    # Update cached result
                    self.session_speaker_states[session_id]["speaker_info"] = result
            
            logger.info(f"✅ Background speaker identification completed for session {session_id}: {result.get('status', 'unknown')}")
            logger.info(f"🎯 Speaker info cached for session {session_id} - ready for next conversation turn")
            
        except Exception as e:
            logger.error(f"❌ Background speaker identification failed for session {session_id}: {e}")
            # Fallback to guest
            self.session_speaker_states[session_id] = {
                "status": "completed",
                "speaker_info": {"user_id": "guest", "name": "Friend", "is_new": False}
            }

    async def passively_accumulate_speaker_audio(self, audio_bytes: bytes, session_id: str):
        """🚀 PASSIVE: Accumulate audio chunks in background without blocking conversation"""
        try:
            # Get or create buffer manager for this session
            if session_id not in self.session_audio_buffers:
                self.session_audio_buffers[session_id] = AudioBufferManager()
            
            buffer_manager = self.session_audio_buffers[session_id]
            
            # Convert audio to float array and add to buffer
            import numpy as np
            float_array = np.frombuffer(audio_bytes, dtype=np.float32)
            buffer_ready = buffer_manager.add_audio_chunk(float_array)
            
            if buffer_ready:
                # We have 10 seconds! Start background identification
                logger.info(f"🔄 Background: 10 seconds accumulated for session {session_id} - starting speaker identification")
                asyncio.create_task(self._perform_background_speaker_identification(session_id, buffer_manager))
                
        except Exception as e:
            logger.error(f"❌ Passive audio accumulation failed for session {session_id}: {e}")

# Initialize orchestrator
orchestrator = VoiceOrchestrator()

@app.get("/health")
async def health():
    """Health check endpoint"""
    return JSONResponse({
        "status": "ok", 
        "memory_enabled": orchestrator.memory_enabled,
        "timestamp": time.time()
    })

@app.post("/start-conversation")
async def start_conversation():
    """Starts a new conversation and returns a session ID and greeting"""
    session_id = f"session_{uuid.uuid4().hex}"
    greeting = "Hello! I am Maestro, your personal voice assistant. How can I help you today?"
    
    # Initialize session state
    orchestrator.sentence_detector.reset_session(session_id)
    orchestrator.session_history[session_id] = []
    orchestrator.session_speaker_states[session_id] = {"status": "not_started"}
    
    return JSONResponse({
        "session_id": session_id,
        "greeting": greeting
    })


@app.get("/whisper-info")
async def whisper_info():
    """Get WhisperLive connection info"""
    return {
        "whisper_live_url": f"ws://{orchestrator.whisper_host}:{orchestrator.whisper_port}",
        "message": "Connect frontend directly to WhisperLive, send transcripts to /process-transcript"
    }

class TranscriptRequest(BaseModel):
    transcript: str
    session_id: str = "default"
    audio_data: Optional[str] = None  # base64 encoded audio for speaker identification (optional)

class InterruptRequest(BaseModel):
    session_id: str

@app.post("/interrupt-tts")
async def interrupt_tts_endpoint(request: InterruptRequest):
    """Endpoint to interrupt TTS playback for a given session"""
    logger.info(f"Received interrupt request for session: {request.session_id}")
    
    try:
        # Call the orchestrator's interrupt method
        success = orchestrator.interrupt_tts(request.session_id)
        
        if success:
            return JSONResponse({"status": "interrupted", "session_id": request.session_id})
        else:
            return JSONResponse({"status": "not_found", "session_id": request.session_id}, status_code=404)
            
    except Exception as e:
        logger.error(f"Error interrupting TTS for session {request.session_id}: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/ultra-fast-stream")
async def ultra_fast_stream(request: TranscriptRequest):
    """🚀 REAL-TIME STREAMING with Server-Sent Events returning JSON with base64 audio"""
    session_id = request.session_id
    transcript = request.transcript
    audio_data = request.audio_data

    # Start embeddings in background FIRE-AND-FORGET (absolutely no blocking)
    if orchestrator.memory_enabled and orchestrator.voice_service and audio_data:
        try:
            # Fire and forget - don't wait for this
            asyncio.create_task(orchestrator.passively_accumulate_speaker_audio(
                base64.b64decode(audio_data), session_id
            ))
        except Exception as e:
            # Silently ignore embedding errors - never block conversation
            logger.warning(f"Embedding task failed to start: {e}")
            pass

    async def generate_sse_response():
        """Generate Server-Sent Events with JSON containing base64 audio"""
        try:
            # Generate response immediately (don't wait for embeddings)
            response = await orchestrator.generate_response(transcript)
            
            if not response or not response.strip():
                response = "I'm sorry, I didn't understand that. Could you please try again?"
            
            # Generate TTS audio
            audio_bytes = await orchestrator.synthesize(response)
            
            if audio_bytes and len(audio_bytes) > 0:
                # Convert audio to base64 for JSON transport
                import base64
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                
                # Send as single sentence event (matches frontend expectation)
                event_data = {
                    "type": "sentence_audio",
                    "sequence": 1,
                    "text": response,
                    "audio_data": audio_base64
                }
                
                yield f"data: {json.dumps(event_data)}\n\n"
                
                # Send completion event
                completion_data = {
                    "type": "complete",
                    "total_sentences": 1,
                    "full_text": response
                }
                yield f"data: {json.dumps(completion_data)}\n\n"
                
            else:
                # TTS failed - send error event
                error_data = {
                    "type": "error",
                    "message": "TTS synthesis failed",
                    "text_response": response
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                
        except Exception as e:
            logger.error(f"❌ SSE generation error: {e}")
            error_data = {
                "type": "error", 
                "message": str(e)
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    # Return Server-Sent Events response
    return StreamingResponse(
        generate_sse_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

async def handle_recognized_conversation(session_id: str, transcript: str):
    session_data = conversation_manager.get_session_data(session_id)
    user_id = session_data.get("user_id")
    
    # Load conversation history
    history = await orchestrator.memory_service.get_user_memory(user_id, "conversation")
    context = "\n".join([f"User: {item['user']}\nAssistant: {item['assistant']}" for item in history])

    response = await orchestrator.generate_response(transcript, context=context)
    
    # Save current exchange to memory
    await orchestrator.memory_service.add_user_memory(user_id, {
        "event_type": "conversation",
        "user": transcript,
        "assistant": response
    })

    audio_data = await orchestrator.synthesize(response)
    
    # Handle TTS errors gracefully
    if not audio_data or len(audio_data) == 0:
        logger.error("TTS synthesis failed, returning error response")
        return JSONResponse({
            "error": "TTS synthesis failed",
            "message": response  # At least provide the text response
        }, status_code=500)
    
    return StreamingResponse(io.BytesIO(audio_data), media_type="audio/wav")

async def handle_incognito_conversation(transcript: str, session_id: str = "default", audio_data: str = None):
    # Continue passive audio collection for speaker recognition (only if audio_data provided)
    # This runs in background and doesn't affect response time
    if orchestrator.memory_enabled and orchestrator.voice_service and audio_data:
        try:
            asyncio.create_task(orchestrator.passively_accumulate_speaker_audio(
                base64.b64decode(audio_data), session_id
            ))
        except Exception as e:
            logger.warning(f"Failed to process audio data for speaker recognition: {e}")
            # Continue without speaker recognition
    
    # Check if speaker has been identified in background and upgrade conversation
    if orchestrator.memory_enabled:
        speaker_info = orchestrator.get_session_speaker_info(session_id)
        if speaker_info and speaker_info.get("status") == "identified":
            # Upgrade to recognized conversation
            user_id = speaker_info.get("user_id")
            if user_id:
                logger.info(f"🎉 Upgrading session {session_id} to RECOGNIZED state for user {user_id}")
                conversation_manager.set_state(session_id, ConversationState.RECOGNIZED)
                conversation_manager.set_session_data(session_id, {"user_id": user_id})
                return await handle_recognized_conversation(session_id, transcript)
    
    # Simplified conversation loop without memory
    logger.info(f"🎯 Processing incognito conversation for session {session_id}")
    
    # Generate response and TTS in parallel when possible
    response = await orchestrator.generate_response(transcript)
    
    if not response or not response.strip():
        logger.error("LLM generated empty response")
        response = "I'm sorry, I didn't understand that. Could you please try again?"
    
    audio_data = await orchestrator.synthesize(response)
    
    # Handle TTS errors gracefully
    if not audio_data or len(audio_data) == 0:
        logger.error("TTS synthesis failed in incognito mode, returning text response")
        return JSONResponse({
            "type": "text_response",
            "message": response  # At least provide the text response
        })
    
    logger.info(f"✅ Incognito conversation completed for session {session_id}")
    return StreamingResponse(io.BytesIO(audio_data), media_type="audio/wav")

async def handle_greeting(session_id: str, audio_data: str):
    try:
        if not audio_data:
            return JSONResponse({
                "type": "audio_prompt", 
                "message": "I didn't hear anything. Please speak to start."
            })

        logger.info(f"🔄 Processing greeting for session {session_id} - starting conversation immediately")
        
        # Start passive audio collection in background (non-blocking)
        if orchestrator.memory_enabled and orchestrator.voice_service and audio_data:
            asyncio.create_task(orchestrator.passively_accumulate_speaker_audio(
                base64.b64decode(audio_data), session_id
            ))
        
        # Move directly to conversation - speaker ID will happen in background
        conversation_manager.set_state(session_id, ConversationState.INCOGNITO)
        
        # Generate welcome audio immediately without waiting for speaker identification
        welcome_message = "Hello! I'm ready to chat. What can I help you with?"
        audio_data = await orchestrator.synthesize(welcome_message)
        
        # Handle TTS errors gracefully
        if not audio_data or len(audio_data) == 0:
            logger.error("TTS synthesis failed for greeting, returning text response")
            return JSONResponse({
                "type": "conversation_ready",
                "message": welcome_message
            })
        
        return StreamingResponse(io.BytesIO(audio_data), media_type="audio/wav")
        
    except Exception as e:
        logger.error(f"❌ Error in handle_greeting for session {session_id}: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return JSONResponse({"error": f"Internal error: {str(e)}"}, status_code=500)

async def handle_enrollment(session_id: str, audio_data: str):
    session_data = conversation_manager.get_session_data(session_id)
    embeddings = session_data.get("embeddings", [])
    step = session_data.get("enrollment_step", 1)

    if not audio_data:
        return JSONResponse({
            "type": "audio_prompt",
            "message": "I didn't hear you. Please repeat the sentence."
        })

    embedding = await orchestrator.voice_service.get_embedding(base64.b64decode(audio_data))
    if not embedding:
        return JSONResponse({"error": "Failed to generate embedding"}, status_code=500)

    embeddings.append(embedding)
    conversation_manager.set_session_data(session_id, {"embeddings": embeddings, "enrollment_step": step + 1})

    if step < 3:
        return JSONResponse({
            "type": "enrollment_continue",
            "message": f"Thank you. Please repeat the sentence {3 - step} more time(s)."
        })
    else:
        conversation_manager.set_state(session_id, ConversationState.CONFIRMING_ENROLLMENT)
        return JSONResponse({
            "type": "enrollment_complete",
            "message": "Thank you. I've recorded your voice. What name should I associate with it?"
        })

async def handle_enrollment_confirmation(session_id: str, transcript: str):
    session_data = conversation_manager.get_session_data(session_id)
    embeddings = session_data.get("embeddings", [])
    name = transcript.strip()

    if not name:
        return JSONResponse({
            "type": "audio_prompt",
            "message": "I didn't catch that. Please tell me your name."
        })

    # Average the embeddings to create a single representative embedding
    avg_embedding = np.mean(embeddings, axis=0).tolist()
    user_id = await orchestrator.memory_service.create_speaker_profile(avg_embedding)
    await orchestrator.memory_service.update_speaker_name(user_id, name)

    conversation_manager.set_state(session_id, ConversationState.RECOGNIZED)
    conversation_manager.set_session_data(session_id, {"user_id": user_id})

    return JSONResponse({
        "type": "enrollment_confirmed",
        "message": f"Thank you, {name}! I'll remember you."
    })



@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Voice Orchestrator starting up...")
    logger.info(f"Memory enabled: {orchestrator.memory_enabled}")
    
    # Initialize memory service if enabled
    if orchestrator.memory_enabled and orchestrator.memory_service:
        await orchestrator.memory_service.initialize_chroma_client()

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    logger.info("Voice Orchestrator shutting down...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)