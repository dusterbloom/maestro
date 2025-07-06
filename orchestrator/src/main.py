import asyncio
import base64
import json
import logging
import os
import time
import re
from typing import Optional
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
            self.voice_service = VoiceService()
            self.memory_service = MemoryService()
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
        try:
            async with httpx.AsyncClient(timeout=config.OLLAMA_TIMEOUT) as client:
                response = await client.post(
                    f"{self.tts_url}/v1/audio/speech",  # Correct Kokoro endpoint
                    json={
                        "model": "kokoro",  # Required parameter from source
                        "input": text,
                        "voice": config.TTS_VOICE,
                        "response_format": "wav",  # Keep WAV for non-streaming
                        "stream": False,
                        "speed": config.TTS_SPEED,
                        "volume_multiplier": config.TTS_VOLUME
                    }
                )
                response.raise_for_status()
                return response.content
                
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
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
            import numpy as np
            float_array = np.frombuffer(audio_data, dtype=np.float32)
            buffer_ready = buffer_manager.add_audio_chunk(float_array)
            
            # Update session state to accumulating
            if current_state["status"] == "not_started":
                self.session_speaker_states[session_id] = {"status": "accumulating"}
                logger.info(f"🎤 Session {session_id} started accumulating audio")
            
            if buffer_ready:
                # We have 10 seconds! Perform definitive speaker identification
                logger.info(f"🎯 Session {session_id} reached 10 seconds - performing definitive identification")
                
                # Get WAV format audio for Diglett
                wav_bytes = buffer_manager.get_buffer_as_wav()
                
                # Get embedding from Diglett
                embedding = await self.voice_service.get_embedding(wav_bytes)
                if not embedding:
                    logger.error(f"Failed to get embedding for session {session_id}")
                    return {"user_id": "guest", "name": "Guest", "is_new": False, "greeting": ""}
                
                # Perform definitive identification
                result = await self.voice_service._identify_or_register_speaker_definitively(embedding, session_id)
                
                # Store result in session state to avoid re-identification
                self.session_speaker_states[session_id] = {
                    "status": "identified",
                    "speaker_info": result
                }
                
                # Get agentic context
                if self.agentic_speaker_system and result.get("status") in ["identified", "registered"]:
                    agentic_context = self.agentic_speaker_system.get_current_speaker_context()
                    if agentic_context:
                        result["greeting"] = agentic_context
                        # Update cached result
                        self.session_speaker_states[session_id]["speaker_info"] = result
                
                logger.info(f"🎭 DEFINITIVE speaker recognition completed for session {session_id}: {result}")
                return result
            
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
    audio_data: Optional[str] = None  # NEW: base64 encoded audio for speaker identification

@app.post("/ultra-fast-stream")
async def ultra_fast_stream(request: TranscriptRequest):
    """🚀 REAL-TIME STREAMING with Speaker Identification"""
    try:
        start_time = time.time()
        
        logger.info(f"🚀 Ultra-Fast-Stream: {request.transcript} (session: {request.session_id})")
        
        # 1. Sentence completion detection (same logic as before)
        transcript_words = request.transcript.strip().split()
        is_likely_interruption = (
            len(transcript_words) <= 2 and  # Very short input
            len(request.transcript.strip()) <= 15  # Short character count
        )
        
        if is_likely_interruption:
            # For short inputs that might be interruptions, apply minimal validation
            cleaned_sentence = request.transcript.strip()
            if len(cleaned_sentence) < 3:  # Skip very short inputs like "a", "I", "oh", "um"
                logger.info(f"Ultra-Fast-Stream: Input too short, skipping: {request.transcript}")
                
                async def empty_stream():
                    yield b""  # Empty binary response
                
                return StreamingResponse(
                    empty_stream(),
                    media_type="audio/wav",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Access-Control-Allow-Origin": "*"
                    }
                )
            logger.info(f"Ultra-Fast-Stream: Processing likely interruption: {cleaned_sentence}")
        else:
            # For longer inputs, use normal sentence completion detection
            is_complete, cleaned_sentence = orchestrator.sentence_detector.is_sentence_complete(request.transcript, request.session_id)
            
            if not is_complete:
                logger.info(f"Ultra-Fast-Stream: Sentence not complete, skipping: {request.transcript}")
                
                async def empty_stream():
                    yield b""  # Empty binary response
                
                return StreamingResponse(
                    empty_stream(),
                    media_type="audio/wav",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Access-Control-Allow-Origin": "*"
                    }
                )
            
            logger.info(f"Ultra-Fast-Stream: Processing complete sentence: {cleaned_sentence}")
        
        # 2. ENHANCED: Definitive speaker recognition with 10-second buffering
        speaker_context = ""
        name_learning_response = None
        
        if request.audio_data and orchestrator.memory_enabled:
            try:
                audio_bytes = base64.b64decode(request.audio_data)
                speaker_result = await orchestrator.accumulate_speaker_audio(audio_bytes, request.session_id)
                
                if speaker_result.get("status") == "accumulating":
                    # Still buffering - show progress
                    progress = speaker_result.get("progress", 0)
                    duration = speaker_result.get("buffer_duration", 0)
                    logger.info(f"🎤 Accumulating audio: {duration:.1f}s / 10.0s ({progress:.1%} complete)")
                elif speaker_result.get("status") in ["identified", "registered"]:
                    # Speaker recognition completed definitively
                    speaker_context = speaker_result.get("greeting", "")
                    if speaker_result.get("is_definitive"):
                        logger.info(f"🎭 DEFINITIVE RECOGNITION: {speaker_result['name']} ({speaker_result.get('confidence', 1.0):.2f})")
                    else:
                        logger.info(f"Speaker result: {speaker_result['name']} (status: {speaker_result['status']})")
                        
            except Exception as e:
                logger.warning(f"Speaker recognition failed: {e}")
        
        # 3. Check for name learning in transcript
        if orchestrator.agentic_speaker_system:
            name_learning_response = await orchestrator.handle_name_learning(cleaned_sentence, request.session_id)
        
        # 3. Real-time streaming with speaker-aware prompting
        async def realtime_sentence_stream():
            """Stream individual sentence audio with speaker awareness"""
            try:
                # Initialize session history if not exists
                if request.session_id not in orchestrator.session_history:
                    orchestrator.session_history[request.session_id] = []
                
                # Build context-aware prompt with speaker greeting
                history = orchestrator.session_history[request.session_id]
                context = ""
                if history:
                    # Include recent conversation turns
                    recent_exchanges = []
                    for exchange in history[-3:]:
                        recent_exchanges.append(f"User: {exchange['user']}")
                        recent_exchanges.append(f"Assistant: {exchange['assistant']}")
                    context = "\n".join(recent_exchanges) + "\n"
                
                logger.info("🚀 Starting real-time token streaming with sentence buffering...")
                
                # Initialize token buffer for sentence detection
                token_buffer = RealtimeTokenBuffer(min_chars=15, max_buffer_time=2.0)
                sentence_count = 0
                full_response = ""
                
                # ENHANCED: Handle name learning and speaker context
                if name_learning_response:
                    # Speaker told us their name - use personalized response
                    effective_prompt = f"{name_learning_response}How can I help you?"
                elif speaker_context:
                    # Add speaker context to system context, not user prompt
                    # This tells the LLM who they're talking to
                    context = f"[SPEAKER CONTEXT: {speaker_context.strip()}]\n{context}"
                    effective_prompt = cleaned_sentence
                    logger.info(f"🎭 Added speaker context to LLM: {speaker_context.strip()}")
                else:
                    # Default prompt
                    effective_prompt = cleaned_sentence
                
                # Start streaming LLM tokens and process in real-time
                async for token in orchestrator.stream_llm_tokens(effective_prompt, context):
                    full_response += token
                    
                    # Feed token to buffer and check for sentence fragment
                    sentence_fragment = token_buffer.feed(token)
                    
                    if sentence_fragment:
                        # Skip fragments that are just dots/ellipsis
                        if sentence_fragment.strip() in ['...', '....', '.....', '..', '.', '']:
                            logger.info(f"🔍 Skipping ellipsis fragment: '{sentence_fragment.strip()}'")
                            continue
                            
                        sentence_count += 1
                        logger.info(f"📝 Sentence {sentence_count}: {sentence_fragment[:50]}...")
                        
                        # Generate TTS for this sentence fragment immediately
                        try:
                            audio_data = await orchestrator.synthesize(sentence_fragment)
                            
                            if audio_data and len(audio_data) > 0:
                                # Convert to base64 for JSON streaming
                                audio_b64 = base64.b64encode(audio_data).decode()
                                
                                # Stream as Server-Sent Event
                                event_data = {
                                    'type': 'sentence_audio',
                                    'sequence': sentence_count,
                                    'text': sentence_fragment,
                                    'audio_data': audio_b64,
                                    'size_bytes': len(audio_data)
                                }
                                yield f"data: {json.dumps(event_data)}\n\n"
                                
                                logger.info(f"🎵 Streamed sentence {sentence_count} audio ({len(audio_data)} bytes)")
                            else:
                                logger.warning(f"⚠️ No audio generated for sentence {sentence_count}")
                                
                        except Exception as e:
                            logger.error(f"TTS error for sentence {sentence_count}: {e}")
                
                # Handle any remaining buffer content
                final_fragment = token_buffer.flush()
                if final_fragment:
                    # Skip final fragments that are just dots/ellipsis
                    if final_fragment.strip() in ['...', '....', '.....', '..', '.', '']:
                        logger.info(f"🔍 Skipping final ellipsis fragment: '{final_fragment.strip()}'")
                    else:
                        sentence_count += 1
                        logger.info(f"📝 Final sentence {sentence_count}: {final_fragment[:50]}...")
                        
                        try:
                            audio_data = await orchestrator.synthesize(final_fragment)
                            
                            if audio_data and len(audio_data) > 0:
                                audio_b64 = base64.b64encode(audio_data).decode()
                                
                                event_data = {
                                    'type': 'sentence_audio',
                                    'sequence': sentence_count,
                                    'text': final_fragment,
                                    'audio_data': audio_b64,
                                    'size_bytes': len(audio_data)
                                }
                                yield f"data: {json.dumps(event_data)}\n\n"
                                
                                logger.info(f"🎵 Streamed final sentence {sentence_count} audio ({len(audio_data)} bytes)")
                        except Exception as e:
                            logger.error(f"TTS error for final sentence: {e}")
                
                # Store conversation in session history
                if full_response:
                    exchange = {"user": cleaned_sentence, "assistant": full_response}
                    orchestrator.session_history[request.session_id].append(exchange)
                    
                    # Keep session history reasonable
                    if len(orchestrator.session_history[request.session_id]) > 10:
                        orchestrator.session_history[request.session_id] = orchestrator.session_history[request.session_id][-10:]
                
                # Send completion event
                completion_data = {
                    'type': 'complete',
                    'total_sentences': sentence_count,
                    'full_text': full_response
                }
                yield f"data: {json.dumps(completion_data)}\n\n"
                
                logger.info(f"🎉 Real-time streaming complete! {sentence_count} sentences")
                
            except Exception as e:
                logger.error(f"Real-time streaming error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
        return StreamingResponse(
            realtime_sentence_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control"
            }
        )
        
    except Exception as e:
        logger.error(f"Ultra-fast-stream error: {e}")
        return {"error": str(e)}, 500

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