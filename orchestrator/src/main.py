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

class StreamingSentenceDetector:
    """Simple sentence boundary detection for streaming pipeline (legacy)"""
    
    def __init__(self):
        # Simple patterns for sentence boundaries
        self.sentence_endings = re.compile(r'[.!?]+')
        
    def find_sentence_boundary(self, text_buffer: str) -> tuple[str, str]:
        """
        Find the first complete sentence in text buffer
        Returns: (complete_sentence, remaining_buffer)
        """
        if not text_buffer.strip():
            return "", text_buffer
            
        # Find first sentence ending
        match = self.sentence_endings.search(text_buffer)
        if not match:
            return "", text_buffer
            
        # Include the punctuation in the sentence
        end_pos = match.end()
        sentence = text_buffer[:end_pos].strip()
        remaining = text_buffer[end_pos:].strip()
        
        # Keep punctuation-based validation - safer approach
        if len(sentence.split()) < config.MIN_WORD_COUNT:  # Configurable validation
            return "", text_buffer
            
        return sentence, remaining

class VoiceOrchestrator:
    def __init__(self):
        self.whisper_host = config.WHISPER_URL.split("://")[1].split(":")[0]
        self.whisper_port = int(config.WHISPER_URL.split(":")[-1])
        self.ollama_url = config.OLLAMA_URL
        self.tts_url = config.TTS_URL
        self.diglett_url = config.DIGLETT_URL
        self.memory_enabled = config.MEMORY_ENABLED
        self.amem_url = config.AMEM_URL
        
        # Initialize sentence completion detector
        self.sentence_detector = SentenceCompletionDetector()
        
        # Initialize streaming sentence detector for pipeline
        self.streaming_detector = StreamingSentenceDetector()
        
        # Simple session memory for conversation continuity
        self.session_history = {}  # session_id -> list of {"user": str, "assistant": str}
        
        # Track active TTS sessions for interruption capability
        self.active_tts_sessions = {}  # session_id -> {"thread": Thread, "queue": Queue, "abort_flag": Event}
        
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
    
    async def synthesize_stream(self, text: str):
        """Stream text to speech using Kokoro FastAPI with optimized parameters"""
        try:
            async with httpx.AsyncClient(timeout=config.OLLAMA_TIMEOUT) as client:
                # Use Kokoro FastAPI streaming with optimized parameters for ultra-low latency
                async with client.stream(
                    "POST",
                    f"{self.tts_url}/v1/audio/speech",  # Correct Kokoro endpoint
                    json={
                        "model": "kokoro",
                        "input": text,
                        "voice": config.TTS_VOICE,
                        "response_format": "wav",  # WAV format for browser compatibility
                        "stream": True,
                        "speed": config.TTS_SPEED,
                        "volume_multiplier": config.TTS_VOLUME
                    }
                ) as response:
                    response.raise_for_status()
                    
                    # Stream WAV chunks - browser can decode these directly
                    async for chunk in response.aiter_bytes(chunk_size=config.CHUNK_SIZE):
                        if chunk:
                            yield chunk
                            
        except Exception as e:
            logger.error(f"TTS streaming error: {e}")
            yield b""

    async def stream_llm_tokens(self, text: str, context: str = ""):
        """Stream LLM tokens using native ollama client for maximum performance"""
        prompt = f"{context}\n\nUser: {text}\nAssistant:" if context else f"User: {text}\nAssistant:"
        
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

    def stream_direct_with_real_time_streaming(self, text: str, session_id: str = "default"):
        """Real-time streaming: yields audio chunks as each sentence completes"""
        try:
            import requests
            import threading
            from queue import Queue
            import base64
            import json
            
            logger.info(f"🚀 REAL-TIME STREAMING: Starting sentence-by-sentence audio streaming")
            
            # Create abort flag for this session
            abort_flag = threading.Event()
            
            # Initialize session history if not exists
            if session_id not in self.session_history:
                self.session_history[session_id] = []
            
            # Track this session for interruption capability
            self.active_tts_sessions[session_id] = {
                "abort_flag": abort_flag,
                "start_time": time.time()
            }
            
            # Build context-aware prompt with recent conversation history
            history = self.session_history[session_id]
            prompt_parts = []
            
            # Include recent conversation turns (last 3 exchanges for context)
            for exchange in history[-3:]:
                prompt_parts.append(f"User: {exchange['user']}")
                prompt_parts.append(f"Assistant: {exchange['assistant']}")
            
            # Add current user input
            prompt_parts.append(f"User: {text}")
            prompt_parts.append("Assistant:")
            
            prompt = "\n".join(prompt_parts)
            
            # Log session context for debugging
            logger.info(f"Session {session_id} history: {len(history)} exchanges")
            logger.info(f"Prompt context: {prompt[:200]}..." if len(prompt) > 200 else f"Prompt: {prompt}")
            
            # Real-time streaming coordination
            token_buffer = ""
            full_response = ""
            tts_queue = Queue()
            audio_result_queue = Queue()  # For getting audio chunks back from worker
            
            def tts_worker():
                """Background worker that yields audio chunks in real-time"""
                sentence_count = 0
                while True:
                    # Check for interruption before processing next sentence
                    if abort_flag.is_set():
                        logger.info(f"🛑 TTS Worker: Interruption detected, stopping TTS generation")
                        audio_result_queue.put({"type": "interrupted"})
                        break
                    
                    sentence = tts_queue.get()
                    if sentence == "DONE":
                        audio_result_queue.put({"type": "done"})
                        break
                    elif sentence == "INTERRUPTED":
                        logger.info(f"🛑 TTS Worker: Received interruption signal from LLM processor")
                        audio_result_queue.put({"type": "interrupted"})
                        break
                    
                    # Check again after getting sentence from queue
                    if abort_flag.is_set():
                        logger.info(f"🛑 TTS Worker: Interruption detected after queue get, stopping")
                        audio_result_queue.put({"type": "interrupted"})
                        break
                    
                    try:
                        sentence_count += 1
                        logger.info(f"🎵 TTS Worker: Processing sentence {sentence_count}: {sentence[:50]}...")
                        
                        tts_response = requests.post(
                            f"{self.tts_url}/v1/audio/speech",
                            json={
                                "model": "kokoro",
                                "input": sentence,
                                "voice": config.TTS_VOICE,
                                "response_format": "wav",
                                "stream": False,
                                "speed": config.TTS_SPEED,
                            },
                            timeout=config.TTS_TIMEOUT
                        )
                        
                        if tts_response.status_code == 200:
                            # Put audio chunk in result queue for immediate streaming
                            audio_result_queue.put({
                                "type": "audio",
                                "sequence": sentence_count,
                                "text": sentence,
                                "audio_data": tts_response.content
                            })
                            logger.info(f"✅ TTS Worker: Sentence {sentence_count} ready! {len(tts_response.content)} bytes")
                        else:
                            logger.error(f"❌ TTS Worker: Sentence {sentence_count} failed with status {tts_response.status_code}")
                            audio_result_queue.put({
                                "type": "error",
                                "sequence": sentence_count,
                                "message": f"TTS failed with status {tts_response.status_code}"
                            })
                            
                    except Exception as e:
                        logger.error(f"💥 TTS Worker error on sentence {sentence_count}: {e}")
                        audio_result_queue.put({
                            "type": "error",
                            "sequence": sentence_count,
                            "message": str(e)
                        })
                    finally:
                        tts_queue.task_done()
            
            # Start TTS worker thread
            tts_thread = threading.Thread(target=tts_worker, daemon=True)
            tts_thread.start()
            
            # Stream LLM tokens with sentence boundary detection
            def llm_processor():
                """Process LLM stream and detect sentence boundaries"""
                nonlocal token_buffer, full_response
                
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": config.LLM_MODEL,
                        "prompt": prompt,
                        "stream": True,
                        "options": {
                            "num_predict": config.LLM_MAX_TOKENS,
                            "temperature": config.LLM_TEMPERATURE,
                            "num_ctx": 2048,
                            "top_p": 0.9
                        }
                    },
                    timeout=config.OLLAMA_TIMEOUT,
                    stream=True
                )
                
                logger.info(f"📡 LLM streaming status: {response.status_code}")
                
                # Process streaming tokens with sentence boundary detection
                for line in response.iter_lines():
                    # Check for interruption before processing each token
                    if abort_flag.is_set():
                        logger.info(f"🛑 LLM Processor: Interruption detected, stopping token processing")
                        break
                        
                    if line:
                        chunk = json.loads(line)
                        if chunk.get('response'):
                            token = chunk['response']
                            token_buffer += token
                            full_response += token
                            
                            # Word-by-word streaming: send every 3-5 words to TTS immediately
                            words = token_buffer.split()
                            if len(words) >= 4:  # Send every 4 words for ultra-low latency
                                # Check for interruption before sending to TTS
                                if abort_flag.is_set():
                                    logger.info(f"🛑 LLM Processor: Interruption detected, stopping word streaming")
                                    break
                                    
                                # Take first 4 words and send to TTS
                                words_to_send = words[:4]
                                chunk_text = " ".join(words_to_send)
                                
                                logger.info(f"🚀 WORD STREAMING: {chunk_text}")
                                tts_queue.put(chunk_text)
                                
                                # Keep remaining words in buffer
                                remaining_words = words[4:]
                                token_buffer = " ".join(remaining_words)
                        
                        if chunk.get('done'):
                            # Process any remaining words as final chunk
                            if token_buffer.strip() and not abort_flag.is_set():
                                logger.info(f"🏁 Final words: {token_buffer[:50]}...")
                                tts_queue.put(token_buffer.strip())
                            break
                
                # Signal TTS worker completion (or interruption)
                if abort_flag.is_set():
                    logger.info(f"🛑 LLM Processor: Session interrupted, signaling TTS worker to stop")
                    tts_queue.put("INTERRUPTED")
                else:
                    tts_queue.put("DONE")
            
            # Start LLM processing in background
            llm_thread = threading.Thread(target=llm_processor, daemon=True)
            llm_thread.start()
            
            # Yield audio chunks in real-time as they become available
            sentences_received = 0
            complete_audio_chunks = []
            
            while True:
                audio_item = audio_result_queue.get()
                
                if audio_item["type"] == "done":
                    logger.info(f"🎉 All sentences processed! Total: {sentences_received}")
                    break
                elif audio_item["type"] == "interrupted":
                    logger.info(f"🛑 TTS generation interrupted! Stopping stream.")
                    yield {
                        "type": "interrupted",
                        "message": "TTS generation was interrupted",
                        "sentences_completed": sentences_received
                    }
                    break
                elif audio_item["type"] == "audio":
                    sentences_received += 1
                    sequence = audio_item["sequence"]
                    text_part = audio_item["text"]
                    audio_data = audio_item["audio_data"]
                    
                    complete_audio_chunks.append(audio_data)
                    
                    # Yield this chunk immediately for real-time playback
                    yield {
                        "type": "text_chunk",
                        "sequence": sequence,
                        "text": text_part,
                        "total_sentences": sentences_received
                    }
                    
                    # Split large audio into smaller chunks to avoid JSON parsing issues
                    chunk_size = 32768  # 32KB chunks
                    audio_b64 = base64.b64encode(audio_data).decode()
                    
                    if len(audio_b64) > chunk_size:
                        # Send audio in multiple smaller chunks
                        for i in range(0, len(audio_b64), chunk_size):
                            chunk_part = audio_b64[i:i+chunk_size]
                            is_last_chunk = (i + chunk_size >= len(audio_b64))
                            
                            yield {
                                "type": "audio_chunk_part",
                                "sequence": sequence,
                                "chunk_index": i // chunk_size,
                                "audio_data": chunk_part,
                                "is_last": is_last_chunk,
                                "total_size": len(audio_data)
                            }
                    else:
                        # Small enough to send as single chunk
                        yield {
                            "type": "audio_chunk", 
                            "sequence": sequence,
                            "audio_data": audio_b64,
                            "size_bytes": len(audio_data)
                        }
                    
                    logger.info(f"🎵 STREAMED sentence {sequence} to frontend immediately!")
                    
                elif audio_item["type"] == "error":
                    logger.error(f"❌ Error in sentence {audio_item['sequence']}: {audio_item['message']}")
            
            # Wait for threads to complete
            llm_thread.join(timeout=2)
            tts_thread.join(timeout=2)
            
            # Store conversation in session history for context
            if full_response:
                exchange = {
                    "user": text,
                    "assistant": full_response
                }
                self.session_history[session_id].append(exchange)
                logger.info(f"💾 Stored exchange in session {session_id}: User: '{text}' -> Assistant: '{full_response[:50]}...'")
                
                # Keep session history reasonable (last 10 exchanges)
                if len(self.session_history[session_id]) > 10:
                    self.session_history[session_id] = self.session_history[session_id][-10:]
                    logger.info(f"🧹 Trimmed session {session_id} history to last 10 exchanges")
            
            # Final completion message
            yield {
                "type": "complete",
                "total_sentences": sentences_received,
                "full_text": full_response,
                "total_audio_bytes": sum(len(chunk) for chunk in complete_audio_chunks)
            }
            
            # Clean up the session
            self.cleanup_completed_session(session_id)
            
        except Exception as e:
            logger.error(f"💥 Real-time streaming error: {e}")
            # Clean up the session on error
            self.cleanup_completed_session(session_id)
            yield {
                "type": "error",
                "message": str(e)
            }

    async def tts_processing_worker(self, sentence_queue: asyncio.Queue, audio_queue: asyncio.Queue):
        """Worker that processes sentences through TTS streaming"""
        try:
            while True:
                # Get next sentence from queue
                item = await sentence_queue.get()
                
                if item["type"] == "done":
                    logger.info("TTS Worker: Received done signal")
                    await audio_queue.put({"type": "done"})
                    break
                    
                if item["type"] == "error":
                    logger.error(f"TTS Worker: Received error: {item['message']}")
                    await audio_queue.put(item)
                    break
                
                if item["type"] in ["sentence", "final"]:
                    sequence = item["sequence"]
                    text = item["text"]
                    logger.info(f"TTS Worker: Processing sentence {sequence}: {text[:30]}...")
                    
                    # Stream TTS audio for this sentence
                    audio_chunks = []
                    async for audio_chunk in self.synthesize_stream(text):
                        if audio_chunk:
                            audio_chunks.append(audio_chunk)
                    
                    # Queue the complete audio for this sentence
                    if audio_chunks:
                        await audio_queue.put({
                            "sequence": sequence,
                            "audio_chunks": audio_chunks,
                            "text": text,
                            "type": "audio"
                        })
                        logger.info(f"TTS Worker: Completed sentence {sequence}, {len(audio_chunks)} chunks")
                    
        except Exception as e:
            logger.error(f"TTS processing worker error: {e}")
            await audio_queue.put({"type": "error", "message": str(e)})

    async def audio_response_worker(self, audio_queue: asyncio.Queue, session_id: str):
        """Worker that coordinates audio chunks and streams response"""
        try:
            sentence_audio = {}  # Store audio by sequence number
            next_sequence = 1
            complete_response = ""
            
            while True:
                # Get next audio item from queue
                item = await audio_queue.get()
                
                if item["type"] == "done":
                    logger.info("Audio Response Worker: Received done signal")
                    break
                    
                if item["type"] == "error":
                    logger.error(f"Audio Response Worker: Received error: {item['message']}")
                    # Fallback to existing pipeline
                    return
                
                if item["type"] == "audio":
                    sequence = item["sequence"]
                    audio_chunks = item["audio_chunks"]
                    text = item["text"]
                    
                    # Store audio by sequence
                    sentence_audio[sequence] = {
                        "audio_chunks": audio_chunks,
                        "text": text
                    }
                    
                    logger.info(f"Audio Response Worker: Stored sentence {sequence}")
                    
                    # Yield any consecutive sentences starting from next_sequence
                    while next_sequence in sentence_audio:
                        sentence_data = sentence_audio[next_sequence]
                        complete_response += sentence_data["text"] + " "
                        
                        # Yield text first
                        yield {
                            "type": "text",
                            "sequence": next_sequence,
                            "text": sentence_data["text"],
                            "complete_text": complete_response.strip()
                        }
                        
                        # Then yield audio chunks
                        for chunk in sentence_data["audio_chunks"]:
                            yield {
                                "type": "audio",
                                "sequence": next_sequence,
                                "audio_chunk": chunk
                            }
                        
                        # Clean up and move to next
                        del sentence_audio[next_sequence]
                        next_sequence += 1
                        
                        logger.info(f"Audio Response Worker: Streamed sentence {next_sequence - 1}")
            
            # Yield completion
            yield {"type": "complete", "complete_text": complete_response.strip()}
            
        except Exception as e:
            logger.error(f"Audio response worker error: {e}")
            yield {"type": "error", "message": str(e)}
    
    async def retrieve_context(self, query: str, session_id: str) -> str:
        """Retrieve conversation context from A-MEM"""
        try:
            async with httpx.AsyncClient(timeout=config.AMEM_TIMEOUT) as client:
                response = await client.post(
                    f"{self.amem_url}/retrieve",
                    json={"query": query, "session_id": session_id, "k": 5}
                )
                response.raise_for_status()
                return response.json().get("context", "")
        except Exception as e:
            logger.error(f"Context retrieval error: {e}")
            return ""
    
    async def store_interaction(self, user_input: str, ai_response: str, session_id: str):
        """Store conversation in A-MEM"""
        try:
            async with httpx.AsyncClient(timeout=config.AMEM_TIMEOUT) as client:
                await client.post(
                    f"{self.amem_url}/store",
                    json={
                        "user_input": user_input,
                        "ai_response": ai_response,
                        "session_id": session_id
                    }
                )
        except Exception as e:
            logger.error(f"Context storage error: {e}")
    
    def interrupt_tts_session(self, session_id: str) -> bool:
        """Interrupt active TTS generation for a specific session"""
        try:
            if session_id in self.active_tts_sessions:
                session_data = self.active_tts_sessions[session_id]
                
                # Set abort flag to stop any ongoing processing
                if 'abort_flag' in session_data:
                    session_data['abort_flag'].set()
                    logger.info(f"TTS interruption signal sent for session {session_id}")
                
                # Clear the session from active tracking
                del self.active_tts_sessions[session_id]
                logger.info(f"TTS session {session_id} interrupted and cleaned up")
                return True
            else:
                logger.warning(f"No active TTS session found for session {session_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error interrupting TTS session {session_id}: {e}")
            return False
    
    def cleanup_completed_session(self, session_id: str):
        """Clean up a completed TTS session"""
        try:
            if session_id in self.active_tts_sessions:
                del self.active_tts_sessions[session_id]
                logger.info(f"Cleaned up completed TTS session {session_id}")
        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {e}")

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

@app.post("/warmup")
async def warmup():
    """Warm up the Ollama model for faster subsequent responses"""
    try:
        start_time = time.time()
        
        # Quick warm-up generation
        client = ollama.AsyncClient(host=orchestrator.ollama_url)
        await client.generate(
            model=config.LLM_MODEL,
            prompt="Hi",
            options={"num_predict": 1}
        )
        
        warmup_time = (time.time() - start_time) * 1000
        logger.info(f"Model warmed up in {warmup_time:.2f}ms")
        
        return {
            "status": "warmed_up",
            "warmup_time_ms": warmup_time,
            "model": config.LLM_MODEL
        }
        
    except Exception as e:
        logger.error(f"Warmup error: {e}")
        return {"error": str(e)}, 500

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

class InterruptRequest(BaseModel):
    session_id: str

@app.post("/process-transcript")
async def process_transcript(request: TranscriptRequest):
    """Process transcript through LLM and TTS pipeline with sentence completion detection"""
    try:
        start_time = time.time()
        
        # 1. Check if sentence is complete and ready for processing
        is_complete, cleaned_sentence = orchestrator.sentence_detector.is_sentence_complete(request.transcript, request.session_id)
        
        if not is_complete:
            logger.info(f"Sentence not complete, skipping: {request.transcript}")
            return {
                "response_text": "",
                "audio_data": None,
                "latency_ms": 0,
                "sentence_complete": False
            }
        
        logger.info(f"Processing complete sentence: {cleaned_sentence}")
        
        # 2. Retrieve context if memory enabled
        context = ""
        if orchestrator.memory_enabled:
            try:
                context = await orchestrator.retrieve_context(cleaned_sentence, request.session_id)
            except Exception as e:
                logger.warning(f"Memory retrieval failed: {e}")
        
        # 3. Generate response using Ollama
        response = await orchestrator.generate_response(cleaned_sentence, context)
        logger.info(f"LLM Response: {response}")
        
        # 4. Store interaction if memory enabled
        if orchestrator.memory_enabled:
            try:
                await orchestrator.store_interaction(cleaned_sentence, response, request.session_id)
            except Exception as e:
                logger.warning(f"Memory storage failed: {e}")
        
        # 5. Synthesize speech using Kokoro
        audio_response = await orchestrator.synthesize(response)
        
        total_time = (time.time() - start_time) * 1000
        logger.info(f"Total pipeline latency: {total_time:.2f}ms")
        
        return {
            "response_text": response,
            "audio_data": base64.b64encode(audio_response).decode() if audio_response else None,
            "latency_ms": total_time,
            "sentence_complete": True
        }
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        return {"error": str(e)}, 500

@app.post("/process-transcript-stream")
async def process_transcript_stream(request: TranscriptRequest):
    """Process transcript through LLM and stream TTS response with sentence completion detection"""
    try:
        start_time = time.time()
        
        # 1. Check if sentence is complete and ready for processing
        is_complete, cleaned_sentence = orchestrator.sentence_detector.is_sentence_complete(request.transcript, request.session_id)
        
        if not is_complete:
            logger.info(f"Sentence not complete, skipping: {request.transcript}")
            # Return empty stream for incomplete sentences
            async def empty_stream():
                yield f"data: {json.dumps({'type': 'incomplete', 'message': 'Sentence not complete'})}\n\n"
            
            return StreamingResponse(
                empty_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Cache-Control"
                }
            )
        
        logger.info(f"Processing complete sentence: {cleaned_sentence}")
        
        # 2. Retrieve context if memory enabled
        context = ""
        if orchestrator.memory_enabled:
            try:
                context = await orchestrator.retrieve_context(cleaned_sentence, request.session_id)
            except Exception as e:
                logger.warning(f"Memory retrieval failed: {e}")
        
        # 3. Generate response using Ollama
        response = await orchestrator.generate_response(cleaned_sentence, context)
        logger.info(f"LLM Response: {response}")
        
        # 4. Store interaction if memory enabled
        if orchestrator.memory_enabled:
            try:
                await orchestrator.store_interaction(cleaned_sentence, response, request.session_id)
            except Exception as e:
                logger.warning(f"Memory storage failed: {e}")
        
        # 5. Stream TTS response
        async def generate_audio_stream():
            """Generate streaming audio response"""
            yield f"data: {json.dumps({'type': 'text', 'data': response})}\n\n"
            
            async for audio_chunk in orchestrator.synthesize_stream(response):
                if audio_chunk:
                    # Convert binary chunk to base64 for streaming
                    import base64
                    chunk_b64 = base64.b64encode(audio_chunk).decode()
                    yield f"data: {json.dumps({'type': 'audio', 'data': chunk_b64})}\n\n"
            
            total_time = (time.time() - start_time) * 1000
            yield f"data: {json.dumps({'type': 'complete', 'latency_ms': total_time})}\n\n"
        
        return StreamingResponse(
            generate_audio_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control"
            }
        )
        
    except Exception as e:
        logger.error(f"Streaming pipeline error: {e}")
        return {"error": str(e)}, 500

@app.post("/ultra-fast")
async def ultra_fast(request: TranscriptRequest):
    """Ultra-fast direct pipeline - sub-second target with session management"""
    try:
        start_time = time.time()
        
        logger.info(f"Ultra-Fast: {request.transcript} (session: {request.session_id})")
        
        # 1. Simple heuristic: if input is very short or looks like interruption sound, 
        # skip sentence completion logic to allow for quick interruptions
        transcript_words = request.transcript.strip().split()
        is_likely_interruption = (
            len(transcript_words) <= 2 and  # Very short input
            len(request.transcript.strip()) <= 15  # Short character count
        )
        
        if is_likely_interruption:
            # For short inputs that might be interruptions, apply minimal validation
            cleaned_sentence = request.transcript.strip()
            if len(cleaned_sentence) < 3:  # Skip very short inputs like "a", "I", "oh", "um"
                logger.info(f"Ultra-Fast: Input too short, skipping: {request.transcript}")
                return {
                    "response_text": "",
                    "audio_data": None,
                    "latency_ms": 0,
                    "sentence_complete": False,
                    "method": "direct_sync"
                }
            logger.info(f"Ultra-Fast: Processing likely interruption: {cleaned_sentence}")
        else:
            # For longer inputs, use normal sentence completion detection
            is_complete, cleaned_sentence = orchestrator.sentence_detector.is_sentence_complete(request.transcript, request.session_id)
            
            if not is_complete:
                logger.info(f"Ultra-Fast: Sentence not complete, skipping: {request.transcript}")
                return {
                    "response_text": "",
                    "audio_data": None,
                    "latency_ms": 0,
                    "sentence_complete": False,
                    "method": "direct_sync"
                }
            
            logger.info(f"Ultra-Fast: Processing complete sentence: {cleaned_sentence}")
        
        # 2. Direct streaming call with sentence-by-sentence TTS
        text_response, audio_data = orchestrator.stream_direct_with_sentence_streaming(cleaned_sentence, request.session_id)
        
        total_time = (time.time() - start_time) * 1000
        logger.info(f"Ultra-Fast: Total = {total_time:.2f}ms")
        
        return {
            "response_text": text_response,
            "audio_data": base64.b64encode(audio_data).decode() if audio_data else None,
            "latency_ms": total_time,
            "sentence_complete": True,
            "method": "direct_sync"
        }
        
    except Exception as e:
        logger.error(f"Ultra-fast error: {e}")
        return {"error": str(e)}, 500

@app.post("/ultra-fast-stream")
async def ultra_fast_stream(request: TranscriptRequest):
    """🚀 REAL-TIME BINARY STREAMING: Stream raw audio chunks like Kokoro"""
    try:
        start_time = time.time()
        
        logger.info(f"🚀 Ultra-Fast-Stream: {request.transcript} (session: {request.session_id})")
        
        # 🛑 CRITICAL: Interrupt any existing TTS session for this user before starting new one
        if request.session_id in orchestrator.active_tts_sessions:
            logger.info(f"🛑 INTERRUPTING existing TTS session for {request.session_id}")
            orchestrator.interrupt_tts_session(request.session_id)
        
        # 1. Sentence completion detection (same logic as ultra-fast)
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
        
        # 2. Real-time streaming with sentence-level TTS (RealtimeTTS pattern)
        async def realtime_sentence_stream():
            """Stream individual sentence WAVs as they complete"""
            try:
                # Initialize session history if not exists
                if request.session_id not in orchestrator.session_history:
                    orchestrator.session_history[request.session_id] = []
                
                # Build context-aware prompt
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
                
                # Start streaming LLM tokens and process in real-time
                async for token in orchestrator.stream_llm_tokens(cleaned_sentence, context):
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
                                import base64
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
                                import base64
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

@app.get("/debug/sessions")
async def debug_sessions():
    """Debug endpoint to check active TTS sessions"""
    try:
        session_details = {}
        for session_id, data in orchestrator.active_tts_sessions.items():
            session_details[session_id] = {
                "start_time": data.get("start_time", "unknown"),
                "abort_flag_set": data.get("abort_flag").is_set() if data.get("abort_flag") else False,
                "has_abort_flag": "abort_flag" in data
            }
        
        return {
            "active_sessions": list(orchestrator.active_tts_sessions.keys()),
            "session_count": len(orchestrator.active_tts_sessions),
            "session_details": session_details,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Debug sessions error: {e}")
        return {"error": str(e)}, 500

@app.post("/interrupt-tts")
async def interrupt_tts(request: InterruptRequest):
    """Interrupt active TTS generation for a specific session"""
    try:
        start_time = time.time()
        
        logger.info(f"🛑 TTS INTERRUPTION: Received interrupt request for session {request.session_id}")
        
        # Attempt to interrupt the session
        success = orchestrator.interrupt_tts_session(request.session_id)
        
        interrupt_time = (time.time() - start_time) * 1000
        
        if success:
            logger.info(f"✅ TTS INTERRUPTION: Successfully interrupted session {request.session_id} in {interrupt_time:.2f}ms")
            return {
                "status": "interrupted",
                "session_id": request.session_id,
                "interrupt_time_ms": interrupt_time,
                "message": "TTS generation interrupted successfully"
            }
        else:
            logger.warning(f"⚠️ TTS INTERRUPTION: No active session found for {request.session_id}")
            return {
                "status": "no_active_session",
                "session_id": request.session_id,
                "interrupt_time_ms": interrupt_time,
                "message": "No active TTS session to interrupt"
            }
            
    except Exception as e:
        logger.error(f"💥 TTS INTERRUPTION ERROR: {e}")
        return {"error": str(e)}, 500

@app.post("/process-transcript-pipeline")
async def process_transcript_pipeline(request: TranscriptRequest):
    """Ultra-low latency pipeline: LLM streaming + sentence-based TTS with overlapping processing"""
    try:
        start_time = time.time()
        
        # 1. Check if sentence is complete and ready for processing (using existing logic)
        is_complete, cleaned_sentence = orchestrator.sentence_detector.is_sentence_complete(request.transcript)
        
        if not is_complete:
            logger.info(f"Sentence not complete, skipping: {request.transcript}")
            # Return empty stream for incomplete sentences
            async def empty_stream():
                yield f"data: {json.dumps({'type': 'incomplete', 'message': 'Sentence not complete'})}\n\n"
            
            return StreamingResponse(
                empty_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Cache-Control"
                }
            )
        
        logger.info(f"Ultra-Low Latency Pipeline: Processing sentence: {cleaned_sentence[:50]}...")
        
        # 2. Retrieve context if memory enabled
        context = ""
        if orchestrator.memory_enabled:
            try:
                context = await orchestrator.retrieve_context(cleaned_sentence, request.session_id)
            except Exception as e:
                logger.warning(f"Memory retrieval failed: {e}")
        
        # 3. Ultra-Low Latency Streaming Pipeline
        async def ultra_low_latency_stream():
            """Coordinate LLM streaming + sentence-based TTS processing"""
            try:
                # Use direct stream method for maximum speed
                start_time = time.time()
                text_response, audio_data = orchestrator.stream_direct_with_sentence_streaming(cleaned_sentence, request.session_id)
                
                # Stream response as JSON events
                yield f"data: {json.dumps({'type': 'text', 'text': text_response})}\\n\\n"
                
                if audio_data:
                    import base64
                    audio_b64 = base64.b64encode(audio_data).decode()
                    yield f"data: {json.dumps({'type': 'audio', 'data': audio_b64})}\\n\\n"
                
                total_time = (time.time() - start_time) * 1000
                yield f"data: {json.dumps({'type': 'complete', 'latency_ms': total_time})}\\n\\n"
                
            except Exception as e:
                logger.error(f"Ultra-low latency stream error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\\n\\n"
        
        return StreamingResponse(
            ultra_low_latency_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control"
            }
        )
        
    except Exception as e:
        logger.error(f"Ultra-low latency pipeline error: {e}")
        return {"error": str(e)}, 500

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Voice Orchestrator starting up...")
    logger.info(f"Memory enabled: {orchestrator.memory_enabled}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    logger.info("Voice Orchestrator shutting down...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)