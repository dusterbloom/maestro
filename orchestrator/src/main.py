import asyncio
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
# from whisper_live.client import TranscriptionClient

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
        
        # Track conversation state
        self.last_processed_sentence = ""
        self.current_session_sentences = set()
    
    def is_sentence_complete(self, text: str) -> tuple[bool, str]:
        """
        Determine if a sentence is complete and ready for processing
        Returns: (is_complete, cleaned_sentence)
        """
        if not text or not text.strip():
            return False, ""
            
        cleaned = text.strip()
        
        # Skip if this exact sentence was already processed
        if cleaned in self.current_session_sentences:
            logger.info(f"Sentence already processed: {cleaned}")
            return False, ""
        
        # Check for ellipsis (indicates incomplete thought)
        if self.ellipsis_pattern.search(cleaned):
            return False, ""
        
        # Check for sentence ending punctuation
        if not self.sentence_endings.search(cleaned):
            return False, ""
        
        # Check if last word suggests continuation
        words = cleaned.split()
        if words:
            last_word = words[-1].lower().rstrip('.,!?')
            if last_word in self.continuation_words:
                return False, ""
        
        # Additional heuristics for completeness
        if len(words) < 3:  # Very short sentences might be incomplete
            return False, ""
        
        # Mark as processed and return complete
        self.current_session_sentences.add(cleaned)
        self.last_processed_sentence = cleaned
        return True, cleaned
    
    def reset_session(self):
        """Reset state for new conversation session"""
        self.current_session_sentences.clear()
        self.last_processed_sentence = ""

class StreamingSentenceDetector:
    """Simple sentence boundary detection for streaming pipeline"""
    
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
        if len(sentence.split()) < 3:
            return "", text_buffer
            
        return sentence, remaining

class VoiceOrchestrator:
    def __init__(self):
        self.whisper_host = os.getenv("WHISPER_HOST", "localhost")
        self.whisper_port = int(os.getenv("WHISPER_PORT", "9090"))
        self.ollama_url = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
        self.tts_url = os.getenv("TTS_URL", "http://kokoro:8880")
        self.memory_enabled = os.getenv("MEMORY_ENABLED", "false").lower() == "true"
        self.amem_url = os.getenv("AMEM_URL", "http://a-mem:8001")
        
        # Initialize sentence completion detector
        self.sentence_detector = SentenceCompletionDetector()
        
        # Initialize streaming sentence detector for pipeline
        self.streaming_detector = StreamingSentenceDetector()
        
        # WhisperLive client for direct transcription (disabled for now)
        # self.whisper_client = None
    
    # def get_whisper_client(self) -> TranscriptionClient:
    #     """Get or create WhisperLive client"""
    #     if self.whisper_client is None:
    #         self.whisper_client = TranscriptionClient(
    #             host=self.whisper_host,
    #             port=self.whisper_port,
    #             lang="en",
    #             translate=False,
    #             model="tiny",
    #             use_vad=True
    #         )
    #     return self.whisper_client
    # 
    # async def transcribe_audio(self, audio_file_path: str) -> str:
    #     """Transcribe audio file using WhisperLive client"""
    #     try:
    #         client = self.get_whisper_client()
    #         # Use synchronous transcription for audio files
    #         result = client.transcribe_file(audio_file_path)
    #         return result.get("text", "")
    #     except Exception as e:
    #         logger.error(f"Transcription error: {e}")
    #         return ""
    
    async def generate_response(self, text: str, context: str = "") -> str:
        """Generate response using Ollama with streaming"""
        prompt = f"{context}\n\nUser: {text}\nAssistant:" if context else f"User: {text}\nAssistant:"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": os.getenv("LLM_MODEL", "gemma3n:latest"),
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
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.tts_url}/v1/audio/speech",  # Verified endpoint from source
                    json={
                        "model": "kokoro",  # Required parameter from source
                        "input": text,
                        "voice": os.getenv("TTS_VOICE", "af_bella"),
                        "response_format": "wav",  # Keep WAV for non-streaming
                        "stream": False,
                        "speed": float(os.getenv("TTS_SPEED", "1.3")),
                        "volume_multiplier": float(os.getenv("TTS_VOLUME", "1.0"))
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
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Use Kokoro FastAPI streaming with optimized parameters for ultra-low latency
                async with client.stream(
                    "POST",
                    f"{self.tts_url}/v1/audio/speech",  # Verified endpoint from source
                    json={
                        "model": "kokoro",
                        "input": text,
                        "voice": os.getenv("TTS_VOICE", "af_bella"),
                        "response_format": "pcm",  # PCM for lowest latency
                        "stream": True,
                        "speed": float(os.getenv("TTS_SPEED", "1.3")),  # Optimized speed
                        "volume_multiplier": float(os.getenv("TTS_VOLUME", "1.0"))
                    }
                ) as response:
                    response.raise_for_status()
                    
                    # Stream PCM chunks with smaller chunk size for lower latency
                    async for chunk in response.aiter_bytes(chunk_size=256):
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
                model=os.getenv("LLM_MODEL", "gemma3n:latest"),
                prompt=prompt,
                stream=True,
                options={
                    "num_predict": 64,      # Very short responses for low latency
                    "temperature": 0.3,     # Lower temperature for faster generation
                    "top_p": 0.8,
                    "num_ctx": 1024,        # Minimal context for speed
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

    def stream_direct(self, text: str):
        """Direct streaming without async overhead - maximum speed"""
        try:
            import requests
            
            logger.info(f"Direct stream: Calling {self.ollama_url}/api/generate")
            
            # Very aggressive settings for speed
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "llama3.2:3b",  # Use smallest fast model
                    "prompt": f"Answer briefly: {text}",
                    "stream": False,  # Non-streaming for now to debug
                    "options": {
                        "num_predict": 16,      # Very short responses
                        "temperature": 0.1,     # Deterministic
                        "num_ctx": 256,         # Minimal context
                        "top_p": 0.9,
                        "stop": ["\n\n"]        # Stop early
                    }
                },
                timeout=10  # 10 second timeout
            )
            
            logger.info(f"Ollama response status: {response.status_code}")
            result = response.json()
            full_text = result.get('response', 'No response')
            
            logger.info(f"Generated text: {full_text[:50]}...")
            
            # Skip TTS for now to debug LLM speed
            return full_text, b"fake_audio"
            
        except Exception as e:
            logger.error(f"Direct stream error: {e}")
            return f"Error: {str(e)}", b""

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
            async with httpx.AsyncClient(timeout=5.0) as client:
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
            async with httpx.AsyncClient(timeout=5.0) as client:
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
            model=os.getenv("LLM_MODEL", "gemma3n:latest"),
            prompt="Hi",
            options={"num_predict": 1}
        )
        
        warmup_time = (time.time() - start_time) * 1000
        logger.info(f"Model warmed up in {warmup_time:.2f}ms")
        
        return {
            "status": "warmed_up",
            "warmup_time_ms": warmup_time,
            "model": os.getenv("LLM_MODEL", "gemma3n:latest")
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

@app.post("/process-transcript")
async def process_transcript(request: TranscriptRequest):
    """Process transcript through LLM and TTS pipeline with sentence completion detection"""
    try:
        start_time = time.time()
        
        # 1. Check if sentence is complete and ready for processing
        is_complete, cleaned_sentence = orchestrator.sentence_detector.is_sentence_complete(request.transcript)
        
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
            "audio_data": audio_response.hex() if audio_response else None,
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
    """Ultra-fast direct pipeline - sub-second target"""
    try:
        start_time = time.time()
        
        logger.info(f"Ultra-Fast: {request.transcript}")
        
        # Direct synchronous call
        text_response, audio_data = orchestrator.stream_direct(request.transcript)
        
        total_time = (time.time() - start_time) * 1000
        logger.info(f"Ultra-Fast: Total = {total_time:.2f}ms")
        
        return {
            "response_text": text_response,
            "audio_data": audio_data.hex() if audio_data else None,
            "latency_ms": total_time,
            "method": "direct_sync"
        }
        
    except Exception as e:
        logger.error(f"Ultra-fast error: {e}")
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
                # Use the new ultra-low latency method
                first_response_time = None
                async for response_item in orchestrator.stream_llm_to_tts(cleaned_sentence, context):
                    
                    if first_response_time is None:
                        first_response_time = time.time()
                        ttfr = (first_response_time - start_time) * 1000  # Time to first response
                        logger.info(f"Ultra-Low Latency Pipeline: TTFR = {ttfr:.2f}ms")
                    
                    if response_item["type"] == "text":
                        # Send text response immediately
                        data = {
                            'type': 'text',
                            'sequence': response_item['sequence'],
                            'text': response_item['text'],
                            'complete_text': response_item['complete_text']
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                        
                    elif response_item["type"] == "audio":
                        # Send audio chunk
                        import base64
                        chunk_b64 = base64.b64encode(response_item["audio_chunk"]).decode()
                        data = {
                            'type': 'audio',
                            'sequence': response_item['sequence'],
                            'data': chunk_b64
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                        
                    elif response_item["type"] == "complete":
                        # Pipeline completed
                        total_time = (time.time() - start_time) * 1000
                        logger.info(f"Ultra-Low Latency Pipeline: Total time = {total_time:.2f}ms")
                        
                        # Store interaction if memory enabled
                        if orchestrator.memory_enabled:
                            try:
                                await orchestrator.store_interaction(
                                    cleaned_sentence, 
                                    response_item["complete_text"], 
                                    request.session_id
                                )
                            except Exception as e:
                                logger.warning(f"Memory storage failed: {e}")
                        
                        data = {
                            'type': 'complete',
                            'complete_text': response_item['complete_text'],
                            'latency_ms': total_time,
                            'ttfr_ms': (first_response_time - start_time) * 1000 if first_response_time else 0
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                        break
                        
                    elif response_item["type"] == "error":
                        # Error in pipeline - fallback to existing endpoint
                        logger.error(f"Pipeline error: {response_item['message']}, falling back to sequential processing")
                        data = {
                            'type': 'fallback',
                            'message': 'Pipeline error, falling back to sequential processing'
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                        break
                
                # Clean up workers
                for worker in workers:
                    if not worker.done():
                        worker.cancel()
                await asyncio.gather(*workers, return_exceptions=True)
                
            except Exception as e:
                logger.error(f"Ultra-low latency stream error: {e}")
                # Fallback error message
                data = {
                    'type': 'error',
                    'message': f'Pipeline error: {str(e)}'
                }
                yield f"data: {json.dumps(data)}\n\n"
        
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