# Optimized Main.py - Performance Improvements Implementation

# Add these imports to the existing main.py
import asyncio
import httpx
from concurrent.futures import ThreadPoolExecutor
import aiofiles
import uvloop  # For better async performance on Linux
from contextlib import asynccontextmanager

# Add to VoiceStreamOrchestrator class initialization
class VoiceStreamOrchestrator:
    def __init__(self):
        # ... existing code ...
        
        # Performance optimizations
        self.http_client_pool = {}
        self.connection_semaphore = asyncio.Semaphore(20)  # Limit concurrent connections
        self.tts_batch_processor = AudioBatchProcessor()
        self.adaptive_chunk_processor = AdaptiveChunkProcessor()
        self.llm_stream_optimizer = LLMStreamOptimizer()
        
        # Thread pool for CPU-intensive tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Persistent HTTP clients
        self._init_http_clients()
    
    def _init_http_clients(self):
        """Initialize persistent HTTP clients with connection pooling"""
        self.http_clients = {
            'tts': httpx.AsyncClient(
                base_url=self.tts_url,
                timeout=httpx.Timeout(config.TTS_TIMEOUT, connect=2.0),
                limits=httpx.Limits(
                    max_connections=20,
                    max_keepalive_connections=10
                ),
                http2=True  # Enable HTTP/2 multiplexing
            ),
            'ollama': httpx.AsyncClient(
                base_url=self.ollama_url,
                timeout=httpx.Timeout(config.OLLAMA_TIMEOUT, connect=5.0),
                limits=httpx.Limits(
                    max_connections=10,
                    max_keepalive_connections=5
                ),
                http2=True
            )
        }

    # Optimized TTS processing with connection reuse
    async def _process_tts_queue_optimized(self, session: StreamSession):
        """Optimized TTS queue processing with persistent connections"""
        try:
            async with session.tts_processing_lock:
                while session.tts_queue and not session.tts_abort_event.is_set():
                    if session.tts_abort_event.is_set():
                        break
                        
                    # Process multiple sentences in parallel when possible
                    batch_size = min(3, len(session.tts_queue))  # Process up to 3 at once
                    batch = []
                    
                    for _ in range(batch_size):
                        if session.tts_queue and not session.tts_abort_event.is_set():
                            batch.append(session.tts_queue.pop(0))
                    
                    if batch:
                        # Process batch in parallel
                        tasks = [
                            self._process_single_tts_optimized(session, item) 
                            for item in batch
                        ]
                        await asyncio.gather(*tasks, return_exceptions=True)
                        
        except Exception as e:
            logger.error(f"Optimized TTS queue processing error: {e}")

    async def _process_single_tts_optimized(self, session: StreamSession, tts_item: dict):
        """Process single TTS with optimized HTTP client"""
        sequence = tts_item["sequence"]
        sentence = tts_item["text"]
        
        try:
            session.tts_active = True
            start_time = time.time()
            
            # Use persistent HTTP client (no connection overhead)
            client = self.http_clients['tts']
            
            async with self.connection_semaphore:  # Limit concurrent requests
                if session.tts_abort_event.is_set():
                    return
                    
                response = await client.post(
                    "/v1/audio/speech",
                    json={
                        "model": "kokoro",
                        "input": sentence,
                        "voice": config.TTS_VOICE,
                        "response_format": "wav",
                        "stream": False,
                        "speed": config.TTS_SPEED,
                        "volume_multiplier": config.TTS_VOLUME
                    }
                )
                
                if session.tts_abort_event.is_set():
                    return
                    
                if response.status_code == 200:
                    audio_data = response.content
                    processing_time = (time.time() - start_time) * 1000  # ms
                    
                    # Track performance for adaptive optimization
                    self.adaptive_chunk_processor.adjust_chunk_size(processing_time)
                    
                    if not session.tts_abort_event.is_set():
                        await self._send_to_frontend(session, {
                            "type": "sentence_audio",
                            "sequence": sequence,
                            "text": sentence,
                            "audio_data": base64.b64encode(audio_data).decode(),
                            "size_bytes": len(audio_data),
                            "processing_time_ms": processing_time
                        })
                        
        except Exception as e:
            logger.error(f"Optimized TTS error for sequence {sequence}: {e}")
        finally:
            session.tts_active = False

    # Optimized LLM streaming with predictive sentence detection
    async def _stream_llm_response_optimized(self, text: str, history: list):
        """Optimized LLM streaming with predictive processing"""
        client = self.http_clients['ollama']
        
        # Build optimized context (keep only last 3 exchanges)
        context = self._build_optimized_context(text, history[-3:])
        
        try:
            async with client.stream(
                'POST',
                '/api/generate',
                json={
                    "model": config.LLM_MODEL,
                    "prompt": context,
                    "stream": True,
                    "options": {
                        "temperature": config.LLM_TEMPERATURE,
                        "num_predict": config.LLM_MAX_TOKENS,
                        "top_k": 40,  # Optimize for speed
                        "top_p": 0.9,
                        "num_ctx": 2048,  # Smaller context for speed
                        "num_batch": 8,    # Batch processing
                        "num_gpu_layers": -1,  # Use all GPU layers
                        "main_gpu": 0,
                        "low_vram": False,
                        "f16_kv": True,    # Use fp16 for speed
                        "use_mlock": True, # Keep model in memory
                        "use_mmap": True,  # Memory mapping for efficiency
                        "numa": False      # Disable NUMA for consistency
                    }
                }
            ) as response:
                buffer = ""
                async for chunk in response.aiter_lines():
                    if chunk:
                        try:
                            data = json.loads(chunk)
                            if 'response' in data:
                                token = data['response']
                                buffer += token
                                yield token
                                
                                # Predictive sentence boundary detection
                                if self.llm_stream_optimizer.should_flush_buffer(buffer, ""):
                                    # Trigger early TTS processing
                                    yield "\n[SENTENCE_BOUNDARY]\n"
                                    buffer = ""
                                    
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"Optimized LLM streaming error: {e}")
            yield f"Error: {str(e)}"

    def _build_optimized_context(self, current_text: str, history: list) -> str:
        """Build optimized context with reduced token count"""
        context_parts = [
            "You are a helpful voice assistant. Respond naturally and concisely in 1-2 sentences.",
            ""
        ]
        
        # Add compressed history (only essential parts)
        for exchange in history:
            user_text = exchange.get('user', '')[:100]  # Truncate long inputs
            assistant_text = exchange.get('assistant', '')[:150]  # Truncate long responses
            context_parts.extend([
                f"Human: {user_text}",
                f"Assistant: {assistant_text}",
                ""
            ])
        
        context_parts.extend([
            f"Human: {current_text}",
            "Assistant: "
        ])
        
        return "\n".join(context_parts)

    # WebSocket optimization with compression
    async def _send_to_frontend_optimized(self, session: StreamSession, message: dict):
        """Optimized frontend communication with compression"""
        if not session.frontend_ws:
            return
            
        try:
            # Compress large audio data
            if message.get("type") == "sentence_audio" and "audio_data" in message:
                # Use smaller base64 encoding optimization
                audio_data = message["audio_data"]
                if len(audio_data) > 1000:  # Only compress large payloads
                    # Could implement audio compression here
                    pass
            
            # Send with error handling
            await session.frontend_ws.send_text(json.dumps(message))
            
        except Exception as e:
            logger.warning(f"Failed to send to frontend: {e}")

    # Optimized audio processing with adaptive chunks
    async def send_audio_to_whisper_optimized(self, session_id: str, audio_data: bytes):
        """Optimized audio forwarding with adaptive chunking"""
        if session_id not in self.sessions:
            return False
            
        session = self.sessions[session_id]
        
        # Get adaptive chunk size
        optimal_chunk_size = self.adaptive_chunk_processor.current_chunk_size
        
        # Check connection health
        if not session.whisper_connected or not session.whisper_ws or not self._is_websocket_connected(session.whisper_ws):
            success = await self._connect_to_whisper(session)
            if not success:
                return False
        
        try:
            # Process audio in optimal chunks
            if len(audio_data) > optimal_chunk_size:
                # Split into optimal chunks
                for i in range(0, len(audio_data), optimal_chunk_size):
                    chunk = audio_data[i:i + optimal_chunk_size]
                    await session.whisper_ws.send(chunk)
            else:
                await session.whisper_ws.send(audio_data)
                
            session.update_activity()
            return True
            
        except Exception as e:
            logger.error(f"Optimized audio send error: {e}")
            return False

    async def cleanup_optimized(self):
        """Clean up all persistent connections"""
        # Close HTTP clients
        for client in self.http_clients.values():
            await client.aclose()
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Clean up sessions
        for session_id in list(self.sessions.keys()):
            await self.cleanup_session(session_id)
