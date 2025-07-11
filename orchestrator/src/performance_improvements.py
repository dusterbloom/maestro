# Performance Improvements for Maestro Voice Orchestrator

import asyncio
import httpx
from typing import Dict
import time

class HTTPClientPool:
    """Persistent HTTP client pool to avoid connection overhead"""
    
    def __init__(self):
        self.clients: Dict[str, httpx.AsyncClient] = {}
        self.client_locks: Dict[str, asyncio.Lock] = {}
    
    async def get_client(self, service_name: str, base_url: str) -> httpx.AsyncClient:
        """Get or create a persistent client for a service"""
        if service_name not in self.clients:
            self.clients[service_name] = httpx.AsyncClient(
                base_url=base_url,
                timeout=httpx.Timeout(10.0, connect=5.0),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                http2=True  # Enable HTTP/2 for multiplexing
            )
            self.client_locks[service_name] = asyncio.Lock()
        
        return self.clients[service_name]
    
    async def cleanup(self):
        """Clean up all clients"""
        for client in self.clients.values():
            await client.aclose()
        self.clients.clear()

class AudioBatchProcessor:
    """Batch and pipeline audio processing for better throughput"""
    
    def __init__(self, batch_size: int = 4):
        self.batch_size = batch_size
        self.pending_sentences = []
        self.processing_lock = asyncio.Lock()
    
    async def add_sentence(self, session_id: str, sentence: str, sequence: int):
        """Add sentence to processing batch"""
        async with self.processing_lock:
            self.pending_sentences.append({
                'session_id': session_id,
                'sentence': sentence,
                'sequence': sequence,
                'queued_at': time.time()
            })
            
            # Process batch when full or after timeout
            if len(self.pending_sentences) >= self.batch_size:
                await self._process_batch()
    
    async def _process_batch(self):
        """Process batch of sentences in parallel"""
        if not self.pending_sentences:
            return
            
        batch = self.pending_sentences[:self.batch_size]
        self.pending_sentences = self.pending_sentences[self.batch_size:]
        
        # Process all sentences in parallel
        tasks = [self._process_single_tts(item) for item in batch]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_single_tts(self, item: dict):
        """Process single TTS request"""
        # Implementation would call TTS service
        pass

class AdaptiveChunkProcessor:
    """Dynamically adjust chunk sizes based on performance"""
    
    def __init__(self):
        self.current_chunk_size = 256
        self.min_chunk_size = 128
        self.max_chunk_size = 512
        self.latency_history = []
        self.adjustment_threshold = 10  # Adjust after 10 measurements
    
    def adjust_chunk_size(self, processing_latency_ms: float):
        """Dynamically adjust chunk size based on performance"""
        self.latency_history.append(processing_latency_ms)
        
        if len(self.latency_history) >= self.adjustment_threshold:
            avg_latency = sum(self.latency_history) / len(self.latency_history)
            
            # If latency is high, reduce chunk size for faster processing
            if avg_latency > 200:  # ms
                self.current_chunk_size = max(
                    self.min_chunk_size, 
                    int(self.current_chunk_size * 0.8)
                )
            # If latency is low, increase chunk size for efficiency
            elif avg_latency < 100:  # ms
                self.current_chunk_size = min(
                    self.max_chunk_size,
                    int(self.current_chunk_size * 1.2)
                )
            
            # Reset history
            self.latency_history = []
            
        return self.current_chunk_size

class LLMStreamOptimizer:
    """Optimize LLM streaming with predictive sentence boundary detection"""
    
    def __init__(self):
        self.sentence_markers = ['.', '!', '?', ':', ';']
        self.buffer_size = 50  # Characters
        self.word_buffer = []
    
    def is_likely_sentence_end(self, text: str, next_token: str = "") -> bool:
        """Predict if current text is likely a sentence end"""
        if not text:
            return False
            
        # Check for clear sentence markers
        if text.rstrip()[-1:] in self.sentence_markers:
            # Look ahead to confirm (avoid abbreviations)
            if next_token and next_token[0].isupper():
                return True
            # If no next token, assume end
            if not next_token:
                return True
                
        # Check for natural pauses (comma + conjunction)
        if text.rstrip().endswith(',') and next_token.lower() in ['and', 'but', 'or', 'so']:
            return True
            
        return False
    
    def should_flush_buffer(self, buffer: str, new_token: str) -> bool:
        """Determine if buffer should be flushed for TTS"""
        # Flush on sentence boundaries
        if self.is_likely_sentence_end(buffer, new_token):
            return True
            
        # Flush if buffer is getting too long
        if len(buffer) > 100:  # characters
            # Find a good breaking point
            words = buffer.split()
            if len(words) >= 8:  # At least 8 words
                return True
                
        return False
