"""Memory plugin for non-blocking context caching."""

import asyncio
import logging
from typing import Any, Dict, List, Optional
import json
import time
from datetime import datetime, timedelta
from .base_plugin import BasePlugin, Event, PluginConfig

logger = logging.getLogger(__name__)


class MemoryPlugin(BasePlugin):
    """Plugin for caching conversation context and user preferences."""
    
    def __init__(self, config: Optional[PluginConfig] = None):
        super().__init__(config or PluginConfig(
            enabled=True,
            priority=1,  # High priority
            max_workers=2,
            timeout=5.0  # Short timeout to avoid blocking
        ))
        self._cache: Dict[str, Any] = {}
        self._conversation_history: List[Dict[str, Any]] = []
        self._user_profiles: Dict[str, Dict[str, Any]] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Initialize the memory plugin."""
        logger.info("Initializing memory plugin")
        
        # Start background cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_cache())
        
        # Load any existing memory from file (optional)
        await self._load_memory()
    
    async def process_event(self, event: Event) -> None:
        """Process events for memory caching."""
        try:
            if event.event_type == "transcription_complete":
                await self._cache_transcription(event.data)
            elif event.event_type == "user_speaking":
                await self._update_user_context(event.data)
            elif event.event_type == "conversation_end":
                await self._save_conversation_summary(event.data)
            elif event.event_type == "user_identified":
                await self._update_user_profile(event.data)
                
        except Exception as e:
            logger.error(f"Error processing memory event: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown the memory plugin."""
        logger.info("Shutting down memory plugin")
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        await self._save_memory()
    
    async def _cache_transcription(self, data: Dict[str, Any]) -> None:
        """Cache transcription data."""
        text = data.get("text", "")
        user_id = data.get("user_id", "default")
        timestamp = data.get("timestamp", time.time())
        
        # Cache recent transcription
        cache_key = f"transcription:{user_id}:{int(timestamp)}"
        self._cache[cache_key] = {
            "text": text,
            "timestamp": timestamp,
            "user_id": user_id,
            "expires": time.time() + 3600  # 1 hour
        }
        
        # Add to conversation history
        self._conversation_history.append({
            "type": "user",
            "text": text,
            "timestamp": timestamp,
            "user_id": user_id
        })
        
        # Keep only last 100 messages
        if len(self._conversation_history) > 100:
            self._conversation_history = self._conversation_history[-100:]
        
        logger.debug(f"Cached transcription for user {user_id}")
    
    async def _update_user_context(self, data: Dict[str, Any]) -> None:
        """Update user speaking context."""
        user_id = data.get("user_id", "default")
        duration = data.get("duration", 0)
        
        if user_id not in self._user_profiles:
            self._user_profiles[user_id] = {
                "total_speaking_time": 0,
                "conversation_count": 0,
                "last_activity": time.time()
            }
        
        profile = self._user_profiles[user_id]
        profile["total_speaking_time"] += duration
        profile["last_activity"] = time.time()
        
        logger.debug(f"Updated context for user {user_id}")
    
    async def _save_conversation_summary(self, data: Dict[str, Any]) -> None:
        """Save conversation summary."""
        user_id = data.get("user_id", "default")
        summary = data.get("summary", {})
        
        # Generate quick summary
        user_messages = [
            msg for msg in self._conversation_history 
            if msg.get("user_id") == user_id
        ]
        
        if user_messages:
            summary_data = {
                "user_id": user_id,
                "message_count": len(user_messages),
                "total_duration": sum(
                    msg.get("duration", 0) for msg in user_messages
                ),
                "topics": self._extract_topics(user_messages),
                "timestamp": time.time()
            }
            
            # Cache summary
            cache_key = f"summary:{user_id}:{int(time.time())}"
            self._cache[cache_key] = {
                "data": summary_data,
                "expires": time.time() + 86400  # 24 hours
            }
            
            logger.debug(f"Saved conversation summary for user {user_id}")
    
    async def _update_user_profile(self, data: Dict[str, Any]) -> None:
        """Update user profile with identification data."""
        user_id = data.get("user_id")
        speaker_id = data.get("speaker_id")
        
        if user_id and speaker_id:
            if user_id not in self._user_profiles:
                self._user_profiles[user_id] = {}
            
            self._user_profiles[user_id]["speaker_id"] = speaker_id
            self._user_profiles[user_id]["identified_at"] = time.time()
            
            logger.debug(f"Updated user profile for {user_id} with speaker {speaker_id}")
    
    def _extract_topics(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Extract topics from messages (simplified)."""
        # Simple keyword extraction - in real implementation, use NLP
        keywords = []
        for msg in messages:
            text = msg.get("text", "").lower()
            words = text.split()
            # Simple keyword extraction based on word frequency
            for word in words:
                if len(word) > 4 and word not in ["the", "and", "that", "this"]:
                    keywords.append(word)
        
        # Return top 5 keywords
        from collections import Counter
        counter = Counter(keywords)
        return [word for word, _ in counter.most_common(5)]
    
    async def _cleanup_expired_cache(self) -> None:
        """Background task to cleanup expired cache entries."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                current_time = time.time()
                expired_keys = [
                    key for key, value in self._cache.items()
                    if value.get("expires", 0) < current_time
                ]
                
                for key in expired_keys:
                    del self._cache[key]
                
                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
    
    async def _load_memory(self) -> None:
        """Load memory from persistent storage (optional)."""
        # Placeholder for persistent storage loading
        pass
    
    async def _save_memory(self) -> None:
        """Save memory to persistent storage (optional)."""
        # Placeholder for persistent storage saving
        pass
    
    def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get cached context for a user (synchronous for fast access)."""
        return {
            "profile": self._user_profiles.get(user_id, {}),
            "recent_messages": [
                msg for msg in self._conversation_history[-10:]
                if msg.get("user_id") == user_id
            ],
            "cache_keys": [
                key for key in self._cache.keys()
                if key.startswith(f"transcription:{user_id}")
            ]
        }
    
    def get_conversation_summary(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation summary for a user."""
        summaries = []
        for key, value in self._cache.items():
            if key.startswith(f"summary:{user_id}"):
                summaries.append(value.get("data", {}))
        
        return sorted(summaries, key=lambda x: x.get("timestamp", 0), reverse=True)[:limit]