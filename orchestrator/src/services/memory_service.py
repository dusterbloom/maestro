import chromadb
import redis.asyncio as redis
import uuid
import asyncio
import json
from config import config

class MemoryService:
    def __init__(self):
        self.redis_client = redis.from_url(config.REDIS_URL, decode_responses=True)
        self.chroma_client = None
        self.collection = None

    async def initialize_chroma_client(self):
        max_retries = 30  # Increased retries
        retry_delay = 2  # seconds

        for i in range(max_retries):
            try:
                # Use official ChromaDB client syntax per docs
                self.chroma_client = chromadb.HttpClient(host="chromadb", port=8000)
                # Test connection with heartbeat
                self.chroma_client.heartbeat()
                # Create collection for speaker embeddings
                self.collection = self.chroma_client.get_or_create_collection("speaker_embeddings")
                print(f"Successfully connected to ChromaDB after {i+1} attempts.")
                return
            except Exception as e:
                print(f"Attempt {i+1}/{max_retries} to connect to ChromaDB failed: {e}")
                if i < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    raise # Re-raise the last exception if all retries fail

    async def find_speaker_by_embedding(self, embedding: list[float]) -> str | None:
        results = self.collection.query(query_embeddings=[embedding], n_results=1)
        if results and results["ids"][0]:
            return results["ids"][0][0]
        return None

    async def create_speaker_profile(self, embedding: list[float]) -> str:
        user_id = f"user_{uuid.uuid4().hex[:8]}"
        self.collection.add(embeddings=[embedding], ids=[user_id])
        await self.redis_client.hset(f"speaker:{user_id}", mapping={
            "name": f"Speaker {self.collection.count()}",
            "status": "pending_naming"
        })
        return user_id

    async def get_speaker_profile(self, user_id: str) -> dict | None:
        return await self.redis_client.hgetall(f"speaker:{user_id}")

    async def update_speaker_name(self, user_id: str, name: str):
        await self.redis_client.hset(f"speaker:{user_id}", "name", name)
        await self.redis_client.hset(f"speaker:{user_id}", "status", "active")
    
    async def add_user_memory(self, user_id: str, memory_data: dict):
        """Add memory data to user's profile"""
        memory_key = f"memory:{user_id}:{memory_data.get('event_type', 'general')}"
        memory_json = json.dumps(memory_data)
        await self.redis_client.lpush(memory_key, memory_json)
        # Keep only last 100 memories per event type
        await self.redis_client.ltrim(memory_key, 0, 99)
