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
        max_retries = 30
        retry_delay = 2  # seconds

        for i in range(max_retries):
            try:
                print(f"🗄️ Attempting to connect to ChromaDB (attempt {i+1}/{max_retries})...")
                
                # Use official ChromaDB client syntax per docs
                self.chroma_client = chromadb.HttpClient(host="chromadb", port=8000)
                
                # Test connection with heartbeat
                heartbeat_result = self.chroma_client.heartbeat()
                print(f"🗄️ ChromaDB heartbeat successful: {heartbeat_result}")
                
                # Create collection for speaker embeddings
                self.collection = self.chroma_client.get_or_create_collection("speaker_embeddings")
                print(f"🗄️ Successfully connected to ChromaDB collection after {i+1} attempts.")
                
                # Test Redis connection
                await self._test_redis_connection()
                
                return
            except Exception as e:
                print(f"🗄️ Attempt {i+1}/{max_retries} to connect to ChromaDB failed: {e}")
                if i < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    print(f"🗄️ CRITICAL: Failed to connect to ChromaDB after {max_retries} attempts!")
                    raise # Re-raise the last exception if all retries fail

    async def _test_redis_connection(self):
        """Test Redis connection and log result"""
        try:
            await self.redis_client.ping()
            print("🗄️ Redis connection test successful")
        except Exception as redis_error:
            print(f"🗄️ Redis connection test failed: {redis_error}")
            raise

    async def find_speaker_by_embedding(self, embedding: list[float]) -> dict | None:
        results = self.collection.query(query_embeddings=[embedding], n_results=1)
        if results and results["ids"][0]:
            user_id = results["ids"][0][0]
            profile = await self.get_speaker_profile(user_id)
            if profile:
                return {"user_id": user_id, "name": profile.get("name", "Friend")}
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

    async def get_all_speaker_profiles(self) -> list[dict]:
        """Fetch all speaker profiles from ChromaDB and Redis"""
        try:
            # Get all embeddings from ChromaDB
            results = self.collection.get(include=["embeddings", "metadatas"])
            
            if not results or not results["ids"]:
                return []

            profiles = []
            for i, user_id in enumerate(results["ids"]):
                embedding = results["embeddings"][i]
                # Fetch name from Redis
                profile_data = await self.get_speaker_profile(user_id)
                name = profile_data.get("name", "Friend")
                
                profiles.append({
                    "user_id": user_id,
                    "name": name,
                    "embedding": embedding
                })
            return profiles
        except Exception as e:
            print(f"Error getting all speaker profiles: {e}")
            return []
    
    async def add_user_memory(self, user_id: str, memory_data: dict):
        """Add memory data to user's profile"""
        memory_key = f"memory:{user_id}:{memory_data.get('event_type', 'general')}"
        memory_json = json.dumps(memory_data)
        await self.redis_client.lpush(memory_key, memory_json)
        # Keep only last 100 memories per event type
        await self.redis_client.ltrim(memory_key, 0, 99)

    async def get_user_memory(self, user_id: str, event_type: str = "general") -> list[dict]:
        """Get user's memory for a specific event type"""
        memory_key = f"memory:{user_id}:{event_type}"
        memory_json_list = await self.redis_client.lrange(memory_key, 0, -1)
        return [json.loads(memory_json) for memory_json in memory_json_list]
