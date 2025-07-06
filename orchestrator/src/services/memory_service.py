import chromadb
import redis.asyncio as redis
import uuid
from config import CHROMADB_URL, REDIS_URL

class MemoryService:
    def __init__(self):
        self.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        self.chroma_client = chromadb.HttpClient(host=CHROMADB_URL, port=8002) # Default ChromaDB port is 8000, but we mapped it to 8002
        self.collection = self.chroma_client.get_or_create_collection("speaker_embeddings")

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
