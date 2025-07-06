import httpx
from config import config

class VoiceService:
    def __init__(self):
        self.client = httpx.AsyncClient(base_url=config.DIGLETT_URL, timeout=10.0)

    async def get_embedding(self, audio_data: bytes) -> list[float] | None:
        try:
            files = {"file": ("audio.wav", audio_data, "audio/wav")}
            response = await self.client.post("/embed", files=files)
            response.raise_for_status()
            result = response.json()
            return result.get("speaker_embedding")
        except httpx.HTTPError as e:
            print(f"Error getting embedding from Diglett: {e}")
            return None
