import httpx
from config import config

class VoiceService:
    def __init__(self):
        self.client = httpx.AsyncClient(base_url=config.DIGLETT_URL, timeout=10.0)

    async def get_embedding(self, audio_data: bytes) -> list[float] | None:
        try:
            # Convert Float32Array bytes to int16 PCM format that Diglett expects
            import numpy as np
            
            # Convert bytes back to float32 array
            float_array = np.frombuffer(audio_data, dtype=np.float32)
            
            # Convert float32 (-1.0 to 1.0) to int16 (-32768 to 32767)
            int16_array = (float_array * 32767).astype(np.int16)
            
            # Convert to bytes
            pcm_bytes = int16_array.tobytes()
            
            # Send as file upload (Diglett expects File() upload, not raw bytes)
            files = {"file": ("audio.pcm", pcm_bytes, "audio/pcm")}
            response = await self.client.post("/embed", files=files)
            response.raise_for_status()
            result = response.json()
            
            # Diglett returns direct dict format: {"speaker_embedding": [...], "avg_db": float, "speaker_name": str}
            if isinstance(result, dict):
                return result.get("speaker_embedding")
                
            return None
        except Exception as e:
            print(f"Error getting embedding from Diglett: {e}")
            return None
