
import asyncio
import logging
import time
import httpx
from config import config
from services.base_service import BaseService, ServiceResult

logger = logging.getLogger(__name__)

class TTSService(BaseService):
    def __init__(self):
        self.tts_url = config.TTS_URL

    async def process(self, text: str, context=None):
        """Yields audio chunks as soon as they are received from the TTS service."""
        if not text or not text.strip():
            logger.warning("Empty text provided to TTS")
            return

        try:
            start_time = time.time()
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=2.0, read=10.0, write=5.0, pool=None),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            ) as client:
                response = await client.post(
                    f"{self.tts_url}/v1/audio/speech",
                    json={
                        "model": "kokoro",
                        "input": text[:500],
                        "voice": config.TTS_VOICE,
                        "response_format": "wav",
                        "stream": True,
                        "speed": config.TTS_SPEED,
                        "volume_multiplier": config.TTS_VOLUME
                    }
                )
                response.raise_for_status()
                
                chunk_count = 0
                async for chunk in response.aiter_bytes():
                    if chunk:
                        chunk_count += 1
                        if chunk_count == 1:
                            first_chunk_time = time.time() - start_time
                            logger.info(f"🚀 First TTS chunk arrived in {first_chunk_time:.3f}s")
                        yield chunk
                elapsed = time.time() - start_time
                logger.info(f"✅ TTS streaming completed in {elapsed:.2f}s ({chunk_count} chunks)")
        except httpx.TimeoutException:
            elapsed = time.time() - start_time
            logger.error(f"❌ TTS timeout after {elapsed:.2f}s")
            return
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ TTS synthesis error after {elapsed:.2f}s: {e}")
            return

    async def health_check(self) -> bool:
        # In a real scenario, you'd ping the TTS service's health endpoint
        return True

    def get_metrics(self) -> dict:
        return {}
