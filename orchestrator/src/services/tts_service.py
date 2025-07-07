
import asyncio
import logging
import time
import httpx
from orchestrator.src.config import config
from orchestrator.src.services.base_service import BaseService, ServiceResult

logger = logging.getLogger(__name__)

class TTSService(BaseService):
    def __init__(self):
        self.tts_url = config.TTS_URL

    async def process(self, text: str, context=None) -> ServiceResult:
        """Converts text to speech using the TTS service."""
        if not text or not text.strip():
            logger.warning("Empty text provided to TTS")
            return ServiceResult(success=False, error="Empty text provided.")

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
                        "stream": False,
                        "speed": config.TTS_SPEED,
                        "volume_multiplier": config.TTS_VOLUME
                    }
                )
                response.raise_for_status()
                elapsed = time.time() - start_time
                audio_bytes = response.content
                logger.info(f"✅ TTS completed in {elapsed:.2f}s - {len(audio_bytes)} bytes")
                return ServiceResult(success=True, data=audio_bytes)
        except httpx.TimeoutException:
            elapsed = time.time() - start_time
            logger.error(f"❌ TTS timeout after {elapsed:.2f}s")
            return ServiceResult(success=False, error="TTS service timed out.")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ TTS synthesis error after {elapsed:.2f}s: {e}")
            return ServiceResult(success=False, error=f"TTS synthesis failed: {e}")

    async def health_check(self) -> bool:
        # In a real scenario, you'd ping the TTS service's health endpoint
        return True

    def get_metrics(self) -> dict:
        return {}
