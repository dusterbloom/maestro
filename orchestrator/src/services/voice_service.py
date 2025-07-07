import asyncio
import numpy as np
import time
import hashlib
import logging
import json
import redis
from concurrent.futures import ThreadPoolExecutor
from resemblyzer import VoiceEncoder
from orchestrator.src.config import config
import soundfile as sf

logger = logging.getLogger(__name__)

class VoiceService:
    """A stateless service for generating high-quality speaker embeddings from audio data."""
    
    def __init__(self):
        logger.info(f"Initializing Resemblyzer VoiceEncoder on device: {config.RESEMBLYZER_DEVICE}")
        self.voice_encoder = VoiceEncoder(device=config.RESEMBLYZER_DEVICE)
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="embedding_worker")
        self.redis_client = self._initialize_redis()
        self.embedding_cache_ttl = 86400  # 24 hours

    def _initialize_redis(self):
        try:
            client = redis.from_url(config.REDIS_URL, decode_responses=False)
            client.ping()
            logger.info(f"✅ VoiceService connected to Redis for embedding cache.")
            return client
        except Exception as e:
            logger.warning(f"⚠️ VoiceService Redis cache unavailable: {e}")
            return None

    async def get_embedding(self, audio_data: bytes) -> list[float] | None:
        """
        Generates a speaker embedding from audio data.
        This is the primary public method of this service.
        """
        audio_hash = hashlib.sha256(audio_data).hexdigest()[:16]
        if self.redis_client:
            cached_embedding = await self._get_cached_embedding(audio_hash)
            if cached_embedding:
                return cached_embedding

        loop = asyncio.get_event_loop()
        try:
            embedding = await asyncio.wait_for(
                loop.run_in_executor(self.executor, self._generate_embedding_sync, audio_data),
                timeout=config.SPEAKER_EMBEDDING_TIMEOUT
            )
            if embedding and self.redis_client:
                await self._cache_embedding(audio_hash, embedding)
            return embedding
        except asyncio.TimeoutError:
            logger.error(f"Embedding generation timed out after {config.SPEAKER_EMBEDDING_TIMEOUT}s")
            return None
        except Exception as e:
            logger.error(f"Exception in get_embedding: {e}")
            return None

    def _generate_embedding_sync(self, audio_data: bytes) -> list[float] | None:
        """Synchronous wrapper for the CPU-bound embedding generation."""
        try:
            audio_array, sample_rate = self._decode_audio(audio_data)
            if audio_array is None:
                return None
            
            preprocessed_wav = self._preprocess_audio(audio_array, sample_rate)
            if preprocessed_wav is None:
                return None

            embedding = self.voice_encoder.embed_utterance(preprocessed_wav)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Sync embedding generation failed: {e}", exc_info=True)
            return None

    def _decode_audio(self, audio_data: bytes) -> tuple[np.ndarray, int] | tuple[None, None]:
        """Decodes audio bytes into a numpy array, handling WAV or raw formats."""
        try:
            with io.BytesIO(audio_data) as buffer:
                audio_array, sample_rate = sf.read(buffer, dtype='float32')
            return audio_array, sample_rate
        except Exception as e:
            logger.error(f"Failed to decode audio with soundfile: {e}")
            return None, None

    def _preprocess_audio(self, audio_array: np.ndarray, sample_rate: int) -> np.ndarray | None:
        """Applies normalization and ensures audio is suitable for the model."""
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        if np.max(np.abs(audio_array)) == 0:
            logger.error("Audio is completely silent.")
            return None

        # Normalize audio to a target level to improve consistency
        target_dbfs = -20.0
        rms = np.sqrt(np.mean(audio_array**2))
        if rms > 0:
            change_in_dbfs = target_dbfs - (20 * np.log10(rms))
            gain = 10**(change_in_dbfs / 20)
            audio_array = audio_array * gain

        # Ensure minimum duration for Resemblyzer
        min_samples = int(1.6 * sample_rate)
        if len(audio_array) < min_samples:
            if len(audio_array) > 0:
                reps = int(np.ceil(min_samples / len(audio_array)))
                audio_array = np.tile(audio_array, reps)[:min_samples]
            else:
                return None # Cannot process empty audio

        return audio_array

    async def _get_cached_embedding(self, audio_hash: str) -> list[float] | None:
        try:
            cached_data = self.redis_client.get(f"embedding:{audio_hash}")
            if cached_data:
                logger.info(f"Embedding cache hit for hash {audio_hash}")
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Redis cache get failed: {e}")
        return None

    async def _cache_embedding(self, audio_hash: str, embedding: list[float]):
        try:
            self.redis_client.setex(f"embedding:{audio_hash}", self.embedding_cache_ttl, json.dumps(embedding))
            logger.info(f"Cached embedding for hash {audio_hash}")
        except Exception as e:
            logger.warning(f"Redis cache set failed: {e}")