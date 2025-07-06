# orchestrator/config.py
# Single source of truth for all configuration

import os
from typing import Optional

class Config:
    """Centralized configuration - reads from environment with sensible defaults"""
    
    # Service URLs
    WHISPER_URL: str = os.getenv("WHISPER_URL", "http://whisper-live:9090")
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
    TTS_URL: str = os.getenv("TTS_URL", "http://kokoro:8880")
    AMEM_URL: str = os.getenv("AMEM_URL", "http://a-mem:8001")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379")
    CHROMADB_URL: str = os.getenv("CHROMADB_URL", "http://chromadb:8002")
    
    # Model Configuration
    STT_MODEL: str = os.getenv("STT_MODEL", "tiny")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemma3n:latest")
    TTS_VOICE: str = os.getenv("TTS_VOICE", "af_bella")
    
    # Memory
    MEMORY_ENABLED: bool = os.getenv("MEMORY_ENABLED", "false").lower() == "true"
    
    # LLM Parameters
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "512"))  # Reasonable default
    LLM_SHORT_TOKENS: int = int(os.getenv("LLM_SHORT_TOKENS", "128"))  # For quick responses
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    
    # TTS Parameters  
    TTS_SPEED: float = float(os.getenv("TTS_SPEED", "1.0"))
    TTS_VOLUME: float = float(os.getenv("TTS_VOLUME", "1.0"))
    
    # Audio Processing
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "256"))
    AUDIO_SAMPLE_RATE: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    
    # Sentence Processing
    MIN_WORD_COUNT: int = int(os.getenv("MIN_WORD_COUNT", "3"))
    
    # Timeouts (in seconds)
    AMEM_TIMEOUT: float = float(os.getenv("AMEM_TIMEOUT", "5.0"))
    OLLAMA_TIMEOUT: float = float(os.getenv("OLLAMA_TIMEOUT", "30.0"))
    TTS_TIMEOUT: float = float(os.getenv("TTS_TIMEOUT", "10.0"))
    WHISPER_TIMEOUT: float = float(os.getenv("WHISPER_TIMEOUT", "10.0"))
    
    # WhisperLive Settings
    NO_SPEECH_THRESHOLD: float = float(os.getenv("NO_SPEECH_THRESHOLD", "0.45"))
    VAD_ENABLED: bool = os.getenv("VAD_ENABLED", "true").lower() == "true"
    
    # Speaker Recognition VAD Settings
    SPEAKER_VAD_THRESHOLD: float = float(os.getenv("SPEAKER_VAD_THRESHOLD", "0.01"))
    SPEAKER_VAD_MIN_SEGMENT_MS: int = int(os.getenv("SPEAKER_VAD_MIN_SEGMENT_MS", "200"))
    SPEAKER_VAD_PAD_MS: int = int(os.getenv("SPEAKER_VAD_PAD_MS", "100"))
    
    # Resemblyzer Speaker Recognition Settings
    SPEAKER_SIMILARITY_THRESHOLD: float = float(os.getenv("SPEAKER_SIMILARITY_THRESHOLD", "0.7"))
    RESEMBLYZER_DEVICE: str = os.getenv("RESEMBLYZER_DEVICE", "cuda")  # "cpu" or "cuda"

# Single config instance
config = Config()