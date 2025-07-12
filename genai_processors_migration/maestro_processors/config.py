"""
Configuration module for GenAI Processors migration of Maestro
Extends the original config with GenAI Processors specific settings
"""

import os
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel


class GenAIProcessorsConfig(BaseModel):
    """Enhanced configuration for GenAI Processors migration"""
    
    # === Original Maestro Configuration ===
    # Service URLs
    WHISPER_URL: str = os.getenv("WHISPER_URL", "http://whisper-live:9090")
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
    TTS_URL: str = os.getenv("TTS_URL", "http://kokoro:8880")
    AMEM_URL: str = os.getenv("AMEM_URL", "http://a-mem:8001")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379")
    
    # Model Configuration
    STT_MODEL: str = os.getenv("STT_MODEL", "tiny")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemma3n:latest")
    TTS_VOICE: str = os.getenv("TTS_VOICE", "af_bella")
    
    # Memory
    MEMORY_ENABLED: bool = os.getenv("MEMORY_ENABLED", "false").lower() == "true"
    
    # LLM Parameters
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "512"))
    LLM_SHORT_TOKENS: int = int(os.getenv("LLM_SHORT_TOKENS", "128"))
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
    
    # WhisperLive Settings
    NO_SPEECH_THRESHOLD: float = float(os.getenv("NO_SPEECH_THRESHOLD", "0.45"))
    VAD_ENABLED: bool = os.getenv("VAD_ENABLED", "true").lower() == "true"
    
    # === GenAI Processors Specific Configuration ===
    
    # Google API
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    GOOGLE_PROJECT_ID: Optional[str] = os.getenv("GOOGLE_PROJECT_ID")
    
    # GenAI Processors Framework
    PROCESSOR_CONCURRENCY: int = int(os.getenv("PROCESSOR_CONCURRENCY", "4"))
    AUDIO_BUFFER_SIZE: int = int(os.getenv("AUDIO_BUFFER_SIZE", "256"))
    STREAM_TIMEOUT: float = float(os.getenv("STREAM_TIMEOUT", "30.0"))
    
    # Processor Chain Settings
    ENABLE_VAD_MONITORING: bool = os.getenv("ENABLE_VAD_MONITORING", "true").lower() == "true"
    SENTENCE_BOUNDARY_CHARS: str = os.getenv("SENTENCE_BOUNDARY_CHARS", ".!?")
    SENTENCE_MIN_LENGTH: int = int(os.getenv("SENTENCE_MIN_LENGTH", "10"))
    
    # Performance Tuning
    MAX_CONCURRENT_SESSIONS: int = int(os.getenv("MAX_CONCURRENT_SESSIONS", "10"))
    PROCESSOR_QUEUE_SIZE: int = int(os.getenv("PROCESSOR_QUEUE_SIZE", "100"))
    PROCESSOR_BUFFER_TIMEOUT: float = float(os.getenv("PROCESSOR_BUFFER_TIMEOUT", "0.1"))
    
    # Voice Activity Detection
    VAD_ENERGY_THRESHOLD: float = float(os.getenv("VAD_ENERGY_THRESHOLD", "300"))
    VAD_DYNAMIC_THRESHOLD: bool = os.getenv("VAD_DYNAMIC_THRESHOLD", "true").lower() == "true"
    VAD_PAUSE_THRESHOLD: float = float(os.getenv("VAD_PAUSE_THRESHOLD", "0.8"))
    
    # Real-time Processing
    REALTIME_PROCESSING: bool = os.getenv("REALTIME_PROCESSING", "true").lower() == "true"
    ENABLE_INTERRUPTION: bool = os.getenv("ENABLE_INTERRUPTION", "true").lower() == "true"
    INTERRUPTION_DELAY: float = float(os.getenv("INTERRUPTION_DELAY", "0.1"))
    
    # Latency Optimization
    TARGET_LATENCY_MS: int = int(os.getenv("TARGET_LATENCY_MS", "450"))
    STT_LATENCY_TARGET_MS: int = int(os.getenv("STT_LATENCY_TARGET_MS", "100"))
    LLM_LATENCY_TARGET_MS: int = int(os.getenv("LLM_LATENCY_TARGET_MS", "200"))
    TTS_LATENCY_TARGET_MS: int = int(os.getenv("TTS_LATENCY_TARGET_MS", "150"))
    
    # Error Handling
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY: float = float(os.getenv("RETRY_DELAY", "1.0"))
    CIRCUIT_BREAKER_ENABLED: bool = os.getenv("CIRCUIT_BREAKER_ENABLED", "true").lower() == "true"
    
    # Logging and Monitoring
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    ENABLE_PERFORMANCE_LOGGING: bool = os.getenv("ENABLE_PERFORMANCE_LOGGING", "true").lower() == "true"
    METRICS_COLLECTION: bool = os.getenv("METRICS_COLLECTION", "true").lower() == "true"
    
    # Development and Testing
    DEVELOPMENT_MODE: bool = os.getenv("DEVELOPMENT_MODE", "false").lower() == "true"
    TEST_MODE: bool = os.getenv("TEST_MODE", "false").lower() == "true"
    MOCK_SERVICES: bool = os.getenv("MOCK_SERVICES", "false").lower() == "true"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Audio configuration for ProcessorParts
class AudioConfig:
    SAMPLE_RATE = 16000
    CHANNELS = 1
    SAMPLE_WIDTH = 2  # 16-bit
    CHUNK_SIZE = 256
    FORMAT = "PCM_16"  # For compatibility with WhisperLive


# Metadata types for ProcessorParts
class VoiceMetadata:
    """Metadata schema for voice processing pipeline"""
    
    # Content types
    AUDIO_INPUT = "audio_input"
    TRANSCRIPT = "transcript"
    LLM_TEXT = "llm_text"
    TTS_AUDIO = "tts_audio"
    
    # Processing stages
    STAGE_STT = "stt"
    STAGE_LLM = "llm"
    STAGE_TTS = "tts"
    STAGE_OUTPUT = "output"
    
    # Status indicators
    STATUS_PROCESSING = "processing"
    STATUS_COMPLETE = "complete"
    STATUS_INTERRUPTED = "interrupted"
    STATUS_ERROR = "error"


# Processor chain configuration
class ProcessorChainConfig:
    """Configuration for different processor chain compositions"""
    
    # Standard voice conversation chain
    VOICE_CONVERSATION = [
        "audio_input",
        "whisper_live", 
        "session_manager",
        "ollama_stream",
        "kokoro_tts",
        "audio_output"
    ]
    
    # With VAD monitoring (parallel)
    VOICE_WITH_VAD = [
        "audio_input",
        "whisper_live",
        "session_manager", 
        "ollama_stream",
        "kokoro_tts",
        "audio_output",
        "vad_monitor"  # Parallel processor
    ]
    
    # Testing chain (mocked services)
    TESTING_CHAIN = [
        "mock_audio_input",
        "mock_stt",
        "mock_llm",
        "mock_tts",
        "mock_audio_output"
    ]


# Single configuration instance
config = GenAIProcessorsConfig()

# Export commonly used values
WHISPER_HOST = config.WHISPER_URL.replace("http://", "").replace("https://", "").split(":")[0]
WHISPER_PORT = int(config.WHISPER_URL.split(":")[-1]) if ":" in config.WHISPER_URL else 9090
