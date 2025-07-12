"""
Maestro Processors Package
Custom GenAI processors for voice orchestration system migration
"""

from .config import config, GenAIProcessorsConfig, AudioConfig, VoiceMetadata
from .whisper_live_processor import WhisperLiveProcessor
from .ollama_stream_processor import OllamaStreamProcessor  
from .kokoro_tts_processor import KokoroTTSProcessor
from .voice_activity_processor import VoiceActivityProcessor
from .session_manager_processor import SessionManagerProcessor

__version__ = "1.0.0"
__author__ = "Maestro GenAI Migration Team"

__all__ = [
    "config",
    "GenAIProcessorsConfig", 
    "AudioConfig",
    "VoiceMetadata",
    "WhisperLiveProcessor",
    "OllamaStreamProcessor",
    "KokoroTTSProcessor", 
    "VoiceActivityProcessor",
    "SessionManagerProcessor",
]
