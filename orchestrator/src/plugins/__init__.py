"""Plugin system for orchestrator parallel processing."""

from .base_plugin import BasePlugin, PluginManager
from .memory_plugin import MemoryPlugin
from .speaker_plugin import SpeakerPlugin
from .interrupt_plugin import InterruptPlugin

__all__ = ['BasePlugin', 'PluginManager', 'MemoryPlugin', 'SpeakerPlugin', 'InterruptPlugin']