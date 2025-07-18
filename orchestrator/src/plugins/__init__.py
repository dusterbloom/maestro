"""Plugin system for orchestrator parallel processing."""

from plugins.base_plugin import BasePlugin, PluginManager
from plugins.memory_plugin import MemoryPlugin
from plugins.speaker_plugin import SpeakerPlugin
from plugins.interrupt_plugin import InterruptPlugin

__all__ = ['BasePlugin', 'PluginManager', 'MemoryPlugin', 'SpeakerPlugin', 'InterruptPlugin']