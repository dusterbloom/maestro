"""Base plugin system for orchestrator."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class PluginConfig:
    """Configuration for a plugin."""
    enabled: bool = True
    priority: int = 0  # Lower = higher priority
    max_workers: int = 1
    timeout: float = 30.0


class Event:
    """Event object passed to plugins."""
    
    def __init__(self, event_type: str, data: Any, timestamp: Optional[float] = None):
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp or time.time()
        self._metadata = {}
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the event."""
        self._metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata from the event."""
        return self._metadata.get(key, default)


class BasePlugin(ABC):
    """Base class for all orchestrator plugins."""
    
    def __init__(self, config: Optional[PluginConfig] = None):
        self.config = config or PluginConfig()
        self.name = self.__class__.__name__
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the plugin."""
        pass
    
    @abstractmethod
    async def process_event(self, event: Event) -> None:
        """Process an event. Must be non-blocking."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the plugin gracefully."""
        pass
    
    async def start(self) -> None:
        """Start the plugin."""
        if not self.config.enabled:
            logger.info(f"Plugin {self.name} is disabled")
            return
        
        logger.info(f"Starting plugin: {self.name}")
        await self.initialize()
        self._running = True
    
    async def stop(self) -> None:
        """Stop the plugin."""
        if not self._running:
            return
        
        logger.info(f"Stopping plugin: {self.name}")
        self._running = False
        
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        await self.shutdown()
    
    @property
    def is_running(self) -> bool:
        """Check if plugin is running."""
        return self._running


class PluginManager:
    """Manages all orchestrator plugins."""
    
    def __init__(self):
        self.plugins: Dict[str, BasePlugin] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._worker_tasks: List[asyncio.Task] = []
        self._running = False
    
    def register_plugin(self, plugin: BasePlugin) -> None:
        """Register a plugin."""
        if plugin.name in self.plugins:
            logger.warning(f"Plugin {plugin.name} already registered, overwriting")
        
        self.plugins[plugin.name] = plugin
        logger.info(f"Registered plugin: {plugin.name}")
    
    def unregister_plugin(self, name: str) -> None:
        """Unregister a plugin."""
        if name in self.plugins:
            del self.plugins[name]
            logger.info(f"Unregistered plugin: {name}")
    
    async def start_all(self) -> None:
        """Start all registered plugins."""
        if self._running:
            return
        
        self._running = True
        
        # Start all plugins
        for plugin in self.plugins.values():
            try:
                await plugin.start()
            except Exception as e:
                logger.error(f"Failed to start plugin {plugin.name}: {e}")
        
        # Start event processing workers
        for i in range(2):  # 2 worker tasks for parallel processing
            task = asyncio.create_task(self._event_worker())
            self._worker_tasks.append(task)
        
        logger.info("Plugin manager started")
    
    async def stop_all(self) -> None:
        """Stop all plugins."""
        if not self._running:
            return
        
        self._running = False
        
        # Stop all plugins
        for plugin in self.plugins.values():
            try:
                await plugin.stop()
            except Exception as e:
                logger.error(f"Error stopping plugin {plugin.name}: {e}")
        
        # Stop workers
        for task in self._worker_tasks:
            task.cancel()
        
        # Wait for all tasks to complete
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()
        
        logger.info("Plugin manager stopped")
    
    async def emit_event(self, event_type: str, data: Any) -> None:
        """Emit an event to all plugins."""
        if not self._running:
            return
        
        event = Event(event_type, data)
        await self._event_queue.put(event)
    
    async def _event_worker(self) -> None:
        """Worker task to process events."""
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(), 
                    timeout=1.0
                )
                
                # Process event with all enabled plugins
                tasks = []
                for plugin in sorted(
                    self.plugins.values(), 
                    key=lambda p: p.config.priority
                ):
                    if plugin.is_running:
                        task = asyncio.create_task(
                            self._process_plugin_event(plugin, event)
                        )
                        tasks.append(task)
                
                # Don't wait for completion to maintain non-blocking behavior
                if tasks:
                    asyncio.create_task(self._wait_for_tasks(tasks))
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in event worker: {e}")
    
    async def _process_plugin_event(self, plugin: BasePlugin, event: Event) -> None:
        """Process an event for a specific plugin."""
        try:
            await asyncio.wait_for(
                plugin.process_event(event),
                timeout=plugin.config.timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Plugin {plugin.name} timed out processing event")
        except Exception as e:
            logger.error(f"Plugin {plugin.name} error processing event: {e}")
    
    async def _wait_for_tasks(self, tasks: List[asyncio.Task]) -> None:
        """Wait for plugin tasks to complete and handle exceptions."""
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Plugin task error: {result}")