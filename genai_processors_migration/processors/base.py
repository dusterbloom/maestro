from abc import ABC, abstractmethod
from typing import AsyncIterator, Any, Dict
from dataclasses import dataclass
import asyncio

@dataclass
class ProcessorPart:
    """Base data container for processor parts"""
    data: Any
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class Processor(ABC):
    """Base processor interface"""
    
    @abstractmethod
    async def __call__(self, input_stream: AsyncIterator[ProcessorPart]) -> AsyncIterator[ProcessorPart]:
        pass
    
    def __add__(self, other: 'Processor') -> 'Processor':
        """Chain processors together"""
        return ChainedProcessor(self, other)

class ChainedProcessor(Processor):
    """Processor that chains two processors together"""
    
    def __init__(self, first: Processor, second: Processor):
        self.first = first
        self.second = second
    
    async def __call__(self, input_stream: AsyncIterator[ProcessorPart]) -> AsyncIterator[ProcessorPart]:
        intermediate_stream = self.first(input_stream)
        async for part in self.second(intermediate_stream):
            yield part

class ServiceProcessor(Processor):
    """Base class for service-based processors that communicate with external services"""
    
    def __init__(self, service_url: str, **kwargs):
        self.service_url = service_url
        self.config = kwargs

class LocalProcessor(Processor):
    """Base class for local processors that run on the same machine"""
    
    def __init__(self, **kwargs):
        self.config = kwargs