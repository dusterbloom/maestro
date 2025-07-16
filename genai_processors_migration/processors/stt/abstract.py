from abc import abstractmethod
from typing import AsyncIterator
from genai_processors import Processor, ProcessorPart

class STTProcessor(Processor):
    """Abstract base class for Speech-to-Text processors"""
    
    @abstractmethod
    async def __call__(self, input_stream: AsyncIterator[ProcessorPart]) -> AsyncIterator[ProcessorPart]:
        """Process audio input and return transcribed text"""
        pass