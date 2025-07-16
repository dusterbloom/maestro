from abc import abstractmethod
from typing import AsyncIterator
from genai_processors import Processor, ProcessorPart

class LLMProcessor(Processor):
    """Base class for Large Language Model processors"""
    
    @abstractmethod
    async def __call__(self, input_stream: AsyncIterator[ProcessorPart]) -> AsyncIterator[ProcessorPart]:
        """Process text input using LLM"""
        pass