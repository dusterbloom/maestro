from abc import abstractmethod
from typing import AsyncIterator
from genai_processors import Processor, ProcessorPart

class TTSProcessor(Processor):
    """Base class for Text-to-Speech processors"""
    
    @abstractmethod
    async def __call__(self, input_stream: AsyncIterator[ProcessorPart]) -> AsyncIterator[ProcessorPart]:
        """Process text input and return audio"""
        pass