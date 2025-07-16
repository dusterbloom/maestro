from abc import abstractmethod
from typing import AsyncIterator
from genai_processors import Processor, ProcessorPart

class AnalysisProcessor(Processor):
    """Base class for analysis processors"""
    
    @abstractmethod
    async def __call__(self, input_stream: AsyncIterator[ProcessorPart]) -> AsyncIterator[ProcessorPart]:
        """Process input and return analysis results"""
        pass