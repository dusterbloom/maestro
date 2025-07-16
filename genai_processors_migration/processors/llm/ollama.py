import logging
import asyncio
import aiohttp
import json
from typing import AsyncIterator
from processors.base import Processor, ProcessorPart

logger = logging.getLogger(__name__)

class OllamaProcessor(Processor):
    def __init__(self, service_url='http://localhost:11434', model='gemma3n:latest', temperature=0.7, **kwargs):
        super().__init__(**kwargs)
        self.service_url = service_url
        self.model = model
        self.temperature = temperature
    
    async def process(self, text: str) -> AsyncIterator[str]:
        """Process text with Ollama LLM."""
        try:
            url = f"{self.service_url}/api/generate"
            
            payload = {
                "model": self.model,
                "prompt": text,
                "stream": True,
                "temperature": self.temperature
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line.decode('utf-8'))
                                if 'response' in data:
                                    yield data['response']
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            logger.error(f"Error in Ollama processing: {e}")
            yield f"Error: {str(e)}"

class OllamaPart(ProcessorPart):
    def __init__(self, data, metadata=None):
        super().__init__(data, metadata)
        self.processor = OllamaProcessor(**(data or {}))
    
    async def process(self, data: str) -> AsyncIterator[str]:
        """Process text data and return LLM responses."""
        async for result in self.processor.process(data):
            yield result