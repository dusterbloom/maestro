"""LLM processor using official genai-processors patterns."""

import logging
import os
from typing import AsyncIterable, Optional, Dict, Any

from genai_processors import content_api
from genai_processors import processor
from genai_processors.core import genai_model
from google.genai import types as genai_types

logger = logging.getLogger(__name__)


class GenaiModelProcessor(processor.Processor):
    """Gemini model processor using genai-processors built-in GenaiModel."""
    
    def __init__(
        self,
        api_key: str = None,
        model_name: str = "gemini-2.0-flash-001",
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tools: Optional[list] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.api_key = api_key or os.environ.get('GOOGLE_API_KEY')
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tools = tools or []
        
        if not self.api_key:
            raise ValueError("api_key is required for GenaiModelProcessor")
        
        self._setup_model()
    
    def _setup_model(self):
        """Setup the Gemini model processor."""
        try:
            # Prepare system instruction
            instruction_parts = []
            if self.system_instruction:
                instruction_parts.append(self.system_instruction)
            
            # Create generation config
            generation_config = genai_types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                response_modalities=['TEXT'],
            )
            
            # Add system instruction if provided
            if instruction_parts:
                generation_config.system_instruction = instruction_parts
            
            # Add tools if provided
            if self.tools:
                generation_config.tools = self.tools
            
            # Create the model processor
            self.model_processor = genai_model.GenaiModel(
                api_key=self.api_key,
                model_name=self.model_name,
                generate_content_config=generation_config,
                http_options=genai_types.HttpOptions(api_version='v1alpha')
            )
            
            logger.info(f"GenAI model processor initialized: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error setting up GenAI model: {e}")
            raise
    
    async def __call__(
        self, 
        content: AsyncIterable[content_api.ProcessorPart]
    ) -> AsyncIterable[content_api.ProcessorPart]:
        """Process content through Gemini model."""
        try:
            async for part in self.model_processor(content):
                yield part
        except Exception as e:
            logger.error(f"GenAI model processing error: {e}")
            # Create error response
            yield content_api.ProcessorPart(
                data=f"I apologize, but I encountered an error: {str(e)}",
                mimetype="text/plain",
                metadata={"error": True, "model": self.model_name}
            )


class OllamaProcessor(processor.Processor):
    """Ollama LLM processor as fallback."""
    
    def __init__(
        self,
        service_url: str = "http://localhost:11434",
        model_name: str = "gemma2:7b",
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.service_url = service_url
        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = system_prompt
    
    async def __call__(
        self, 
        content: AsyncIterable[content_api.ProcessorPart]
    ) -> AsyncIterable[content_api.ProcessorPart]:
        """Process content through Ollama."""
        # Collect text parts for processing
        text_parts = []
        non_text_parts = []
        
        async for part in content:
            if content_api.is_text(part.mimetype):
                text_parts.append(part.data if hasattr(part, 'data') else str(part))
            else:
                non_text_parts.append(part)
        
        if text_parts:
            try:
                # Combine text parts
                combined_text = " ".join(str(text) for text in text_parts)
                
                # Process with Ollama
                response_text = await self._call_ollama(combined_text)
                
                if response_text:
                    yield content_api.ProcessorPart(
                        data=response_text,
                        mimetype="text/plain",
                        metadata={
                            "model": self.model_name,
                            "source": "ollama",
                            "temperature": self.temperature
                        }
                    )
                    
            except Exception as e:
                logger.error(f"Ollama processing error: {e}")
                yield content_api.ProcessorPart(
                    data="I apologize, but I'm currently unavailable.",
                    mimetype="text/plain",
                    metadata={"error": True, "source": "ollama"}
                )
        
        # Pass through non-text parts
        for part in non_text_parts:
            yield part
    
    async def _call_ollama(self, text: str) -> str:
        """Call Ollama API."""
        try:
            import aiohttp
            import json
            
            url = f"{self.service_url}/api/generate"
            
            payload = {
                "model": self.model_name,
                "prompt": text,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                }
            }
            
            if self.system_prompt:
                payload["system"] = self.system_prompt
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "")
                    else:
                        logger.error(f"Ollama API error: {response.status}")
                        return ""
                        
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return ""


class LLMRouterProcessor(processor.Processor):
    """Router that tries Gemini first, falls back to Ollama."""
    
    def __init__(
        self,
        api_key: str = None,
        prefer_gemini: bool = True,
        gemini_model: str = "gemini-2.0-flash-001",
        ollama_model: str = "gemma2:7b",
        system_instruction: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.prefer_gemini = prefer_gemini
        self.system_instruction = system_instruction
        
        # Initialize processors based on preference and availability
        if self.api_key and self.prefer_gemini:
            try:
                self.primary_llm = GenaiModelProcessor(
                    api_key=self.api_key,
                    model_name=gemini_model,
                    system_instruction=system_instruction,
                    **kwargs
                )
                self.fallback_llm = OllamaProcessor(
                    model_name=ollama_model,
                    system_prompt=system_instruction,
                    **kwargs
                )
                logger.info("LLM router: Gemini primary, Ollama fallback")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {e}")
                self.primary_llm = OllamaProcessor(
                    model_name=ollama_model,
                    system_prompt=system_instruction,
                    **kwargs
                )
                self.fallback_llm = None
                logger.info("LLM router: Ollama only")
        else:
            self.primary_llm = OllamaProcessor(
                model_name=ollama_model,
                system_prompt=system_instruction,
                **kwargs
            )
            self.fallback_llm = None
            logger.info("LLM router: Ollama only")
    
    async def __call__(
        self, 
        content: AsyncIterable[content_api.ProcessorPart]
    ) -> AsyncIterable[content_api.ProcessorPart]:
        """Route LLM processing with fallback."""
        try:
            async for part in self.primary_llm(content):
                yield part
        except Exception as e:
            logger.error(f"Primary LLM failed: {e}")
            if self.fallback_llm:
                logger.info("Switching to fallback LLM")
                async for part in self.fallback_llm(content):
                    yield part
            else:
                # Create fallback response
                yield content_api.ProcessorPart(
                    data="I apologize, but I'm temporarily unavailable.",
                    mimetype="text/plain",
                    metadata={"error": True, "fallback": True}
                )


# Filter processor for removing non-text parts before LLM
@processor.create_filter
def text_only_filter(part: content_api.ProcessorPart) -> bool:
    """Filter to pass only text parts to LLM."""
    return content_api.is_text(part.mimetype)


# Combined text and LLM processor
class TextToLLMProcessor(processor.Processor):
    """Combined processor that filters for text and processes with LLM."""
    
    def __init__(self, llm_config: Dict[str, Any] = None, **kwargs):
        super().__init__(**kwargs)
        self.llm_config = llm_config or {}
        
        # Create the combined processor: filter + LLM
        self.combined_processor = (
            text_only_filter + 
            LLMRouterProcessor(**self.llm_config)
        )
    
    async def __call__(
        self, 
        content: AsyncIterable[content_api.ProcessorPart]
    ) -> AsyncIterable[content_api.ProcessorPart]:
        """Process through text filter and LLM."""
        async for part in self.combined_processor(content):
            yield part