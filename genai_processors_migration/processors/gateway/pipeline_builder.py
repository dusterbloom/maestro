"""Pipeline builder using official genai-processors patterns."""

import importlib
import logging
from typing import List, Dict, Any, Optional, AsyncIterable

from genai_processors import processor
from genai_processors import content_api
from genai_processors import streams

logger = logging.getLogger(__name__)


class PipelineBuilder:
    """Build processing pipelines using genai-processors patterns."""
    
    @staticmethod
    def build_processor(processor_config: Dict[str, Any], context: Dict[str, Any] = None) -> processor.Processor:
        """Build a single processor from configuration."""
        try:
            module_path = processor_config["module"]
            class_name = processor_config["class"]
            params = processor_config.get("params", {})
            
            # Add context to params if provided
            if context:
                params = {**params}
            
            # Import the module and get the class
            module = importlib.import_module(module_path)
            processor_class = getattr(module, class_name)
            
            # Create processor instance
            processor_instance = processor_class(**params)
            
            logger.info(f"Created processor: {class_name}")
            return processor_instance
            
        except Exception as e:
            logger.error(f"Error building processor {processor_config}: {e}")
            raise
    
    @staticmethod
    def build_pipeline(
        processor_configs: List[Dict[str, Any]], 
        context: Dict[str, Any] = None
    ) -> processor.Processor:
        """Build a complete pipeline by chaining processors."""
        if not processor_configs:
            raise ValueError("No processor configurations provided")
        
        try:
            # Build all processors
            processors = []
            for config in processor_configs:
                proc = PipelineBuilder.build_processor(config, context)
                processors.append(proc)
            
            # Chain processors using the + operator (genai-processors pattern)
            pipeline = processors[0]
            for proc in processors[1:]:
                pipeline = pipeline + proc
            
            logger.info(f"Built pipeline with {len(processors)} processors")
            return pipeline
            
        except Exception as e:
            logger.error(f"Error building pipeline: {e}")
            raise
    
    @staticmethod
    def build_parallel_pipeline(
        processor_configs: List[Dict[str, Any]], 
        context: Dict[str, Any] = None
    ) -> processor.Processor:
        """Build a parallel pipeline using the // operator."""
        if not processor_configs:
            raise ValueError("No processor configurations provided")
        
        try:
            # Build all processors
            processors = []
            for config in processor_configs:
                proc = PipelineBuilder.build_processor(config, context)
                processors.append(proc)
            
            # Parallelize processors using the // operator
            if len(processors) == 1:
                return processors[0]
            
            pipeline = processors[0]
            for proc in processors[1:]:
                pipeline = pipeline // proc
            
            logger.info(f"Built parallel pipeline with {len(processors)} processors")
            return pipeline
            
        except Exception as e:
            logger.error(f"Error building parallel pipeline: {e}")
            raise
    
    @staticmethod
    def create_input_stream(data: List[Any]) -> AsyncIterable[content_api.ProcessorPart]:
        """Create an input stream from data."""
        return streams.stream_content(data)
    
    @staticmethod
    def create_text_stream(texts: List[str]) -> AsyncIterable[content_api.ProcessorPart]:
        """Create a stream of text parts."""
        parts = [
            content_api.ProcessorPart(
                data=text,
                mimetype="text/plain"
            ) for text in texts
        ]
        return streams.stream_content(parts)
    
    @staticmethod
    def create_audio_stream(audio_data_list: List[bytes]) -> AsyncIterable[content_api.ProcessorPart]:
        """Create a stream of audio parts."""
        parts = [
            content_api.ProcessorPart(
                data=audio_data,
                mimetype="audio/pcm",
                metadata={"sample_rate": 16000, "channels": 1}
            ) for audio_data in audio_data_list
        ]
        return streams.stream_content(parts)


class PipelineRegistry:
    """Registry for common pipeline configurations."""
    
    @staticmethod
    def get_audio_to_text_pipeline(config: Dict[str, Any] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get audio-to-text pipeline configuration."""
        config = config or {}
        
        return {
            "processors": [
                {
                    "module": "processors.audio.audio_processor",
                    "class": "AudioProcessor",
                    "params": {
                        "target_sample_rate": config.get("sample_rate", 16000),
                        "target_channels": config.get("channels", 1)
                    }
                },
                {
                    "module": "processors.stt.speech_to_text",
                    "class": "STTRouterProcessor", 
                    "params": {
                        "project_id": config.get("project_id"),
                        "prefer_cloud": config.get("prefer_cloud", True)
                    }
                }
            ]
        }
    
    @staticmethod
    def get_text_to_audio_pipeline(config: Dict[str, Any] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get text-to-audio pipeline configuration."""
        config = config or {}
        
        return {
            "processors": [
                {
                    "module": "processors.tts.text_to_speech",
                    "class": "TextToSpeechPipeline",
                    "params": {
                        "tts_config": {
                            "project_id": config.get("project_id"),
                            "prefer_google": config.get("prefer_google", True),
                            "voice": config.get("voice", "af_bella")
                        },
                        "with_audio_output": config.get("with_audio_output", True),
                        "with_rate_limiting": config.get("with_rate_limiting", True)
                    }
                }
            ]
        }
    
    @staticmethod
    def get_full_voice_pipeline(config: Dict[str, Any] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get complete voice processing pipeline."""
        config = config or {}
        
        return {
            "processors": [
                # Audio preprocessing
                {
                    "module": "processors.audio.audio_processor",
                    "class": "AudioProcessor",
                    "params": {
                        "target_sample_rate": config.get("sample_rate", 16000),
                        "target_channels": config.get("channels", 1)
                    }
                },
                # Speech to text
                {
                    "module": "processors.stt.speech_to_text",
                    "class": "STTRouterProcessor",
                    "params": {
                        "project_id": config.get("project_id"),
                        "prefer_cloud": config.get("prefer_cloud", True)
                    }
                },
                # LLM processing
                {
                    "module": "processors.llm.genai_model",
                    "class": "TextToLLMProcessor",
                    "params": {
                        "llm_config": {
                            "api_key": config.get("api_key"),
                            "prefer_gemini": config.get("prefer_gemini", True),
                            "gemini_model": config.get("gemini_model", "gemini-2.0-flash-001"),
                            "system_instruction": config.get("system_instruction")
                        }
                    }
                },
                # Text to speech
                {
                    "module": "processors.tts.text_to_speech",
                    "class": "TextToSpeechPipeline",
                    "params": {
                        "tts_config": {
                            "project_id": config.get("project_id"),
                            "prefer_google": config.get("prefer_google", True),
                            "voice": config.get("voice", "af_bella")
                        },
                        "with_audio_output": config.get("with_audio_output", True),
                        "with_rate_limiting": config.get("with_rate_limiting", True)
                    }
                }
            ]
        }
    
    @staticmethod
    def get_realtime_agent_pipeline(config: Dict[str, Any] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get real-time agent pipeline using genai-processors core components."""
        config = config or {}
        
        return {
            "processors": [
                # Audio input
                {
                    "module": "processors.audio.audio_processor",
                    "class": "AudioInputProcessor",
                    "params": {
                        "sample_rate": config.get("sample_rate", 16000),
                        "use_pcm_mimetype": True
                    }
                },
                # Speech to text
                {
                    "module": "processors.stt.speech_to_text",
                    "class": "SpeechToTextProcessor",
                    "params": {
                        "project_id": config.get("project_id"),
                        "with_interim_results": False
                    }
                },
                # Gemini model
                {
                    "module": "processors.llm.genai_model",
                    "class": "GenaiModelProcessor",
                    "params": {
                        "api_key": config.get("api_key"),
                        "model_name": config.get("model_name", "gemini-2.0-flash-001"),
                        "system_instruction": config.get("system_instruction"),
                        "tools": config.get("tools", [])
                    }
                },
                # Text to speech with rate limiting
                {
                    "module": "processors.tts.text_to_speech",
                    "class": "TTSRouterProcessor",
                    "params": {
                        "project_id": config.get("project_id"),
                        "prefer_google": config.get("prefer_google", True)
                    }
                }
            ]
        }


class PipelineValidator:
    """Validate pipeline configurations."""
    
    @staticmethod
    def validate_processor_config(config: Dict[str, Any]) -> bool:
        """Validate a single processor configuration."""
        required_fields = ["module", "class"]
        
        for field in required_fields:
            if field not in config:
                logger.error(f"Missing required field '{field}' in processor config")
                return False
        
        # Try to import the module
        try:
            module_path = config["module"]
            class_name = config["class"]
            
            module = importlib.import_module(module_path)
            processor_class = getattr(module, class_name)
            
            # Check if it's a valid processor
            if not issubclass(processor_class, processor.Processor):
                logger.error(f"Class {class_name} is not a Processor subclass")
                return False
                
        except ImportError as e:
            logger.error(f"Cannot import module {module_path}: {e}")
            return False
        except AttributeError as e:
            logger.error(f"Class {class_name} not found in module {module_path}: {e}")
            return False
        
        return True
    
    @staticmethod
    def validate_pipeline_config(configs: List[Dict[str, Any]]) -> bool:
        """Validate a complete pipeline configuration."""
        if not configs:
            logger.error("Empty pipeline configuration")
            return False
        
        for i, config in enumerate(configs):
            if not PipelineValidator.validate_processor_config(config):
                logger.error(f"Invalid processor config at index {i}")
                return False
        
        return True


# Utility functions for common patterns
def create_simple_text_processor(system_instruction: str = None) -> processor.Processor:
    """Create a simple text processing pipeline."""
    config = {
        "api_key": None,  # Will use environment variable
        "system_instruction": system_instruction or "You are a helpful assistant."
    }
    
    pipeline_config = PipelineRegistry.get_text_to_audio_pipeline(config)
    return PipelineBuilder.build_pipeline(pipeline_config["processors"])


def create_voice_agent(
    api_key: str = None,
    project_id: str = None,
    system_instruction: str = None
) -> processor.Processor:
    """Create a complete voice agent pipeline."""
    config = {
        "api_key": api_key,
        "project_id": project_id,
        "system_instruction": system_instruction or "You are a helpful voice assistant.",
        "prefer_gemini": True,
        "prefer_cloud": True,
        "with_audio_output": True
    }
    
    pipeline_config = PipelineRegistry.get_full_voice_pipeline(config)
    return PipelineBuilder.build_pipeline(pipeline_config["processors"])