"""WebSocket gateway using official genai-processors patterns."""

import asyncio
import logging
import json
import base64
import os
from typing import Dict, Any, List, AsyncIterable
from pathlib import Path

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from genai_processors import content_api
from genai_processors import streams
from genai_processors import processor

from .pipeline_builder import PipelineBuilder, PipelineRegistry, PipelineValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GenAI Processors Gateway", version="0.2.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_pipelines: Dict[str, processor.Processor] = {}
    
    async def connect(self, websocket: WebSocket, connection_id: str):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        logger.info(f"Client connected: {connection_id}")
    
    def disconnect(self, connection_id: str):
        """Remove a WebSocket connection."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        if connection_id in self.connection_pipelines:
            del self.connection_pipelines[connection_id]
        logger.info(f"Client disconnected: {connection_id}")
    
    async def send_personal_message(self, message: str, connection_id: str):
        """Send a message to a specific connection."""
        if connection_id in self.active_connections:
            try:
                await self.active_connections[connection_id].send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to {connection_id}: {e}")
                self.disconnect(connection_id)
    
    def set_pipeline(self, connection_id: str, pipeline: processor.Processor):
        """Set the processing pipeline for a connection."""
        self.connection_pipelines[connection_id] = pipeline
    
    def get_pipeline(self, connection_id: str) -> processor.Processor:
        """Get the processing pipeline for a connection."""
        return self.connection_pipelines.get(connection_id)


manager = ConnectionManager()


def load_config():
    """Load configuration from YAML files."""
    config_dir = Path(__file__).parent.parent.parent / "config"
    
    processors_config = {}
    pipelines_config = {}
    
    # Load processors config
    processors_file = config_dir / "processors.yaml"
    if processors_file.exists():
        with open(processors_file, 'r') as f:
            processors_config = yaml.safe_load(f)
    
    # Load pipelines config
    pipelines_file = config_dir / "pipelines.yaml"
    if pipelines_file.exists():
        with open(pipelines_file, 'r') as f:
            pipelines_config = yaml.safe_load(f)
    
    return processors_config, pipelines_config


def resolve_processor_configs(
    pipeline_config: List[str], 
    processors_config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Resolve processor names to full configurations."""
    resolved_configs = []
    
    for processor_name in pipeline_config:
        if processor_name in processors_config.get("processors", {}):
            resolved_configs.append(processors_config["processors"][processor_name])
        else:
            raise ValueError(f"Processor '{processor_name}' not found in configuration")
    
    return resolved_configs


class WebSocketAudioInput(processor.Processor):
    """Audio input processor for WebSocket connections."""
    
    def __init__(self, websocket: WebSocket, connection_id: str, **kwargs):
        super().__init__(**kwargs)
        self.websocket = websocket
        self.connection_id = connection_id
        self.audio_buffer = bytearray()
    
    async def __call__(
        self, 
        content: AsyncIterable[content_api.ProcessorPart]
    ) -> AsyncIterable[content_api.ProcessorPart]:
        """Generate audio stream from WebSocket messages."""
        try:
            while True:
                message = await self.websocket.receive()
                
                if message["type"] == "text":
                    try:
                        data = json.loads(message["text"])
                        
                        if data.get("type") == "audio":
                            # Handle base64 encoded audio
                            audio_data = base64.b64decode(data["data"])
                            self.audio_buffer.extend(audio_data)
                            
                            # Yield audio chunks
                            if len(self.audio_buffer) >= 16000:  # 1 second at 16kHz
                                yield content_api.ProcessorPart(
                                    data=bytes(self.audio_buffer),
                                    mimetype="audio/pcm",
                                    metadata={
                                        "sample_rate": data.get("sample_rate", 16000),
                                        "channels": data.get("channels", 1),
                                        "source": "websocket"
                                    }
                                )
                                self.audio_buffer.clear()
                        
                        elif data.get("type") == "text":
                            # Handle text input
                            yield content_api.ProcessorPart(
                                data=data["data"],
                                mimetype="text/plain",
                                metadata={"source": "websocket"}
                            )
                        
                        elif data.get("type") == "audio_end":
                            # Send remaining audio
                            if self.audio_buffer:
                                yield content_api.ProcessorPart(
                                    data=bytes(self.audio_buffer),
                                    mimetype="audio/pcm",
                                    metadata={
                                        "sample_rate": 16000,
                                        "channels": 1,
                                        "source": "websocket"
                                    }
                                )
                                self.audio_buffer.clear()
                    
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON from WebSocket: {e}")
                
                elif message["type"] == "bytes":
                    # Handle raw audio bytes
                    yield content_api.ProcessorPart(
                        data=message["bytes"],
                        mimetype="audio/pcm",
                        metadata={
                            "sample_rate": 16000,
                            "channels": 1,
                            "source": "websocket"
                        }
                    )
        
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {self.connection_id}")
        except Exception as e:
            logger.error(f"WebSocket input error: {e}")


class WebSocketOutput(processor.Processor):
    """Output processor that sends results back to WebSocket."""
    
    def __init__(self, websocket: WebSocket, connection_id: str, **kwargs):
        super().__init__(**kwargs)
        self.websocket = websocket
        self.connection_id = connection_id
    
    async def __call__(
        self, 
        content: AsyncIterable[content_api.ProcessorPart]
    ) -> AsyncIterable[content_api.ProcessorPart]:
        """Send processed results back to WebSocket."""
        async for part in content:
            try:
                # Prepare response based on content type
                if content_api.is_text(part.mimetype):
                    response = {
                        "type": "text",
                        "data": part.data if hasattr(part, 'data') else str(part),
                        "metadata": part.metadata
                    }
                
                elif content_api.is_audio(part.mimetype):
                    # Convert audio to base64 for transmission
                    audio_data = part.data if hasattr(part, 'data') else part
                    if isinstance(audio_data, bytes):
                        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                        response = {
                            "type": "audio",
                            "data": audio_b64,
                            "metadata": part.metadata
                        }
                    else:
                        continue
                
                else:
                    # Generic response for other types
                    response = {
                        "type": "data",
                        "data": str(part.data if hasattr(part, 'data') else part),
                        "metadata": part.metadata
                    }
                
                # Send response to WebSocket
                await manager.send_personal_message(
                    json.dumps(response), 
                    self.connection_id
                )
                
                # Also yield the part for potential further processing
                yield part
                
            except Exception as e:
                logger.error(f"Error sending WebSocket output: {e}")
                yield part


# API Routes
@app.get("/")
async def root():
    """Serve the main HTML page."""
    return FileResponse("test.html")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "framework": "genai-processors"}


@app.get("/processors")
async def list_processors():
    """List available processors."""
    processors_config, _ = load_config()
    return processors_config.get("processors", {})


@app.get("/pipelines")
async def list_pipelines():
    """List available pipelines."""
    _, pipelines_config = load_config()
    return pipelines_config.get("pipelines", {})


@app.get("/pipelines/registry")
async def list_registry_pipelines():
    """List built-in pipeline templates."""
    return {
        "audio_to_text": "Convert audio input to text",
        "text_to_audio": "Convert text to audio output", 
        "full_voice": "Complete voice processing pipeline",
        "realtime_agent": "Real-time voice agent with Gemini"
    }


@app.post("/pipelines/validate")
async def validate_pipeline(pipeline_config: Dict[str, Any]):
    """Validate a pipeline configuration."""
    try:
        processors = pipeline_config.get("processors", [])
        if not processors:
            raise HTTPException(status_code=400, detail="No processors in pipeline")
        
        is_valid = PipelineValidator.validate_pipeline_config(processors)
        
        return {
            "valid": is_valid,
            "processor_count": len(processors)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.websocket("/ws/{pipeline_name}")
async def websocket_endpoint(websocket: WebSocket, pipeline_name: str):
    """WebSocket endpoint for real-time processing."""
    connection_id = f"{pipeline_name}_{id(websocket)}"
    
    await manager.connect(websocket, connection_id)
    
    try:
        # Load pipeline configuration
        processors_config, pipelines_config = load_config()
        
        # Check if pipeline exists in config
        if pipeline_name in pipelines_config.get("pipelines", {}):
            pipeline_config = pipelines_config["pipelines"][pipeline_name]
            processor_configs = resolve_processor_configs(pipeline_config, processors_config)
        
        # Check if pipeline exists in registry
        elif pipeline_name in ["audio_to_text", "text_to_audio", "full_voice", "realtime_agent"]:
            config = {
                "api_key": os.environ.get("GOOGLE_API_KEY"),
                "project_id": os.environ.get("GOOGLE_PROJECT_ID"),
                "system_instruction": "You are a helpful voice assistant.",
                "prefer_gemini": True,
                "prefer_cloud": True
            }
            
            if pipeline_name == "audio_to_text":
                registry_config = PipelineRegistry.get_audio_to_text_pipeline(config)
            elif pipeline_name == "text_to_audio":
                registry_config = PipelineRegistry.get_text_to_audio_pipeline(config)
            elif pipeline_name == "full_voice":
                registry_config = PipelineRegistry.get_full_voice_pipeline(config)
            elif pipeline_name == "realtime_agent":
                registry_config = PipelineRegistry.get_realtime_agent_pipeline(config)
            
            processor_configs = registry_config["processors"]
        
        else:
            await manager.send_personal_message(
                json.dumps({"type": "error", "data": f"Pipeline '{pipeline_name}' not found"}),
                connection_id
            )
            return
        
        # Validate configuration
        if not PipelineValidator.validate_pipeline_config(processor_configs):
            await manager.send_personal_message(
                json.dumps({"type": "error", "data": "Invalid pipeline configuration"}),
                connection_id
            )
            return
        
        # Build the processing pipeline
        context = {"websocket": websocket, "connection_id": connection_id}
        
        try:
            processing_pipeline = PipelineBuilder.build_pipeline(processor_configs, context)
            manager.set_pipeline(connection_id, processing_pipeline)
            
            logger.info(f"Pipeline '{pipeline_name}' initialized for {connection_id}")
            
            # Send ready signal
            await manager.send_personal_message(
                json.dumps({
                    "type": "ready", 
                    "data": f"Pipeline '{pipeline_name}' ready",
                    "processor_count": len(processor_configs)
                }),
                connection_id
            )
            
        except Exception as e:
            logger.error(f"Error building pipeline: {e}")
            await manager.send_personal_message(
                json.dumps({"type": "error", "data": f"Pipeline build error: {str(e)}"}),
                connection_id
            )
            return
        
        # Create input processor for WebSocket
        input_processor = WebSocketAudioInput(websocket, connection_id)
        
        # Create output processor for WebSocket
        output_processor = WebSocketOutput(websocket, connection_id)
        
        # Create complete pipeline: input -> processing -> output
        complete_pipeline = input_processor + processing_pipeline + output_processor
        
        # Create empty input stream to start the pipeline
        empty_stream = streams.stream_content([])
        
        # Process the stream
        try:
            async for result in complete_pipeline(empty_stream):
                # Results are sent via WebSocketOutput processor
                pass
        
        except asyncio.CancelledError:
            logger.info(f"Pipeline cancelled for {connection_id}")
        except Exception as e:
            logger.error(f"Pipeline processing error: {e}")
            await manager.send_personal_message(
                json.dumps({"type": "error", "data": str(e)}),
                connection_id
            )
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await manager.send_personal_message(
                json.dumps({"type": "error", "data": str(e)}),
                connection_id
            )
        except:
            pass
    finally:
        manager.disconnect(connection_id)


@app.websocket("/ws/live/{model_name}")
async def live_websocket_endpoint(websocket: WebSocket, model_name: str = "gemini-2.5-flash-preview-native-audio-dialog"):
    """WebSocket endpoint using genai-processors LiveProcessor."""
    connection_id = f"live_{model_name}_{id(websocket)}"
    
    await manager.connect(websocket, connection_id)
    
    try:
        from genai_processors.core import live_model, audio_io
        from google.genai import types as genai_types
        import pyaudio
        
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            await manager.send_personal_message(
                json.dumps({"type": "error", "data": "GOOGLE_API_KEY not configured"}),
                connection_id
            )
            return
        
        # Create live processor
        live_processor = live_model.LiveProcessor(
            api_key=api_key,
            model_name=model_name,
            realtime_config=genai_types.LiveConnectConfig(
                system_instruction=["You are a helpful voice assistant."],
                tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
                response_modalities=['AUDIO'],
                speech_config={'language_code': 'en-US'},
            ),
            http_options=genai_types.HttpOptions(api_version='v1alpha'),
        )
        
        # Audio I/O setup
        pya = pyaudio.PyAudio()
        audio_input = audio_io.PyAudioIn(pya, use_pcm_mimetype=True)
        audio_output = audio_io.PyAudioOut(pya)
        
        # Create live agent pipeline
        live_agent = audio_input + live_processor + audio_output
        
        # Simple text input for triggering
        from genai_processors.core import text
        input_stream = text.terminal_input()
        
        logger.info(f"Live agent started for {connection_id}")
        
        # Send ready signal
        await manager.send_personal_message(
            json.dumps({"type": "ready", "data": "Live agent ready"}),
            connection_id
        )
        
        # Process the live stream
        async for part in live_agent(input_stream):
            # Send transcription and status updates
            if hasattr(part, 'metadata') and part.metadata:
                await manager.send_personal_message(
                    json.dumps({
                        "type": "status",
                        "data": str(part),
                        "metadata": part.metadata
                    }),
                    connection_id
                )
    
    except Exception as e:
        logger.error(f"Live WebSocket error: {e}")
        await manager.send_personal_message(
            json.dumps({"type": "error", "data": str(e)}),
            connection_id
        )
    finally:
        manager.disconnect(connection_id)


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "framework": "genai-processors"}


@app.exception_handler(500)
async def server_error_handler(request, exc):
    logger.error(f"Server error: {exc}")
    return {"error": "Internal server error", "framework": "genai-processors"}


if __name__ == "__main__":
    uvicorn.run(
        "processors.gateway.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )