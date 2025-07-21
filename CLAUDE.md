# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is **Maestro**, a real-time voice conversation system that orchestrates speech-to-text (STT), large language models (LLM), and text-to-speech (TTS) services. The system is designed for low-latency voice interactions with interrupt capabilities.

## Architecture

The system follows a **Central Orchestrator Architecture** with these core components:

- **Orchestrator** (`orchestrator/src/main.py`): Central coordinator that manages all voice processing streams via WebSocket connections
- **WhisperLive**: External STT service for real-time speech recognition
- **Ollama**: LLM service for generating responses  
- **Kokoro TTS**: Text-to-speech service for audio generation
- **Terminal Client** (`terminal_client.py`): Test client for voice interaction
- **Plugin System**: Modular interrupt detection and memory management

### Key Design Patterns

1. **Event-Driven Processing**: Uses `PipelineEventBus` for fire-and-forget event handling
2. **Session Management**: Each voice session (`StreamSession`) maintains its own state and connections
3. **Sequential TTS Processing**: Prevents voice "avalanche" through queued sentence processing
4. **Real-time Interruption**: Voice activity detection allows natural conversation interrupts

## Common Development Commands

### Build and Deployment
```bash
# Build with no cache (required after changes)
docker-compose build --no-cache

# Start basic services
docker-compose up

# Start with memory services enabled  
docker-compose --profile with-memory up

# Run terminal test client
python terminal_client.py
```

### Development
```bash
# View orchestrator logs
docker-compose logs -f orchestrator

# Debug active sessions
curl http://localhost:8000/debug/sessions

# Health check
curl http://localhost:8000/health
```

## Key Configuration

Configuration is centralized in `orchestrator/src/config.py` and controlled via environment variables in `docker-compose.yml`:

**Critical Settings:**
- `STT_MODEL`: Whisper model size (tiny/small/base)
- `LLM_MODEL`: Ollama model name
- `TTS_VOICE`: Voice name for Kokoro TTS
- `CHUNK_SIZE`: Audio processing chunk size (256 bytes)
- `NO_SPEECH_THRESHOLD`: Voice activity detection threshold

## Core Components

### Orchestrator (orchestrator/src/main.py)

The main service coordinator with these key responsibilities:
- **WebSocket Management**: `/ws/voice` endpoint for frontend connections
- **WhisperLive Integration**: Real-time STT via WebSocket to port 9090
- **Session Management**: Creates/manages `StreamSession` objects
- **Interrupt Handling**: `interrupt_session()` method stops TTS/processing
- **Event Bus**: Fire-and-forget event processing for low latency

### Plugin System (orchestrator/src/plugins/)

Modular plugin architecture:
- **InterruptPlugin**: Voice activity detection during TTS playback
- **MemoryPlugin**: Session memory and context management  
- **SpeakerPlugin**: Speaker identification and voice processing
- **BasePlugin**: Abstract plugin interface with async lifecycle

### Terminal Client (terminal_client.py)

Testing client that demonstrates the audio pipeline:
- Records from microphone (16kHz Float32)
- Sends raw audio to orchestrator via WebSocket
- Plays back TTS audio responses
- Supports real-time interruption with 'i' key

## Audio Processing Pipeline

1. **Audio Capture**: 16kHz Float32 format, 256-byte chunks
2. **STT Processing**: Forward to WhisperLive via WebSocket
3. **Transcript Processing**: Complete sentences trigger LLM processing  
4. **LLM Streaming**: Ollama generates streaming responses
5. **TTS Generation**: Kokoro converts text to audio
6. **Sequential Playback**: Prevents audio overlap through queuing

## Interrupt System

The system supports natural conversation interrupts:

1. **Voice Detection**: InterruptPlugin monitors audio levels during TTS
2. **Immediate Abort**: `interrupt_session()` stops TTS and clears queues
3. **State Recovery**: Session remains active for continued conversation
4. **Event Propagation**: Interrupt events notify all plugins

## Testing and Debugging

### Running Tests
Use the terminal client for end-to-end testing:
```bash
python terminal_client.py
# Press SPACE to toggle recording
# Press 'i' to send interrupt
# Press 'q' to quit
```

### Debugging Audio Issues
- Check `orchestrator/src/main.py:765-812` for detailed audio logging
- Audio stats logged: min/max/mean/std of audio samples
- Binary debug info shows raw WebSocket data format
- Terminal client logs audio levels in real-time

### Common Issues
- **WhisperLive Connection**: Check `WHISPER_URL` environment variable
- **Audio Format**: Must be 16kHz Float32, 256-byte chunks
- **TTS Timeout**: Increase `TTS_TIMEOUT` if Kokoro is slow
- **Memory Leaks**: Monitor `docker-compose logs orchestrator` for session cleanup

## Important Notes

- The system requires GPU support for WhisperLive and Kokoro TTS services
- Audio processing is extremely sensitive to format - Float32 16kHz only
- Plugin events use fire-and-forget pattern for minimal latency impact
- Session cleanup happens automatically after 1 hour of inactivity
- All processing is designed to be interruptible for natural conversation flow