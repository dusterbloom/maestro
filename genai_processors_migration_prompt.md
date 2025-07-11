# GenAI Processors Migration Prompt for Maestro

## Context & Objective

You are tasked with completely refactoring the "maestro" voice orchestration system to leverage Google's **genai-processors** framework (announced December 2024). Maestro is currently a custom ultra-low latency voice conversation system that needs to be migrated to use Google's professional streaming architecture while preserving all performance characteristics.

## Current Maestro Architecture Analysis

### Key Components (Working System)
```
Frontend (React/Next.js) ←→ Orchestrator (FastAPI) ←→ WhisperLive (STT)
                                    ↓
                            Ollama (LLM) + Kokoro (TTS)
```

### Critical Features to Preserve
1. **Ultra-low latency**: < 500ms end-to-end response time
2. **Real-time barge-in**: Voice activity detection during TTS playback
3. **Sentence-level streaming**: TTS starts as soon as LLM completes a sentence
4. **Session management**: Conversation history and context
5. **Current UI/UX**: No changes to frontend user experience
6. **Interruption handling**: Clean TTS stopping when user speaks

### Current File Structure (Reference)
```
maestro/
├── orchestrator/src/main.py          # Main FastAPI orchestrator
├── orchestrator/src/config.py        # Configuration management
├── ui/lib/websocket.ts               # Frontend WebSocket client
├── ui/components/VoiceButton.tsx     # Main voice interface
├── ui/lib/audio.ts                   # Audio recording/playback
└── docker-compose.yml               # Service orchestration
```

### Current Technical Stack
- **STT**: WhisperLive (WebSocket) 
- **LLM**: Ollama with streaming (gemma3n:latest)
- **TTS**: Kokoro FastAPI (sentence-level streaming)
- **Frontend**: React/Next.js with WebSocket + WebRTC
- **Orchestrator**: FastAPI with async/await

### Performance Metrics (Must Match/Exceed)
- **STT Latency**: ~50-100ms (WhisperLive with VAD)
- **LLM TTFT**: ~100-200ms (Ollama streaming)
- **TTS Latency**: ~100-150ms per sentence (Kokoro)
- **Total E2E**: < 500ms for first audio output
- **Barge-in Response**: < 100ms interruption detection

## GenAI Processors Framework Research Required

### Must Examine These Google Examples
1. **Real-Time Live Example**: `examples/realtime_simple_cli.py`
   - Audio-in/Audio-out Live agent
   - Google Search tool integration
   - Real-time streaming patterns

2. **Live Commentary Example**: `examples/live/README.md`
   - Dual agent architecture (event detection + conversation)
   - Real-time processing patterns
   - State management

3. **Research Agent Example**: `examples/research/README.md`
   - Multi-processor chaining
   - ProcessorPart creation patterns
   - Complex workflow orchestration

4. **Core Processors**: `core/` directory
   - Base processor implementations
   - Stream management utilities
   - Concurrency patterns

### Key Classes to Understand
- `Processor` base class
- `LiveProcessor` for real-time streaming  
- `ProcessorPart` for data flow
- Stream utilities (splitting, concatenating, merging)
- Async/concurrent execution patterns

## Target Architecture Design

### Proposed GenAI Processors Flow
```python
# Target processor chain
audio_input → WhisperProcessor → LLMProcessor → TTSProcessor → audio_output
               ↓                    ↓              ↓
           transcripts         text_chunks    audio_chunks
```

### Required Custom Processors
1. **WhisperLiveProcessor**: Integrate existing WhisperLive WebSocket
2. **OllamaStreamProcessor**: Stream LLM tokens with sentence detection
3. **KokoroTTSProcessor**: Convert sentences to audio immediately
4. **VoiceActivityProcessor**: Handle barge-in detection
5. **SessionManagerProcessor**: Maintain conversation context

## Specific Migration Requirements

### 1. WhisperLive Integration
- Must maintain existing WebSocket connection to WhisperLive
- Preserve VAD (Voice Activity Detection) settings
- Convert WhisperLive messages to ProcessorParts
- Handle real-time transcript streaming

### 2. Ollama LLM Integration  
- Stream tokens through genai-processors framework
- Implement sentence boundary detection within streaming
- Maintain conversation context/history
- Support configurable models (currently gemma3n:latest)

### 3. Kokoro TTS Integration
- Process sentence-level chunks immediately (not waiting for full response)
- Stream audio as ProcessorParts
- Maintain current voice settings (af_bella, speed, volume)
- Support interruption/cancellation

### 4. Barge-in Implementation
- Voice activity detection during TTS playback
- Immediate stream interruption capabilities
- Clean processor chain shutdown/restart
- Frontend audio queue management

### 5. Frontend Integration
- Maintain current React/WebSocket architecture
- Convert to communicate with genai-processors backend
- Preserve current UI components unchanged
- Support real-time audio streaming

## Technical Constraints

### Dependencies to Maintain
- **WhisperLive**: External service, cannot modify
- **Ollama**: Local LLM service, streaming API
- **Kokoro**: TTS service, existing API
- **Docker**: Container orchestration must work
- **React/Next.js**: Frontend stack unchanged

### Configuration Requirements
- Environment variable configuration (config.py pattern)
- Docker compose service definition
- Health check endpoints
- Debug/monitoring capabilities

### Performance Requirements
- No degradation in latency metrics
- Efficient memory usage for long conversations
- Proper cleanup of processor chains
- Scalable for multiple concurrent sessions

## Success Criteria

### Functional Requirements
✅ **Voice conversation flow**: Record → Transcribe → Generate → Speak  
✅ **Real-time barge-in**: Interrupt during TTS playback  
✅ **Session persistence**: Conversation history maintained  
✅ **Error recovery**: Graceful handling of service failures  
✅ **Multi-user support**: Concurrent conversation sessions  

### Performance Requirements  
✅ **Latency**: ≤ 500ms end-to-end response  
✅ **Throughput**: Support 10+ concurrent conversations  
✅ **Memory**: Efficient processor chain management  
✅ **Interruption**: < 100ms barge-in response time  

### Code Quality Requirements
✅ **Maintainability**: Clear processor separation  
✅ **Testability**: Unit tests for each processor  
✅ **Debuggability**: Logging and monitoring  
✅ **Documentation**: Clear setup and usage instructions  

## Deliverables Required

### 1. Core Processor Implementations
```python
# maestro/processors/whisper_live.py
class WhisperLiveProcessor(Processor): ...

# maestro/processors/ollama_stream.py  
class OllamaStreamProcessor(Processor): ...

# maestro/processors/kokoro_tts.py
class KokoroTTSProcessor(Processor): ...

# maestro/processors/voice_activity.py
class VoiceActivityProcessor(Processor): ...

# maestro/processors/session_manager.py
class SessionManagerProcessor(Processor): ...
```

### 2. Main Application
```python
# maestro/main.py - GenAI Processors powered orchestrator
# FastAPI app that manages processor chains per session
```

### 3. Frontend Updates
```typescript
// ui/lib/genai-processors-client.ts
// Updated WebSocket client for genai-processors backend
```

### 4. Configuration & Docker
```yaml
# docker-compose.yml - Updated for genai-processors architecture
# Requirements and environment setup
```

### 5. Migration Guide
- Step-by-step migration instructions
- Comparison with current architecture
- Performance benchmarking results
- Troubleshooting guide

## Important Implementation Notes

### GenAI Processors Best Practices
- Use `async def call()` pattern for all processors
- Leverage ProcessorPart metadata for rich data flow
- Implement proper error handling and recovery
- Use stream utilities for complex data manipulation
- Follow Google's concurrency patterns

### Maestro-Specific Considerations
- Preserve exact audio formats (WAV, sample rates)
- Maintain session isolation between users
- Handle network interruptions gracefully
- Support existing environment variable configuration
- Ensure Docker container compatibility

### Integration Points
- WhisperLive WebSocket ↔ ProcessorPart conversion
- Ollama streaming ↔ sentence boundary detection  
- Kokoro HTTP API ↔ async processor calls
- Frontend WebSocket ↔ processor chain events

## Research Strategy

1. **Study Google Examples**: Focus on real-time audio processing patterns
2. **Understand ProcessorPart**: Data flow and metadata patterns
3. **Analyze Stream Management**: Concurrent execution and chaining
4. **Map Current Flow**: Identify exact processor boundaries
5. **Design Processor Chain**: End-to-end data flow architecture
6. **Implement Incrementally**: Start with one processor, build up
7. **Test Performance**: Validate latency requirements throughout

## Expected Outcome

A production-ready maestro system powered by genai-processors that:
- Maintains all current functionality and performance
- Uses Google's professional streaming architecture
- Is significantly more maintainable and scalable
- Serves as a reference implementation for others
- Provides a solid foundation for future AI features

This migration should result in a **best-in-class voice orchestration system** that leverages Google's expertise while preserving maestro's unique ultra-low latency characteristics.