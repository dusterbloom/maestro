# Ultimate Voice Orchestrator Architecture: Stateful, Modular, Ultra-Low Latency

## Vision
A distributed voice assistant that recognizes speakers instantly, remembers conversations, and responds in <200ms while maintaining perfect state consistency across all components.

## Core Principles
1. **Single Source of Truth**: WebSocket-based state machine in orchestrator
2. **Event-Driven Architecture**: All components communicate via events
3. **Async by Design**: No blocking operations in critical path
4. **Modular Services**: Each service has single responsibility
5. **State Persistence**: Redis for session state, ChromaDB for memory

## Architecture Overview

### 1. Central State Machine (Orchestrator)
```
┌─────────────────────────────────────────┐
│           ORCHESTRATOR                  │
│  ┌─────────────────────────────────┐   │
│  │        STATE MACHINE            │   │
│  │  ┌─────────────────────────┐   │   │
│  │  │ Session Management      │   │   │
│  │  │ - Connection State      │   │   │
│  │  │ - Speaker Identity      │   │   │
│  │  │ - Conversation Context  │   │   │
│  │  │ - Audio Pipeline State  │   │   │
│  │  └─────────────────────────┘   │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │       EVENT DISPATCHER          │   │
│  │  - WebSocket Management         │   │
│  │  - Service Coordination         │   │
│  │  - Request Deduplication        │   │
│  │  - Error Recovery               │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

### 2. Modular Service Architecture
```
Frontend ←→ WebSocket ←→ Orchestrator ←→ Services
                             │
                ┌────────────┼────────────┐
                │            │            │
         ┌─────────┐  ┌─────────┐  ┌─────────┐
         │   STT   │  │   LLM   │  │   TTS   │
         │ Service │  │ Service │  │ Service │
         └─────────┘  └─────────┘  └─────────┘
                             │
                ┌────────────┼────────────┐
                │            │            │
         ┌─────────┐  ┌─────────┐  ┌─────────┐
         │ Speaker │  │ Memory  │  │  Redis  │
         │ Service │  │ Service │  │ State   │
         └─────────┘  └─────────┘  └─────────┘
```

## Implementation Plan

### Phase 1: WebSocket-First Architecture (Week 1)

#### 1.1 Replace HTTP with WebSocket
- **Current**: HTTP SSE + WebSocket hybrid
- **New**: Single WebSocket connection for everything
- **Benefits**: Real-time bidirectional communication, natural state management

#### 1.2 Centralized State Machine
```typescript
interface SessionState {
  id: string
  connectionState: 'connecting' | 'connected' | 'authenticated' | 'ready'
  speakerState: {
    status: 'unknown' | 'identifying' | 'recognized' | 'new'
    speakerId?: string
    confidence?: number
    embeddings?: Float32Array
  }
  conversationState: {
    context: ConversationMessage[]
    activeRequest?: string
    lastActivity: timestamp
  }
  audioState: {
    recording: boolean
    processing: boolean
    playing: boolean
    queue: AudioChunk[]
  }
}
```

#### 1.3 Event-Driven Communication
```typescript
// Inbound Events (Frontend → Orchestrator)
type InboundEvent = 
  | { type: 'audio.start' }
  | { type: 'audio.chunk', data: ArrayBuffer }
  | { type: 'audio.end' }
  | { type: 'conversation.interrupt' }
  | { type: 'speaker.identify', audio: ArrayBuffer }

// Outbound Events (Orchestrator → Frontend)
type OutboundEvent =
  | { type: 'session.ready', sessionId: string }
  | { type: 'speaker.identified', speaker: SpeakerInfo }
  | { type: 'transcript.partial', text: string }
  | { type: 'response.audio', audio: ArrayBuffer, text: string }
  | { type: 'error', message: string, recovery?: string }
```

### Phase 2: Async Service Orchestration (Week 2)

#### 2.1 Service Interface Standardization
```python
class ServiceInterface:
    async def process(self, input: Any, context: Dict) -> ServiceResult
    async def health_check() -> bool
    def get_metrics() -> Dict
```

#### 2.2 Speaker Recognition Pipeline
```
Audio Input → VAD Filter → Embedding Queue → Speaker Matcher → State Update
     ↓              ↓            ↓              ↓            ↓
   16ms          50ms         Background     Cache Hit    State Event
                                 (10s)        (1ms)       Notification
```

#### 2.3 Conversation Pipeline
```
Transcript → Context Lookup → LLM Generate → TTS Queue → Audio Stream
    ↓            ↓              ↓             ↓           ↓
  Instant     Redis Cache    Streaming     Parallel    WebSocket
              (5ms)          (200ms)       (100ms)     (10ms)
```

### Phase 3: Memory & Recognition (Week 3)

#### 3.1 Multi-Tier Caching Strategy
```
L1: In-Memory Session Cache (1ms access)
L2: Redis Session Store (5ms access)  
L3: ChromaDB Vector Store (50ms access)
L4: File/Database Persistence (500ms access)
```

#### 3.2 Smart Speaker Recognition
```python
class SpeakerRecognitionService:
    # Immediate recognition from cache
    async def quick_match(audio: bytes) -> Optional[Speaker]
    
    # Background embedding generation  
    async def deep_analysis(audio: bytes) -> SpeakerEmbedding
    
    # Progressive confidence building
    async def update_confidence(speaker_id: str, new_audio: bytes)
```

#### 3.3 Contextual Memory System
```python
class MemoryService:
    # Immediate context (current conversation)
    async def get_active_context(session_id: str) -> ConversationContext
    
    # Personal memory (speaker-specific)
    async def get_speaker_memory(speaker_id: str) -> PersonalMemory
    
    # Long-term memory (semantic search)
    async def search_memories(query: str, speaker_id: str) -> List[Memory]
```

### Phase 4: Performance Optimization (Week 4)

#### 4.1 Request Deduplication
```python
class RequestDeduplicator:
    active_requests: Dict[str, asyncio.Task] = {}
    
    async def process_or_join(request_id: str, coro):
        if request_id in active_requests:
            return await active_requests[request_id]
        
        task = asyncio.create_task(coro)
        active_requests[request_id] = task
        try:
            return await task
        finally:
            del active_requests[request_id]
```

#### 4.2 Predictive Preloading
```python
# Pre-warm TTS for likely responses
async def preload_common_responses(speaker_id: str)

# Pre-fetch conversation context
async def prefetch_context(speaker_id: str)

# Background model warming
async def warm_models_on_connection()
```

#### 4.3 Circuit Breakers & Fallbacks
```python
class ServiceCircuitBreaker:
    # Fail fast on service degradation
    # Graceful degradation paths
    # Automatic recovery detection
```

## Technical Implementation Details

### New File Structure
```
orchestrator/
├── src/
│   ├── core/
│   │   ├── state_machine.py        # Central state management
│   │   ├── event_dispatcher.py     # WebSocket event handling
│   │   └── session_manager.py      # Session lifecycle
│   ├── services/
│   │   ├── base_service.py         # Service interface
│   │   ├── speaker_service.py      # Recognition + embedding
│   │   ├── memory_service.py       # Context + personal memory
│   │   ├── conversation_service.py # LLM + context management
│   │   └── audio_service.py        # TTS + audio processing
│   ├── utils/
│   │   ├── deduplicator.py         # Request deduplication
│   │   ├── cache_manager.py        # Multi-tier caching
│   │   └── performance_monitor.py  # Metrics + health
│   └── main.py                     # WebSocket server
├── tests/
│   ├── integration/                # End-to-end tests
│   ├── performance/                # Latency benchmarks
│   └── unit/                       # Service tests
└── docs/
    ├── architecture.md             # System design
    ├── api.md                      # WebSocket API
    └── deployment.md               # Production setup
```

### Frontend Simplification
```typescript
// Single WebSocket connection
class VoiceOrchestrator {
  private ws: WebSocket
  private state: ClientState
  
  // Simple event handlers
  onSpeakerRecognized(handler: (speaker: Speaker) => void)
  onResponseReceived(handler: (audio: ArrayBuffer, text: string) => void)
  onError(handler: (error: Error) => void)
  
  // Simple actions
  startRecording()
  stopRecording()
  interrupt()
}
```

## Current Issues Analysis (From Session)

### Root Cause: Architectural Mismatch
1. **Multiple Communication Protocols**: HTTP SSE + WebSocket + Direct API calls
2. **Distributed State**: Session state scattered across frontend/backend
3. **Race Conditions**: Duplicate embedding requests within 37ms
4. **CPU Saturation**: Multiple 10.8s Resemblyzer processes running simultaneously
5. **Stream Corruption**: `error: Stream is already ended` from concurrent requests

### Specific Problems Found
```
18:04:05.460 - Background: 10 seconds accumulated for session session_d3ea9fcb3b724dbca850565151f702d4
18:04:05.497 - Background: 10 seconds accumulated for session session_d3ea9fcb3b724dbca850565151f702d4
```
- **Two identical embeddings** triggered within 37ms
- **Each takes 10.8 seconds** on CPU
- **System overwhelmed** despite "background" processing

### Frontend Issues
```
voice-ui-1 | error: failed to pipe response
voice-ui-1 | error: Stream is already ended
```
- **Stream corruption** from rapid requests
- **JSON parsing failures** with SSE responses
- **Connection timeouts** and disconnections

## Success Metrics
- **Latency**: <200ms average response time
- **Recognition**: >95% speaker accuracy after 3 interactions  
- **Reliability**: 99.9% uptime, graceful degradation
- **Memory**: Perfect conversation continuity across sessions
- **Scalability**: Handle 100+ concurrent users

## Risk Mitigation
1. **Incremental Migration**: Parallel deployment, gradual cutover
2. **Comprehensive Testing**: Load testing, chaos engineering  
3. **Monitoring**: Real-time metrics, alerting, health checks
4. **Rollback Plan**: Quick revert to current system if needed

## Next Steps
1. **Review & Approve Plan**: Validate architecture decisions
2. **Setup Development Environment**: Clean workspace for new architecture
3. **Phase 1 Implementation**: Start with WebSocket-first approach
4. **Testing & Validation**: Ensure each phase works before moving forward

This architecture will create a voice assistant that feels magical - instant recognition, perfect memory, ultra-low latency, while being maintainable and scalable.