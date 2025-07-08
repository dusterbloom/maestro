# CONTINUATION PROMPT FOR NEXT SESSION

## Context
You are continuing work on the **Maestro Voice Orchestrator** - an ultra-low latency voice assistant with speaker recognition and memory capabilities. The current system has critical performance issues that require a complete architectural redesign.

## Current State Assessment
The system has these **CRITICAL ISSUES**:

1. **Duplicate Embedding Processing**: Multiple 10.8-second Resemblyzer processes running simultaneously for the same session, causing CPU saturation
2. **Stream Corruption**: Frontend receiving `error: Stream is already ended` from concurrent HTTP SSE requests  
3. **Architectural Mismatch**: Hybrid HTTP SSE + WebSocket system creating race conditions
4. **No Request Deduplication**: Rapid user speech triggers duplicate expensive operations
5. **Distributed State**: Session state scattered across frontend/backend without coordination

**Evidence from logs**: Two identical embedding requests within 37ms for same session, each taking 10.8s, while user experiences 2+ second delays and disconnections.

## Your Mission
Execute **Phase 1** of the architectural redesign plan located at `/docs/tasks/architecture-redesign-plan.md`. 

**CRITICAL**: Read the full plan first, then implement Phase 1: "WebSocket-First Architecture"

## Implementation Strategy

### Step 1: Research Phase (MANDATORY)
Before writing any code, you MUST:

1. **Read the complete plan**: `/docs/tasks/architecture-redesign-plan.md`
2. **Analyze current codebase structure**:
   ```bash
   find orchestrator/src -name "*.py" | head -20
   find ui -name "*.tsx" -o -name "*.ts" | head -20
   ```
3. **Check current state management**:
   - Examine `orchestrator/src/main.py` - session handling
   - Examine `ui/components/VoiceButton.tsx` - frontend state
   - Examine WebSocket usage patterns
4. **Review current WebSocket implementation**:
   - Check `ui/lib/websocket.ts` if exists
   - Identify WhisperLive WebSocket connection patterns
   - Map current communication flows
5. **Examine service interfaces**:
   - Current service organization in `orchestrator/src/services/`
   - Dependencies and coupling between services
   - Async patterns and blocking operations

### Step 2: Architecture Setup
Create the new modular structure:

```
orchestrator/src/
├── core/
│   ├── state_machine.py        # Central session state management
│   ├── event_dispatcher.py     # WebSocket event handling  
│   └── session_manager.py      # Session lifecycle management
├── services/
│   ├── base_service.py         # Abstract service interface
│   └── [existing services...]  # Refactor existing services
└── utils/
    ├── deduplicator.py         # Request deduplication
    └── performance_monitor.py  # Metrics and health checks
```

### Step 3: WebSocket-First Implementation
1. **Replace HTTP SSE with WebSocket**: Single connection for all communication
2. **Implement State Machine**: Centralized session state with clear transitions
3. **Add Request Deduplication**: Prevent duplicate expensive operations
4. **Event-Driven Architecture**: Clean event system between frontend/backend

### Step 4: Validation
1. **Test request deduplication**: Ensure rapid requests don't trigger duplicate embeddings
2. **Verify state consistency**: Session state properly managed across all components  
3. **Measure performance**: Latency improvements and CPU usage reduction
4. **Test error recovery**: Graceful handling of connection issues

## Critical Requirements

### Performance Targets
- **Response latency**: <500ms for first request, <200ms for subsequent
- **No duplicate processing**: Only one embedding per session, ever
- **CPU efficiency**: No more than one Resemblyzer process per unique speaker
- **Memory management**: Proper cleanup of session state

### Code Quality Standards
- **Type hints**: All Python functions must have complete type annotations
- **Error handling**: Comprehensive error boundaries with graceful degradation
- **Logging**: Detailed timing and state transition logs for debugging
- **Documentation**: Each module must have clear docstrings explaining purpose

### Testing Requirements
- **Integration tests**: End-to-end WebSocket communication flow
- **Performance tests**: Latency measurements and load testing
- **Error scenarios**: Connection drops, service failures, rapid requests
- **State consistency**: Verify state machine transitions work correctly

## Potential Research Needs

If you encounter these issues, **STOP and research first**:

1. **WebSocket Library Choice**: Research best Python WebSocket library for FastAPI (likely `websockets` or `socketio`)
2. **State Persistence**: How to integrate Redis for session state without blocking
3. **Event System Design**: Best practices for event-driven architecture in Python
4. **Request Deduplication Patterns**: Proven patterns for preventing duplicate async operations
5. **Service Interface Patterns**: Abstract base classes vs protocols for service standardization

## Success Criteria
- **No duplicate embeddings**: Logs show only one embedding process per session
- **Fast responses**: Sub-second response times for all requests
- **Clean state transitions**: State machine logs show proper session lifecycle
- **Error resilience**: System recovers gracefully from connection issues
- **Frontend simplification**: Single WebSocket connection replaces multiple HTTP calls

## Warning Signs to Watch For
- **Multiple embedding logs** for same session within short timeframe
- **Stream errors** or connection timeout messages  
- **Memory leaks** in session state management
- **Race conditions** in concurrent request handling
- **Blocking operations** in critical response path

## Files to Focus On
**Primary Implementation Files**:
- `orchestrator/src/core/state_machine.py` (new)
- `orchestrator/src/core/event_dispatcher.py` (new)  
- `orchestrator/src/main.py` (major refactor)
- `ui/lib/websocket.ts` (new or major refactor)
- `ui/components/VoiceButton.tsx` (simplify state)

**Key Integration Points**:
- WhisperLive WebSocket connection (keep existing)
- Speaker recognition service (prevent duplicates)
- TTS pipeline (maintain performance)
- Redis state persistence (add)

## Final Notes
- **Read the plan first** - don't start coding until you understand the full architecture
- **Incremental approach** - get WebSocket communication working before adding complexity
- **Test continuously** - validate each component works before moving to next
- **Document decisions** - update plan with any architectural changes needed
- **Measure everything** - log timing and performance metrics throughout

The goal is a **magical voice assistant experience**: instant speaker recognition, perfect conversation memory, ultra-low latency responses. The foundation you build in Phase 1 will enable all future capabilities.