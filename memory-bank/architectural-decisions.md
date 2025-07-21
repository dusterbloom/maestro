# Architectural Decisions Log

## Decision 1: Distributed Event Bus Architecture
**Date**: Session 1 - Initial Rebuild  
**Problem**: Current PipelineEventBus is fake - creates tasks but no coordination  
**Decision**: Replace with Redis-based distributed event bus using pub-sub pattern  
**Rationale**: 
- True cross-service coordination like Daily.co
- Real acknowledgments and state tracking
- Handles service failures gracefully
- Scales to multiple containers

**Implementation**: 
- `event_bus.py`: DistributedEventBus class with Redis pub-sub
- Standardized ServiceEvent structure  
- Correlation IDs for acknowledgment tracking
- Service-specific and global channels

## Decision 2: Service Coordinator Pattern
**Date**: Session 1 - Initial Rebuild  
**Problem**: No coordination of state transitions across services  
**Decision**: Implement centralized ServiceCoordinator for state management  
**Rationale**:
- Mirrors Daily.co's coordinated interrupt pattern
- Ensures all services reach consistent states
- Handles timeouts and failure scenarios
- Clear service state tracking

**Implementation**:
- `service_coordinator.py`: ServiceCoordinator class
- ServiceState enum for consistent state names
- ServiceType enum mapping to biological terms
- Coordinated interrupt workflow with acknowledgments

## Decision 3: Service Integration Strategy - REVISED
**Date**: Session 1 - Initial Rebuild  
**Problem**: Services have no standardized communication protocol  
**Decision**: Use existing service APIs + Event Bus for coordination (no REST wrappers needed)  
**Rationale**:
- Ollama: Already has REST API via ollama.AsyncClient
- Kokoro TTS: Already has REST API (/v1/audio/speech)
- WhisperLive: Already has WebSocket API (port 9090)
- Event Bus for coordination only (interrupts, state changes)
- Avoid reinventing wheels - leverage existing protocols

**Implementation**: 
- Keep existing API calls in orchestrator
- Add ServiceCoordinator integration to existing flows
- Event bus for coordination, not service communication

## Decision 4: Redis as Message Backbone  
**Date**: Session 1 - Initial Rebuild  
**Problem**: No reliable cross-container messaging  
**Decision**: Use existing Redis container for pub-sub messaging  
**Rationale**:
- Already configured in docker-compose.yml
- Proven reliability for distributed systems
- Built-in persistence and clustering support
- Low latency pub-sub capabilities

**Implementation**:
- Added redis and aioredis to requirements.txt
- Channel naming convention: maestro:global, maestro:service:{id}, maestro:interrupt, maestro:state

## Decision 5: Biological Naming Convention
**Date**: Session 1 - Planning  
**Problem**: Technical terminology not intuitive for non-technical users  
**Decision**: Implement biological metaphor throughout system  
**Rationale**:
- EAR (STT), BRAIN (LLM), MOUTH (TTS), NERVOUS SYSTEM (orchestrator)
- Intuitive understanding of data flow
- Natural debugging language ("ear not listening")
- Self-documenting code

**Implementation**: Applied in ServiceType enum, will extend to class names and variables

## Decision 6: Memory Bank Progress Tracking
**Date**: Session 1 - Context Management  
**Problem**: Multi-session rebuild needs persistent progress tracking  
**Decision**: Use memory-bank/ directory with regular file tools  
**Rationale**:
- MCP memory bank tool has path resolution issues
- Regular Read/Write tools work reliably
- Structured markdown documentation
- Git-trackable progress

**Implementation**: 
- progress-tracking-strategy.md for session continuity
- architectural-decisions.md for decision rationale  
- implementation-steps.md for detailed progress tracking