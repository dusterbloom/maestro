# Resilient Session Management and Resource Cleanup Plan

## Overview

Design and implement a robust session management system that achieves resilient session management and resource cleanup with minimal and maintainable code changes, adhering to codebase standards with no hardcoded values.

## Current State Analysis

### WhisperLive Session Management
- Basic `Client` class with minimal reconnection logic (`whisper_live/client.py:17`)
- Simple `wait_before_disconnect()` method with 15-second hardcoded timeout
- `ClientManager.is_client_timeout()` enforces 600-second max connection time
- `ClientManager.remove_client()` calls `client.cleanup()` but cleanup is minimal
- `ServeClientBase.cleanup()` only sets `exit=True` flag

### Our Orchestrator Integration
- No direct WhisperLive client usage - relies on frontend connecting directly to WhisperLive
- Good TTS interruption capabilities with session tracking in `active_tts_sessions`
- Basic session history management in `session_history` dict
- Missing resilient connection patterns and automatic recovery

### Identified Gaps
- No automatic reconnection logic with exponential backoff
- No connection health monitoring or preemptive reconnection
- Basic resource cleanup (just setting flags, no verification)
- Hard timeout limits without graceful warning systems
- No session state persistence for recovery after disconnections
- No circuit breaker pattern for failed service handling

## Implementation Plan

### 1. ResilientWhisperClient Component
**Purpose**: Wrapper around WhisperLive's Client class that adds resilience patterns

**Features**:
- **Exponential Backoff Reconnection**: Configurable retry intervals with exponential backoff
- **Connection Health Monitoring**: Periodic ping/pong to detect connection issues early
- **Automatic Session Recovery**: Persist session state and resume after disconnections
- **Enhanced Cleanup Tracking**: Track all resources and verify cleanup completion

**Implementation**:
```python
class ResilientWhisperClient:
    def __init__(self, config: Config):
        self.base_client = None
        self.retry_config = RetryConfig.from_config(config)
        self.health_monitor = ConnectionHealthMonitor(config)
        self.cleanup_tracker = ResourceCleanupTracker()
        
    async def connect_with_retry(self) -> bool:
        # Exponential backoff implementation
        
    async def ensure_connection_health(self):
        # Proactive health monitoring
        
    async def cleanup_with_verification(self):
        # Tracked resource cleanup
```

### 2. SessionManager Component
**Purpose**: Centralized session lifecycle management

**Features**:
- **Session State Tracking**: Track active sessions and their health status
- **Connection Pooling**: Reuse connections when possible to reduce overhead
- **Circuit Breaker Pattern**: Fail fast when services are consistently down
- **Session Persistence**: Store session state for recovery scenarios

**Implementation**:
```python
class SessionManager:
    def __init__(self, config: Config):
        self.active_sessions = {}  # session_id -> SessionState
        self.connection_pool = ConnectionPool(config)
        self.circuit_breakers = {}  # service -> CircuitBreaker
        self.persistence = SessionPersistence(config)
        
    async def get_or_create_session(self, session_id: str) -> Session:
        # Session lifecycle management
        
    async def handle_session_failure(self, session_id: str, error: Exception):
        # Graceful failure handling
```

### 3. ConnectionHealthMonitor Component
**Purpose**: Background health checking and preemptive maintenance

**Features**:
- **Periodic Health Checks**: Configurable interval health monitoring
- **Preemptive Reconnection**: Reconnect before reaching timeout limits
- **Service Availability Monitoring**: Track service health across the system
- **Health Metrics Collection**: Gather connection health data for optimization

**Implementation**:
```python
class ConnectionHealthMonitor:
    def __init__(self, config: Config):
        self.check_interval = config.HEALTH_CHECK_INTERVAL
        self.preemptive_threshold = config.PREEMPTIVE_RECONNECT_THRESHOLD
        
    async def start_monitoring(self):
        # Background health check loop
        
    async def check_connection_health(self, client: ResilientWhisperClient) -> HealthStatus:
        # Individual connection health assessment
```

### 4. ResourceCleanupTracker Component
**Purpose**: Comprehensive resource cleanup management

**Features**:
- **Resource Registration**: Track all resources that need cleanup per session
- **Cleanup Verification**: Verify that cleanup operations actually succeeded
- **Partial Cleanup Recovery**: Handle scenarios where cleanup partially fails
- **Resource Leak Detection**: Monitor for resource leaks and alert

**Implementation**:
```python
class ResourceCleanupTracker:
    def __init__(self):
        self.tracked_resources = {}  # session_id -> Set[Resource]
        self.cleanup_verifiers = {}  # resource_type -> VerificationFunc
        
    def register_resource(self, session_id: str, resource: Resource):
        # Track resource for cleanup
        
    async def cleanup_session_resources(self, session_id: str) -> CleanupResult:
        # Comprehensive cleanup with verification
```

### 5. Configuration Extensions
**Purpose**: Make all resilience parameters configurable

**New Configuration Parameters**:
```python
# Retry Configuration
RETRY_MAX_ATTEMPTS: int = int(os.getenv("RETRY_MAX_ATTEMPTS", "5"))
RETRY_BASE_DELAY: float = float(os.getenv("RETRY_BASE_DELAY", "1.0"))
RETRY_MAX_DELAY: float = float(os.getenv("RETRY_MAX_DELAY", "60.0"))
RETRY_BACKOFF_MULTIPLIER: float = float(os.getenv("RETRY_BACKOFF_MULTIPLIER", "2.0"))

# Health Monitoring
HEALTH_CHECK_INTERVAL: float = float(os.getenv("HEALTH_CHECK_INTERVAL", "30.0"))
PREEMPTIVE_RECONNECT_THRESHOLD: float = float(os.getenv("PREEMPTIVE_RECONNECT_THRESHOLD", "0.8"))
CONNECTION_TIMEOUT_WARNING: float = float(os.getenv("CONNECTION_TIMEOUT_WARNING", "480.0"))  # 80% of 600s

# Circuit Breaker
CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = int(os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5"))
CIRCUIT_BREAKER_TIMEOUT: float = float(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "60.0"))
CIRCUIT_BREAKER_RECOVERY_TIMEOUT: float = float(os.getenv("CIRCUIT_BREAKER_RECOVERY_TIMEOUT", "300.0"))

# Session Persistence
SESSION_PERSISTENCE_ENABLED: bool = os.getenv("SESSION_PERSISTENCE_ENABLED", "true").lower() == "true"
SESSION_STATE_TTL: int = int(os.getenv("SESSION_STATE_TTL", "3600"))  # 1 hour
```

## Implementation Strategy

### Phase 1: Foundation Components
1. Implement `ResourceCleanupTracker` as standalone component
2. Add configuration extensions to `config.py`
3. Create `ConnectionHealthMonitor` base implementation
4. Unit tests for core components

### Phase 2: Session Management Integration
1. Implement `SessionManager` component
2. Add circuit breaker pattern implementation
3. Integrate with existing orchestrator session tracking
4. Integration tests with WhisperLive

### Phase 3: Resilient Client Implementation
1. Implement `ResilientWhisperClient` wrapper
2. Add exponential backoff reconnection logic
3. Integrate health monitoring with session management
4. End-to-end resilience testing

### Phase 4: Production Integration
1. Add optional resilient client to orchestrator service
2. Maintain backward compatibility with direct frontend connections
3. Add monitoring and metrics collection
4. Performance testing and optimization

## Design Principles

### Minimal Code Changes
- Extend existing patterns rather than replacing them
- Use composition over inheritance for WhisperLive integration
- Maintain compatibility with current frontend connection approach

### Maintainable Architecture
- Single responsibility principle for each component
- Clear interfaces and dependency injection
- Comprehensive error handling and logging

### Configurable Behavior
- All timeout values and retry parameters configurable
- Environment variable based configuration following existing patterns
- Sensible defaults that work out-of-the-box

### No Hardcoded Values
- All resilience parameters sourced from configuration
- Use existing configuration patterns (config.py)
- Runtime adjustable parameters where appropriate

## Expected Benefits

1. **Reduced Connection Failures**: Automatic reconnection with exponential backoff
2. **Faster Failure Recovery**: Preemptive reconnection before timeouts
3. **Resource Leak Prevention**: Comprehensive cleanup tracking and verification
4. **Improved User Experience**: Seamless session recovery after network issues
5. **Better System Reliability**: Circuit breaker pattern prevents cascade failures
6. **Operational Visibility**: Health monitoring and metrics for system optimization

## Future Considerations

- Integration with distributed tracing for session flow visibility
- Advanced session affinity for multi-instance deployments
- Adaptive timeout adjustment based on historical performance
- Session migration capabilities for zero-downtime maintenance