# Ultra-Low-Latency Voice Agent Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Client Layer                               │
│  ┌─────────────────┐                                                │
│  │   Next.js PWA   │  ← Web Audio API, Opus 48kHz                  │
│  │  (React 18+)    │  ← Adaptive Jitter Buffer                     │
│  └────────┬────────┘                                                │
│           │ WebSocket (duplex)                                      │
└───────────┼─────────────────────────────────────────────────────────┘
            │
┌───────────┼─────────────────────────────────────────────────────────┐
│           ▼                    Backend Layer                         │
│  ┌─────────────────┐                                                │
│  │  FastAPI Server │  ← WebSocket Handler                          │
│  │   (Stateless)   │  ← Asyncio Queues & Back-pressure             │
│  └────────┬────────┘                                                │
│           │                                                          │
│     ┌─────┴─────┬──────────┬───────────┐                          │
│     ▼           ▼          │           ▼                          │
│ ┌────────┐ ┌────────┐     │     ┌────────┐                      │
│ │  STT   │ │External│     │     │  TTS   │                      │
│ │Whisper │ │ Ollama │     │     │ Kokoro │                      │
│ │Live+VAD│ │Service │     │     │82M+CUDA│                      │
│ │  CUDA  │ │  HTTP  │     │     │  ONNX  │                      │
│ └────────┘ └────────┘     │     └────────┘                      │
│                           │           │                            │
│                           │           └─ Piper (CPU/ARM)          │
└───────────────────────────┼─────────────────────────────────────────┘
                            │
                   External Service
                            │
                    ┌───────┴────────┐
                    │  Ollama Docker │
                    │  (Pre-existing) │
                    │  Port: 11434   │
                    └────────────────┘
```

## Latency Budget Allocation

| Component | Budget | Measured Target | Notes |
|-----------|--------|-----------------|-------|
| Audio Capture | 20ms | 16ms | 16kHz mono, 320ms chunks |
| Network (Client→Server) | 10ms | 8ms | Local network |
| STT+VAD (WhisperLive) | 150ms | 120ms | Tiny model, streaming, CUDA |
| LLM (Ollama External) | 200ms | 180ms | Gemma-2B via HTTP API |
| TTS (Kokoro) | 100ms | 80ms | 82M model, ONNX CUDA |
| Audio Synthesis | 10ms | 8ms | Direct waveform generation |
| Network (Server→Client) | 5ms | 4ms | WebSocket push |
| **Total E2E** | **495ms** | **406ms** | 89ms buffer |

## Sequence Diagram - Streaming Pipeline

```
Client          FastAPI      WhisperLive+VAD    Ollama(External)   Kokoro+CUDA
  │               │              │                    │                │
  ├─Audio Chunk──►│              │                    │                │
  │  (320ms)      ├─16kHz mono──►│                    │                │
  │               │              ├─VAD+Transcript─────►│                │
  │               │              │    (streaming)     │                │
  │               │              │                    ├─HTTP Stream───►│
  │               │              │                    │   Tokens       │
  │               │◄─────────────┼────────────────────┼───Audio Chunk─┤
  │◄─Opus Frame───┤              │                    │    (48kHz)     │
  │  (streaming)  │              │                    │                │
```

## Component Interfaces

### WebSocket Message Schema
```json
{
  "client_to_server": {
    "type": "audio_chunk",
    "data": "base64_encoded_opus",
    "timestamp": 1234567890,
    "sequence": 1
  },
  "server_to_client": {
    "type": "audio_response | transcript | status",
    "data": "base64_audio | text | status_info",
    "timestamp": 1234567890,
    "latency_ms": 145
  }
}
```

### gRPC Service Definitions
```proto
service VoiceAgent {
  rpc StreamAudio(stream AudioChunk) returns (stream AudioResponse);
}

message AudioChunk {
  bytes data = 1;
  int64 timestamp = 2;
  int32 sequence = 3;
}

message AudioResponse {
  oneof response {
    bytes audio_data = 1;
    string transcript = 2;
    Status status = 3;
  }
  int64 timestamp = 4;
  int32 latency_ms = 5;
}
```

: 4G
        reservations:
          devices:
            - capabilities: [gpu]
              count: all

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    environment:
      - NEXT_PUBLIC_WS_URL=${WS_URL:-ws://voice-agent:8000/ws}
    ports:
      - "3001:3000"
```

## Performance Optimization Strategies

### CPU/Memory
- Thread affinity for audio processing
- Zero-copy audio buffers
- Preallocated memory pools
- NUMA-aware allocation on multi-socket systems

### ARM Specific (Orange Pi 5)
- NEON SIMD for audio processing
- INT8 quantization for Whisper
- ONNX Runtime with ARM compute library
- Thermal throttling mitigation

### Network
- Binary WebSocket frames (no JSON parsing overhead)
- TCP_NODELAY for minimal latency
- Connection pooling for internal services
- Protobuf for compact serialization

## Monitoring & Observability

```python
class LatencyTracker:
    buckets = {
        "audio_capture": Histogram(),
        "vad_process": Histogram(),
        "stt_inference": Histogram(),
        "llm_generation": Histogram(),
        "tts_synthesis": Histogram(),
        "network_rtt": Histogram()
    }
    
    def emit_metrics(self):
        return {
            "p50": self.calculate_percentile(50),
            "p95": self.calculate_percentile(95),
            "p99": self.calculate_percentile(99)
        }
```

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| ARM quantization degradation | Quality loss | Dual-path: FP16 for quality, INT8 for speed |
| Thermal throttling on Orange Pi | Latency spikes | Active cooling, frequency governor tuning |
| WebSocket reconnection storms | Service disruption | Exponential backoff, connection pooling |
| Memory pressure from model loading | OOM kills | Model lazy loading, swap to zram |
| Opus encoding latency | Budget overrun | Pre-allocated encoder state, tuned complexity |