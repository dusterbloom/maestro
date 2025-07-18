# Audio Input Validation Guide

## Quick Start Validation

### 1. Service Health Check
```bash
# Start all services
docker-compose up -d

# Check all services are running
docker-compose ps

# Expected output:
# orchestrator    Up
# whisper-live    Up
# kokoro          Up
# voice-ui        Up
```

### 2. WebSocket Connection Test
```bash
# Test WebSocket endpoint
curl -i -N \
  -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Key: test" \
  -H "Sec-WebSocket-Version: 13" \
  http://localhost:8000/ws/voice
```

### 3. Browser Integration Test

#### Manual Test Steps:
1. Open browser to `http://localhost:3001`
2. Open browser DevTools (F12)
3. Navigate to Console tab
4. Click the voice button and speak
5. Watch for these console messages:

**Expected Console Output:**
```
🔗 WebSocket connected to orchestrator: ws://localhost:8000/ws/voice
🎤 Sending 4096 float32 samples (16384 bytes) to orchestrator
✅ Successfully sent audio to WhisperLive
📝 Transcription: [your spoken text]
🗣️ Processing: [AI response]
🔊 Playing audio response...
```

## Technical Validation

### Audio Format Verification
The system expects **Float32Array @ 16kHz** format:

```javascript
// Verify audio format in browser console
navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000 } })
  .then(stream => {
    const audioContext = new AudioContext({ sampleRate: 16000 });
    const source = audioContext.createMediaStreamSource(stream);
    
    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = (e) => {
      const data = e.inputBuffer.getChannelData(0);
      console.log('Audio format:', {
        sampleRate: audioContext.sampleRate,
        length: data.length,
        type: data.constructor.name,
        bytes: data.buffer.byteLength
      });
    };
    
    source.connect(processor);
  });
```

### WebSocket Message Flow

#### Client → Orchestrator
```javascript
// Binary audio data (Float32Array)
const audioData = new Float32Array(4096);
// Send as ArrayBuffer
websocket.send(audioData.buffer);
```

#### Orchestrator → WhisperLive
```python
# Forward binary data unchanged
await session.whisper_ws.send(audio_data)  # Float32Array bytes
```

#### WhisperLive → Orchestrator
```json
{
  "segments": [{
    "text": "transcribed text",
    "start": 0.0,
    "end": 1.5,
    "confidence": 0.95
  }]
}
```

## Automated Testing Script

### 1. Service Readiness Check
```bash
#!/bin/bash
# save as test-readiness.sh

echo "🔍 Testing service readiness..."

# Test orchestrator health
if curl -s http://localhost:8000/health > /dev/null; then
  echo "✅ Orchestrator healthy"
else
  echo "❌ Orchestrator not responding"
  exit 1
fi

# Test WhisperLive connection
if docker-compose logs whisper-live | grep -q "listening"; then
  echo "✅ WhisperLive ready"
else
  echo "❌ WhisperLive not ready"
  exit 1
fi

echo "🚀 All services ready for audio testing"
```

### 2. Audio Flow Test
```bash
#!/bin/bash
# save as test-audio-flow.sh

echo "🎤 Testing complete audio flow..."

# Start monitoring logs
echo "📊 Monitoring logs for audio flow..."
docker-compose logs -f orchestrator | grep -E "(audio|transcript|WhisperLive)" &
LOG_PID=$!

# Wait for test completion
sleep 10

# Check for successful audio processing
if docker-compose logs orchestrator | grep -q "Successfully sent audio to WhisperLive"; then
  echo "✅ Audio successfully reaching WhisperLive"
else
  echo "❌ Audio not reaching WhisperLive"
fi

# Check for transcription
if docker-compose logs orchestrator | grep -q "transcription"; then
  echo "✅ Transcription working"
else
  echo "❌ No transcription detected"
fi

kill $LOG_PID
```

## Performance Benchmarks

### Target Latencies
| Component | Target | Measured |
|-----------|--------|----------|
| Audio Capture | 20ms | TBD |
| Network (Client→Orchestrator) | 10ms | TBD |
| STT (WhisperLive) | 150ms | TBD |
| LLM Response | 200ms | TBD |
| TTS Generation | 100ms | TBD |
| **Total E2E** | **480ms** | **TBD** |

### Measurement Commands
```bash
# Monitor end-to-end latency
docker-compose logs orchestrator | grep -E "latency|processing time"

# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor network latency
ping -c 10 localhost
```

## Troubleshooting Quick Reference

### Common Issues & Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| **WebSocket Connection Failed** | "WebSocket connection failed" in console | Check `docker-compose ps`, restart orchestrator |
| **No Audio Data** | No "Sending audio" logs | Check microphone permissions, browser console |
| **WhisperLive Timeout** | "Failed to connect to WhisperLive" | Check GPU availability, restart whisper-live |
| **Format Mismatch** | "Cannot interpret as Float32" | Verify resampling in voice-pipeline.ts |
| **No Transcription** | Audio sent but no text returned | Check WhisperLive logs, verify model loading |

### Debug Commands
```bash
# Real-time orchestrator logs
docker-compose logs -f orchestrator

# Check WhisperLive status
docker-compose logs whisper-live | tail -20

# Test WebSocket manually
websocat ws://localhost:8000/ws/voice

# Check audio format
docker-compose logs orchestrator | grep -A5 "First 16 bytes"
```

## Validation Checklist

- [ ] All services running (`docker-compose ps`)
- [ ] WebSocket connection successful
- [ ] Microphone permissions granted
- [ ] Audio data format correct (Float32Array @ 16kHz)
- [ ] Audio reaching WhisperLive
- [ ] Transcription received
- [ ] LLM response generated
- [ ] TTS audio played back
- [ ] Latency within acceptable range (<500ms)
- [ ] Error handling working (connection loss, etc.)

## Quick Validation Command
```bash
# One-command validation
./scripts/validate-audio-flow.sh
```

This will test the complete flow and report any issues.