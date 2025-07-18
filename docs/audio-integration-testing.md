# Audio Input Integration Testing Guide

## Pre-Testing Setup

### 1. Environment Configuration
```bash
# .env file
NEXT_PUBLIC_ORCHESTRATOR_WS_URL=ws://localhost:8000
WHISPER_URL=ws://whisper-live:9090
OLLAMA_URL=http://host.docker.internal:11434
TTS_URL=http://kokoro:8880
```

### 2. Service Health Checks
```bash
# Check orchestrator health
curl http://localhost:8000/health

# Check WhisperLive connectivity
docker-compose logs whisper-live | grep "listening"

# Check orchestrator logs
docker-compose logs orchestrator | grep "WhisperLive"
```

## Testing Audio Flow

### Phase 1: Basic Connectivity
1. **WebSocket Connection Test**
   ```javascript
   // Browser console test
   const ws = new WebSocket('ws://localhost:8000/ws/voice');
   ws.onopen = () => console.log('Connected to ultra-fast endpoint');
   ws.onmessage = (e) => console.log('Received:', e.data);
   ```

2. **Audio Permission Test**
   ```javascript
   // Check microphone access
   navigator.mediaDevices.getUserMedia({ audio: true })
     .then(stream => console.log('Microphone access granted'))
     .catch(err => console.error('Microphone access denied:', err));
   ```

### Phase 2: Audio Streaming Test
1. **Manual Audio Test**
   - Open UI at http://localhost:3001
   - Click voice button and speak
   - Check browser console for audio data logs
   - Check orchestrator logs for received audio

2. **Expected Log Sequence**
   ```
   Client: "🎤 Sending 4096 float32 samples (16384 bytes) to orchestrator"
   Orchestrator: "🎤 [Ultra-fast] Received 16384 bytes of audio data from frontend"
   Orchestrator: "📤 Sending 16384 bytes of audio to WhisperLive"
   WhisperLive: "Transcription: [your speech text]"
   ```

### Phase 3: End-to-End Flow
1. **Complete Pipeline Test**
   - Speak into microphone
   - Verify transcription appears in UI
   - Verify LLM response generation
   - Verify TTS audio playback

### Phase 4: Error Handling
1. **Connection Loss Test**
   - Stop WhisperLive container mid-conversation
   - Verify reconnection attempts
   - Verify graceful degradation

2. **Audio Format Test**
   - Test with different sample rates
   - Verify resampling works correctly

## Debugging Commands

### Real-time Monitoring
```bash
# Monitor all services
docker-compose logs -f

# Monitor specific components
docker-compose logs -f orchestrator | grep -E "(WhisperLive|audio|transcript)"
docker-compose logs -f whisper-live | grep -E "(transcription|client)"
```

### WebSocket Debugging
```bash
# Test WebSocket connection
websocat ws://localhost:8000/ws/voice

# Send test audio (base64 encoded)
echo '{"type":"test_audio","data":"base64_encoded_audio"}' | websocat ws://localhost:8000/ws/voice
```

### Performance Metrics
```bash
# Check latency
curl http://localhost:8000/debug/sessions

# Monitor GPU usage
nvidia-smi -l 1
```

## Troubleshooting Guide

### Common Issues

1. **"WebSocket connection failed"**
   - Check orchestrator is running: `docker-compose ps orchestrator`
   - Verify port 8000 is accessible: `curl http://localhost:8000/health`

2. **"No transcription received"**
   - Check WhisperLive logs: `docker-compose logs whisper-live`
   - Verify audio format: should be Float32Array @ 16kHz
   - Check audio data size in logs

3. **"Audio data format error"**
   - Verify client sends Float32Array
   - Check buffer alignment (multiple of 4 bytes)
   - Ensure 16kHz resampling is working

4. **"WhisperLive connection timeout"**
   - Check WhisperLive container: `docker-compose logs whisper-live`
   - Verify GPU availability: `nvidia-smi`
   - Check model loading status

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
docker-compose restart orchestrator

# Test with minimal setup
docker-compose -f docker-compose.minimal.yml up