# Voice Interruption Debug Test

## Test Plan
This test will systematically check every component of the interruption system to find where it's failing.

## Frontend Debug Test

### Step 1: Test Audio Player State Tracking
Add this to browser console and run:

```javascript
// Test 1: Check if audio player state is being tracked correctly
setInterval(() => {
  const audioElements = document.querySelectorAll('audio');
  const webAudioSources = window.audioContext ? 'WebAudio context exists' : 'No WebAudio context';
  console.log(`🔍 AUDIO STATE CHECK: DOM audio elements=${audioElements.length}, ${webAudioSources}`);
}, 2000);
```

### Step 2: Test isPlaying State
Add this debug code to VoiceButton component (add after line 16):

```typescript
// Debug: Log state changes
useEffect(() => {
  console.log(`🔍 STATE CHANGE: isPlaying=${isPlaying}, isRecording=${isRecording}, status=${status}`);
}, [isPlaying, isRecording, status]);
```

### Step 3: Test Voice Activity Detection
Add this to test if voice activity is being detected:

```typescript
// Add after recorder initialization (around line 325)
recorder.onAudioLevel((level) => {
  const isVoiceActive = recorder.isVoiceActive();
  console.log(`🎤 VOICE LEVEL: ${level.toFixed(4)}, active=${isVoiceActive}, threshold=${recorder.voiceActivityThreshold}`);
  
  // If we detect voice activity while TTS is playing, trigger barge-in
  if (isVoiceActive && isPlaying && !isRecording) {
    console.log('🛑 VOICE ACTIVITY BARGE-IN: Voice level:', level, '- Audio pipeline: INTERRUPT');
    handleBargeIn();
  }
});
```

## Backend Debug Test

### Step 4: Test Session Tracking
Add this endpoint to test session management:

```python
@app.get("/debug/sessions")
async def debug_sessions():
    """Debug endpoint to check active sessions"""
    return {
        "active_sessions": list(orchestrator.active_tts_sessions.keys()),
        "session_details": {
            session_id: {
                "start_time": data.get("start_time", "unknown"),
                "abort_flag_set": data.get("abort_flag", {}).is_set() if data.get("abort_flag") else False
            }
            for session_id, data in orchestrator.active_tts_sessions.items()
        }
    }
```

### Step 5: Test Interruption Endpoint Directly
```bash
# Test the interrupt endpoint directly
curl -X POST http://localhost:8000/interrupt-tts \
  -H "Content-Type: application/json" \
  -d '{"session_id": "session_1751668221265"}'
```

## Comprehensive Test Script

Create this test file to run all checks: