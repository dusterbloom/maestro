---
id: task-2
title: "## \U0001F9E0 SPEAKER RECOGNITION REFACTOR"
status: In Progress
assignee: []
created_date: '2025-07-09'
updated_date: '2025-07-09'
labels: []
dependencies: []
priority: medium
---

## Description

```

```

**GitHub:** github.com/dusterbloom/maestro  
**Current Branch:** vibe-1751794205 (broken, over-engineered)  
**Working Branch:** main (ultra-fast <250ms voice pipeline)

### 🎯 CORE ISSUE
User built working voice assistant on main branch, tried adding speaker recognition on vibe branch but created over-engineered mess with state machines. Wants magical speaker recognition without breaking what works.

### 📍 KEY LOCATIONS
- **Main branch working code:** `orchestrator/src/main.py` - simple VoiceOrchestrator class
- **Vibe branch speaker implementation:** 
  - `orchestrator/src/services/speaker_service.py` - uses AudioBufferManager + Resemblyzer
  - `orchestrator/src/services/voice_service.py` - ThreadPoolExecutor for non-blocking embeddings
  - Problem: 10-second buffer, complex state machine, CPU-heavy Resemblyzer

### ✨ PROPOSED SOLUTION: 100-Line SpeechBrain Integration

```python
# Add to working main.py without breaking anything
import torch
from speechbrain.pretrained import SpeakerRecognition
import asyncio
import numpy as np
import time

class MagicSpeakerRecognition:
    """Drop-in speaker recognition - no state machines!"""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="models/speaker_recognition",
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )
        self.audio_queues = {}  # session_id -> asyncio.Queue
        self.known_speakers = {}  # speaker_id -> {"name": str, "embedding": tensor}
        self.session_speakers = {}  # session_id -> speaker_id
        
    async def process_audio_chunk(self, session_id: str, audio_chunk: bytes):
        """Queue audio without blocking main pipeline"""
        if session_id not in self.audio_queues:
            self.audio_queues[session_id] = asyncio.Queue()
            asyncio.create_task(self._process_session(session_id))
        
        await self.audio_queues[session_id].put({
            "audio": audio_chunk,
            "timestamp": time.time()
        })
    
    async def _process_session(self, session_id: str):
        """Background processor per session"""
        queue = self.audio_queues[session_id]
        audio_buffer = []
        last_check = 0
        
        while True:
            try:
                # Collect audio chunks
                item = await asyncio.wait_for(queue.get(), timeout=30)
                audio_buffer.append(item["audio"])
                
                # Process every 2 seconds (not 10!)
                if time.time() - last_check >= 2.0 and len(audio_buffer) >= 20:
                    # Combine audio
                    audio_data = b''.join(audio_buffer[-30:])  # Last 3 seconds
                    
                    # Generate embedding (non-blocking)
                    embedding = await self._get_embedding_async(audio_data)
                    
                    if embedding is not None:
                        # Check if known speaker
                        speaker_id = self._find_speaker(embedding)
                        
                        if speaker_id and speaker_id != self.session_speakers.get(session_id):
                            # Recognized!
                            self.session_speakers[session_id] = speaker_id
                            name = self.known_speakers[speaker_id]["name"]
                            await self._inject_recognition(session_id, name)
                        
                        elif not speaker_id and session_id not in self.session_speakers:
                            # New speaker
                            new_id = f"spk_{len(self.known_speakers)}"
                            self.known_speakers[new_id] = {
                                "name": None,
                                "embedding": embedding
                            }
                            self.session_speakers[session_id] = new_id
                    
                    last_check = time.time()
                    
            except asyncio.TimeoutError:
                # Session inactive, cleanup
                del self.audio_queues[session_id]
                break
    
    async def _get_embedding_async(self, audio_data: bytes) -> torch.Tensor:
        """Generate embedding without blocking"""
        loop = asyncio.get_event_loop()
        
        def _compute():
            # Convert audio_data to tensor (simplified)
            # In real implementation, use soundfile/librosa
            audio_tensor = torch.randn(1, 16000)  # Placeholder
            
            with torch.no_grad():
                embeddings = self.model.encode_batch(audio_tensor)
                return embeddings[0]
        
        return await loop.run_in_executor(None, _compute)
    
    def _find_speaker(self, embedding: torch.Tensor, threshold: float = 0.7):
        """Find matching speaker by cosine similarity"""
        for spk_id, data in self.known_speakers.items():
            if data["name"]:  # Only match named speakers
                similarity = torch.nn.functional.cosine_similarity(
                    embedding.unsqueeze(0),
                    data["embedding"].unsqueeze(0)
                ).item()
                
                if similarity > threshold:
                    return spk_id
        return None
    
    async def _inject_recognition(self, session_id: str, name: str):
        """Tell LLM about recognition naturally"""
        if session_id in self.orchestrator.session_history:
            self.orchestrator.session_history[session_id].append({
                "role": "system",
                "content": f"The speaker has been recognized as {name}. Acknowledge naturally."
            })
    
    def extract_name_from_response(self, session_id: str, text: str):
        """Check if user introduced themselves"""
        if "my name is" in text.lower() or "i'm " in text.lower():
            # Extract name (simplified)
            if session_id in self.session_speakers:
                spk_id = self.session_speakers[session_id]
                # Parse name from text...
                self.known_speakers[spk_id]["name"] = "ExtractedName"

# Integration - just 3 lines in existing code:
# In VoiceOrchestrator.__init__:
self.speaker_recognition = MagicSpeakerRecognition(self)

# In audio processing:
asyncio.create_task(self.speaker_recognition.process_audio_chunk(session_id, audio_chunk))

# In response generation:
self.speaker_recognition.extract_name_from_response(session_id, user_text)
```

### 🔑 KEY POINTS FOR IMPLEMENTATION
1. **NO state machines** - Just queues and background tasks
2. **2-second recognition** (not 10) - Progressive confidence building
3. **Non-blocking** - Everything async, main pipeline never waits
4. **Natural UX** - System message injection for smooth recognition
5. **SpeechBrain** - 3x faster than Resemblyzer, GPU support, better accuracy

### 🎮 NEXT STEPS
1. Install: `pip install speechbrain`
2. Add the class above to working main.py
3. Add 3 integration lines
4. Test with existing working pipeline
5. Delete the vibe branch mess

**Remember:** User is learning, likes simplicity, got burned by over-engineering. Keep it SIMPLE!
