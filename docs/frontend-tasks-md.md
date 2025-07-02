# Frontend Implementation Tasks

## FRONTEND-001: Next.js Voice UI Implementation

### Objective
Create a minimal Next.js 14 PWA with WebSocket-based push-to-talk voice interface.

### Directory Structure
```
ui/
├── app/
│   ├── page.tsx
│   ├── layout.tsx
│   └── globals.css
├── components/
│   ├── VoiceButton.tsx
│   ├── Waveform.tsx
│   └── StatusIndicator.tsx
├── lib/
│   ├── websocket.ts
│   └── audio.ts
├── public/
│   └── manifest.json
├── package.json
├── tsconfig.json
├── tailwind.config.ts
└── Dockerfile
```

### Files to Create

#### ui/package.json
```json
{
  "name": "voice-ui",
  "version": "1.0.0",
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start"
  },
  "dependencies": {
    "next": "14.1.0",
    "react": "^18",
    "react-dom": "^18"
  },
  "devDependencies": {
    "@types/node": "^20",
    "@types/react": "^18",
    "@types/react-dom": "^18",
    "autoprefixer": "^10.0.1",
    "postcss": "^8",
    "tailwindcss": "^3.3.0",
    "typescript": "^5"
  }
}
```

#### ui/lib/websocket.ts
```typescript
export class VoiceWebSocket {
  private ws: WebSocket | null = null;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  
  private onConnectCallback?: () => void;
  private onAudioCallback?: (audio: ArrayBuffer) => void;
  private onErrorCallback?: (error: string) => void;
  
  constructor(private url: string) {}
  
  connect(sessionId?: string) {
    try {
      this.ws = new WebSocket(this.url);
      this.ws.binaryType = 'arraybuffer';
      
      if (sessionId) {
        // Note: Headers can't be set on WebSocket in browser
        // Session ID should be sent as first message or query param
      }
      
      this.ws.onopen = () => {
        this.reconnectAttempts = 0;
        this.onConnectCallback?.();
      };
      
      this.ws.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) {
          this.onAudioCallback?.(event.data);
        }
      };
      
      this.ws.onerror = (error) => {
        this.onErrorCallback?.('WebSocket error');
      };
      
      this.ws.onclose = () => {
        this.scheduleReconnect();
      };
    } catch (error) {
      this.onErrorCallback?.('Failed to connect');
      this.scheduleReconnect();
    }
  }
  
  private scheduleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      this.onErrorCallback?.('Max reconnection attempts reached');
      return;
    }
    
    this.reconnectTimer = setTimeout(() => {
      this.reconnectAttempts++;
      this.connect();
    }, this.reconnectDelay * this.reconnectAttempts);
  }
  
  sendAudio(audioData: ArrayBuffer) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(audioData);
    }
  }
  
  disconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }
    this.ws?.close();
  }
  
  onConnect(callback: () => void) {
    this.onConnectCallback = callback;
  }
  
  onAudio(callback: (audio: ArrayBuffer) => void) {
    this.onAudioCallback = callback;
  }
  
  onError(callback: (error: string) => void) {
    this.onErrorCallback = callback;
  }
}
```

#### ui/lib/audio.ts
```typescript
export class AudioRecorder {
  private mediaRecorder: MediaRecorder | null = null;
  private audioContext: AudioContext | null = null;
  private chunks: Blob[] = [];
  
  async initialize() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.audioContext = new AudioContext({ sampleRate: 16000 });
    
    this.mediaRecorder = new MediaRecorder(stream, {
      mimeType: 'audio/webm;codecs=opus'
    });
    
    this.mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        this.chunks.push(event.data);
      }
    };
  }
  
  start() {
    this.chunks = [];
    this.mediaRecorder?.start(100); // 100ms chunks
  }
  
  async stop(): Promise<ArrayBuffer> {
    return new Promise((resolve) => {
      this.mediaRecorder!.onstop = async () => {
        const blob = new Blob(this.chunks, { type: 'audio/webm' });
        const arrayBuffer = await blob.arrayBuffer();
        resolve(arrayBuffer);
      };
      this.mediaRecorder?.stop();
    });
  }
  
  getAudioLevel(): number {
    // Simplified - implement actual level detection if needed
    return Math.random();
  }
}

export class AudioPlayer {
  private audioContext: AudioContext;
  
  constructor() {
    this.audioContext = new AudioContext();
  }
  
  async play(audioData: ArrayBuffer) {
    const audioBuffer = await this.audioContext.decodeAudioData(audioData);
    const source = this.audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this.audioContext.destination);
    source.start();
  }
}
```

#### ui/components/VoiceButton.tsx
```typescript
'use client';

import { useState, useRef, useEffect } from 'react';
import { VoiceWebSocket } from '@/lib/websocket';
import { AudioRecorder, AudioPlayer } from '@/lib/audio';

export default function VoiceButton() {
  const [isRecording, setIsRecording] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<VoiceWebSocket | null>(null);
  const recorderRef = useRef<AudioRecorder | null>(null);
  const playerRef = useRef<AudioPlayer | null>(null);
  
  useEffect(() => {
    // Initialize WebSocket
    const ws = new VoiceWebSocket(process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws');
    wsRef.current = ws;
    
    ws.onConnect(() => setIsConnected(true));
    ws.onError(() => setIsConnected(false));
    ws.onAudio(async (audio) => {
      if (playerRef.current) {
        await playerRef.current.play(audio);
      }
    });
    
    ws.connect();
    
    // Initialize audio
    recorderRef.current = new AudioRecorder();
    playerRef.current = new AudioPlayer();
    recorderRef.current.initialize();
    
    return () => {
      ws.disconnect();
    };
  }, []);
  
  const handleMouseDown = () => {
    if (!isConnected) return;
    setIsRecording(true);
    recorderRef.current?.start();
  };
  
  const handleMouseUp = async () => {
    if (!isRecording) return;
    setIsRecording(false);
    
    const audioData = await recorderRef.current?.stop();
    if (audioData) {
      wsRef.current?.sendAudio(audioData);
    }
  };
  
  return (
    <button
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onTouchStart={handleMouseDown}
      onTouchEnd={handleMouseUp}
      disabled={!isConnected}
      className={`
        w-32 h-32 rounded-full transition-all duration-200
        ${isRecording 
          ? 'bg-red-500 scale-110 shadow-lg' 
          : 'bg-blue-500 hover:bg-blue-600 shadow-md'
        }
        ${!isConnected && 'opacity-50 cursor-not-allowed'}
        text-white font-bold text-lg
      `}
    >
      {isRecording ? 'Listening...' : 'Push to Talk'}
    </button>
  );
}
```

#### ui/app/page.tsx
```typescript
import VoiceButton from '@/components/VoiceButton';
import Waveform from '@/components/Waveform';
import StatusIndicator from '@/components/StatusIndicator';

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24 bg-gray-50">
      <div className="flex flex-col items-center gap-8">
        <h1 className="text-4xl font-bold text-gray-800">Voice Assistant</h1>
        
        <StatusIndicator />
        
        <div className="relative">
          <VoiceButton />
          <Waveform />
        </div>
        
        <p className="text-gray-600 text-center max-w-md">
          Hold the button and speak. Release to send your message.
        </p>
      </div>
    </main>
  );
}
```

#### ui/Dockerfile
```dockerfile
FROM node:18-alpine AS base

FROM base AS deps
RUN apk add --no-cache libc6-compat
WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci

FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

RUN npm run build

FROM base AS runner
WORKDIR /app

ENV NODE_ENV production

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

ENV PORT 3000

CMD ["node", "server.js"]
```

### Acceptance Criteria
1. Push-to-talk button works on desktop and mobile
2. WebSocket connects and reconnects automatically
3. Audio recording and playback functional
4. Status indicator shows connection state
5. PWA installable on mobile

---

## FRONTEND-002: Settings Panel Implementation

### Objective
Add a configuration panel for model selection and preferences.

### File to Create

#### ui/components/SettingsPanel.tsx
```typescript
'use client';

import { useState, useEffect } from 'react';

interface Settings {
  sttModel: string;
  ttsVoice: string;
  memoryEnabled: boolean;
}

export default function SettingsPanel() {
  const [isOpen, setIsOpen] = useState(false);
  const [settings, setSettings] = useState<Settings>({
    sttModel: 'tiny',
    ttsVoice: 'af_bella',
    memoryEnabled: false
  });
  
  useEffect(() => {
    // Load settings from localStorage
    const saved = localStorage.getItem('voiceSettings');
    if (saved) {
      setSettings(JSON.parse(saved));
    }
  }, []);
  
  const saveSettings = () => {
    localStorage.setItem('voiceSettings', JSON.stringify(settings));
    // TODO: Send settings to backend
    setIsOpen(false);
  };
  
  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className="fixed top-4 right-4 p-2 bg-gray-200 rounded-lg hover:bg-gray-300"
      >
        ⚙️ Settings
      </button>
      
      {isOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-lg p-6 max-w-md w-full">
            <div className="flex gap-2 mt-6">
              <button
                onClick={saveSettings}
                className="flex-1 bg-blue-500 text-white p-2 rounded hover:bg-blue-600"
              >
                Save
              </button>
              <button
                onClick={() => setIsOpen(false)}
                className="flex-1 bg-gray-300 p-2 rounded hover:bg-gray-400"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}<h2 className="text-2xl font-bold mb-4">Settings</h2>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">
                  STT Model
                </label>
                <select
                  value={settings.sttModel}
                  onChange={(e) => setSettings({...settings, sttModel: e.target.value})}
                  className="w-full p-2 border rounded"
                >
                  <option value="tiny">Tiny (Fastest)</option>
                  <option value="base">Base</option>
                  <option value="small">Small</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-1">
                  TTS Voice
                </label>
                <select
                  value={settings.ttsVoice}
                  onChange={(e) => setSettings({...settings, ttsVoice: e.target.value})}
                  className="w-full p-2 border rounded"
                >
                  <option value="af_bella">Bella (Female)</option>
                  <option value="am_adam">Adam (Male)</option>
                  <option value="af_sarah">Sarah (Female)</option>
                </select>
              </div>
              
              <div>
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={settings.memoryEnabled}
                    onChange={(e) => setSettings({...settings, memoryEnabled: e.target.checked})}
                    className="mr-2"
                  />
                  Enable conversation memory
                </label>
              </div>
            </div>
            
            