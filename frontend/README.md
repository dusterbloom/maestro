# Maestro Frontend

A modern, real-time voice assistant frontend built with Next.js, Jotai, and TypeScript.

## Features

- **Ultra-low latency** voice interactions via WebSocket
- **Push-to-talk** interface with keyboard support
- **Real-time transcription** display with live updates
- **Audio visualization** with waveform display
- **Event-driven architecture** using fire-and-forget pattern
- **Atomic state management** with Jotai
- **Fully typed** with TypeScript

## Tech Stack

- **Framework**: Next.js 14 with App Router
- **State Management**: Jotai (atomic state)
- **Styling**: Tailwind CSS + Radix UI
- **WebSocket**: Custom event bus implementation
- **Audio**: Web Audio API with 16kHz resampling

## Getting Started

1. Install dependencies:
```bash
npm install
```

2. Set environment variables (optional):
```bash
# Create .env.local file
NEXT_PUBLIC_ORCHESTRATOR_WS_URL=ws://localhost:8000
```

3. Run the development server:
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000)

## Architecture

### WebSocket Integration
- Connects to orchestrator's `/ws/voice` endpoint
- Implements event bus pattern for fire-and-forget messaging
- Automatic reconnection with exponential backoff

### Audio Processing
- Captures audio at native sample rate
- Resamples to 16kHz for orchestrator compatibility
- Sends Float32Array audio data via WebSocket

### State Management
- **Voice atoms**: Recording, transcription, processing states
- **Session atoms**: Connection, WebSocket, audio instances
- **Settings atoms**: User preferences and feature flags

### Components
- **VoiceButton**: Push-to-talk interface with visual feedback
- **TranscriptDisplay**: Real-time transcript with segments
- **AudioWaveform**: Live audio visualization
- **StatusIndicator**: Connection and session metrics

## Usage

1. **Hold the microphone button** to record
2. **Release** to send audio for processing
3. **Click while processing** to interrupt
4. **Press Space** for hands-free recording

## Development

```bash
# Build for production
npm run build

# Start production server
npm start

# Type checking
npm run type-check
```