import { atom } from 'jotai'
import { VoiceWebSocket } from '@/lib/websocket/VoiceWebSocket'
import { AudioProcessor } from '@/lib/audio/AudioProcessor'
import { AudioPlayer } from '@/lib/audio/AudioPlayer'

// Connection state
export const isConnectedAtom = atom(false)
export const sessionIdAtom = atom<string | null>(null)
export const connectionStatusAtom = atom<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected')

// Core instances
export const voiceWebSocketAtom = atom<VoiceWebSocket | null>(null)
export const audioProcessorAtom = atom<AudioProcessor | null>(null)
export const audioPlayerAtom = atom<AudioPlayer | null>(null)

// Session metrics
export const sessionMetricsAtom = atom({
  startTime: null as number | null,
  messageCount: 0,
  audioBytesSent: 0,
  lastActivity: null as number | null
})

// Derived atoms
export const sessionDurationAtom = atom(get => {
  const metrics = get(sessionMetricsAtom)
  if (!metrics.startTime) return 0
  return Date.now() - metrics.startTime
})