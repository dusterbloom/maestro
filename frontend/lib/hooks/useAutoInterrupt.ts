'use client'

import { useEffect, useRef } from 'react'
import { useAtomValue } from 'jotai'
import { 
  isProcessingAtom, 
  isPlayingAtom, 
  isRecordingAtom 
} from '@/atoms/voice.atoms'
import { 
  voiceWebSocketAtom 
} from '@/atoms/session.atoms'

export function useAutoInterrupt() {
  const isProcessing = useAtomValue(isProcessingAtom)
  const isPlaying = useAtomValue(isPlayingAtom)
  const isRecording = useAtomValue(isRecordingAtom)
  const voiceWebSocket = useAtomValue(voiceWebSocketAtom)
  
  // Note: With continuous streaming, the AudioProcessor now handles all audio monitoring
  // This hook is kept for potential future enhancements but auto-interrupt is now handled
  // by the orchestrator's InterruptPlugin based on continuous audio stream
  
  const shouldMonitorForInterruption = !isRecording && (isProcessing || isPlaying)
  
  useEffect(() => {
    // With continuous streaming architecture, interrupt detection is now handled
    // server-side by the InterruptPlugin based on the continuous audio stream
    // This provides more accurate and faster interrupt detection
    
    if (shouldMonitorForInterruption && voiceWebSocket) {
      console.log('🎯 Auto-interrupt is now handled server-side via continuous audio stream')
      
      // The VoiceButton component now keeps the audio processor running in monitoring mode
      // so that the server can detect voice activity during TTS and trigger interrupts
    }
    
    // No cleanup needed since audio monitoring is handled by AudioProcessor
    return () => {
      // No-op: cleanup is handled by AudioProcessor lifecycle
    }
  }, [shouldMonitorForInterruption, voiceWebSocket])
}