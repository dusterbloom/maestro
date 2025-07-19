'use client'

import { useCallback, useEffect, useRef } from 'react'
import { useAtom, useAtomValue, useSetAtom } from 'jotai'
import { Mic, MicOff, Loader2, Square } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils/cn'
import {
  isRecordingAtom,
  isProcessingAtom,
  isPlayingAtom,
  errorAtom,
  audioLevelAtom
} from '@/atoms/voice.atoms'
import {
  isConnectedAtom,
  voiceWebSocketAtom,
  audioProcessorAtom
} from '@/atoms/session.atoms'

export function VoiceButton() {
  // Track monitoring mode (for barge-in/interrupt detection)
  const isMonitoringRef = useRef(false);
  const [isRecording, setIsRecording] = useAtom(isRecordingAtom)
  const [isConnected] = useAtom(isConnectedAtom)
  const isProcessing = useAtomValue(isProcessingAtom)
  const isPlaying = useAtomValue(isPlayingAtom)
  const setError = useSetAtom(errorAtom)
  const setAudioLevel = useSetAtom(audioLevelAtom)
  const audioLevel = useAtomValue(audioLevelAtom)
  
  const voiceWebSocket = useAtomValue(voiceWebSocketAtom)
  const audioProcessor = useAtomValue(audioProcessorAtom)
  
  const buttonRef = useRef<HTMLButtonElement>(null)
  
  // Toggle recording on/off
  const toggleRecording = useCallback(async () => {
    // Always stop monitoring mode before starting a new recording
    if (isMonitoringRef.current) {
      audioProcessor?.stopProcessing();
      isMonitoringRef.current = false;
      console.log('🛑 Stopped monitoring mode before new recording');
    }
    // Always reset audio processor to recording mode after an interrupt
    // This ensures that after an interrupt, the next utterance is treated as a new segment
    if (!isConnected || !voiceWebSocket || !audioProcessor) {
      setError('Not connected to voice service')
      return
    }

    // If we are in monitoring mode (after interrupt), force stop and reset before starting new recording
    if (!isRecording && !isProcessing && !isPlaying) {
      // Stop any monitoring mode
      audioProcessor.stopProcessing();
      console.log('🔄 Resetting audio processor to recording mode after interrupt');
    }
    if (!isConnected || !voiceWebSocket || !audioProcessor) {
      setError('Not connected to voice service')
      return
    }
    
    if (isProcessing || isPlaying) {
      // Send interrupt signal
      voiceWebSocket.interrupt()
      return
    }
    
    if (isRecording) {
      console.log('🛑 User stopped recording, sending end_audio (should always happen after utterance, even post-interrupt)')
      isMonitoringRef.current = true; // Next, we enter monitoring mode
      // Stop recording but keep processing for interrupt detection
      console.log('🛑 Stopping recording...')
      setIsRecording(false)
      setAudioLevel(0)
      
      // Signal end of audio to WebSocket but keep audio processor running for interrupt detection
      if (voiceWebSocket) {
        voiceWebSocket.endAudio()
        console.log('📡 End audio signal sent')
      }
      
      // Switch audio processor to monitoring mode (continuous streaming for interrupt detection)
      if (audioProcessor) {
        audioProcessor.startProcessing((audioData) => {
          if (!isMonitoringRef.current) return; // Only send audio if in monitoring mode
          // Only send audio for monitoring, not for STT
          voiceWebSocket.sendAudio(audioData)
          
          // Update audio level for visual feedback
          const level = audioProcessor.getAudioLevel(audioData)
          setAudioLevel(level * 0.5) // Reduced level for monitoring mode
        })
        console.log('🎤 Audio processing switched to monitoring mode')
      }
    } else {
      // Start recording
      try {
        isMonitoringRef.current = false;
        console.log('🎤 Starting recording (should always be in recording mode, not monitoring, after interrupt or new utterance)...')
        setIsRecording(true)
        setError(null)
        
        // Start audio processing
        audioProcessor.startProcessing((audioData) => {
          // Send audio to WebSocket
          voiceWebSocket.sendAudio(audioData)
          
          // Update audio level
          const level = audioProcessor.getAudioLevel(audioData)
          setAudioLevel(level)
        })
        
        console.log('✅ Recording started successfully')
        
      } catch (error) {
        console.error('Failed to start recording:', error)
        setError('Failed to start recording')
        setIsRecording(false)
      }
    }
  }, [isConnected, isRecording, isProcessing, isPlaying, voiceWebSocket, audioProcessor, setIsRecording, setError, setAudioLevel])
  
  // Keyboard support
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space' && !e.repeat) {
        e.preventDefault()
        toggleRecording()
      }
    }
    
    window.addEventListener('keydown', handleKeyDown)
    
    return () => {
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [toggleRecording])
  
  const isActive = isRecording || isProcessing || isPlaying
  const showLoader = isProcessing || isPlaying
  
  return (
    <Button
      ref={buttonRef}
      size="icon"
      variant={isActive ? 'default' : 'outline'}
      className={cn(
        'relative h-16 w-16 rounded-full transition-all',
        isRecording && 'scale-110 ring-4 ring-primary/50 animate-pulse-scale',
        (isProcessing || isPlaying) && 'ring-2 ring-primary/30',
        !isConnected && 'opacity-50 cursor-not-allowed'
      )}
      onClick={toggleRecording}
      disabled={!isConnected}
      aria-label={isRecording ? 'Stop recording' : 'Start recording'}
      title={
        !isConnected ? 'Connecting to voice service...' :
        isProcessing || isPlaying ? 'Click to interrupt' :
        isRecording ? 'Click to stop recording' :
        'Click to start recording'
      }
    >
      {showLoader ? (
        <Loader2 className="h-6 w-6 animate-spin" />
      ) : isRecording ? (
        <Square className="h-5 w-5" />
      ) : (
        <Mic className="h-6 w-6" />
      )}
      
      {/* Audio level indicator */}
      {isRecording && (
        <div 
          className="absolute inset-0 rounded-full bg-primary/20 scale-[var(--scale)]"
          style={{
            '--scale': `${1 + audioLevel * 0.5}`
          } as React.CSSProperties}
        />
      )}
    </Button>
  )
}