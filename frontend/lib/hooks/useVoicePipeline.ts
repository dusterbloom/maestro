'use client'

import { useEffect, useRef } from 'react'
import { useSetAtom, useAtom } from 'jotai'
import { VoiceWebSocket } from '@/lib/websocket/VoiceWebSocket'
import { AudioProcessor } from '@/lib/audio/AudioProcessor'
import { AudioPlayer } from '@/lib/audio/AudioPlayer'
import {
  isRecordingAtom,
  liveTranscriptAtom,
  isProcessingAtom,
  processingTextAtom,
  isPlayingAtom,
  isPausedAtom,
  errorAtom,
  transcriptSegmentsAtom,
  audioQueueLengthAtom
} from '@/atoms/voice.atoms'
import {
  isConnectedAtom,
  sessionIdAtom,
  connectionStatusAtom,
  voiceWebSocketAtom,
  audioProcessorAtom,
  audioPlayerAtom,
  sessionMetricsAtom
} from '@/atoms/session.atoms'

export function useVoicePipeline() {
  const setIsConnected = useSetAtom(isConnectedAtom)
  const setSessionId = useSetAtom(sessionIdAtom)
  const setConnectionStatus = useSetAtom(connectionStatusAtom)
  const setError = useSetAtom(errorAtom)
  
  const setLiveTranscript = useSetAtom(liveTranscriptAtom)
  const setIsProcessing = useSetAtom(isProcessingAtom)
  const setProcessingText = useSetAtom(processingTextAtom)
  const setIsPlaying = useSetAtom(isPlayingAtom)
  const setIsPaused = useSetAtom(isPausedAtom) 
  const setTranscriptSegments = useSetAtom(transcriptSegmentsAtom)
  const setAudioQueueLength = useSetAtom(audioQueueLengthAtom)
  
  const [voiceWebSocket, setVoiceWebSocket] = useAtom(voiceWebSocketAtom)
  const [audioProcessor, setAudioProcessor] = useAtom(audioProcessorAtom)
  const [audioPlayer, setAudioPlayer] = useAtom(audioPlayerAtom)
  const [sessionMetrics, setSessionMetrics] = useAtom(sessionMetricsAtom)
  
  const initializingRef = useRef(false)
  
  useEffect(() => {
    if (initializingRef.current) return
    initializingRef.current = true
    
    const initialize = async () => {
      try {
        setConnectionStatus('connecting')
        
        // Create instances
        const ws = new VoiceWebSocket()
        const processor = new AudioProcessor()
        const player = new AudioPlayer()
        
        // Set up WebSocket event handlers
        ws.addEventListener('connected', () => {
          setIsConnected(true)
          setConnectionStatus('connected')
          setError(null)
          setSessionMetrics({
            startTime: Date.now(),
            messageCount: 0,
            audioBytesSent: 0,
            lastActivity: Date.now()
          })
        })
        
        ws.addEventListener('disconnected', () => {
          setIsConnected(false)
          setConnectionStatus('disconnected')
        })
        
        ws.addEventListener('error', (event: any) => {
          setConnectionStatus('error')
          setError(event.detail?.message || 'Connection error')
        })
        
        // Set up simplified message handlers - trust orchestrator as source of truth
        ws.bus.on('ready', (message) => {
          setSessionId(message.session_id)
        })

        // Inside useVoicePipeline useEffect
        ws.bus.on('pause_tts', () => {
          console.log('Orchestrator requested TTS pause');
          setIsPaused(true);
        });

        ws.bus.on('resume_tts', () => {
          console.log('Orchestrator requested TTS resume');
          setIsPaused(false);
        });
        
        ws.bus.on('live_transcript', (message) => {
          setLiveTranscript(message.text)
          setSessionMetrics(prev => ({
            ...prev,
            messageCount: prev.messageCount + 1,
            lastActivity: Date.now()
          }))
        })
        
        ws.bus.on('segments', (message) => {
          setTranscriptSegments(message.segments)
        })
        
        ws.bus.on('processing_started', (message) => {
          setIsProcessing(true)
          setProcessingText(message.text)
        })
        
        ws.bus.on('processing_complete', () => {
          setIsProcessing(false)
          setProcessingText('')
        })
        
        ws.bus.on('sentence_audio', async (message) => {
          setIsPlaying(true)
          setIsPaused(false)
          await player.playAudio(
            message.audio_data,
            message.sequence,
            message.text
          )
          setAudioQueueLength(player.queueLength)
          
          // Let orchestrator control playing state - simplified logic
          if (player.queueLength === 0 && !player.playing) {
            setIsPlaying(false)
          }
        })
        
        ws.bus.on('interrupted', () => {
          // Trust orchestrator's interrupt signal completely
          setIsProcessing(false)
          setProcessingText('')
          setIsPlaying(false)
          setIsPaused(false)
          player.interrupt()
        })
        
        ws.bus.on('error', (message) => {
          setError(message.message)
        })
        
        // Initialize audio processor
        await processor.initialize()
        
        // Connect WebSocket
        await ws.connect()
        
        // Store instances
        setVoiceWebSocket(ws)
        setAudioProcessor(processor)
        setAudioPlayer(player)
        
      } catch (error) {
        console.error('Failed to initialize voice pipeline:', error)
        setConnectionStatus('error')
        setError('Failed to initialize voice service')
      }
    }
    
    initialize()
    
    // Cleanup
    return () => {
      const cleanup = async () => {
        if (voiceWebSocket) {
          voiceWebSocket.disconnect()
        }
        if (audioProcessor) {
          await audioProcessor.cleanup()
        }
        if (audioPlayer) {
          audioPlayer.interrupt()
        }
        
        setVoiceWebSocket(null)
        setAudioProcessor(null)
        setAudioPlayer(null)
        setIsConnected(false)
        setConnectionStatus('disconnected')
      }
      
      cleanup()
    }
  }, []) // Empty dependency array - only run once
  
  return {
    isConnected: voiceWebSocket?.isConnected ?? false,
    sessionId: voiceWebSocket?.currentSessionId ?? null
  }
}