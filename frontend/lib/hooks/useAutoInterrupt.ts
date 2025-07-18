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
    if (!shouldMonitorForInterruption || !voiceWebSocket) {
      // Clean up if we should stop monitoring
      if (voiceActivityTimeoutRef.current) {
        clearTimeout(voiceActivityTimeoutRef.current)
      }
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
      if (audioContextRef.current) {
        audioContextRef.current.close()
        audioContextRef.current = null
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
        streamRef.current = null
      }
      analyserRef.current = null
      return
    }
    
    // Don't start if already running
    if (analyserRef.current && audioContextRef.current) {
      return
    }
    
    console.log('🎯 Starting lightweight automatic interruption monitoring')
    
    // Use Web Audio API AnalyserNode for lightweight voice detection
    const startVoiceActivityDetection = async () => {
      try {
        // Get microphone stream
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            sampleRate: 16000,
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true
          }
        })
        
        streamRef.current = stream
        
        // Create audio context and analyser
        const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)()
        const analyser = audioContext.createAnalyser()
        const source = audioContext.createMediaStreamSource(stream)
        
        analyser.fftSize = 256
        analyser.smoothingTimeConstant = 0.8
        source.connect(analyser)
        
        audioContextRef.current = audioContext
        analyserRef.current = analyser
        
        // Start voice activity monitoring
        const dataArray = new Uint8Array(analyser.frequencyBinCount)
        
        const checkVoiceActivity = () => {
          if (!analyserRef.current || !shouldMonitorForInterruption) {
            return
          }
          
          analyser.getByteFrequencyData(dataArray)
          
          // Calculate average volume
          const volume = dataArray.reduce((sum, value) => sum + value, 0) / dataArray.length
          
          // Simple voice activity detection threshold
          const voiceThreshold = 20
          
          if (volume > voiceThreshold) {
            // Clear any existing timeout
            if (voiceActivityTimeoutRef.current) {
              clearTimeout(voiceActivityTimeoutRef.current)
            }
            
            // Debounce voice activity
            voiceActivityTimeoutRef.current = setTimeout(() => {
              const now = Date.now()
              const timeSinceLastInterrupt = now - lastInterruptTimeRef.current
              
              // Only interrupt if enough time has passed since last interrupt
              if (timeSinceLastInterrupt > 2000) { // Increased to 2 seconds to prevent loops
                console.log('🛑 Auto-interrupting due to voice activity (lightweight detection)')
                voiceWebSocket.interrupt()
                lastInterruptTimeRef.current = now
              }
            }, 200) // Increased debounce to 200ms
          }
          
          // Continue monitoring
          animationFrameRef.current = requestAnimationFrame(checkVoiceActivity)
        }
        
        // Start monitoring
        checkVoiceActivity()
        
      } catch (error) {
        console.error('Failed to start voice activity detection:', error)
      }
    }
    
    startVoiceActivityDetection()
    
    return () => {
      console.log('🎯 Stopping lightweight automatic interruption monitoring')
      
      // Clean up timeouts
      if (voiceActivityTimeoutRef.current) {
        clearTimeout(voiceActivityTimeoutRef.current)
      }
      
      // Clean up animation frame
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
      
      // Clean up audio context
      if (audioContextRef.current) {
        audioContextRef.current.close()
        audioContextRef.current = null
      }
      
      // Clean up stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
        streamRef.current = null
      }
      
      analyserRef.current = null
    }
  }, [shouldMonitorForInterruption, voiceWebSocket])
  
  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (voiceActivityTimeoutRef.current) {
        clearTimeout(voiceActivityTimeoutRef.current)
      }
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
      if (audioContextRef.current) {
        audioContextRef.current.close()
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
      }
    }
  }, [])
}