'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { VoiceWebSocket } from '@/lib/websocket';
import { AudioRecorder, AudioPlayer } from '@/lib/audio';

interface VoiceButtonProps {
  onStatusChange?: (status: 'idle' | 'connecting' | 'connected' | 'recording' | 'processing' | 'error') => void;
  onTranscript?: (transcript: string) => void;
  onError?: (error: string) => void;
}

export default function VoiceButton({ onStatusChange, onTranscript, onError }: VoiceButtonProps) {
  const [status, setStatus] = useState<'idle' | 'connecting' | 'connected' | 'recording' | 'processing' | 'error'>('idle');
  const [isRecording, setIsRecording] = useState(false);
  
  const whisperWsRef = useRef<VoiceWebSocket | null>(null);
  const recorderRef = useRef<AudioRecorder | null>(null);
  const playerRef = useRef<AudioPlayer | null>(null);
  const sessionIdRef = useRef<string>(`session_${Date.now()}`);
  
  const updateStatus = useCallback((newStatus: typeof status) => {
    setStatus(newStatus);
    onStatusChange?.(newStatus);
  }, [onStatusChange]);
  
  const handleError = useCallback((error: string) => {
    console.error('Voice error:', error);
    updateStatus('error');
    onError?.(error);
  }, [onError, updateStatus]);
  
  useEffect(() => {
    let mounted = true;
    
    async function initializeServices() {
      try {
        updateStatus('connecting');
        
        // Initialize direct connection to WhisperLive
        const whisperWsUrl = process.env.NEXT_PUBLIC_WHISPER_WS_URL || 'ws://localhost:9090';
        const whisperWs = new VoiceWebSocket(whisperWsUrl);
        whisperWsRef.current = whisperWs;
        
        whisperWs.onConnect(() => {
          if (mounted) updateStatus('connected');
        });
        
        whisperWs.onError((error) => {
          if (mounted) handleError(`WhisperLive: ${error}`);
        });
        
        whisperWs.onDisconnect(() => {
          if (mounted) updateStatus('idle');
        });
        
        whisperWs.onTranscript((transcript) => {
          if (mounted) {
            console.log('Received transcript:', transcript);
            onTranscript?.(transcript);
          }
        });
        
        whisperWs.onSentence(async (sentence) => {
          if (mounted) {
            console.log('Complete sentence received:', sentence);
            
            // Send complete sentence to orchestrator for streaming LLM+TTS processing
            try {
              updateStatus('processing');
              const response = await fetch('/api/process-transcript-stream', {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                  transcript: sentence,
                  session_id: sessionIdRef.current
                })
              });
              
              if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
              }
              
              // Handle streaming response
              const reader = response.body?.getReader();
              const decoder = new TextDecoder();
              
              if (reader) {
                let buffer = '';
                while (true) {
                  const { done, value } = await reader.read();
                  if (done) break;
                  
                  buffer += decoder.decode(value, { stream: true });
                  const lines = buffer.split('\n');
                  buffer = lines.pop() || '';
                  
                  for (const line of lines) {
                    if (line.startsWith('data: ')) {
                      const data = JSON.parse(line.slice(6));
                      
                      if (data.type === 'text') {
                        console.log('LLM Response:', data.data);
                      } else if (data.type === 'audio' && playerRef.current) {
                        // Decode PCM chunk and play streaming audio
                        const audioBytes = Uint8Array.from(atob(data.data), c => c.charCodeAt(0));
                        await playerRef.current.playPCMChunk(audioBytes.buffer);
                      } else if (data.type === 'complete') {
                        console.log('Streaming complete, latency:', data.latency_ms);
                        updateStatus('connected');
                      }
                    }
                  }
                }
              }
              
            } catch (error) {
              handleError('Failed to process sentence');
            }
          }
        });
        
        whisperWs.connect(sessionIdRef.current);
        
        // Initialize audio recorder
        const recorder = new AudioRecorder();
        const initialized = await recorder.initialize();
        
        if (!initialized) {
          throw new Error('Failed to access microphone');
        }
        
        recorderRef.current = recorder;
        
        // Initialize audio player
        playerRef.current = new AudioPlayer();
        
      } catch (error) {
        if (mounted) {
          handleError(error instanceof Error ? error.message : 'Initialization failed');
        }
      }
    }
    
    initializeServices();
    
    return () => {
      mounted = false;
      if (whisperWsRef.current) {
        whisperWsRef.current.disconnect();
      }
      if (recorderRef.current) {
        recorderRef.current.cleanup();
      }
      if (playerRef.current) {
        playerRef.current.cleanup();
      }
    };
  }, []); // Remove unstable dependencies that cause re-connections
  
  const startRecording = useCallback(async () => {
    if (!whisperWsRef.current?.isConnected() || !recorderRef.current) {
      handleError('Services not ready');
      return;
    }
    
    try {
      setIsRecording(true);
      updateStatus('recording');
      
      // Start recording with real-time audio streaming to WhisperLive
      recorderRef.current.start((audioData: Float32Array) => {
        if (whisperWsRef.current?.isConnected()) {
          // Send resampled audio data directly to WhisperLive
          whisperWsRef.current.sendAudio(audioData.buffer);
        }
      });
    } catch (error) {
      handleError('Failed to start recording');
      setIsRecording(false);
    }
  }, [handleError, updateStatus]);
  
  const stopRecording = useCallback(async () => {
    if (!isRecording || !recorderRef.current || !whisperWsRef.current) {
      return;
    }
    
    try {
      setIsRecording(false);
      // Keep status as recording until we get transcript
      
      // Stop recording
      recorderRef.current.stop();
      
      // Send end of audio signal to WhisperLive
      whisperWsRef.current.sendEndOfAudio();
      
    } catch (error) {
      handleError('Failed to process recording');
    }
  }, [isRecording, handleError]);
  
  const handleToggleRecording = useCallback(() => {
    if (status === 'connected') {
      if (isRecording) {
        stopRecording();
      } else {
        startRecording();
      }
    }
  }, [status, isRecording, startRecording, stopRecording]);
  
  const getButtonText = () => {
    switch (status) {
      case 'idle':
        return 'Connecting...';
      case 'connecting':
        return 'Connecting...';
      case 'connected':
        return isRecording ? 'Stop Recording' : 'Start Recording';
      case 'recording':
        return 'Stop Recording';
      case 'processing':
        return 'Processing...';
      case 'error':
        return 'Error - Retry';
      default:
        return 'Start Recording';
    }
  };
  
  const getButtonStyle = () => {
    const baseClasses = "w-32 h-32 rounded-full transition-all duration-200 font-bold text-lg shadow-lg";
    
    switch (status) {
      case 'idle':
      case 'connecting':
        return `${baseClasses} bg-gray-400 text-white cursor-not-allowed opacity-70`;
      case 'connected':
        return `${baseClasses} bg-blue-500 hover:bg-blue-600 text-white cursor-pointer ${isRecording ? 'scale-110 bg-red-500 hover:bg-red-600' : ''}`;
      case 'recording':
        return `${baseClasses} bg-red-500 scale-110 text-white cursor-pointer`;
      case 'processing':
        return `${baseClasses} bg-yellow-500 text-white cursor-not-allowed animate-pulse`;
      case 'error':
        return `${baseClasses} bg-red-600 hover:bg-red-700 text-white cursor-pointer`;
      default:
        return `${baseClasses} bg-gray-400 text-white cursor-not-allowed`;
    }
  };
  
  const isDisabled = status !== 'connected' && status !== 'recording';
  
  return (
    <button
      onClick={handleToggleRecording}
      disabled={isDisabled}
      className={getButtonStyle()}
      aria-label={getButtonText()}
    >
      {getButtonText()}
    </button>
  );
}