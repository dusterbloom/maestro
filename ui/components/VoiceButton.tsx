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
  const [isPlaying, setIsPlaying] = useState(false);
  
  const whisperWsRef = useRef<VoiceWebSocket | null>(null);
  const recorderRef = useRef<AudioRecorder | null>(null);
  const playerRef = useRef<AudioPlayer | null>(null);
  const sessionIdRef = useRef<string>(`session_${Date.now()}`);
  
  // TTS interruption control
  const currentStreamControllerRef = useRef<AbortController | null>(null);
  const voiceActivityRef = useRef<boolean>(false);
  const bargeInTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  
  const updateStatus = useCallback((newStatus: typeof status) => {
    setStatus(newStatus);
    onStatusChange?.(newStatus);
  }, [onStatusChange]);
  
  const handleError = useCallback((error: string) => {
    console.error('Voice error:', error);
    updateStatus('error');
    onError?.(error);
  }, [onError, updateStatus]);
  
  const handleBargeIn = useCallback(() => {
    console.log('Handling barge-in: stopping TTS and starting recording');
    
    // Cancel any ongoing TTS stream
    if (currentStreamControllerRef.current) {
      currentStreamControllerRef.current.abort();
      currentStreamControllerRef.current = null;
    }
    
    // Stop all audio playback immediately
    if (playerRef.current) {
      playerRef.current.stopAll();
    }
    
    // Start recording immediately if not already recording and connected
    if (!isRecording && status === 'connected') {
      // Use direct recording start to avoid circular dependency
      if (whisperWsRef.current?.isConnected() && recorderRef.current) {
        setIsRecording(true);
        updateStatus('recording');
        
        recorderRef.current.start((audioData: Float32Array) => {
          if (whisperWsRef.current?.isConnected()) {
            whisperWsRef.current.sendAudio(audioData.buffer);
          }
        });
      }
    }
  }, [isRecording, status, updateStatus]);
  
  const abortCurrentStream = useCallback(() => {
    if (currentStreamControllerRef.current) {
      console.log('Aborting current TTS stream');
      currentStreamControllerRef.current.abort();
      currentStreamControllerRef.current = null;
    }
  }, []);
  
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
            
            // Abort any ongoing TTS stream before starting new one
            abortCurrentStream();
            
            // Send complete sentence to orchestrator for streaming LLM+TTS processing
            try {
              updateStatus('processing');
              
              // Create new AbortController for this stream
              const streamController = new AbortController();
              currentStreamControllerRef.current = streamController;
              
              const response = await fetch('/api/process-transcript-pipeline', {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                  transcript: sentence,
                  session_id: sessionIdRef.current
                }),
                signal: streamController.signal  // Add abort signal
              });
              
              if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
              }
              
              // Handle streaming response with interruption support
              const reader = response.body?.getReader();
              const decoder = new TextDecoder();
              
              if (reader) {
                let buffer = '';
                try {
                  while (true) {
                    // Check if stream was aborted
                    if (streamController.signal.aborted) {
                      console.log('TTS stream aborted by barge-in');
                      break;
                    }
                    
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop() || '';
                    
                    for (const line of lines) {
                      if (line.startsWith('data: ')) {
                        const data = JSON.parse(line.slice(6));
                        
                        if (data.type === 'text') {
                          console.log('LLM Response (Sequence', data.sequence + '):', data.text);
                          console.log('Complete text so far:', data.complete_text);
                        } else if (data.type === 'audio' && playerRef.current) {
                          // Check again before playing audio chunk
                          if (streamController.signal.aborted) {
                            console.log('Audio chunk skipped due to abort');
                            break;
                          }
                          
                          // Decode PCM chunk and play streaming audio
                          const audioBytes = Uint8Array.from(atob(data.data), c => c.charCodeAt(0));
                          await playerRef.current.playPCMChunk(audioBytes.buffer);
                        } else if (data.type === 'complete') {
                          console.log('Ultra-low latency pipeline complete');
                          console.log('Final response:', data.complete_text);
                          console.log('Total latency:', data.latency_ms + 'ms');
                          console.log('Time to first response (TTFR):', data.ttfr_ms + 'ms');
                          updateStatus('connected');
                          // Clear the controller reference
                          if (currentStreamControllerRef.current === streamController) {
                            currentStreamControllerRef.current = null;
                          }
                        }
                      }
                    }
                  }
                } catch (streamError: any) {
                  if (streamError.name === 'AbortError') {
                    console.log('TTS stream was aborted');
                  } else {
                    throw streamError;
                  }
                } finally {
                  // Ensure reader is closed
                  reader.releaseLock();
                }
              }
              
            } catch (error: any) {
              // Don't treat abort as error
              if (error.name === 'AbortError') {
                console.log('TTS request aborted due to barge-in');
                updateStatus('connected');
              } else {
                handleError('Failed to process sentence');
              }
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
        
        // Set up voice activity detection for barge-in
        recorder.onAudioLevel((level) => {
          const isVoiceActive = recorder.isVoiceActive();
          voiceActivityRef.current = isVoiceActive;
          
          // If we detect voice activity while TTS is playing, trigger barge-in
          if (isVoiceActive && isPlaying && !isRecording) {
            console.log('Barge-in detected! Voice level:', level);
            handleBargeIn();
          }
        });
        
        recorderRef.current = recorder;
        
        // Initialize audio player
        const player = new AudioPlayer();
        
        // Set up playback state tracking
        player.onPlaybackStart(() => {
          console.log('TTS playback started');
          setIsPlaying(true);
        });
        
        player.onPlaybackEnd(() => {
          console.log('TTS playback ended');
          setIsPlaying(false);
        });
        
        playerRef.current = player;
        
      } catch (error) {
        if (mounted) {
          handleError(error instanceof Error ? error.message : 'Initialization failed');
        }
      }
    }
    
    initializeServices();
    
    return () => {
      mounted = false;
      
      // Clean up TTS stream
      if (currentStreamControllerRef.current) {
        currentStreamControllerRef.current.abort();
        currentStreamControllerRef.current = null;
      }
      
      // Clean up timeouts
      if (bargeInTimeoutRef.current) {
        clearTimeout(bargeInTimeoutRef.current);
        bargeInTimeoutRef.current = null;
      }
      
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