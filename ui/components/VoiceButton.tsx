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
  
  // Audio queue management at component level for interruption access
  const audioQueueRef = useRef<Array<{sequence: number, audioData: string, text: string}>>([]);
  const nextToPlayRef = useRef<number>(1);
  
  // ENHANCED: Store audio chunks for magical speaker recognition (5-second buffering)
  const lastRecordedAudioRef = useRef<ArrayBuffer | null>(null);
  const audioChunksRef = useRef<Float32Array[]>([]);
  const speakerProgressRef = useRef<number>(0);
  
  const updateStatus = useCallback((newStatus: typeof status) => {
    setStatus(newStatus);
    onStatusChange?.(newStatus);
  }, [onStatusChange]);
  
  const handleError = useCallback((error: string) => {
    console.error('Voice error:', error);
    updateStatus('error');
    onError?.(error);
  }, [onError, updateStatus]);
  
  const handleBargeIn = useCallback(async () => {
    console.log('🛑 Audio Pipeline: USER INTERRUPT - Stopping TTS and starting recording');
    
    // 1. Send server-side TTS interruption request immediately
    if (whisperWsRef.current) {
      try {
        console.log('   → Sending server TTS interruption request');
        const result = await whisperWsRef.current.sendInterruptTts(sessionIdRef.current);
        if (result.success) {
          console.log('   ✅ Server TTS interruption successful');
        } else {
          console.warn('   ⚠️ Server TTS interruption failed:', result.message);
        }
      } catch (error) {
        console.error('   ❌ Server TTS interruption error:', error);
      }
    }
    
    // 2. Abort any ongoing TTS generation request (frontend control)
    if (currentStreamControllerRef.current) {
      console.log('   → Aborting frontend TTS generation request');
      currentStreamControllerRef.current.abort();
      currentStreamControllerRef.current = null;
    }
    
    // 3. Stop all audio playback immediately (frontend audio control)
    if (playerRef.current) {
      console.log('   → Stopping audio playback');
      playerRef.current.stopAll();
    }
    
    // 4. Update isPlaying state since we interrupted the audio
    setIsPlaying(false);
    console.log('   → Set isPlaying=false due to interruption');
    
    // 5. Start recording immediately if not already recording and connected
    if (!isRecording && status === 'connected') {
      // Use direct recording start to avoid circular dependency
      if (whisperWsRef.current?.isConnected() && recorderRef.current) {
        setIsRecording(true);
        updateStatus('recording');
        
        recorderRef.current.start((audioData: Float32Array) => {
          if (whisperWsRef.current?.isConnected()) {
            whisperWsRef.current.sendAudio(audioData.buffer);
            
            // ENHANCED: Store chunks for magical speaker recognition
            audioChunksRef.current.push(new Float32Array(audioData));
            
            // Keep last 5 seconds of audio (approximately 16000 * 5 samples)
            const maxSamples = 16000 * 5; // 5 seconds at 16kHz
            let totalSamples = audioChunksRef.current.reduce((sum, chunk) => sum + chunk.length, 0);
            
            while (totalSamples > maxSamples && audioChunksRef.current.length > 1) {
              const removed = audioChunksRef.current.shift();
              if (removed) totalSamples -= removed.length;
            }
            
            // Store the full 5-second buffer for speaker identification
            if (audioChunksRef.current.length > 0) {
              const combinedLength = audioChunksRef.current.reduce((sum, chunk) => sum + chunk.length, 0);
              const combinedAudio = new Float32Array(combinedLength);
              let offset = 0;
              
              for (const chunk of audioChunksRef.current) {
                combinedAudio.set(chunk, offset);
                offset += chunk.length;
              }
              
              lastRecordedAudioRef.current = combinedAudio.buffer;
            }
          }
        });
      }
    }
  }, [isRecording, status, updateStatus]);
  
  const clearAudioQueue = useCallback(() => {
    console.log(`🧹 Clearing audio queue: ${audioQueueRef.current.length} pending sentences removed`);
    audioQueueRef.current = [];
    nextToPlayRef.current = 1;
  }, []);

  const abortCurrentStream = useCallback(() => {
    if (currentStreamControllerRef.current) {
      console.log('Aborting current TTS stream - Audio pipeline control');
      currentStreamControllerRef.current.abort();
      currentStreamControllerRef.current = null;
    }
    
    // Also stop any playing audio immediately
    if (playerRef.current) {
      playerRef.current.stopAll();
      setIsPlaying(false);
    }
    
    // Clear pending audio queue
    clearAudioQueue();
  }, [clearAudioQueue]);
  
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
        
        // NEW: Modified to handle speaker-aware processing
        whisperWs.onSentence(async (sentence, audioData) => {
          if (mounted) {
            console.log('Complete sentence received:', sentence);
            console.log(`🔍 SENTENCE CHECK: isPlaying=${isPlaying}, isRecording=${isRecording}, mounted=${mounted}`);
            
            // 🚨 CRITICAL: Completely block new TTS requests while audio is playing
            if (isPlaying) {
              console.log('🚨 BLOCKING NEW TTS: Audio is currently playing - ignoring sentence to prevent cascading voices');
              return; // Do NOT process new sentences while TTS is playing
            }
            
            // Only process if no audio is currently playing
            console.log(`📝 Processing sentence (audio not playing): ${sentence}`);
            
            // Abort any ongoing TTS stream before starting new one (audio pipeline control)
            console.log('🎯 Audio Pipeline: NEW REQUEST - Interrupting any current audio');
            abortCurrentStream();
            
            // Send complete sentence to orchestrator for REAL-TIME streaming with speaker identification
            try {
              updateStatus('processing');
              
              // Create new AbortController for this stream
              const streamController = new AbortController();
              currentStreamControllerRef.current = streamController;
              
              console.log('🚀 REAL-TIME STREAMING: Using streaming endpoint for instant audio playback WITH SPEAKER ID');
              
              // NEW: Prepare request with audio data for speaker identification
              const requestBody: any = {
                transcript: sentence,
                session_id: sessionIdRef.current
              };
              
              // Include audio data if available for speaker identification
              if (lastRecordedAudioRef.current) {
                // Convert ArrayBuffer to base64
                const audioBytes = new Uint8Array(lastRecordedAudioRef.current);
                const audioBase64 = btoa(String.fromCharCode(...audioBytes));
                requestBody.audio_data = audioBase64;
                console.log('🎭 Including 5-second audio buffer for magical speaker recognition');
              }
              
              const response = await fetch('/api/process-transcript', {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                  'X-Use-Streaming': 'true'  // Enable real-time streaming
                },
                body: JSON.stringify(requestBody),
                signal: streamController.signal  // Add abort signal
              });
              
              if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
              }
              
              // Handle Server-Sent Events with sentence-level audio streaming
              if (!response.body) {
                throw new Error('No response stream available');
              }
              
              const eventSource = new ReadableStream({
                start(controller) {
                  const reader = response.body!.getReader();
                  
                  function pump(): Promise<void> {
                    return reader.read().then(({ done, value }) => {
                      if (done) {
                        controller.close();
                        return;
                      }
                      controller.enqueue(value);
                      return pump();
                    }).catch(error => {
                      controller.error(error);
                    });
                  }
                  
                  return pump();
                }
              });
              
              const reader = eventSource.getReader();
              const decoder = new TextDecoder();
              let buffer = '';
              let sentenceCount = 0;
              // Use component-level queue refs for interruption access
              audioQueueRef.current = []; // Reset queue for new stream
              nextToPlayRef.current = 1;
              
              const playNextInQueue = async () => {
                if (isPlaying) return;
                
                // Find the next sentence to play
                const nextItem = audioQueueRef.current.find(item => item.sequence === nextToPlayRef.current);
                if (!nextItem) return;
                
                setIsPlaying(true);  // ✅ FIX: Use React state setter, not local variable
                console.log(`🎵 Playing sentence ${nextItem.sequence}: ${nextItem.text.slice(0, 30)}...`);
                
                try {
                  // Decode base64 audio data
                  const audioBytes = Uint8Array.from(atob(nextItem.audioData), c => c.charCodeAt(0));
                  
                  if (playerRef.current) {
                    await playerRef.current.play(audioBytes.buffer);
                    console.log(`✅ Sentence ${nextItem.sequence} played successfully!`);
                  }
                  
                  // Remove played item and move to next
                  const index = audioQueueRef.current.findIndex(item => item.sequence === nextToPlayRef.current);
                  if (index >= 0) {
                    audioQueueRef.current.splice(index, 1);
                  }
                  nextToPlayRef.current++;
                  
                  // Check if more sentences are queued, only set isPlaying=false when done
                  if (audioQueueRef.current.length === 0) {
                    setIsPlaying(false);  // ✅ Only stop when queue is empty
                    console.log('🎵 All audio sentences completed, setting isPlaying=false');
                  }
                  
                  // Play next item if available
                  setTimeout(() => playNextInQueue(), 50); // Small delay between sentences
                  
                } catch (audioError) {
                  console.error(`❌ Failed to play sentence ${nextItem.sequence}:`, audioError);
                  
                  // Remove failed item and check if queue is empty
                  const index = audioQueueRef.current.findIndex(item => item.sequence === nextToPlayRef.current);
                  if (index >= 0) {
                    audioQueueRef.current.splice(index, 1);
                  }
                  nextToPlayRef.current++;
                  
                  if (audioQueueRef.current.length === 0) {
                    setIsPlaying(false);
                    console.log('🎵 Audio queue empty after error, setting isPlaying=false');
                  }
                  
                  setTimeout(() => playNextInQueue(), 50);
                }
              };
              
              while (true) {
                const { done, value } = await reader.read();
                
                if (done) {
                  console.log('🎉 Real-time streaming complete!');
                  break;
                }
                
                // Check if stream was aborted
                if (streamController.signal.aborted) {
                  console.log('🛑 Stream aborted by user');
                  reader.cancel();
                  break;
                }
                
                if (value) {
                  buffer += decoder.decode(value, { stream: true });
                  
                  // Process complete lines
                  const lines = buffer.split('\n');
                  buffer = lines.pop() || ''; // Keep incomplete line
                  
                  for (const line of lines) {
                    if (line.startsWith('data: ')) {
                      try {
                        const data = JSON.parse(line.slice(6));
                        
                        if (data.type === 'sentence_audio') {
                          sentenceCount++;
                          console.log(`📝 Received sentence ${data.sequence}: ${data.text.slice(0, 30)}...`);
                          
                          // Add to audio queue
                          audioQueueRef.current.push({
                            sequence: data.sequence,
                            audioData: data.audio_data,
                            text: data.text
                          });
                          
                          // Start playing if this is the next expected sentence
                          if (data.sequence === nextToPlayRef.current) {
                            playNextInQueue();
                          }
                          
                        } else if (data.type === 'complete') {
                          console.log(`🎉 All ${data.total_sentences} sentences received!`);
                          console.log(`📄 Complete response: ${data.full_text?.slice(0, 100)}...`);
                        } else if (data.type === 'error') {
                          console.error('❌ Streaming error:', data.message);
                        }
                      } catch (e) {
                        console.error('Failed to parse streaming data:', e);
                      }
                    }
                  }
                }
              }
              
              updateStatus('connected');
              // Clear the controller reference
              if (currentStreamControllerRef.current === streamController) {
                currentStreamControllerRef.current = null;
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
        
        // Set up voice activity detection for IMMEDIATE barge-in (< 100ms)
        recorder.onAudioLevel((level) => {
          const isVoiceActive = recorder.isVoiceActive();
          voiceActivityRef.current = isVoiceActive;
          
          // 🔍 DEBUG: Log voice activity levels to diagnose detection issues
          if (level > 0.001) { // Only log when there's some audio
            console.log(`🎤 VOICE DEBUG: level=${level.toFixed(4)}, active=${isVoiceActive}, isPlaying=${isPlaying}, isRecording=${isRecording}, threshold=${recorder.getVoiceActivityThreshold()}`);
          }
          
          // 🚨 IMMEDIATE INTERRUPTION: If user speaks while TTS is playing OR queued
          const hasPendingAudio = audioQueueRef.current.length > 0;
          if (isVoiceActive && (isPlaying || hasPendingAudio) && !isRecording) {
            console.log('🚨 IMMEDIATE BARGE-IN: Voice detected while TTS playing/queued - INSTANT STOP');
            console.log(`   → isPlaying=${isPlaying}, pendingAudio=${hasPendingAudio}, queueSize=${audioQueueRef.current.length}`);
            
            // 1. INSTANT: Stop all audio playback
            if (playerRef.current) {
              playerRef.current.stopAll();
              console.log('   ⚡ Audio stopped instantly');
            }
            
            // 2. INSTANT: Clear all pending TTS sentences to prevent cascading voices
            clearAudioQueue();
            
            // 3. INSTANT: Update state
            setIsPlaying(false);
            
            // 4. INSTANT: Start recording 
            if (whisperWsRef.current?.isConnected() && recorderRef.current) {
              setIsRecording(true);
              updateStatus('recording');
              
              recorderRef.current.start((audioData: Float32Array) => {
                if (whisperWsRef.current?.isConnected()) {
                  whisperWsRef.current.sendAudio(audioData.buffer);
                  
                  // ENHANCED: Store chunks for magical speaker recognition
                  audioChunksRef.current.push(new Float32Array(audioData));
                  
                  // Keep last 5 seconds of audio
                  const maxSamples = 16000 * 5;
                  let totalSamples = audioChunksRef.current.reduce((sum, chunk) => sum + chunk.length, 0);
                  
                  while (totalSamples > maxSamples && audioChunksRef.current.length > 1) {
                    const removed = audioChunksRef.current.shift();
                    if (removed) totalSamples -= removed.length;
                  }
                  
                  // Update combined buffer
                  if (audioChunksRef.current.length > 0) {
                    const combinedLength = audioChunksRef.current.reduce((sum, chunk) => sum + chunk.length, 0);
                    const combinedAudio = new Float32Array(combinedLength);
                    let offset = 0;
                    
                    for (const chunk of audioChunksRef.current) {
                      combinedAudio.set(chunk, offset);
                      offset += chunk.length;
                    }
                    
                    lastRecordedAudioRef.current = combinedAudio.buffer;
                  }
                }
              });
              console.log('   ⚡ Recording started instantly');
            }
            
            // 5. BACKGROUND: Server interruption (don't wait for this)
            if (whisperWsRef.current) {
              whisperWsRef.current.sendInterruptTts(sessionIdRef.current).then(result => {
                if (result.success) {
                  console.log('   ✅ Server interruption completed in background');
                } else {
                  console.warn('   ⚠️ Server interruption failed:', result.message);
                }
              }).catch(error => {
                console.error('   ❌ Server interruption error:', error);
              });
            }
            
            // 6. BACKGROUND: Abort frontend stream (don't wait)
            if (currentStreamControllerRef.current) {
              currentStreamControllerRef.current.abort();
              currentStreamControllerRef.current = null;
              console.log('   ✅ Frontend stream aborted in background');
            }
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
      
      // Clear magical speaker recognition buffers
      lastRecordedAudioRef.current = null;
      audioChunksRef.current = [];
      speakerProgressRef.current = 0;
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
          
          // ENHANCED: Store chunks for magical speaker recognition
          audioChunksRef.current.push(new Float32Array(audioData));
          
          // Keep last 5 seconds of audio
          const maxSamples = 16000 * 5;
          let totalSamples = audioChunksRef.current.reduce((sum, chunk) => sum + chunk.length, 0);
          
          while (totalSamples > maxSamples && audioChunksRef.current.length > 1) {
            const removed = audioChunksRef.current.shift();
            if (removed) totalSamples -= removed.length;
          }
          
          // Update combined buffer
          if (audioChunksRef.current.length > 0) {
            const combinedLength = audioChunksRef.current.reduce((sum, chunk) => sum + chunk.length, 0);
            const combinedAudio = new Float32Array(combinedLength);
            let offset = 0;
            
            for (const chunk of audioChunksRef.current) {
              combinedAudio.set(chunk, offset);
              offset += chunk.length;
            }
            
            lastRecordedAudioRef.current = combinedAudio.buffer;
          }
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