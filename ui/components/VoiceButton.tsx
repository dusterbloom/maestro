'use client';

import { useState, useRef, useEffect, useCallback, useReducer } from 'react';
import { VoiceWebSocket } from '@/lib/websocket';
import { AudioRecorder, AudioPlayer } from '@/lib/audio';

// Audio queue management with useReducer for better state control
interface AudioQueueItem {
  sequence: number;
  audioData: string;
  text: string;
}

interface AudioQueueState {
  queue: AudioQueueItem[];
  nextToPlay: number;
}

type AudioQueueAction = 
  | { type: 'ADD_AUDIO'; payload: AudioQueueItem }
  | { type: 'CLEAR_QUEUE' }
  | { type: 'REMOVE_PLAYED'; sequence: number }
  | { type: 'RESET' };

const audioQueueReducer = (state: AudioQueueState, action: AudioQueueAction): AudioQueueState => {
  switch (action.type) {
    case 'ADD_AUDIO':
      return {
        ...state,
        queue: [...state.queue, action.payload]
      };
    case 'CLEAR_QUEUE':
      return {
        queue: [],
        nextToPlay: 1
      };
    case 'REMOVE_PLAYED':
      return {
        ...state,
        queue: state.queue.filter(item => item.sequence !== action.sequence),
        nextToPlay: state.nextToPlay + 1
      };
    case 'RESET':
      return {
        queue: [],
        nextToPlay: 1
      };
    default:
      return state;
  }
};

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
  
  // TTS interruption control with proper coordination
  const currentStreamControllerRef = useRef<AbortController | null>(null);
  const voiceActivityRef = useRef<boolean>(false);
  const bargeInTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const isProcessingInterruptionRef = useRef<boolean>(false);
  
  // Audio queue management using useReducer for better state control
  const [audioQueueState, dispatchAudioQueue] = useReducer(audioQueueReducer, {
    queue: [],
    nextToPlay: 1
  });
  
  // Use refs for audio queue access in async contexts to avoid stale closures
  const audioQueueRef = useRef<AudioQueueItem[]>([]);
  const nextToPlayRef = useRef<number>(1);
  
  // Sync reducer state with refs for async access
  useEffect(() => {
    audioQueueRef.current = audioQueueState.queue;
    nextToPlayRef.current = audioQueueState.nextToPlay;
  }, [audioQueueState]);
  
  const updateStatus = useCallback((newStatus: typeof status) => {
    setStatus(newStatus);
    onStatusChange?.(newStatus);
  }, [onStatusChange]);
  
  const handleError = useCallback((error: string) => {
    console.error('Voice error:', error);
    updateStatus('error');
    onError?.(error);
  }, [onError, updateStatus]);
  

  const clearAudioQueue = useCallback(() => {
    console.log(`🧹 Clearing audio queue: ${audioQueueRef.current.length} pending sentences removed`);
    dispatchAudioQueue({ type: 'CLEAR_QUEUE' });
    // Refs will be updated by the useEffect above
  }, []);

  const handleBargeIn = useCallback(async () => {
    // Prevent multiple simultaneous interruptions
    if (isProcessingInterruptionRef.current) {
      console.log('🛑 Interruption already in progress, skipping duplicate request');
      return;
    }
    
    isProcessingInterruptionRef.current = true;
    console.log('🛑 Audio Pipeline: USER INTERRUPT - User can ALWAYS stop assistant');
    
    try {
      // 1. INSTANT: Stop all audio playback first (ALWAYS allowed)
      if (playerRef.current) {
        console.log('   → Stopping audio playback immediately (user override)');
        playerRef.current.stopAll();
      }
      
      // 2. INSTANT: Clear all pending TTS sentences (user takes priority)
      console.log('   → Clearing audio queue - user interruption overrides all');
      clearAudioQueue();
      
      // 3. INSTANT: Update state to reflect stopped audio
      setIsPlaying(false);
      console.log('   → Set isPlaying=false due to user interruption');
      
      // 4. INSTANT: Start recording if conditions are met
      if (!isRecording && status === 'connected' && whisperWsRef.current?.isConnected() && recorderRef.current) {
        console.log('   → Starting recording immediately (user wants to speak)');
        setIsRecording(true);
        updateStatus('recording');
        
        recorderRef.current.start((audioData: Float32Array) => {
          if (whisperWsRef.current?.isConnected()) {
            whisperWsRef.current.sendAudio(audioData.buffer);
          }
        });
      }
      
      // 5. BACKGROUND: Send server-side TTS interruption request (don't await)
      if (whisperWsRef.current) {
        console.log('   → Sending server TTS interruption request (background)');
        whisperWsRef.current.sendInterruptTts(sessionIdRef.current).then(result => {
          if (result.success) {
            console.log('   ✅ Server TTS interruption successful');
          } else {
            console.warn('   ⚠️ Server TTS interruption failed:', result.message);
          }
        }).catch(error => {
          console.error('   ❌ Server TTS interruption error:', error);
        });
      }
      
      // 6. BACKGROUND: Abort any ongoing TTS generation request (frontend control)
      if (currentStreamControllerRef.current) {
        console.log('   → Aborting frontend TTS generation request');
        currentStreamControllerRef.current.abort();
        currentStreamControllerRef.current = null;
      }
      
      console.log('   ✅ User interruption completed - assistant stopped, user can speak');
      
    } catch (error) {
      console.error('   ❌ Error during user interruption:', error);
    } finally {
      // Reset the processing flag after a short delay to prevent rapid retriggering
      setTimeout(() => {
        isProcessingInterruptionRef.current = false;
      }, 100);
    }
  }, [isRecording, status, updateStatus, clearAudioQueue]);
  


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
        
        // Initialize direct connection to WhisperLive with validation
        const whisperWsUrl = process.env.NEXT_PUBLIC_WHISPER_WS_URL || 'ws://localhost:9090';
        
        // Validate WebSocket URL format
        try {
          new URL(whisperWsUrl);
        } catch (urlError) {
          throw new Error(`Invalid WhisperLive WebSocket URL: ${whisperWsUrl}`);
        }
        
        console.log(`🔌 Initializing WhisperLive connection to: ${whisperWsUrl}`);
        const whisperWs = new VoiceWebSocket(whisperWsUrl);
        whisperWsRef.current = whisperWs;
        
        whisperWs.onConnect(() => {
          if (mounted) updateStatus('connected');
        });
        
        whisperWs.onError((error) => {
          if (mounted) handleError(`WhisperLive: ${error}`);
        });
        
        whisperWs.onDisconnect(() => {
          if (mounted) {
            console.log('🔌 WhisperLive disconnected - updating status to idle');
            console.log(`   → Connection state when disconnected: isRecording=${isRecording}, isPlaying=${isPlaying}`);
            updateStatus('idle');
          }
        });
        
        whisperWs.onTranscript((transcript) => {
          if (mounted) {
            console.log('Received transcript:', transcript);
            onTranscript?.(transcript);
          }
        });
        
        whisperWs.onSentence(async (sentence) => {
          if (mounted) {
            console.log('📝 Complete sentence received:', sentence);
            console.log(`🔍 SENTENCE CHECK: isPlaying=${isPlaying}, isRecording=${isRecording}, mounted=${mounted}, wsState=${whisperWs.isConnected() ? 'connected' : 'disconnected'}`);
            console.log(`🔍 QUEUE STATE: queueLength=${audioQueueRef.current.length}, nextToPlay=${nextToPlayRef.current}`);
            
            // ================================================================================================
            // 🚨 IMPORTANT DISTINCTION:
            // 
            // 1. SENTENCE PROCESSING BLOCK (below): Prevents multiple LLM responses from overlapping
            //    - Blocks new sentence processing while assistant is speaking
            //    - Prevents cascading voices and audio chaos
            //    - This is a TECHNICAL protection for audio pipeline stability
            //
            // 2. VOICE INTERRUPTION (above): User can ALWAYS interrupt at any moment  
            //    - Voice activity detection works regardless of any blocks
            //    - User interruption overrides ALL sentence processing blocks
            //    - This is USER CONTROL - user is always in charge
            // ================================================================================================
            
            // 🚨 CRITICAL: Block new TTS sentence processing while audio is actively playing
            // BUT still allow voice interruption via barge-in (this only blocks sentence processing)
            if (isPlaying) {
              console.log('🚨 BLOCKING NEW SENTENCE: Audio is currently playing - ignoring new sentence to prevent overlapping voices');
              console.log(`   → isPlaying=${isPlaying} (user can still interrupt via voice)`);
              return; // Do NOT process new sentences while TTS is actively playing
            }
            
            // Also block if there are queued sentences from the current response
            if (audioQueueRef.current.length > 0) {
              console.log('🚨 BLOCKING NEW SENTENCE: Audio queue has pending sentences - ignoring new sentence');
              console.log(`   → queueLength=${audioQueueRef.current.length} (user can still interrupt via voice)`);
              return; // Do NOT process new sentences while queue has pending audio
            }
            
            // Only process if no audio is currently playing
            console.log(`📝 Processing sentence (audio not playing): ${sentence}`);
            
            // Abort any ongoing TTS stream before starting new one (audio pipeline control)
            console.log('🎯 Audio Pipeline: NEW REQUEST - Interrupting any current audio');
            abortCurrentStream();
            
            // Send complete sentence to orchestrator for REAL-TIME streaming
            try {
              updateStatus('processing');
              
              // Create new AbortController for this stream
              const streamController = new AbortController();
              currentStreamControllerRef.current = streamController;
              
              console.log('🚀 REAL-TIME STREAMING: Using streaming endpoint for instant audio playback');
              
              const response = await fetch('/api/process-transcript', {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                  'X-Use-Streaming': 'true'  // Enable real-time streaming
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
              
              // Reset queue for new stream using reducer
              dispatchAudioQueue({ type: 'RESET' });
              
              const playNextInQueue = async (currentQueue?: AudioQueueItem[]) => {
                // Use passed queue or fall back to ref
                const queueToUse = currentQueue || audioQueueRef.current;
                const nextToPlay = nextToPlayRef.current;
                
                console.log(`🎭 playNextInQueue called: isPlaying=${isPlaying}, queueLength=${queueToUse.length}, nextToPlay=${nextToPlay}`);
                
                if (isPlaying) {
                  console.log('🛑 Already playing, skipping playNextInQueue');
                  return;
                }
                
                // Find the next sentence to play using passed queue or refs
                const nextItem = queueToUse.find((item: AudioQueueItem) => item.sequence === nextToPlay);
                if (!nextItem) {
                  console.log(`❌ No item found for sequence ${nextToPlay} in queue of ${queueToUse.length} items`);
                  console.log(`🔍 Queue contents:`, queueToUse.map(item => `seq${item.sequence}`).join(', '));
                  return;
                }
                
                setIsPlaying(true);  // ✅ FIX: Use React state setter, not local variable
                console.log(`🎵 Playing sentence ${nextItem.sequence}: ${nextItem.text.slice(0, 30)}...`);
                
                try {
                  // Decode base64 audio data
                  console.log(`🔊 Decoding audio data for sentence ${nextItem.sequence}: ${nextItem.audioData.slice(0, 100)}...`);
                  const audioBytes = Uint8Array.from(atob(nextItem.audioData), c => c.charCodeAt(0));
                  console.log(`🔊 Decoded ${audioBytes.length} bytes for sentence ${nextItem.sequence}`);
                  
                  if (playerRef.current) {
                    console.log(`🎵 Starting audio playback for sentence ${nextItem.sequence}`);
                    await playerRef.current.play(audioBytes.buffer);
                    console.log(`✅ Sentence ${nextItem.sequence} played successfully!`);
                  } else {
                    console.error(`❌ No audio player available for sentence ${nextItem.sequence}`);
                  }
                  
                  // Remove played item using reducer action
                  console.log(`🗑️ Removing played sentence ${nextToPlayRef.current}, queue had ${audioQueueRef.current.length} items`);
                  dispatchAudioQueue({ type: 'REMOVE_PLAYED', sequence: nextToPlayRef.current });
                  
                  // Check if more sentences are queued, only set isPlaying=false when done
                  if (audioQueueRef.current.length <= 1) { // Will be 0 after removal
                    setIsPlaying(false);  // ✅ Only stop when queue is empty
                    console.log('🎵 All audio sentences completed, setting isPlaying=false');
                  } else {
                    console.log(`🎵 More sentences queued (${audioQueueRef.current.length - 1} remaining), continuing playback`);
                  }
                  
                  // Play next item if available
                  setTimeout(() => playNextInQueue(), 50); // Small delay between sentences
                  
                } catch (audioError) {
                  console.error(`❌ Failed to play sentence ${nextItem.sequence}:`, audioError);
                  
                  // Remove failed item using reducer action
                  dispatchAudioQueue({ type: 'REMOVE_PLAYED', sequence: nextToPlayRef.current });
                  
                  if (audioQueueRef.current.length <= 1) { // Will be 0 after removal
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
                          console.log(`🎵 Audio data length: ${data.audio_data?.length || 0} characters`);
                          
                          // Add to audio queue using reducer
                          dispatchAudioQueue({
                            type: 'ADD_AUDIO',
                            payload: {
                              sequence: data.sequence,
                              audioData: data.audio_data,
                              text: data.text
                            }
                          });
                          
                          console.log(`📊 Queue state: nextToPlay=${nextToPlayRef.current}, queueLength=${audioQueueRef.current.length + 1}`);
                          
                          // Start playing if this is the next expected sentence
                          if (data.sequence === nextToPlayRef.current) {
                            console.log(`🚀 Starting playback for sequence ${data.sequence}`);
                            // Pass the updated queue directly to avoid ref synchronization delay
                            const updatedQueue = [...audioQueueRef.current, {
                              sequence: data.sequence,
                              audioData: data.audio_data,
                              text: data.text
                            }];
                            playNextInQueue(updatedQueue);
                          } else {
                            console.log(`⏳ Queuing sequence ${data.sequence}, waiting for ${nextToPlayRef.current}`);
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
            // console.log(`🎤 VOICE DEBUG: level=${level.toFixed(4)}, active=${isVoiceActive}, isPlaying=${isPlaying}, isRecording=${isRecording}, threshold=${recorder.getVoiceActivityThreshold()}`);
          }
          
          // 🚨 IMMEDIATE INTERRUPTION: User can ALWAYS interrupt assistant at any moment
          // This voice interruption logic overrides ALL sentence processing blocks
          const hasPendingAudio = audioQueueRef.current.length > 0;
          if (isVoiceActive && (isPlaying || hasPendingAudio) && !isRecording) {
            console.log('🚨 IMMEDIATE BARGE-IN: User interrupting assistant - ALWAYS ALLOWED');
            console.log(`   → isPlaying=${isPlaying}, pendingAudio=${hasPendingAudio}, queueSize=${audioQueueRef.current.length}`);
            console.log('   → This interruption overrides any sentence processing blocks');
            
            // Use the coordinated handleBargeIn function for proper async coordination
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