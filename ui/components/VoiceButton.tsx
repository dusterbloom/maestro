// components/VoiceButton.tsx (Improved Session Management)
'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { VoiceOrchestratorClient, SimpleAudioQueue, SentenceAudioData } from '@/lib/voice-orchestrator-client';
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
  const [connectionAttempts, setConnectionAttempts] = useState(0);
  
  // Core service references
  const orchestratorRef = useRef<VoiceOrchestratorClient | null>(null);
  const recorderRef = useRef<AudioRecorder | null>(null);
  const playerRef = useRef<AudioPlayer | null>(null);
  const audioQueueRef = useRef<SimpleAudioQueue | null>(null);
  
  // Connection management
  const initializationRef = useRef<boolean>(false);
  const cleanupRef = useRef<boolean>(false);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  
  // Voice activity detection for barge-in
  const voiceActivityRef = useRef<boolean>(false);
  
  const updateStatus = useCallback((newStatus: typeof status) => {
    console.log(`🔄 Status change: ${status} -> ${newStatus}`);
    setStatus(newStatus);
    onStatusChange?.(newStatus);
  }, [status, onStatusChange]);
  
  const handleError = useCallback((error: string) => {
    console.error('🚨 Voice error:', error);
    updateStatus('error');
    onError?.(error);
  }, [onError, updateStatus]);
  
  const handleBargein = useCallback(() => {
    console.log('🛑 BARGE-IN: User interrupt detected');
    
    // 1. Immediately stop all audio playback (failsafe)
    if (playerRef.current) {
      playerRef.current.stopAll();
    }
    
    // 2. Clear local audio queue aggressively
    if (audioQueueRef.current) {
      audioQueueRef.current.interrupt();
    }
    
    // 3. Interrupt orchestrator (if connected)
    if (orchestratorRef.current) {
      try {
        orchestratorRef.current.interrupt();
      } catch (e) {
        console.warn('Failed to send interrupt to orchestrator:', e);
      }
    }
    
    // 4. Update local state immediately
    setIsPlaying(false);
    
    // 5. Start recording if not already (to capture new input)
    if (!isRecording && status === 'connected') {
      startRecording();
    }
  }, [isRecording, status]);
  
  // Cleanup function to prevent memory leaks
  const cleanup = useCallback(async () => {
    if (cleanupRef.current) return;
    cleanupRef.current = true;
    
    console.log('🧹 Cleaning up VoiceButton resources');
    
    // Clear any reconnection timeout
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    // Stop recording
    if (recorderRef.current) {
      try {
        recorderRef.current.stop();
        recorderRef.current.cleanup();
      } catch (e) {
        console.warn('Error cleaning up recorder:', e);
      }
      recorderRef.current = null;
    }
    
    // Stop audio playback
    if (playerRef.current) {
      try {
        playerRef.current.cleanup();
      } catch (e) {
        console.warn('Error cleaning up player:', e);
      }
      playerRef.current = null;
    }
    
    // Clear audio queue
    if (audioQueueRef.current) {
      try {
        audioQueueRef.current.clear();
      } catch (e) {
        console.warn('Error clearing audio queue:', e);
      }
      audioQueueRef.current = null;
    }
    
    // Disconnect orchestrator
    if (orchestratorRef.current) {
      try {
        orchestratorRef.current.disconnect();
      } catch (e) {
        console.warn('Error disconnecting orchestrator:', e);
      }
      orchestratorRef.current = null;
    }
    
    // Reset state
    setIsRecording(false);
    setIsPlaying(false);
    setConnectionAttempts(0);
    updateStatus('idle');
    
    console.log('✅ VoiceButton cleanup complete');
  }, [updateStatus]);
  
  // Initialize services with improved error handling and connection management
  const initializeServices = useCallback(async (retryCount: number = 0) => {
    if (initializationRef.current || cleanupRef.current) {
      console.log('⚠️ Initialization already in progress or cleanup initiated');
      return;
    }
    
    const maxRetries = 3;
    const retryDelay = Math.min(1000 * Math.pow(2, retryCount), 10000); // Exponential backoff, max 10s
    
    try {
      initializationRef.current = true;
      setConnectionAttempts(retryCount + 1);
      
      console.log(`🚀 Initializing services (attempt ${retryCount + 1}/${maxRetries})`);
      updateStatus('connecting');
      
      // 1. Initialize audio recorder with error handling
      console.log('🎤 Initializing audio recorder...');
      const recorder = new AudioRecorder();
      const recorderInitialized = await recorder.initialize();
      
      if (!recorderInitialized) {
        throw new Error('Failed to access microphone - check permissions');
      }
      
      // Set up voice activity detection for barge-in
      recorder.onAudioLevel((level) => {
        const isVoiceActive = recorder.isVoiceActive();
        voiceActivityRef.current = isVoiceActive;
        
        // More aggressive barge-in detection with multiple triggers
        const hasPendingAudio = audioQueueRef.current ? audioQueueRef.current.getQueueSize() > 0 : false;
        const isTTSActive = isPlaying || hasPendingAudio || status === 'processing';
        
        // Trigger barge-in immediately when voice is detected during TTS
        if (isVoiceActive && isTTSActive && !isRecording) {
          console.log('🚨 BARGE-IN TRIGGERED: Interrupting TTS');
          handleBargein();
        }
        
        // Additional hair-trigger interruption on any significant audio level
        if (level > 0.08 && (isPlaying || status === 'processing') && !isRecording) {
          console.log('🚨 VOICE DETECTED: Quick interrupt on audio level', level);
          handleBargein();
        }
      });
      
      recorderRef.current = recorder;
      console.log('✅ Audio recorder initialized');
      
      // 2. Initialize audio player
      console.log('🔊 Initializing audio player...');
      const player = new AudioPlayer();
      playerRef.current = player;
      console.log('✅ Audio player initialized');
      
      // 3. Initialize audio queue
      const audioQueue = new SimpleAudioQueue(player);
      audioQueueRef.current = audioQueue;
      console.log('✅ Audio queue initialized');
      
      // 4. Initialize orchestrator client with stable session ID
      console.log('🎭 Initializing orchestrator client...');
      const sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const orchestrator = new VoiceOrchestratorClient(sessionId);
      orchestratorRef.current = orchestrator;
      
      // Set up orchestrator event handlers
      orchestrator.onReady(() => {
        if (!cleanupRef.current) {
          console.log('🎯 Orchestrator ready');
          updateStatus('connected');
          setConnectionAttempts(0); // Reset retry counter on success
          initializationRef.current = false;
        }
      });
      
      orchestrator.onError((error) => {
        if (!cleanupRef.current) {
          console.error('🚨 Orchestrator error:', error);
          handleError(`Connection: ${error}`);
          initializationRef.current = false;
          
          // Attempt retry if under max attempts
          if (retryCount < maxRetries - 1) {
            console.log(`🔄 Retrying connection in ${retryDelay}ms...`);
            reconnectTimeoutRef.current = setTimeout(() => {
              initializeServices(retryCount + 1);
            }, retryDelay);
          } else {
            console.error('❌ Max connection attempts reached');
          }
        }
      });
      
      orchestrator.onDisconnect(() => {
        if (!cleanupRef.current) {
          console.log('🔌 Orchestrator disconnected');
          updateStatus('idle');
          initializationRef.current = false;
          
          // Attempt reconnection if not at max retries
          if (retryCount < maxRetries - 1) {
            console.log(`🔄 Attempting reconnection in ${retryDelay}ms...`);
            reconnectTimeoutRef.current = setTimeout(() => {
              initializeServices(retryCount + 1);
            }, retryDelay);
          }
        }
      });
      
      orchestrator.onLiveTranscript((transcript) => {
        if (!cleanupRef.current) {
          console.log('📝 Live transcript:', transcript);
          onTranscript?.(transcript);
        }
      });
      
      orchestrator.onProcessing((isProcessing) => {
        if (!cleanupRef.current) {
          if (isProcessing) {
            updateStatus('processing');
          } else if (status === 'processing') {
            updateStatus('connected');
          }
        }
      });
      
      orchestrator.onSentenceAudio((data: SentenceAudioData) => {
        if (!cleanupRef.current && audioQueueRef.current) {
          console.log(`🎵 Received sentence audio ${data.sequence}`);
          audioQueueRef.current.addSentence(data);
          
          // Update playing status
          if (!isPlaying) {
            setIsPlaying(true);
          }
        }
      });
      
      // Connect to orchestrator
      console.log('🔌 Connecting to orchestrator...');
      orchestrator.connect();
      
    } catch (error) {
      console.error(`❌ Initialization failed (attempt ${retryCount + 1}):`, error);
      initializationRef.current = false;
      
      if (!cleanupRef.current) {
        if (retryCount < maxRetries - 1) {
          console.log(`🔄 Retrying initialization in ${retryDelay}ms...`);
          reconnectTimeoutRef.current = setTimeout(() => {
            initializeServices(retryCount + 1);
          }, retryDelay);
        } else {
          handleError(error instanceof Error ? error.message : 'Initialization failed after multiple attempts');
        }
      }
    }
  }, [updateStatus, handleError, handleBargein, isPlaying, status, isRecording, onTranscript]);
  
  // Initialize on mount, cleanup on unmount
  useEffect(() => {
    console.log('🎬 VoiceButton component mounting');
    cleanupRef.current = false;
    initializeServices();
    
    return () => {
      console.log('🎬 VoiceButton component unmounting');
      cleanup();
    };
  }, []); // Empty dependency array - only run once
  
  const startRecording = useCallback(async () => {
    if (!orchestratorRef.current?.isConnected() || !recorderRef.current) {
      handleError('Services not ready');
      return;
    }
    
    try {
      setIsRecording(true);
      updateStatus('recording');
      
      console.log('🎤 Starting recording...');
      
      // Start recording with real-time audio streaming
      recorderRef.current.start((audioData: Float32Array) => {
        if (orchestratorRef.current?.isConnected()) {
          orchestratorRef.current.sendAudio(audioData.buffer);
        }
      });
      
      console.log('🎤 Recording started successfully');
      
    } catch (error) {
      console.error('❌ Failed to start recording:', error);
      handleError('Failed to start recording');
      setIsRecording(false);
    }
  }, [handleError, updateStatus]);
  
  const stopRecording = useCallback(async () => {
    if (!isRecording || !recorderRef.current || !orchestratorRef.current) {
      return;
    }
    
    try {
      console.log('🛑 Stopping recording...');
      setIsRecording(false);
      
      // Stop recording
      recorderRef.current.stop();
      
      // Send end of audio signal
      orchestratorRef.current.sendEndOfAudio();
      
      console.log('🛑 Recording stopped successfully');
      
    } catch (error) {
      console.error('❌ Failed to stop recording:', error);
      handleError('Failed to stop recording');
    }
  }, [isRecording, handleError]);
  
  const handleToggleRecording = useCallback(() => {
    if (status === 'connected') {
      if (isRecording) {
        stopRecording();
      } else {
        startRecording();
      }
    } else if (status === 'error') {
      // Retry connection on error
      console.log('🔄 Retrying connection...');
      cleanup().then(() => {
        setTimeout(() => {
          cleanupRef.current = false;
          initializeServices();
        }, 1000);
      });
    }
  }, [status, isRecording, startRecording, stopRecording, cleanup, initializeServices]);
  
  // UI helpers
  const getButtonText = () => {
    switch (status) {
      case 'idle':
        return 'Initializing...';
      case 'connecting':
        return connectionAttempts > 1 ? `Retrying... (${connectionAttempts}/3)` : 'Connecting...';
      case 'connected':
        return isRecording ? 'Stop Recording' : 'Start Recording';
      case 'recording':
        return 'Stop Recording';
      case 'processing':
        return 'Processing...';
      case 'error':
        return 'Error - Tap to Retry';
      default:
        return 'Start Recording';
    }
  };
  
  const getButtonStyle = () => {
    const baseClasses = "w-32 h-32 rounded-full transition-all duration-200 font-bold text-lg shadow-lg";
    
    switch (status) {
      case 'idle':
      case 'connecting':
        return `${baseClasses} bg-gray-400 text-white cursor-not-allowed opacity-70 ${
          status === 'connecting' ? 'animate-pulse' : ''
        }`;
      case 'connected':
        return `${baseClasses} bg-blue-500 hover:bg-blue-600 text-white cursor-pointer ${
          isRecording ? 'scale-110 bg-red-500 hover:bg-red-600' : ''
        }`;
      case 'recording':
        return `${baseClasses} bg-red-500 scale-110 text-white cursor-pointer animate-pulse`;
      case 'processing':
        return `${baseClasses} bg-yellow-500 text-white cursor-not-allowed animate-pulse`;
      case 'error':
        return `${baseClasses} bg-red-600 hover:bg-red-700 text-white cursor-pointer animate-bounce`;
      default:
        return `${baseClasses} bg-gray-400 text-white cursor-not-allowed`;
    }
  };
  
  const isDisabled = status !== 'connected' && status !== 'recording' && status !== 'error';
  
  return (
    <div className="relative">
      <button
        onClick={handleToggleRecording}
        disabled={isDisabled}
        className={getButtonStyle()}
        aria-label={getButtonText()}
      >
        {getButtonText()}
      </button>
      
      {/* Interrupt button when audio is playing */}
      {(isPlaying || status === 'processing') && (
        <button
          onClick={handleBargein}
          className="absolute -right-16 top-1/2 transform -translate-y-1/2 bg-red-500 hover:bg-red-600 text-white px-3 py-2 rounded-full text-sm font-medium transition-colors"
          aria-label="Stop & Interrupt"
          title="Stop audio and start speaking"
        >
          🛑 Stop
        </button>
      )}
      
      {/* Status indicators */}
      <div className="absolute -bottom-8 left-1/2 transform -translate-x-1/2 text-xs text-gray-600">
        {isPlaying && (
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span>Playing</span>
          </div>
        )}
        {audioQueueRef.current && audioQueueRef.current.getQueueSize() > 0 && (
          <div className="text-blue-600">
            Queue: {audioQueueRef.current.getQueueSize()}
          </div>
        )}
        {connectionAttempts > 0 && status === 'connecting' && (
          <div className="text-orange-600 text-xs">
            Attempt {connectionAttempts}/3
          </div>
        )}
      </div>
    </div>
  );
}