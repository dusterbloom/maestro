'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { VoiceWebSocket } from '@/lib/websocket';
import { AudioRecorder, AudioPlayer } from '@/lib/audio';

interface VoiceButtonProps {
  onStatusChange?: (status: 'idle' | 'connecting' | 'connected' | 'recording' | 'processing' | 'error') => void;
  onError?: (error: string) => void;
}

export default function VoiceButton({ onStatusChange, onError }: VoiceButtonProps) {
  const [status, setStatus] = useState<'idle' | 'connecting' | 'connected' | 'recording' | 'processing' | 'error'>('idle');
  const [isRecording, setIsRecording] = useState(false);
  
  const wsRef = useRef<VoiceWebSocket | null>(null);
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
        
        // Initialize WebSocket
        const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws';
        const ws = new VoiceWebSocket(wsUrl);
        wsRef.current = ws;
        
        ws.onConnect(() => {
          if (mounted) updateStatus('connected');
        });
        
        ws.onError((error) => {
          if (mounted) handleError(`WebSocket: ${error}`);
        });
        
        ws.onDisconnect(() => {
          if (mounted) updateStatus('idle');
        });
        
        ws.onAudio(async (audio) => {
          if (mounted && playerRef.current) {
            try {
              await playerRef.current.play(audio);
              updateStatus('connected');
            } catch (error) {
              handleError('Failed to play audio response');
            }
          }
        });
        
        ws.connect(sessionIdRef.current);
        
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
      if (wsRef.current) {
        wsRef.current.disconnect();
      }
      if (recorderRef.current) {
        recorderRef.current.cleanup();
      }
      if (playerRef.current) {
        playerRef.current.cleanup();
      }
    };
  }, [updateStatus, handleError]);
  
  const startRecording = useCallback(async () => {
    if (!wsRef.current?.isConnected() || !recorderRef.current) {
      handleError('Services not ready');
      return;
    }
    
    try {
      setIsRecording(true);
      updateStatus('recording');
      recorderRef.current.start();
    } catch (error) {
      handleError('Failed to start recording');
      setIsRecording(false);
    }
  }, [handleError, updateStatus]);
  
  const stopRecording = useCallback(async () => {
    if (!isRecording || !recorderRef.current || !wsRef.current) {
      return;
    }
    
    try {
      setIsRecording(false);
      updateStatus('processing');
      
      const audioData = await recorderRef.current.stop();
      wsRef.current.sendAudio(audioData);
      
    } catch (error) {
      handleError('Failed to process recording');
    }
  }, [isRecording, handleError, updateStatus]);
  
  const handleMouseDown = useCallback(() => {
    if (status === 'connected') {
      startRecording();
    }
  }, [status, startRecording]);
  
  const handleMouseUp = useCallback(() => {
    if (isRecording) {
      stopRecording();
    }
  }, [isRecording, stopRecording]);
  
  const getButtonText = () => {
    switch (status) {
      case 'idle':
        return 'Connecting...';
      case 'connecting':
        return 'Connecting...';
      case 'connected':
        return isRecording ? 'Listening...' : 'Push to Talk';
      case 'recording':
        return 'Listening...';
      case 'processing':
        return 'Processing...';
      case 'error':
        return 'Error - Retry';
      default:
        return 'Push to Talk';
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
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onTouchStart={handleMouseDown}
      onTouchEnd={handleMouseUp}
      disabled={isDisabled}
      className={getButtonStyle()}
      aria-label={getButtonText()}
    >
      {getButtonText()}
    </button>
  );
}