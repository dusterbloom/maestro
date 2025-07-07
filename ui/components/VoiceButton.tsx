
'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { MaestroWebSocket } from '@/lib/websocket';
import { AudioRecorder, AudioPlayer } from '@/lib/audio';

interface VoiceButtonProps {
  onStatusChange?: (status: 'idle' | 'connecting' | 'connected' | 'recording' | 'processing' | 'error') => void;
  onTranscript?: (transcript: string) => void;
  onError?: (error: string) => void;
  sessionId?: string;
}

export default function VoiceButton({ onStatusChange, onTranscript, onError, sessionId }: VoiceButtonProps) {
  const [status, setStatus] = useState<'idle' | 'connecting' | 'connected' | 'recording' | 'processing' | 'error'>('idle');
  const [isRecording, setIsRecording] = useState(false);
  
  const maestroWsRef = useRef<MaestroWebSocket | null>(null);
  const recorderRef = useRef<AudioRecorder | null>(null);
  const playerRef = useRef<AudioPlayer | null>(null);
  const sessionIdRef = useRef<string>(sessionId || `session_${Date.now()}`);

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

    async function initialize() {
      if (!mounted) return;
      updateStatus('connecting');

      // Initialize Maestro WebSocket
      const maestroWs = new MaestroWebSocket(sessionIdRef.current);
      maestroWsRef.current = maestroWs;

      maestroWs.onConnect(() => {
        if (mounted) updateStatus('connected');
      });

      maestroWs.onError((error) => {
        if (mounted) handleError(`Maestro: ${error}`);
      });

      maestroWs.onDisconnect(() => {
        if (mounted) updateStatus('idle');
      });

      maestroWs.onEvent((event) => {
        if (!mounted) return;
        // Handle events from the orchestrator
        switch (event.type) {
          case 'transcript.partial':
            onTranscript?.(event.text);
            break;
          case 'response.audio':
            playerRef.current?.play(event.audio);
            break;
          case 'error':
            handleError(event.message);
            break;
        }
      });

      maestroWs.connect();

      // Initialize audio recorder
      const recorder = new AudioRecorder();
      await recorder.initialize();
      recorderRef.current = recorder;

      // Initialize audio player
      const player = new AudioPlayer();
      playerRef.current = player;
    }

    initialize();

    return () => {
      mounted = false;
      maestroWsRef.current?.disconnect();
      recorderRef.current?.cleanup();
      playerRef.current?.cleanup();
    };
  }, [handleError, onTranscript, updateStatus]);

  const startRecording = useCallback(async () => {
    if (status !== 'connected' || !recorderRef.current) return;
    
    setIsRecording(true);
    updateStatus('recording');
    
    recorderRef.current.start((audioData) => {
      maestroWsRef.current?.send({
        type: 'audio.chunk',
        data: audioData.buffer,
      });
    });
  }, [status, updateStatus]);

  const stopRecording = useCallback(() => {
    if (!isRecording || !recorderRef.current) return;

    setIsRecording(false);
    updateStatus('processing');
    recorderRef.current.stop();
    maestroWsRef.current?.send({ type: 'audio.end' });
  }, [isRecording, updateStatus]);

  const handleToggleRecording = useCallback(() => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  }, [isRecording, startRecording, stopRecording]);

  const getButtonText = () => {
    switch (status) {
      case 'idle':
      case 'connecting':
        return 'Connecting...';
      case 'connected':
        return isRecording ? 'Stop' : 'Record';
      case 'recording':
        return 'Stop';
      case 'processing':
        return 'Processing...';
      case 'error':
        return 'Error';
      default:
        return 'Record';
    }
  };

  return (
    <button onClick={handleToggleRecording} disabled={status !== 'connected' && status !== 'recording'}>
      {getButtonText()}
    </button>
  );
}
