'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { MaestroWebSocket } from '@/lib/websocket';
import { AudioRecorder, AudioPlayer } from '@/lib/audio';

// Define the status types for clarity
type Status = 'idle' | 'connecting' | 'connected' | 'recording' | 'processing' | 'error';

interface VoiceButtonProps {
  onStatusChange?: (status: Status) => void;
  onTranscript?: (transcript: string) => void;
  onError?: (error: string) => void;
  sessionId?: string;
}

export default function VoiceButton({ onStatusChange, onTranscript, onError, sessionId }: VoiceButtonProps) {
  const [status, setStatus] = useState<Status>('idle');
  const [isRecording, setIsRecording] = useState(false);
  
  const maestroWsRef = useRef<MaestroWebSocket | null>(null);
  const recorderRef = useRef<AudioRecorder | null>(null);
  const playerRef = useRef<AudioPlayer | null>(null);
  const sessionIdRef = useRef<string>(sessionId || `session_${Date.now()}`);

  const updateStatus = useCallback((newStatus: Status) => {
    setStatus(newStatus);
    onStatusChange?.(newStatus);
  }, [onStatusChange]);

  const handleError = useCallback((error: string) => {
    console.error('VoiceButton Error:', error);
    updateStatus('error');
    onError?.(error);
  }, [onError, updateStatus]);

  useEffect(() => {
    let mounted = true;

    async function initialize() {
      if (!mounted) return;
      updateStatus('connecting');

      try {
        // Initialize Maestro WebSocket
        const maestroWs = new MaestroWebSocket(sessionIdRef.current);
        maestroWsRef.current = maestroWs;

        maestroWs.onConnect(() => {
          if (!mounted) return;
          updateStatus('connected');
          console.log('Maestro WebSocket connected!');
          // Send a ping to test the connection
          maestroWs.send({ type: 'ping', data: { message: 'Hello from client!' } });
        });

        maestroWs.onError((error) => {
          if (mounted) handleError(`WebSocket Error: ${error}`);
        });

        maestroWs.onDisconnect(() => {
          if (mounted) {
            updateStatus('idle');
            console.log('Maestro WebSocket disconnected.');
          }
        });

        maestroWs.onEvent((event) => {
          if (!mounted) return;
          console.log('Received event from orchestrator:', event);
          // Handle events from the orchestrator
          switch (event.type) {
            case 'session.ready':
              console.log('Session is ready:', event.data);
              break;
            case 'transcript.partial':
              onTranscript?.(event.data.text);
              break;
            case 'response.audio.chunk':
              // This will be handled in the next phase
              // playerRef.current?.play(event.data.audio);
              break;
            case 'session.error':
              handleError(event.data.message);
              break;
            default:
              console.warn('Received unknown event type:', event.type);
          }
        });

        maestroWs.connect();

        // Initialize audio recorder and player
        const recorder = new AudioRecorder();
        await recorder.initialize();
        recorderRef.current = recorder;

        const player = new AudioPlayer();
        playerRef.current = player;

      } catch (err) {
        if (mounted) {
          handleError(err instanceof Error ? err.message : 'Initialization failed');
        }
      }
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
    maestroWsRef.current?.send({ type: 'audio.start' });
    
    recorderRef.current.start((audioData) => {
      maestroWsRef.current?.send({
        type: 'audio.chunk',
        data: audioData.buffer, // Sending raw buffer
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
      case 'idle': return 'Connecting...';
      case 'connecting': return 'Connecting...';
      case 'connected': return isRecording ? 'Stop' : 'Record';
      case 'recording': return 'Stop';
      case 'processing': return 'Processing...';
      case 'error': return 'Error';
      default: return 'Record';
    }
  };

  return (
    <button onClick={handleToggleRecording} disabled={status !== 'connected' && status !== 'recording'}>
      {getButtonText()}
    </button>
  );
}