
'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { MaestroWebSocket } from '@/lib/websocket';
import { AudioRecorder, AudioPlayer } from '@/lib/audio';

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
        const maestroWs = new MaestroWebSocket(sessionIdRef.current);
        maestroWsRef.current = maestroWs;

        maestroWs.onConnect(() => {
          if (mounted) updateStatus('connected');
        });

        maestroWs.onError((error) => {
          if (mounted) handleError(`WebSocket Error: ${error}`);
        });

        maestroWs.onDisconnect(() => {
          if (mounted) updateStatus('idle');
        });

        maestroWs.onEvent((event) => {
          if (!mounted) return;
          
          switch (event.type) {
            case 'session.ready':
              console.log('Session ready:', event.data.session_id);
              break;
            case 'transcript.final':
              onTranscript?.(event.data.transcript);
              break;
            case 'response.audio.chunk':
              console.log('🔊 Received audio chunk:', event.data.audio_chunk?.length, 'characters');
              try {
                const audioBytes = Uint8Array.from(atob(event.data.audio_chunk), c => c.charCodeAt(0));
                console.log('🔊 Decoded audio chunk:', audioBytes.length, 'bytes');
                
                if (playerRef.current) {
                  console.log('🔊 Playing audio chunk...');
                  await playerRef.current.play(audioBytes.buffer);
                  console.log('🔊 Audio chunk played successfully');
                } else {
                  console.error('🔊 Audio player not available');
                }
              } catch (audioError) {
                console.error('🔊 Audio playback error:', audioError);
                handleError(`Audio playback failed: ${audioError.message}`);
              }
              break;
            case 'session.error':
              handleError(event.data.message);
              break;
          }
        });

        maestroWs.connect();

        const recorder = new AudioRecorder();
        await recorder.initialize();
        recorderRef.current = recorder;

        const player = new AudioPlayer();
        playerRef.current = player;

      } catch (err) {
        if (mounted) handleError(err instanceof Error ? err.message : 'Initialization failed');
      }
    }

    initialize();

    return () => {
      mounted = false;
      maestroWsRef.current?.disconnect();
      recorderRef.current?.cleanup();
      playerRef.current?.cleanup();
    };
  }, []); // Remove dependencies to prevent reconnection loops

  const startRecording = useCallback(() => {
    if (status !== 'connected' || !recorderRef.current) return;
    
    setIsRecording(true);
    updateStatus('recording');
    
    recorderRef.current.start((audioData) => {
        const reader = new FileReader();
        reader.onload = () => {
            if (typeof reader.result === 'string') {
                const base64Audio = reader.result.split(',')[1];
                maestroWsRef.current?.send({
                    type: 'audio.chunk',
                    data: { audio_chunk: base64Audio },
                });
            }
        };
        reader.readAsDataURL(new Blob([audioData]));
    });
  }, [status, updateStatus]);

  const stopRecording = useCallback(() => {
    if (!isRecording || !recorderRef.current) return;
    setIsRecording(false);
    updateStatus('processing');
    recorderRef.current.stop();
  }, [isRecording, updateStatus]);

  const handleToggleRecording = useCallback(() => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  }, [isRecording, startRecording, stopRecording]);

  return (
    <button onClick={handleToggleRecording} disabled={status !== 'connected' && status !== 'recording'}>
      {isRecording ? 'Stop' : 'Record'}
    </button>
  );
}
