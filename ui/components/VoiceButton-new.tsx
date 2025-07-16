// components/VoiceButton-new.tsx - Ultra-slim presentation component
'use client';

import { useEffect } from 'react';
import { useVoiceStore, selectIsRecording, selectIsConnected, selectIsPlaying, selectError } from '@/stores/voice';
import { getAudioService } from '@/lib/audio-service';
import { getOrchestratorWS } from '@/lib/orchestrator-ws';

interface VoiceButtonProps {
  onStatusChange?: (status: any) => void;
  onTranscript?: (transcript: string) => void;
  onError?: (error: string) => void;
}

export default function VoiceButton({ onStatusChange, onTranscript, onError }: VoiceButtonProps) {
  const isRecording = useVoiceStore(selectIsRecording);
  const isConnected = useVoiceStore(selectIsConnected);
  const isPlaying = useVoiceStore(selectIsPlaying);
  const error = useVoiceStore(selectError);

  const status = error ? 'error' : isConnected ? (isRecording ? 'recording' : 'connected') : 'connecting';

  useEffect(() => {
    const audio = getAudioService();
    const ws = getOrchestratorWS();
    
    audio.initialize().catch(e => onError?.(e.message));
    ws.connect();
    
    const handleTranscript = (data: any) => onTranscript?.(data.text);
    const handleError = (data: any) => onError?.(data.message);
    
    ws.on('transcript', handleTranscript);
    ws.on('error', handleError);
    
    return () => {
      ws.off('transcript', handleTranscript);
      ws.off('error', handleError);
    };
  }, [onTranscript, onError]);

  useEffect(() => onStatusChange?.(status), [status, onStatusChange]);

  const toggle = () => isRecording ? getAudioService().stopRecording() : getAudioService().startRecording();
  const bargein = () => getAudioService().cleanup();

  const text = error ? 'Error - Retry' : !isConnected ? 'Connecting...' : isRecording ? 'Stop' : 'Start';
  const style = `w-32 h-32 rounded-full font-bold text-lg shadow-lg transition-all ${error ? 'bg-red-600 animate-bounce' : !isConnected ? 'bg-gray-400 animate-pulse' : isRecording ? 'bg-red-500 scale-110 animate-pulse' : 'bg-blue-500 hover:bg-blue-600'} text-white`;

  return (
    <div className="relative">
      <button onClick={toggle} disabled={!isConnected && !error} className={style}>{text}</button>
      {isPlaying && (
        <button onClick={bargein} className="absolute -right-16 top-1/2 -translate-y-1/2 bg-red-500 hover:bg-red-600 text-white px-3 py-2 rounded-full text-sm">
          🛑 Stop
        </button>
      )}
      {isPlaying && (
        <div className="absolute -bottom-6 left-1/2 -translate-x-1/2 text-xs flex items-center gap-1">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
          <span>Playing</span>
        </div>
      )}
    </div>
  );
}