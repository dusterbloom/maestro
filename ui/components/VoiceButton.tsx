'use client';

import { useEffect, useState } from 'react';
import { useVoiceStore } from '@/stores/voice';
import { getVoicePipeline } from '@/lib/voice-pipeline';

interface VoiceButtonProps {
  onStatusChange?: (status: 'idle' | 'connecting' | 'connected' | 'recording' | 'processing' | 'error') => void;
  onTranscript?: (transcript: string) => void;
  onError?: (error: string) => void;
}

export default function VoiceButton({ onStatusChange, onTranscript, onError }: VoiceButtonProps) {
  const [isInitialized, setIsInitialized] = useState(false);
  
  const isRecording = useVoiceStore(state => state.isRecording);
  const isConnected = useVoiceStore(state => state.isConnected);
  const isPlaying = useVoiceStore(state => state.isPlaying);
  const isProcessing = useVoiceStore(state => state.isProcessing);
  const transcript = useVoiceStore(state => state.transcript);
  const error = useVoiceStore(state => state.error);

  const status = error ? 'error' : 
                !isConnected ? 'connecting' : 
                isRecording ? 'recording' : 
                isProcessing ? 'processing' : 'connected';

  useEffect(() => {
    const pipeline = getVoicePipeline();
    
    pipeline.initialize()
      .then(() => {
        setIsInitialized(true);
      })
      .catch((err) => {
        useVoiceStore.getState().setError(err.message);
      });

    const handleReady = () => {
      console.log('Pipeline ready');
    };

    const handleError = (data: any) => {
      onError?.(data.message);
    };

    pipeline.on('ready', handleReady);
    pipeline.on('error', handleError);

    return () => {
      pipeline.off('ready', handleReady);
      pipeline.off('error', handleError);
    };
  }, [onError]);

  useEffect(() => {
    onStatusChange?.(status);
  }, [status, onStatusChange]);

  useEffect(() => {
    if (transcript) {
      onTranscript?.(transcript);
    }
  }, [transcript, onTranscript]);

  const handleToggleRecording = async () => {
    if (!isInitialized) return;

    const pipeline = getVoicePipeline();
    
    try {
      if (isRecording) {
        pipeline.stopRecording();
      } else {
        await pipeline.startRecording();
      }
    } catch (err) {
      useVoiceStore.getState().setError(err instanceof Error ? err.message : 'Recording failed');
    }
  };

  const handleInterrupt = () => {
    if (!isInitialized) return;
    
    const pipeline = getVoicePipeline();
    pipeline.interrupt();
  };

  const getButtonConfig = () => {
    if (error) {
      return {
        text: 'Error - Retry',
        className: 'bg-red-600 animate-bounce text-white',
        disabled: false
      };
    }
    
    if (!isConnected) {
      return {
        text: 'Connecting...',
        className: 'bg-gray-400 animate-pulse text-white',
        disabled: true
      };
    }
    
    if (isRecording) {
      return {
        text: 'Stop',
        className: 'bg-red-500 scale-110 animate-pulse text-white',
        disabled: false
      };
    }
    
    if (isProcessing) {
      return {
        text: 'Processing...',
        className: 'bg-yellow-500 animate-pulse text-white',
        disabled: true
      };
    }
    
    return {
      text: 'Start',
      className: 'bg-blue-500 hover:bg-blue-600 text-white',
      disabled: false
    };
  };

  const { text, className, disabled } = getButtonConfig();

  return (
    <div className="relative">
      <button
        onClick={handleToggleRecording}
        disabled={disabled || !isInitialized}
        className={`w-32 h-32 rounded-full font-bold text-lg shadow-lg transition-all ${className}`}
      >
        {text}
      </button>

      {(isPlaying || isProcessing) && (
        <button
          onClick={handleInterrupt}
          className="absolute -right-16 top-1/2 -translate-y-1/2 bg-red-500 hover:bg-red-600 text-white px-3 py-2 rounded-full text-sm"
        >
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