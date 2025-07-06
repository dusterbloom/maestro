'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { AudioRecorder } from '@/lib/audio';

interface VoiceButtonProps {
  onStatusChange: (status: 'idle' | 'connecting' | 'connected' | 'recording' | 'processing' | 'error') => void;
  onAudioData: (audioData: ArrayBuffer) => void;
  onError: (error: string) => void;
}

export default function VoiceButton({ onStatusChange, onAudioData, onError }: VoiceButtonProps) {
  const [isRecording, setIsRecording] = useState(false);
  const recorderRef = useRef<AudioRecorder | null>(null);

  useEffect(() => {
    recorderRef.current = new AudioRecorder();
    recorderRef.current.initialize().catch(onError);
    return () => {
      recorderRef.current?.cleanup();
    };
  }, [onError]);

  const startRecording = useCallback(() => {
    if (recorderRef.current) {
      setIsRecording(true);
      onStatusChange('recording');
      recorderRef.current.start((audioData) => {
        onAudioData(audioData.buffer);
      });
    }
  }, [onStatusChange, onAudioData]);

  const stopRecording = useCallback(() => {
    if (recorderRef.current) {
      setIsRecording(false);
      onStatusChange('processing');
      recorderRef.current.stop();
    }
  }, [onStatusChange]);

  const handleMouseDown = () => {
    startRecording();
  };

  const handleMouseUp = () => {
    stopRecording();
  };

  return (
    <button
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
      onTouchStart={handleMouseDown}
      onTouchEnd={handleMouseUp}
      className={`w-32 h-32 rounded-full transition-all duration-200 font-bold text-lg shadow-lg ${isRecording ? 'bg-red-500 scale-110' : 'bg-blue-500'}`}>
      {isRecording ? 'Recording...' : 'Hold to Speak'}
    </button>
  );
}
