'use client';

import { useVoiceStore } from '@/stores/voice';
import { useEffect, useState } from 'react';

interface TranscriptDisplayProps {
  className?: string;
}

export default function EnhancedTranscriptDisplay({ className = '' }: TranscriptDisplayProps) {
  const transcript = useVoiceStore(state => state.transcript);
  const isRecording = useVoiceStore(state => state.isRecording);
  const isProcessing = useVoiceStore(state => state.isProcessing);
  
  const [transcriptHistory, setTranscriptHistory] = useState<string[]>([]);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  useEffect(() => {
    if (transcript && transcript.trim()) {
      setTranscriptHistory(prev => [...prev, transcript]);
      setLastUpdate(new Date());
      console.log('📝 Transcript updated:', transcript);
    }
  }, [transcript]);

  // Debug logging
  useEffect(() => {
    console.log('🔍 Transcript Display State:', {
      transcript,
      isRecording,
      isProcessing,
      history: transcriptHistory,
      lastUpdate: lastUpdate.toISOString()
    });
  }, [transcript, isRecording, isProcessing, transcriptHistory]);

  if (!transcript && transcriptHistory.length === 0) {
    return (
      <div className={`text-center text-gray-500 ${className}`}>
        <p className="text-sm italic">No transcript yet - start speaking to see live transcription</p>
      </div>
    );
  }

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Current Transcript */}
      {transcript && (
        <div className="bg-white/90 rounded-lg p-4 backdrop-blur-sm border border-blue-300 shadow-lg">
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm font-semibold text-blue-600">Live Transcript</p>
            {isRecording && (
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                <span className="text-xs text-red-600">Recording</span>
              </div>
            )}
            {isProcessing && (
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-yellow-500 rounded-full animate-spin"></div>
                <span className="text-xs text-yellow-600">Processing</span>
              </div>
            )}
          </div>
          <p className="text-lg text-gray-800 font-medium">"{transcript}"</p>
          <p className="text-xs text-gray-500 mt-1">
            Updated: {lastUpdate.toLocaleTimeString()}
          </p>
        </div>
      )}

      {/* Transcript History */}
      {transcriptHistory.length > 0 && (
        <div className="bg-gray-50/80 rounded-lg p-3 backdrop-blur-sm">
          <p className="text-sm font-medium text-gray-600 mb-2">History</p>
          <div className="space-y-1 max-h-32 overflow-y-auto">
            {transcriptHistory.slice(-5).map((t, index) => (
              <div key={index} className="text-sm text-gray-700 p-1 bg-white/50 rounded">
                {t}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}