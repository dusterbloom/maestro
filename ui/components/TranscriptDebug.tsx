'use client';

import { useVoiceStore } from '@/stores/voice';
import { useEffect } from 'react';

export default function TranscriptDebug() {
  const transcript = useVoiceStore(state => state.transcript);
  const isConnected = useVoiceStore(state => state.isConnected);
  const isRecording = useVoiceStore(state => state.isRecording);
  const isProcessing = useVoiceStore(state => state.isProcessing);

  // Debug logging
  useEffect(() => {
    console.log('📝 Voice Store State:', {
      transcript,
      isConnected,
      isRecording,
      isProcessing,
      timestamp: new Date().toISOString()
    });
  }, [transcript, isConnected, isRecording, isProcessing]);

  return (
    <div className="fixed top-4 right-4 bg-black/80 text-white p-4 rounded-lg max-w-sm z-50">
      <h3 className="font-bold mb-2">🎯 Transcript Debug</h3>
      <div className="text-xs space-y-1">
        <div>Connected: {isConnected ? '✅' : '❌'}</div>
        <div>Recording: {isRecording ? '🔴' : '⭕'}</div>
        <div>Processing: {isProcessing ? '⚡' : '⭕'}</div>
        <div className="mt-2">
          <strong>Transcript:</strong>
          <div className="bg-gray-700 p-2 rounded mt-1 break-all">
            {transcript || '(empty)'}
          </div>
        </div>
      </div>
    </div>
  );
}