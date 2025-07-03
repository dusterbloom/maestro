'use client';

import { useState } from 'react';
import VoiceButton from '@/components/VoiceButton';
import StatusIndicator from '@/components/StatusIndicator';
import Waveform from '@/components/Waveform';

export default function Home() {
  const [status, setStatus] = useState<'idle' | 'connecting' | 'connected' | 'recording' | 'processing' | 'error'>('idle');
  const [error, setError] = useState<string>('');
  const [lastTranscript, setLastTranscript] = useState<string>('');
  
  const handleStatusChange = (newStatus: typeof status) => {
    setStatus(newStatus);
    if (newStatus !== 'error') {
      setError('');
    }
  };
  
  const handleError = (errorMessage: string) => {
    setError(errorMessage);
  };
  
  const handleTranscript = (transcript: string) => {
    setLastTranscript(transcript);
  };
  
  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex flex-col items-center justify-center p-6">
      <div className="max-w-md w-full text-center space-y-8">
        {/* Header */}
        <div className="space-y-2">
          <h1 className="text-4xl font-bold text-gray-800">
            Voice Assistant
          </h1>
          <p className="text-gray-600">
            Ultra-low-latency voice orchestration
          </p>
        </div>
        
        {/* Status Indicator */}
        <div className="flex justify-center">
          <StatusIndicator status={status} error={error} />
        </div>
        
        {/* Main Voice Interface */}
        <div className="relative flex justify-center py-8">
          <VoiceButton 
            onStatusChange={handleStatusChange}
            onTranscript={handleTranscript}
            onError={handleError}
          />
          <Waveform 
            isRecording={status === 'recording'}
            audioLevel={0.5} // This could be connected to real audio level detection
          />
        </div>
        
        {/* Transcript Display */}
        {lastTranscript && (
          <div className="bg-white/70 rounded-lg p-4 backdrop-blur-sm border border-blue-200">
            <p className="font-medium text-blue-800 mb-2">You said:</p>
            <p className="text-gray-700 italic">"{lastTranscript}"</p>
          </div>
        )}
        
        {/* Instructions */}
        <div className="space-y-4 text-sm text-gray-600">
          <div className="bg-white/50 rounded-lg p-4 backdrop-blur-sm">
            <p className="font-medium mb-2">How to use:</p>
            <ol className="text-left space-y-1 list-decimal list-inside">
              <li>Wait for the "Ready" status</li>
              <li>Hold the button and speak clearly</li>
              <li>Release when finished</li>
              <li>Listen for the AI response</li>
            </ol>
          </div>
          
          {status === 'error' && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-red-700">
              <p className="font-medium">Troubleshooting:</p>
              <ul className="text-xs mt-1 space-y-1 list-disc list-inside">
                <li>Check your microphone permissions</li>
                <li>Ensure a stable internet connection</li>
                <li>Try refreshing the page</li>
              </ul>
            </div>
          )}
        </div>
        
        {/* Footer Info */}
        <div className="text-xs text-gray-500 space-y-1">
          <p>Powered by WhisperLive + Ollama + Kokoro</p>
          <p>Target latency: &lt;500ms</p>
        </div>
      </div>
    </main>
  );
}