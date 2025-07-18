'use client';

import { useState, useEffect, useRef } from 'react';
import { useVoiceStore } from '@/stores/voice';
import { getVoicePipeline } from '@/lib/voice-pipeline';
import EnhancedTranscriptDisplay from './EnhancedTranscriptDisplay';

interface VoiceInterfaceProps {
  className?: string;
  onTranscriptComplete?: (transcript: string) => void;
  onResponse?: (response: string) => void;
  showDebug?: boolean;
}

export default function VoiceInterface({ 
  className = '', 
  onTranscriptComplete, 
  onResponse,
  showDebug = false 
}: VoiceInterfaceProps) {
  const [isInitialized, setIsInitialized] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pipelineRef = useRef<ReturnType<typeof getVoicePipeline> | null>(null);

  // Voice store state
  const transcript = useVoiceStore(state => state.transcript);
  const isRecording = useVoiceStore(state => state.isRecording);
  const isProcessing = useVoiceStore(state => state.isProcessing);
  const isConnected = useVoiceStore(state => state.isConnected);
  const storeError = useVoiceStore(state => state.error);

  useEffect(() => {
    initializePipeline();
    
    return () => {
      if (pipelineRef.current) {
        pipelineRef.current.cleanup();
      }
    };
  }, []);

  const initializePipeline = async () => {
    try {
      setError(null);
      const pipeline = getVoicePipeline();
      await pipeline.initialize();
      pipelineRef.current = pipeline;
      setIsInitialized(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to initialize voice pipeline');
    }
  };

  const handleStartRecording = async () => {
    if (!pipelineRef.current) return;
    
    try {
      await pipelineRef.current.startRecording();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start recording');
    }
  };

  const handleStopRecording = () => {
    if (!pipelineRef.current) return;
    
    pipelineRef.current.stopRecording();
  };

  const handleInterrupt = () => {
    if (!pipelineRef.current) return;
    
    pipelineRef.current.interrupt();
  };

  const handleRetry = () => {
    initializePipeline();
  };

  // Handle completed transcripts
  useEffect(() => {
    if (transcript && onTranscriptComplete) {
      const timer = setTimeout(() => {
        onTranscriptComplete(transcript);
      }, 1000);
      
      return () => clearTimeout(timer);
    }
  }, [transcript, onTranscriptComplete]);

  if (error || storeError) {
    return (
      <div className={`bg-red-50 border border-red-200 rounded-lg p-6 ${className}`}>
        <div className="flex items-center gap-2 text-red-700 mb-4">
          <span className="text-xl">⚠️</span>
          <span className="font-semibold">Voice Interface Error</span>
        </div>
        <p className="text-red-600 mb-4">{error || storeError}</p>
        <button
          onClick={handleRetry}
          className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
        >
          Retry Connection
        </button>
      </div>
    );
  }

  if (!isInitialized) {
    return (
      <div className={`bg-gray-50 border border-gray-200 rounded-lg p-6 ${className}`}>
        <div className="flex items-center justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <span className="ml-2 text-gray-600">Initializing voice interface...</span>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-white border border-gray-200 rounded-lg shadow-lg ${className}`}>
      {/* Header */}
      <div className="border-b border-gray-200 p-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-800">Voice Interface</h3>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
            <span className="text-sm text-gray-600">{isConnected ? 'Connected' : 'Disconnected'}</span>
          </div>
        </div>
      </div>

      {/* Transcript Display */}
      <div className="p-4">
        <EnhancedTranscriptDisplay />
      </div>

      {/* Controls */}
      <div className="border-t border-gray-200 p-4">
        <div className="flex items-center justify-center gap-4">
          {!isRecording ? (
            <button
              onClick={handleStartRecording}
              disabled={!isConnected || isProcessing}
              className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <span>🎤</span>
              Start Recording
            </button>
          ) : (
            <button
              onClick={handleStopRecording}
              className="flex items-center gap-2 px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
            >
              <span>⏹️</span>
              Stop Recording
            </button>
          )}
          
          <button
            onClick={handleInterrupt}
            disabled={!isConnected}
            className="flex items-center gap-2 px-4 py-3 bg-yellow-500 text-white rounded-lg hover:bg-yellow-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <span>⏸️</span>
            Interrupt
          </button>
        </div>

        {showDebug && (
          <div className="mt-4 p-3 bg-gray-50 rounded text-xs">
            <div className="font-mono text-gray-600">
              <div>Transcript: {transcript || 'None'}</div>
              <div>Recording: {isRecording ? 'Yes' : 'No'}</div>
              <div>Processing: {isProcessing ? 'Yes' : 'No'}</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}