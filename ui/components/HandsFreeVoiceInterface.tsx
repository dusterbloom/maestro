'use client';

import { useState, useEffect, useRef } from 'react';
import { useVoiceStore } from '@/stores/voice';
import { getVoicePipeline } from '@/lib/voice-pipeline';
import EnhancedTranscriptDisplay from './EnhancedTranscriptDisplay';

export default function HandsFreeVoiceInterface() {
  const [isInitialized, setIsInitialized] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [lastResponse, setLastResponse] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const pipelineRef = useRef<ReturnType<typeof getVoicePipeline> | null>(null);
  const silenceTimerRef = useRef<NodeJS.Timeout | null>(null);
  const responseTimerRef = useRef<NodeJS.Timeout | null>(null);
  const initTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Voice store state
  const transcript = useVoiceStore(state => state.transcript);
  const isProcessing = useVoiceStore(state => state.isProcessing);
  const isConnected = useVoiceStore(state => state.isConnected);
  const storeError = useVoiceStore(state => state.error);

  useEffect(() => {
    initializePipeline();
    
    return () => {
      if (pipelineRef.current) {
        pipelineRef.current.cleanup();
      }
      if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
      if (responseTimerRef.current) clearTimeout(responseTimerRef.current);
      if (initTimerRef.current) clearTimeout(initTimerRef.current);
    };
  }, []);

  const initializePipeline = async () => {
    try {
      setError(null);
      const pipeline = getVoicePipeline();
      
      // Set up automatic response handling
      pipeline.on('audio', async (data) => {
        console.log('🔊 Playing TTS response:', data.text);
        setLastResponse(data.text);
        
        // Auto-play the response
        try {
          const audioData = atob(data.audioData);
          const arrayBuffer = new ArrayBuffer(audioData.length);
          const uint8Array = new Uint8Array(arrayBuffer);
          
          for (let i = 0; i < audioData.length; i++) {
            uint8Array[i] = audioData.charCodeAt(i);
          }

          const audioBlob = new Blob([uint8Array], { type: 'audio/wav' });
          const audioUrl = URL.createObjectURL(audioBlob);
          const audio = new Audio(audioUrl);

          audio.onended = () => {
            console.log('✅ TTS response finished, resuming listening...');
            URL.revokeObjectURL(audioUrl);
            // Resume listening after response
            setTimeout(() => {
              if (isListening && pipelineRef.current && pipelineRef.current.isConnected) {
                pipelineRef.current.startRecording().catch(console.error);
              }
            }, 500);
          };

          audio.onerror = (error) => {
            console.error('❌ TTS playback error:', error);
            URL.revokeObjectURL(audioUrl);
            // Resume listening even on error
            if (isListening && pipelineRef.current && pipelineRef.current.isConnected) {
              pipelineRef.current.startRecording().catch(console.error);
            }
          };

          await audio.play();
          
        } catch (error) {
          console.error('❌ Error playing TTS:', error);
          if (isListening && pipelineRef.current && pipelineRef.current.isConnected) {
            pipelineRef.current.startRecording().catch(console.error);
          }
        }
      });

      await pipeline.initialize();
      pipelineRef.current = pipeline;
      setIsInitialized(true);
      
      console.log('✅ Pipeline initialized successfully');
      
    } catch (err) {
      console.error('Failed to initialize:', err);
      setError(err instanceof Error ? err.message : 'Failed to initialize voice pipeline');
    }
  };

  const startContinuousListening = async () => {
    if (!pipelineRef.current || !pipelineRef.current.isConnected) {
      console.log('⏳ Waiting for pipeline to be ready...');
      // Retry after a short delay
      setTimeout(() => {
        if (pipelineRef.current && pipelineRef.current.isConnected) {
          startContinuousListening();
        }
      }, 1000);
      return;
    }
    
    try {
      setIsListening(true);
      await pipelineRef.current.startRecording();
      console.log('🎤 Continuous listening started');
    } catch (err) {
      console.error('Failed to start recording:', err);
      setError(err instanceof Error ? err.message : 'Failed to start recording');
    }
  };

  const stopContinuousListening = () => {
    if (!pipelineRef.current) return;
    
    setIsListening(false);
    pipelineRef.current.stopRecording();
    console.log('🛑 Continuous listening stopped');
  };

  // Handle voice-based interruption
  useEffect(() => {
    if (!isListening || !pipelineRef.current) return;

    // Monitor for new speech during processing
    const resetSilenceTimer = () => {
      if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
      
      silenceTimerRef.current = setTimeout(() => {
        // If there's transcript and processing is done, it's a natural break
        if (transcript && !isProcessing) {
          console.log('🎯 Natural conversation break detected');
        }
      }, 1500); // 1.5 second silence threshold
    };

    // Reset timer on transcript changes
    if (transcript) {
      resetSilenceTimer();
    }

    return () => {
      if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
    };
  }, [transcript, isListening, isProcessing]);

  // Handle processing completion
  useEffect(() => {
    if (!isListening || !pipelineRef.current) return;

    if (!isProcessing && transcript) {
      // Processing complete, resume listening after a brief pause
      responseTimerRef.current = setTimeout(() => {
        if (isListening && pipelineRef.current && pipelineRef.current.isConnected) {
          pipelineRef.current.startRecording().catch(console.error);
        }
      }, 1000);
    }

    return () => {
      if (responseTimerRef.current) clearTimeout(responseTimerRef.current);
    };
  }, [isProcessing, isListening, transcript]);

  const toggleListening = () => {
    if (isListening) {
      stopContinuousListening();
    } else {
      startContinuousListening();
    }
  };

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-6">
        <div className="flex items-center gap-2 text-red-700 mb-4">
          <span className="text-xl">⚠️</span>
          <span className="font-semibold">Voice Interface Error</span>
        </div>
        <p className="text-red-600 mb-4">{error}</p>
        <button
          onClick={initializePipeline}
          className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
        >
          Retry Connection
        </button>
      </div>
    );
  }

  if (!isInitialized) {
    return (
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-6">
        <div className="flex items-center justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <span className="ml-2 text-gray-600">Initializing hands-free voice interface...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white border border-gray-200 rounded-lg shadow-lg">
      {/* Status Bar */}
      <div className="border-b border-gray-200 p-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-800">Hands-Free Voice Assistant</h3>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className="text-sm text-gray-600">{isConnected ? 'Connected' : 'Disconnected'}</span>
            </div>
            <button
              onClick={toggleListening}
              disabled={!isConnected}
              className={`px-4 py-2 rounded-lg text-white transition-colors disabled:opacity-50 ${
                isListening 
                  ? 'bg-red-600 hover:bg-red-700' 
                  : 'bg-green-600 hover:bg-green-700'
              }`}
            >
              {isListening ? 'Stop Listening' : 'Start Listening'}
            </button>
          </div>
        </div>
      </div>

      {/* Transcript Display */}
      <div className="p-4">
        <EnhancedTranscriptDisplay />
      </div>

      {/* Status */}
      <div className="border-t border-gray-200 p-4">
        <div className="text-center">
          <div className="text-sm text-gray-600 mb-2">
            {isListening ? (
              <span className="text-green-600 font-semibold">
                🎤 Listening... Speak naturally and the system will respond
              </span>
            ) : (
              <span className="text-gray-500">
                ⏸️ Paused - Click "Start Listening" to begin
              </span>
            )}
          </div>
          
          {lastResponse && (
            <div className="text-sm text-blue-600">
              <strong>Last Response:</strong> {lastResponse}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}