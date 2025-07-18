'use client';

import { useState, useEffect, useRef } from 'react';
import { useVoiceStore } from '@/stores/voice';
import EnhancedTranscriptDisplay from './EnhancedTranscriptDisplay';

export default function WorkingHandsFreeInterface() {
  const [isInitialized, setIsInitialized] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [lastResponse, setLastResponse] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);

  // Voice store state
  const transcript = useVoiceStore(state => state.transcript);
  const isProcessing = useVoiceStore(state => state.isProcessing);
  const storeError = useVoiceStore(state => state.error);

  useEffect(() => {
    initializeSystem();
    
    return () => {
      cleanup();
    };
  }, []);

  const initializeSystem = async () => {
    try {
      setError(null);
      
      // Initialize WebSocket connection
      const wsUrl = process.env.NEXT_PUBLIC_ORCHESTRATOR_WS_URL || 'ws://localhost:8000';
      const fullUrl = `${wsUrl}/ws/voice`;
      
      wsRef.current = new WebSocket(fullUrl);
      
      wsRef.current.onopen = () => {
        console.log('🔗 WebSocket connected to orchestrator:', fullUrl);
        setIsInitialized(true);
      };

      wsRef.current.onmessage = (event) => {
        if (typeof event.data === 'string') {
          try {
            const message = JSON.parse(event.data);
            handleMessage(message);
          } catch (error) {
            console.error('Failed to parse message:', error);
          }
        }
      };

      wsRef.current.onclose = () => {
        console.log('WebSocket disconnected');
        setIsInitialized(false);
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setError('Failed to connect to voice service');
      };

    } catch (err) {
      console.error('Failed to initialize:', err);
      setError(err instanceof Error ? err.message : 'Failed to initialize voice system');
    }
  };

  const handleMessage = (message: any) => {
    console.log('📨 Received:', message);
    
    switch (message.type) {
      case 'ready':
        console.log('✅ System ready, session:', message.session_id);
        break;
        
      case 'live_transcript':
        useVoiceStore.getState().setTranscript(message.text);
        break;
        
      case 'processing_started':
        useVoiceStore.getState().setProcessing(true);
        break;
        
      case 'processing_complete':
        useVoiceStore.getState().setProcessing(false);
        break;
        
      case 'sentence_audio':
        console.log('🔊 Playing TTS response:', message.text);
        setLastResponse(message.text);
        playAudioResponse(message.audio_data);
        break;
        
      case 'interrupted':
        useVoiceStore.getState().setProcessing(false);
        break;
        
      case 'error':
        console.error('❌ Error:', message.message);
        setError(message.message);
        break;
    }
  };

  const startListening = async () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setError('Not connected to voice service');
      return;
    }

    try {
      // Get microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });

      mediaStreamRef.current = stream;
      
      // Create audio context and processing
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      sourceRef.current = audioContextRef.current.createMediaStreamSource(stream);
      scriptProcessorRef.current = audioContextRef.current.createScriptProcessor(4096, 1, 1);

      scriptProcessorRef.current.onaudioprocess = (event) => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          const inputData = event.inputBuffer.getChannelData(0);
          
          // Convert to 16-bit PCM
          const pcmData = new Int16Array(inputData.length);
          for (let i = 0; i < inputData.length; i++) {
            pcmData[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
          }
          
          // Send raw PCM data
          wsRef.current.send(pcmData.buffer);
        }
      };

      sourceRef.current.connect(scriptProcessorRef.current);
      scriptProcessorRef.current.connect(audioContextRef.current.destination);
      
      setIsListening(true);
      useVoiceStore.getState().setRecording(true);
      console.log('🎤 Continuous listening started');
      
    } catch (err) {
      console.error('Failed to start listening:', err);
      setError(err instanceof Error ? err.message : 'Failed to access microphone');
    }
  };

  const stopListening = () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
    }
    
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
    
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'end_audio' }));
    }
    
    setIsListening(false);
    useVoiceStore.getState().setRecording(false);
    console.log('🛑 Listening stopped');
  };

  const playAudioResponse = async (base64Audio: string) => {
    try {
      const audioData = atob(base64Audio);
      const arrayBuffer = new ArrayBuffer(audioData.length);
      const uint8Array = new Uint8Array(arrayBuffer);
      
      for (let i = 0; i < audioData.length; i++) {
        uint8Array[i] = audioData.charCodeAt(i);
      }

      const audioBlob = new Blob([uint8Array], { type: 'audio/wav' });
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);

      audio.onended = () => {
        console.log('✅ TTS response finished');
        URL.revokeObjectURL(audioUrl);
      };

      audio.onerror = (error) => {
        console.error('❌ TTS playback error:', error);
        URL.revokeObjectURL(audioUrl);
      };

      await audio.play();
      
    } catch (error) {
      console.error('❌ Error playing TTS:', error);
    }
  };

  const cleanup = () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
    }
    
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
    
    if (wsRef.current) {
      wsRef.current.close();
    }
    
    useVoiceStore.getState().setRecording(false);
    useVoiceStore.getState().setProcessing(false);
  };

  const toggleListening = () => {
    if (isListening) {
      stopListening();
    } else {
      startListening();
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
          onClick={initializeSystem}
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
          <span className="ml-2 text-gray-600">Initializing voice system...</span>
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
              <div className={`w-2 h-2 rounded-full ${isListening ? 'bg-green-500' : 'bg-gray-400'}`}></div>
              <span className="text-sm text-gray-600">{isListening ? 'Listening' : 'Ready'}</span>
            </div>
            <button
              onClick={toggleListening}
              className={`px-4 py-2 rounded-lg text-white transition-colors ${
                isListening 
                  ? 'bg-red-600 hover:bg-red-700' 
                  : 'bg-green-600 hover:bg-green-700'
              }`}
            >
              {isListening ? 'Stop' : 'Start'}
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
                ⏸️ Ready - Click "Start" to begin hands-free conversation
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