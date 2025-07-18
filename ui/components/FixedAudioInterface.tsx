'use client';

import { useState, useEffect, useRef } from 'react';
import { useVoiceStore } from '@/stores/voice';
import EnhancedTranscriptDisplay from './EnhancedTranscriptDisplay';

export default function FixedAudioInterface() {
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [audioLevel, setAudioLevel] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const animationRef = useRef<number | null>(null);

  // Voice store state
  const transcript = useVoiceStore(state => state.transcript);
  const isProcessing = useVoiceStore(state => state.isProcessing);
  const storeError = useVoiceStore(state => state.error);

  useEffect(() => {
    connectToOrchestrator();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  const connectToOrchestrator = () => {
    try {
      const wsUrl = 'ws://localhost:8000/ws/voice';
      wsRef.current = new WebSocket(wsUrl);
      
      wsRef.current.onopen = () => {
        console.log('🔗 Connected to orchestrator:', wsUrl);
        setIsConnected(true);
        setError(null);
      };

      wsRef.current.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          console.log('📨 Received:', message);
          
          switch (message.type) {
            case 'ready':
              console.log('✅ Session ready:', message.session_id);
              break;
              
            case 'live_transcript':
              console.log('📝 Live transcript:', message.text);
              useVoiceStore.getState().setTranscript(message.text);
              break;
              
            case 'segments':
              console.log('📊 Segments received:', message.segments);
              break;
              
            case 'processing_started':
              useVoiceStore.getState().setProcessing(true);
              break;
              
            case 'processing_complete':
              useVoiceStore.getState().setProcessing(false);
              break;
              
            case 'sentence_audio':
              console.log('🔊 Playing TTS response:', message.text);
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
        } catch (error) {
          console.error('Failed to parse message:', error);
        }
      };

      wsRef.current.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setError('Failed to connect to voice service');
      };

    } catch (err) {
      console.error('Failed to connect:', err);
      setError(err instanceof Error ? err.message : 'Failed to connect');
    }
  };

  const startRecording = async () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setError('Not connected to voice service');
      return;
    }

    try {
      // Get microphone access with better constraints
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        }
      });

      // Create audio context for monitoring
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: 16000
      });
      
      sourceRef.current = audioContextRef.current.createMediaStreamSource(stream);
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 256;
      
      sourceRef.current.connect(analyserRef.current);
      
      // Start audio level monitoring
      startAudioLevelMonitoring();

      // Create MediaRecorder with proper configuration
      const options = {
        mimeType: 'audio/webm;codecs=opus',
        audioBitsPerSecond: 16000
      };

      if (!MediaRecorder.isTypeSupported(options.mimeType)) {
        console.warn('WebM/Opus not supported, trying alternatives');
        options.mimeType = 'audio/webm';
      }

      mediaRecorderRef.current = new MediaRecorder(stream, options);

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN && event.data.size > 0) {
          wsRef.current.send(event.data);
        }
      };

      mediaRecorderRef.current.onstart = () => {
        console.log('🎤 MediaRecorder started');
      };

      mediaRecorderRef.current.onstop = () => {
        console.log('🛑 MediaRecorder stopped');
        if (audioContextRef.current) {
          audioContextRef.current.close();
        }
        if (animationRef.current) {
          cancelAnimationFrame(animationRef.current);
        }
      };

      mediaRecorderRef.current.onerror = (event) => {
        console.error('❌ MediaRecorder error:', event);
        setError('Recording error occurred');
      };

      // Start recording with 100ms intervals
      mediaRecorderRef.current.start(100);
      setIsRecording(true);
      useVoiceStore.getState().setRecording(true);
      
    } catch (err) {
      console.error('Failed to start recording:', err);
      setError(err instanceof Error ? err.message : 'Failed to access microphone');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
    }
    
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'end_audio' }));
    }
    
    setIsRecording(false);
    useVoiceStore.getState().setRecording(false);
    
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
  };

  const startAudioLevelMonitoring = () => {
    if (!analyserRef.current) return;
    
    const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
    
    const updateLevel = () => {
      analyserRef.current!.getByteFrequencyData(dataArray);
      const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
      setAudioLevel(average);
      
      if (isRecording) {
        animationRef.current = requestAnimationFrame(updateLevel);
      }
    };
    
    updateLevel();
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

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  if (error || storeError) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-6">
        <div className="flex items-center gap-2 text-red-700 mb-4">
          <span className="text-xl">⚠️</span>
          <span className="font-semibold">Voice Interface Error</span>
        </div>
        <p className="text-red-600 mb-4">{error || storeError}</p>
        <button
          onClick={connectToOrchestrator}
          className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
        >
          Retry Connection
        </button>
      </div>
    );
  }

  if (!isConnected) {
    return (
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-6">
        <div className="flex items-center justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <span className="ml-2 text-gray-600">Connecting to voice service...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white border border-gray-200 rounded-lg shadow-lg">
      {/* Status Bar */}
      <div className="border-b border-gray-200 p-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-800">Voice Pipeline Interface</h3>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className="text-sm text-gray-600">{isConnected ? 'Connected' : 'Disconnected'}</span>
            </div>
            <div className="w-16 h-2 bg-gray-200 rounded">
              <div 
                className="h-full bg-blue-500 rounded transition-all duration-100" 
                style={{ width: `${Math.min(audioLevel * 2, 100)}%` }}
              ></div>
            </div>
            <button
              onClick={toggleRecording}
              className={`px-4 py-2 rounded-lg text-white transition-colors ${
                isRecording 
                  ? 'bg-red-600 hover:bg-red-700' 
                  : 'bg-green-600 hover:bg-green-700'
              }`}
            >
              {isRecording ? 'Stop Recording' : 'Start Recording'}
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
            {isRecording ? (
              <span className="text-green-600 font-semibold">
                🎤 Recording... Speak and watch live transcripts appear
              </span>
            ) : (
              <span className="text-gray-500">
                ⏸️ Ready - Click "Start Recording" to begin
              </span>
            )}
          </div>
          
          {transcript && (
            <div className="text-sm text-blue-600">
              <strong>Current:</strong> {transcript}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}