'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { VoicePipelineService, VoicePipelineState } from '@/lib/voice-pipeline-service';

interface VoiceButtonProps {
  onStatusChange?: (status: VoicePipelineState) => void;
  onTranscript?: (transcript: string) => void;
  onError?: (error: string) => void;
}

/**
 * Production-grade VoiceButton using Service Layer Architecture
 * 
 * This component is now a thin UI layer that delegates all complex
 * state management to the VoicePipelineService, eliminating React
 * rendering issues and circular dependencies.
 */
export default function VoiceButton({ onStatusChange, onTranscript, onError }: VoiceButtonProps) {
  const [status, setStatus] = useState<VoicePipelineState>('idle');
  const [audioLevel, setAudioLevel] = useState<number>(0);
  const serviceRef = useRef<VoicePipelineService | null>(null);
  const initializationRef = useRef<boolean>(false);
  
  // Initialize service once on mount
  useEffect(() => {
    if (initializationRef.current) return;
    initializationRef.current = true;
    
    let mounted = true;
    
    async function initializeService() {
      try {
        // Ensure we're in browser environment to prevent hydration issues
        if (typeof window === 'undefined') {
          return;
        }
        // Service configuration
        const config = {
          whisperWsUrl: process.env.NEXT_PUBLIC_WHISPER_WS_URL || 'ws://localhost:9090',
          orchestratorUrl: process.env.NEXT_PUBLIC_ORCHESTRATOR_URL || 'http://localhost:8000',
          voiceActivityThreshold: 0.1,
          maxSilenceMs: 1000
        };
        
        // Create service instance
        const service = new VoicePipelineService(config);
        serviceRef.current = service;
        
        // Subscribe to service events
        service.on('state-changed', (newState) => {
          if (mounted) {
            setStatus(newState);
            onStatusChange?.(newState);
          }
        });
        
        service.on('transcript-received', (transcript) => {
          if (mounted) {
            onTranscript?.(transcript);
          }
        });
        
        service.on('error', (error) => {
          if (mounted) {
            onError?.(error);
          }
        });
        
        service.on('audio-level', (level) => {
          if (mounted) {
            setAudioLevel(level);
          }
        });
        
        // Initialize the service
        const initialized = await service.initialize();
        if (!initialized && mounted) {
          console.error('Failed to initialize voice pipeline service');
        }
        
      } catch (error) {
        if (mounted) {
          console.error('Service initialization error:', error);
        }
      }
    }
    
    initializeService();
    
    return () => {
      mounted = false;
      
      // Cleanup service
      if (serviceRef.current) {
        serviceRef.current.destroy();
        serviceRef.current = null;
      }
    };
  }, [onStatusChange, onTranscript, onError]);
  
  // Handle recording toggle
  const handleToggleRecording = useCallback(async () => {
    if (serviceRef.current?.isReady()) {
      await serviceRef.current.toggleRecording();
    }
  }, []);
  
  // Get button text based on state
  const getButtonText = useCallback(() => {
    switch (status) {
      case 'idle':
        return 'Initializing...';
      case 'connecting':
        return 'Connecting...';
      case 'connected':
        return 'Start Recording';
      case 'recording':
        return 'Stop Recording';
      case 'processing':
        return 'Processing...';
      case 'playing':
        return 'Speaking...';
      case 'error':
        return 'Error - Retry';
      default:
        return 'Start Recording';
    }
  }, [status]);
  
  // Get button styling based on state
  const getButtonStyle = useCallback(() => {
    const baseClasses = "w-32 h-32 rounded-full transition-all duration-200 font-bold text-lg shadow-lg relative overflow-hidden";
    
    switch (status) {
      case 'idle':
      case 'connecting':
        return `${baseClasses} bg-gray-400 text-white cursor-not-allowed opacity-70`;
      case 'connected':
        return `${baseClasses} bg-blue-500 hover:bg-blue-600 text-white cursor-pointer`;
      case 'recording':
        return `${baseClasses} bg-red-500 scale-110 text-white cursor-pointer animate-pulse`;
      case 'processing':
        return `${baseClasses} bg-yellow-500 text-white cursor-not-allowed animate-pulse`;
      case 'playing':
        return `${baseClasses} bg-green-500 text-white cursor-not-allowed animate-pulse`;
      case 'error':
        return `${baseClasses} bg-red-600 hover:bg-red-700 text-white cursor-pointer`;
      default:
        return `${baseClasses} bg-gray-400 text-white cursor-not-allowed`;
    }
  }, [status]);
  
  // Check if button should be disabled
  const isDisabled = useCallback(() => {
    return !serviceRef.current?.isReady() || status === 'processing' || status === 'playing';
  }, [status]);
  
  // Voice activity visualization
  const renderVoiceActivity = useCallback(() => {
    if (status !== 'recording' && status !== 'connected') return null;
    
    const intensity = Math.min(audioLevel * 10, 1); // Scale audio level
    const opacity = 0.3 + (intensity * 0.7); // Dynamic opacity
    
    return (
      <div 
        className="absolute inset-0 bg-white rounded-full pointer-events-none"
        style={{ 
          opacity: status === 'recording' ? opacity : 0,
          transform: `scale(${1 + intensity * 0.1})`,
          transition: 'opacity 0.1s ease-out, transform 0.1s ease-out'
        }}
      />
    );
  }, [status, audioLevel]);
  
  return (
    <div className="flex flex-col items-center space-y-4">
      <button
        onClick={handleToggleRecording}
        disabled={isDisabled()}
        className={getButtonStyle()}
        aria-label={getButtonText()}
      >
        {renderVoiceActivity()}
        <span className="relative z-10">
          {getButtonText()}
        </span>
      </button>
      
      {/* Status indicator */}
      <div className="text-sm text-gray-600 text-center">
        <div className="font-medium">Status: {status}</div>
        {status === 'recording' && (
          <div className="text-xs">
            Audio Level: {(audioLevel * 100).toFixed(0)}%
          </div>
        )}
      </div>
      
      {/* Debug info in development */}
      {process.env.NODE_ENV === 'development' && (
        <div className="text-xs text-gray-500 text-center max-w-xs">
          Service Ready: {serviceRef.current?.isReady() ? 'Yes' : 'No'}
        </div>
      )}
    </div>
  );
}