import { create } from 'zustand';
import { devtools, subscribeWithSelector } from 'zustand/middleware';

export type VoiceState = {
  isRecording: boolean;
  isPlaying: boolean;
  isConnected: boolean;
  transcript: string;
  response: string;
  audioLevel: number;
  sessionId: string | null;
  error: string | null;
  isProcessing: boolean;
};

export type VoiceActions = {
  setRecording: (recording: boolean) => void;
  setPlaying: (playing: boolean) => void;
  setConnected: (connected: boolean) => void;
  setTranscript: (transcript: string) => void;
  setResponse: (response: string) => void;
  setAudioLevel: (level: number) => void;
  setSessionId: (sessionId: string | null) => void;
  setError: (error: string | null) => void;
  setProcessing: (processing: boolean) => void;
  reset: () => void;
};

const initialState: VoiceState = {
  isRecording: false,
  isPlaying: false,
  isConnected: false,
  transcript: '',
  response: '',
  audioLevel: 0,
  sessionId: null,
  error: null,
  isProcessing: false,
};

export const useVoiceStore = create<VoiceState & VoiceActions>()(
  devtools(
    subscribeWithSelector((set) => ({
      ...initialState,
      
      setRecording: (recording) => set({ isRecording: recording }),
      setPlaying: (playing) => set({ isPlaying: playing }),
      setConnected: (connected) => set({ isConnected: connected }),
      setTranscript: (transcript) => set({ transcript }),
      setResponse: (response) => set({ response }),
      setAudioLevel: (level) => set({ audioLevel: level }),
      setSessionId: (sessionId) => set({ sessionId }),
      setError: (error) => set({ error }),
      setProcessing: (processing) => set({ isProcessing: processing }),
      reset: () => set(initialState),
    })),
    {
      name: 'voice-store',
    }
  )
);

// Selectors for better performance
export const selectIsRecording = (state: VoiceState) => state.isRecording;
export const selectIsPlaying = (state: VoiceState) => state.isPlaying;
export const selectIsConnected = (state: VoiceState) => state.isConnected;
export const selectTranscript = (state: VoiceState) => state.transcript;
export const selectResponse = (state: VoiceState) => state.response;
export const selectAudioLevel = (state: VoiceState) => state.audioLevel;
export const selectSessionId = (state: VoiceState) => state.sessionId;
export const selectError = (state: VoiceState) => state.error;
export const selectIsProcessing = (state: VoiceState) => state.isProcessing;