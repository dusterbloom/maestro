import { useVoiceStore } from '@/stores/voice';
import { getOrchestratorWS } from './orchestrator-ws';

export interface AudioServiceConfig {
  sampleRate?: number;
  channels?: number;
  bufferSize?: number;
  vadThreshold?: number;
  vadAggressiveness?: number;
}

export interface AudioMetrics {
  level: number;
  isSpeaking: boolean;
  duration: number;
}

export class AudioService {
  private mediaRecorder: MediaRecorder | null = null;
  private audioContext: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private audioStream: MediaStream | null = null;
  private recordedChunks: Blob[] = [];
  private isRecording = false;
  private audioWorklet: AudioWorkletNode | null = null;
  private vadProcessor: AudioWorkletNode | null = null;
  
  private config: Required<AudioServiceConfig>;
  private metricsInterval: NodeJS.Timeout | null = null;

  constructor(config: AudioServiceConfig = {}) {
    this.config = {
      sampleRate: config.sampleRate || 16000,
      channels: config.channels || 1,
      bufferSize: config.bufferSize || 4096,
      vadThreshold: config.vadThreshold || 0.02,
      vadAggressiveness: config.vadAggressiveness || 3,
    };
  }

  async initialize(): Promise<void> {
    try {
      // Request microphone access
      this.audioStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: this.config.sampleRate,
          channelCount: this.config.channels,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      // Create audio context
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: this.config.sampleRate,
      });

      // Create analyser for audio level monitoring
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 256;
      this.analyser.smoothingTimeConstant = 0.8;

      // Connect audio source to analyser
      this.source = this.audioContext.createMediaStreamSource(this.audioStream);
      this.source.connect(this.analyser);

      // Start monitoring audio levels
      this.startAudioLevelMonitoring();

      console.log('Audio service initialized successfully');
    } catch (error) {
      console.error('Failed to initialize audio service:', error);
      throw new Error(`Failed to access microphone: ${error}`);
    }
  }

  async startRecording(): Promise<void> {
    if (!this.audioStream) {
      await this.initialize();
    }

    if (!this.audioStream || !this.audioContext) {
      throw new Error('Audio service not initialized');
    }

    try {
      // Create MediaRecorder
      this.mediaRecorder = new MediaRecorder(this.audioStream, {
        mimeType: 'audio/webm;codecs=opus',
      });

      this.recordedChunks = [];

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.recordedChunks.push(event.data);
          this.processAudioChunk(event.data);
        }
      };

      this.mediaRecorder.onstart = () => {
        this.isRecording = true;
        useVoiceStore.getState().setRecording(true);
        console.log('Recording started');
      };

      this.mediaRecorder.onstop = () => {
        this.isRecording = false;
        useVoiceStore.getState().setRecording(false);
        console.log('Recording stopped');
      };

      // Start recording with timeslice for real-time processing
      this.mediaRecorder.start(100); // Collect data every 100ms
    } catch (error) {
      console.error('Failed to start recording:', error);
      throw new Error(`Failed to start recording: ${error}`);
    }
  }

  stopRecording(): void {
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop();
    }
  }

  async playAudio(base64Audio: string): Promise<void> {
    try {
      useVoiceStore.getState().setPlaying(true);

      // Decode base64 audio
      const audioData = atob(base64Audio);
      const arrayBuffer = new ArrayBuffer(audioData.length);
      const uint8Array = new Uint8Array(arrayBuffer);
      
      for (let i = 0; i < audioData.length; i++) {
        uint8Array[i] = audioData.charCodeAt(i);
      }

      // Create audio blob and play
      const audioBlob = new Blob([uint8Array], { type: 'audio/webm;codecs=opus' });
      const audioUrl = URL.createObjectURL(audioBlob);
      
      const audio = new Audio(audioUrl);
      
      audio.onended = () => {
        useVoiceStore.getState().setPlaying(false);
        URL.revokeObjectURL(audioUrl);
      };

      audio.onerror = (error) => {
        console.error('Audio playback error:', error);
        useVoiceStore.getState().setPlaying(false);
        useVoiceStore.getState().setError('Failed to play audio');
      };

      await audio.play();
    } catch (error) {
      console.error('Failed to play audio:', error);
      useVoiceStore.getState().setPlaying(false);
      useVoiceStore.getState().setError(`Failed to play audio: ${error}`);
    }
  }

  getAudioBlob(): Blob | null {
    if (this.recordedChunks.length === 0) {
      return null;
    }

    return new Blob(this.recordedChunks, { type: 'audio/webm;codecs=opus' });
  }

  getAudioArrayBuffer(): Promise<ArrayBuffer | null> {
    const blob = this.getAudioBlob();
    if (!blob) {
      return Promise.resolve(null);
    }

    return blob.arrayBuffer();
  }

  private processAudioChunk(blob: Blob): void {
    // Convert blob to ArrayBuffer and send via WebSocket
    blob.arrayBuffer().then((arrayBuffer) => {
      const ws = getOrchestratorWS();
      if (ws.isConnected) {
        ws.sendAudioData(arrayBuffer, this.config.sampleRate, this.config.channels);
      }
    }).catch((error) => {
      console.error('Failed to process audio chunk:', error);
    });
  }

  private startAudioLevelMonitoring(): void {
    if (!this.analyser) return;

    const dataArray = new Uint8Array(this.analyser.frequencyBinCount);

    this.metricsInterval = setInterval(() => {
      if (!this.analyser) return;

      this.analyser.getByteFrequencyData(dataArray);
      
      // Calculate audio level (0-100)
      const average = dataArray.reduce((sum, value) => sum + value, 0) / dataArray.length;
      const level = Math.round((average / 255) * 100);
      
      // Simple VAD (Voice Activity Detection)
      const isSpeaking = level > this.config.vadThreshold * 100;
      
      useVoiceStore.getState().setAudioLevel(level);
    }, 100); // Update 10 times per second
  }

  private stopAudioLevelMonitoring(): void {
    if (this.metricsInterval) {
      clearInterval(this.metricsInterval);
      this.metricsInterval = null;
    }
  }

  getCurrentMetrics(): AudioMetrics {
    const level = useVoiceStore.getState().audioLevel;
    return {
      level,
      isSpeaking: level > this.config.vadThreshold * 100,
      duration: 0, // TODO: Calculate actual duration
    };
  }

  async cleanup(): Promise<void> {
    this.stopAudioLevelMonitoring();
    
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop();
    }

    if (this.audioContext) {
      await this.audioContext.close();
      this.audioContext = null;
    }

    if (this.audioStream) {
      this.audioStream.getTracks().forEach(track => track.stop());
      this.audioStream = null;
    }

    this.mediaRecorder = null;
    this.analyser = null;
    this.source = null;
    this.audioWorklet = null;
    this.vadProcessor = null;
    
    this.recordedChunks = [];
    this.isRecording = false;
    
    useVoiceStore.getState().setRecording(false);
    useVoiceStore.getState().setPlaying(false);
    useVoiceStore.getState().setAudioLevel(0);
  }

  get isRecordingActive(): boolean {
    return this.isRecording;
  }

  get isInitialized(): boolean {
    return this.audioStream !== null && this.audioContext !== null;
  }
}

// Singleton instance
let audioService: AudioService | null = null;

export const getAudioService = (): AudioService => {
  if (!audioService) {
    audioService = new AudioService();
  }
  return audioService;
};

export const resetAudioService = (): void => {
  if (audioService) {
    audioService.cleanup();
    audioService = null;
  }
};