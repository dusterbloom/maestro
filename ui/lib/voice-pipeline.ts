import { EventEmitter } from 'events';
import { useVoiceStore } from '@/stores/voice';

export interface PipelineEvent {
  type: string;
  data: any;
  timestamp: number;
}

export class VoicePipeline extends EventEmitter {
  private ws: WebSocket | null = null;
  private sessionId: string;
  private audioRecorder: MediaRecorder | null = null;
  private audioStream: MediaStream | null = null;
  private audioContext: AudioContext | null = null;
  private isRecording = false;
  
  constructor() {
    super();
    this.sessionId = `session_${Date.now()}`;
  }

  async initialize(): Promise<void> {
    try {
      await this.initializeAudio();
      await this.connectToOrchestrator();
      this.setupStateSync();
    } catch (error) {
      this.emit('error', { message: `Initialization failed: ${error}` });
      throw error;
    }
  }

  private async initializeAudio(): Promise<void> {
    this.audioStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: 16000,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      }
    });

    // Create audio context for raw PCM processing (matching WhisperLive Chrome extension)
    this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    const source = this.audioContext.createMediaStreamSource(this.audioStream);
    
    // Use ScriptProcessorNode for raw audio access (same as WhisperLive Chrome extension)
    const processor = this.audioContext.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = (event) => {
      if (this.isRecording && this.ws?.readyState === WebSocket.OPEN) {
        const inputData = event.inputBuffer.getChannelData(0);
        
        // Resample to 16kHz if needed (same as WhisperLive extension)
        const audioData16kHz = this.resampleTo16kHZ(inputData, this.audioContext!.sampleRate);
        
        // Send Float32Array data exactly like WhisperLive Chrome extension
        // WhisperLive expects raw Float32Array data directly
        console.log(`🎤 Sending ${audioData16kHz.length} float32 samples (${audioData16kHz.buffer.byteLength} bytes) to orchestrator`);
        
        // Ensure buffer is properly aligned (length must be multiple of 4 for float32)
        if (audioData16kHz.buffer.byteLength % 4 !== 0) {
          console.warn(`⚠️ Buffer size ${audioData16kHz.buffer.byteLength} is not aligned to 4-byte boundary`);
          // Create aligned buffer
          const alignedLength = Math.floor(audioData16kHz.buffer.byteLength / 4) * 4;
          const alignedBuffer = audioData16kHz.buffer.slice(0, alignedLength);
          this.ws.send(alignedBuffer);
        } else {
          this.ws.send(audioData16kHz.buffer);
        }
      }
    };
    
    // Connect audio nodes (same as WhisperLive extension)
    source.connect(processor);
    processor.connect(this.audioContext.destination);
    
    this.audioRecorder = null; // Not using MediaRecorder
  }

  // Note: Removed PCM conversion - WhisperLive expects Float32Array directly

  // Resample function from WhisperLive Chrome extension
  private resampleTo16kHZ(audioData: Float32Array, origSampleRate: number = 44100): Float32Array {
    const targetSampleRate = 16000;
    
    // If already 16kHz, return as-is
    if (origSampleRate === targetSampleRate) {
      return audioData;
    }
    
    // Calculate the desired length of the resampled data
    const targetLength = Math.round(audioData.length * (targetSampleRate / origSampleRate));
    
    // Create a new Float32Array for the resampled data
    const resampledData = new Float32Array(targetLength);
    
    // Calculate the spring factor and initialize the first and last values
    const springFactor = (audioData.length - 1) / (targetLength - 1);
    resampledData[0] = audioData[0];
    resampledData[targetLength - 1] = audioData[audioData.length - 1];
    
    // Resample the audio data
    for (let i = 1; i < targetLength - 1; i++) {
      const index = i * springFactor;
      const leftIndex = Math.floor(index);
      const rightIndex = Math.ceil(index);
      const fraction = index - leftIndex;
      resampledData[i] = audioData[leftIndex] + (audioData[rightIndex] - audioData[leftIndex]) * fraction;
    }
    
    return resampledData;
  }

  // Note: Removed convertWebMToPCM - using direct Float32Array streaming instead

  private resampleAudio(data: Float32Array, fromRate: number, toRate: number): Float32Array {
    if (fromRate === toRate) return data;
    
    const ratio = fromRate / toRate;
    const outputLength = Math.floor(data.length / ratio);
    const output = new Float32Array(outputLength);
    
    for (let i = 0; i < outputLength; i++) {
      const sourceIndex = i * ratio;
      const sourceIndexFloor = Math.floor(sourceIndex);
      const sourceIndexCeil = Math.min(sourceIndexFloor + 1, data.length - 1);
      const fraction = sourceIndex - sourceIndexFloor;
      
      // Linear interpolation
      output[i] = data[sourceIndexFloor] * (1 - fraction) + data[sourceIndexCeil] * fraction;
    }
    
    return output;
  }

  private async connectToOrchestrator(): Promise<void> {
    const wsUrl = process.env.NEXT_PUBLIC_ORCHESTRATOR_WS_URL || 'ws://localhost:8000';
    const fullUrl = `${wsUrl}/ws/voice`;
    
    this.ws = new WebSocket(fullUrl);
    
    return new Promise((resolve, reject) => {
      if (!this.ws) {
        reject(new Error('Failed to create WebSocket'));
        return;
      }

      this.ws.onopen = () => {
        console.log('🔗 WebSocket connected to orchestrator:', fullUrl);
        this.emit('connected');
        resolve();
      };

      this.ws.onmessage = (event) => {
        if (typeof event.data === 'string') {
          try {
            const message = JSON.parse(event.data);
            this.handleOrchestratorMessage(message);
          } catch (error) {
            console.error('Failed to parse orchestrator message:', error);
          }
        }
      };

      this.ws.onclose = (event) => {
        this.emit('disconnected');
      };

      this.ws.onerror = (error) => {
        this.emit('error', { message: 'WebSocket connection failed' });
        reject(error);
      };

      setTimeout(() => {
        if (this.ws?.readyState !== WebSocket.OPEN) {
          reject(new Error('Connection timeout'));
        }
      }, 10000);
    });
  }

  private handleOrchestratorMessage(message: any): void {
    switch (message.type) {
      case 'ready':
        if (message.mode === 'ultra_fast') {
          this.sessionId = message.session_id;
        }
        this.emit('ready');
        break;

      case 'live_transcript':
        this.emit('transcript', { text: message.text, isFinal: false });
        break;

      case 'processing_started':
        this.emit('processing', { started: true, text: message.text });
        break;

      case 'sentence_audio':
        this.emit('audio', { 
          sequence: message.sequence,
          text: message.text,
          audioData: message.audio_data,
          sizeBytes: message.size_bytes
        });
        break;

      case 'processing_complete':
        this.emit('processing', { started: false });
        break;

      case 'interrupted':
        this.emit('interrupted');
        break;

      case 'error':
        this.emit('error', { message: message.message });
        break;
    }
  }

  private setupStateSync(): void {
    this.on('connected', () => {
      useVoiceStore.getState().setConnected(true);
      useVoiceStore.getState().setError(null);
    });

    this.on('disconnected', () => {
      useVoiceStore.getState().setConnected(false);
    });

    this.on('transcript', (data) => {
      useVoiceStore.getState().setTranscript(data.text);
    });

    this.on('processing', (data) => {
      useVoiceStore.getState().setProcessing(data.started);
    });

    this.on('audio', async (data) => {
      await this.playAudioSequentially(data);
    });

    this.on('error', (data) => {
      useVoiceStore.getState().setError(data.message);
    });

    this.on('interrupted', () => {
      useVoiceStore.getState().setProcessing(false);
      useVoiceStore.getState().setPlaying(false);
    });
  }

  async startRecording(): Promise<void> {
    if (!this.audioContext || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('Pipeline not ready for recording');
    }

    this.isRecording = true;
    useVoiceStore.getState().setRecording(true);
    this.emit('recording_started');
  }

  stopRecording(): void {
    if (this.isRecording) {
      this.isRecording = false;
      
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'end_audio' }));
      }
    }
    
    useVoiceStore.getState().setRecording(false);
    this.emit('recording_stopped');
  }

  interrupt(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'interrupt' }));
      this.emit('interrupt_sent');
    }
  }

  sendUltraFastText(text: string): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ 
        type: 'ultra_fast_text', 
        text: text 
      }));
    }
  }

  private async playAudioSequentially(data: any): Promise<void> {
    try {
      useVoiceStore.getState().setPlaying(true);

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
        useVoiceStore.getState().setPlaying(false);
        URL.revokeObjectURL(audioUrl);
      };

      audio.onerror = (error) => {
        useVoiceStore.getState().setPlaying(false);
        URL.revokeObjectURL(audioUrl);
      };

      await audio.play();
      
    } catch (error) {
      useVoiceStore.getState().setPlaying(false);
    }
  }

  async cleanup(): Promise<void> {
    this.removeAllListeners();
    
    if (this.audioRecorder && this.isRecording) {
      this.audioRecorder.stop();
    }
    
    if (this.audioStream) {
      this.audioStream.getTracks().forEach(track => track.stop());
    }
    
    if (this.audioContext && this.audioContext.state !== 'closed') {
      await this.audioContext.close();
    }
    
    if (this.ws) {
      this.ws.close(1000, 'Client cleanup');
    }
    
    useVoiceStore.getState().setRecording(false);
    useVoiceStore.getState().setPlaying(false);
    useVoiceStore.getState().setConnected(false);
    useVoiceStore.getState().setProcessing(false);
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  get currentSessionId(): string {
    return this.sessionId;
  }

  get isUltraFastMode(): boolean {
    return this.ws?.url.includes('/ws/voice') || false;
  }
}

let voicePipeline: VoicePipeline | null = null;

export const getVoicePipeline = (): VoicePipeline => {
  if (!voicePipeline) {
    voicePipeline = new VoicePipeline();
  }
  return voicePipeline;
};

export const resetVoicePipeline = (): void => {
  if (voicePipeline) {
    voicePipeline.cleanup();
    voicePipeline = null;
  }
};