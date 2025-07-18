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

    // Create audio context for raw PCM processing
    this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    const source = this.audioContext.createMediaStreamSource(this.audioStream);
    
    // Use ScriptProcessorNode for raw audio access
    const processor = this.audioContext.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = (event) => {
      if (this.isRecording && this.ws?.readyState === WebSocket.OPEN) {
        const inputData = event.inputBuffer.getChannelData(0);
        const audioData16kHz = this.resampleTo16kHZ(inputData, this.audioContext!.sampleRate);
        
        console.log(`🎤 Sending ${audioData16kHz.length} float32 samples to orchestrator`);
        
        if (audioData16kHz.buffer.byteLength % 4 !== 0) {
          const alignedLength = Math.floor(audioData16kHz.buffer.byteLength / 4) * 4;
          const alignedBuffer = audioData16kHz.buffer.slice(0, alignedLength);
          this.ws.send(alignedBuffer);
        } else {
          this.ws.send(audioData16kHz.buffer);
        }
      }
    };
    
    source.connect(processor);
    processor.connect(this.audioContext.destination);
  }

  private resampleTo16kHZ(audioData: Float32Array, origSampleRate: number = 44100): Float32Array {
    const targetSampleRate = 16000;
    
    if (origSampleRate === targetSampleRate) {
      return audioData;
    }
    
    const targetLength = Math.round(audioData.length * (targetSampleRate / origSampleRate));
    const resampledData = new Float32Array(targetLength);
    
    const springFactor = (audioData.length - 1) / (targetLength - 1);
    resampledData[0] = audioData[0];
    resampledData[targetLength - 1] = audioData[audioData.length - 1];
    
    for (let i = 1; i < targetLength - 1; i++) {
      const index = i * springFactor;
      const leftIndex = Math.floor(index);
      const rightIndex = Math.ceil(index);
      const fraction = index - leftIndex;
      resampledData[i] = audioData[leftIndex] + (audioData[rightIndex] - audioData[leftIndex]) * fraction;
    }
    
    return resampledData;
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
    console.log('📨 Received message from orchestrator:', message);
    
    switch (message.type) {
      case 'ready':
        if (message.mode === 'ultra_fast') {
          this.sessionId = message.session_id;
        }
        this.emit('ready');
        break;

      case 'live_transcript':
        // Handle live transcript updates from backend
        console.log('📝 Live transcript received:', message.text);
        useVoiceStore.getState().setTranscript(message.text);
        break;

      case 'segments':
        // Handle segment updates for debugging
        console.log('📊 Segments received:', message.segments);
        break;

      case 'processing_started':
        console.log('🔄 Processing started');
        useVoiceStore.getState().setProcessing(true);
        break;

      case 'processing_complete':
        console.log('✅ Processing complete');
        useVoiceStore.getState().setProcessing(false);
        break;

      case 'sentence_audio':
        console.log('🔊 Audio received for sequence:', message.sequence);
        this.emit('audio', { 
          sequence: message.sequence,
          text: message.text,
          audioData: message.audio_data,
          sizeBytes: message.size_bytes
        });
        break;

      case 'interrupted':
        console.log('🛑 Interrupted');
        useVoiceStore.getState().setProcessing(false);
        useVoiceStore.getState().setPlaying(false);
        break;

      case 'error':
        console.error('❌ Error from orchestrator:', message.message);
        useVoiceStore.getState().setError(message.message);
        break;

      default:
        console.log('🤷 Unknown message type:', message.type, message);
    }
  }

  private setupStateSync(): void {
    // Direct state updates from message handler
    // No need for additional event listeners since we handle directly
  }

  async startRecording(): Promise<void> {
    if (!this.audioContext || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('Pipeline not ready for recording');
    }

    this.isRecording = true;
    useVoiceStore.getState().setRecording(true);
    console.log('🎤 Recording started');
  }

  stopRecording(): void {
    if (this.isRecording) {
      this.isRecording = false;
      
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'end_audio' }));
      }
    }
    
    useVoiceStore.getState().setRecording(false);
    console.log('🛑 Recording stopped');
  }

  interrupt(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'interrupt' }));
      console.log('⚡ Interrupt sent');
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

      // Convert base64 audio data to blob
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
        console.error('Error playing audio:', error);
        useVoiceStore.getState().setPlaying(false);
        URL.revokeObjectURL(audioUrl);
      };

      await audio.play();
      
    } catch (error) {
      console.error('Error in audio playback:', error);
      useVoiceStore.getState().setPlaying(false);
    }
  }

  async cleanup(): Promise<void> {
    console.log('🧹 Cleaning up voice pipeline');
    
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