// lib/voice-orchestrator-client.ts
export class VoiceOrchestratorClient {
  private ws: WebSocket | null = null;
  private sessionId: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 3;
  private reconnectDelay = 2000;
  
  // Event callbacks
  private onReadyCallback?: () => void;
  private onLiveTranscriptCallback?: (transcript: string) => void;
  private onSentenceAudioCallback?: (data: SentenceAudioData) => void;
  private onProcessingCallback?: (isProcessing: boolean) => void;
  private onErrorCallback?: (error: string) => void;
  private onDisconnectCallback?: () => void;
  
  constructor(sessionId?: string) {
    this.sessionId = sessionId || `session_${Date.now()}`;
  }
  
  connect() {
    try {
      const wsUrl = process.env.NEXT_PUBLIC_ORCHESTRATOR_WS_URL || 'ws://localhost:8000';
      const fullUrl = `${wsUrl}/ws/${this.sessionId}`;
      
      console.log('🔌 Connecting to Voice Orchestrator:', fullUrl);
      this.ws = new WebSocket(fullUrl);
      this.ws.binaryType = 'arraybuffer';
      
      this.ws.onopen = () => {
        console.log('✅ Connected to Voice Orchestrator');
        this.reconnectAttempts = 0;
      };
      
      this.ws.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) {
          // Shouldn't receive binary data from orchestrator
          console.warn('⚠️ Unexpected binary data from orchestrator');
          return;
        }
        
        try {
          const message = JSON.parse(event.data);
          this.handleMessage(message);
        } catch (error) {
          console.error('❌ Failed to parse message:', error);
        }
      };
      
      this.ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        this.onErrorCallback?.('Connection error');
      };
      
      this.ws.onclose = (event) => {
        console.log('🔌 WebSocket closed:', event.code, event.reason);
        this.onDisconnectCallback?.();
        
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
          this.scheduleReconnect();
        }
      };
      
    } catch (error) {
      console.error('❌ Failed to connect:', error);
      this.onErrorCallback?.('Failed to connect to voice service');
    }
  }
  
  private handleMessage(message: any) {
    console.log('📨 Orchestrator message:', message.type);
    
    switch (message.type) {
      case 'ready':
        console.log('🎯 Voice Orchestrator ready');
        this.onReadyCallback?.();
        break;
        
      case 'whisper_ready':
        console.log('🎤 WhisperLive ready');
        break;
        
      case 'live_transcript':
        this.onLiveTranscriptCallback?.(message.text);
        break;
        
      case 'processing_started':
        console.log('⚙️ Processing started:', message.text);
        this.onProcessingCallback?.(true);
        break;
        
      case 'sentence_audio':
        console.log(`🎵 Received sentence ${message.sequence}: ${message.text.slice(0, 30)}...`);
        this.onSentenceAudioCallback?.({
          sequence: message.sequence,
          text: message.text,
          audioData: message.audio_data,
          sizeBytes: message.size_bytes
        });
        break;
        
      case 'processing_complete':
        console.log('✅ Processing complete');
        this.onProcessingCallback?.(false);
        break;
        
      case 'interrupted':
        console.log('🛑 Session interrupted');
        this.onProcessingCallback?.(false);
        break;
        
      case 'error':
        console.error('❌ Orchestrator error:', message.message);
        this.onErrorCallback?.(message.message);
        break;
        
      default:
        console.log('❓ Unknown message type:', message.type);
    }
  }
  
  private scheduleReconnect() {
    this.reconnectAttempts++;
    const delay = this.reconnectDelay * this.reconnectAttempts;
    
    console.log(`🔄 Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
    
    setTimeout(() => {
      this.connect();
    }, delay);
  }
  
  sendAudio(audioData: ArrayBuffer) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(audioData);
    } else {
      console.warn('⚠️ Cannot send audio - WebSocket not connected');
    }
  }
  
  sendEndOfAudio() {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'end_audio' }));
    }
  }
  
  interrupt() {
    if (this.ws?.readyState === WebSocket.OPEN) {
      console.log('🛑 Sending interrupt signal');
      this.ws.send(JSON.stringify({ type: 'interrupt' }));
    }
  }
  
  disconnect() {
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
  }
  
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
  
  getSessionId(): string {
    return this.sessionId;
  }
  
  // Event listener registration
  onReady(callback: () => void) {
    this.onReadyCallback = callback;
  }
  
  onLiveTranscript(callback: (transcript: string) => void) {
    this.onLiveTranscriptCallback = callback;
  }
  
  onSentenceAudio(callback: (data: SentenceAudioData) => void) {
    this.onSentenceAudioCallback = callback;
  }
  
  onProcessing(callback: (isProcessing: boolean) => void) {
    this.onProcessingCallback = callback;
  }
  
  onError(callback: (error: string) => void) {
    this.onErrorCallback = callback;
  }
  
  onDisconnect(callback: () => void) {
    this.onDisconnectCallback = callback;
  }
}

export interface SentenceAudioData {
  sequence: number;
  text: string;
  audioData: string; // base64 encoded
  sizeBytes: number;
}

// Simplified audio queue manager
export class SimpleAudioQueue {
  private queue: Array<{ sequence: number; audioData: string; text: string }> = [];
  private nextToPlay = 1;
  private isPlaying = false;
  private audioPlayer: AudioPlayer;
  
  constructor(audioPlayer: AudioPlayer) {
    this.audioPlayer = audioPlayer;
  }
  
  addSentence(data: SentenceAudioData) {
    this.queue.push({
      sequence: data.sequence,
      audioData: data.audioData,
      text: data.text
    });
    
    console.log(`📝 Queued sentence ${data.sequence} (queue size: ${this.queue.length})`);
    this.tryPlayNext();
  }
  
  private async tryPlayNext() {
    if (this.isPlaying) return;
    
    const nextItem = this.queue.find(item => item.sequence === this.nextToPlay);
    if (!nextItem) return;
    
    this.isPlaying = true;
    
    try {
      console.log(`🎵 Playing sentence ${nextItem.sequence}: ${nextItem.text.slice(0, 30)}...`);
      
      // Decode base64 audio
      const audioBytes = Uint8Array.from(atob(nextItem.audioData), c => c.charCodeAt(0));
      
      await this.audioPlayer.play(audioBytes.buffer);
      
      // Remove played item and advance
      this.queue = this.queue.filter(item => item.sequence !== this.nextToPlay);
      this.nextToPlay++;
      
      console.log(`✅ Sentence ${nextItem.sequence} played successfully`);
      
    } catch (error) {
      console.error(`❌ Failed to play sentence ${nextItem.sequence}:`, error);
      
      // Skip failed item
      this.queue = this.queue.filter(item => item.sequence !== this.nextToPlay);
      this.nextToPlay++;
      
    } finally {
      this.isPlaying = false;
      
      // Try to play next item
      // Try to play next item immediately (no delay for better sequencing)
      this.tryPlayNext();
    }
  }
  
  clear() {
    console.log(`🧹 Clearing audio queue: ${this.queue.length} items removed`);
    this.queue = [];
    this.nextToPlay = 1;
    this.audioPlayer.stopAll();
  }
  
  interrupt() {
    console.log('🛑 Interrupting audio playback');
    this.clear();
    this.isPlaying = false;
  }
  
  getQueueSize(): number {
    return this.queue.length;
  }
  
  isCurrentlyPlaying(): boolean {
    return this.isPlaying;
  }
}

// Re-export existing AudioPlayer interface (unchanged)
export interface AudioPlayer {
  play(audioBuffer: ArrayBuffer): Promise<void>;
  stopAll(): void;
  cleanup(): void;
}