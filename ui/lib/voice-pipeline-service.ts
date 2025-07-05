/**
 * Production-grade Voice Pipeline State Service
 * 
 * This service handles complex audio pipeline state management outside React's
 * rendering cycle to prevent circular dependencies and improve performance.
 * 
 * Architecture:
 * - State Machine: Clear state transitions with validation
 * - Event-driven: Publish/subscribe pattern for loose coupling
 * - Service Layer: Business logic separated from UI components
 * - Async-safe: Handles concurrent operations and cleanup
 */

import { AudioRecorder, AudioPlayer } from './audio';
import { VoiceWebSocket } from './websocket';

// State machine states
export type VoicePipelineState = 
  | 'idle'
  | 'connecting'
  | 'connected'
  | 'recording'
  | 'processing'
  | 'playing'
  | 'error';

// Audio queue item
export interface AudioQueueItem {
  sequence: number;
  audioData: string;
  text: string;
}

// Events emitted by the service
export interface VoicePipelineEvents {
  'state-changed': (state: VoicePipelineState) => void;
  'transcript-received': (transcript: string) => void;
  'error': (error: string) => void;
  'audio-level': (level: number) => void;
  'speaker-detected': (speakerId: string) => void;
}

// Service configuration
export interface VoicePipelineConfig {
  whisperWsUrl: string;
  orchestratorUrl: string;
  voiceActivityThreshold: number;
  maxSilenceMs: number;
}

/**
 * Voice Pipeline Service - Production State Management
 * 
 * Handles all audio pipeline state transitions and business logic
 * outside of React components to prevent rendering issues.
 */
export class VoicePipelineService {
  private state: VoicePipelineState = 'idle';
  private sessionId: string = '';
  private eventListeners: Map<keyof VoicePipelineEvents, Set<Function>> = new Map();
  
  // Audio components
  private whisperWs: VoiceWebSocket | null = null;
  private recorder: AudioRecorder | null = null;
  private player: AudioPlayer | null = null;
  
  // Audio queue management
  private audioQueue: AudioQueueItem[] = [];
  private nextToPlay: number = 1;
  private currentStreamController: AbortController | null = null;
  
  // Voice activity detection
  private voiceActivityRef: boolean = false;
  private bargeInTimeoutRef: NodeJS.Timeout | null = null;
  
  // Speaker recognition
  private speakerId: string | null = null;
  private embeddingAttempts: number = 0;
  private speakerRecognized: boolean = false;
  
  // Service state
  private isInitialized: boolean = false;
  private isDestroyed: boolean = false;
  
  constructor(private config: VoicePipelineConfig) {
    // Initialize sessionId to empty - will be set during initialization to prevent hydration mismatch
    this.sessionId = '';
    
    // Initialize event listener maps
    const eventKeys: (keyof VoicePipelineEvents)[] = [
      'state-changed',
      'transcript-received', 
      'error',
      'audio-level',
      'speaker-detected'
    ];
    
    eventKeys.forEach(event => {
      this.eventListeners.set(event, new Set());
    });
  }
  
  /**
   * Initialize the voice pipeline service
   */
  async initialize(): Promise<boolean> {
    if (this.isInitialized || this.isDestroyed) {
      return false;
    }
    
    try {
      // Generate session ID on client side only to prevent hydration mismatch
      if (typeof window !== 'undefined') {
        this.sessionId = `session_${Date.now()}`;
      } else {
        // On server side, use a placeholder that will be replaced on client
        this.sessionId = 'session_placeholder';
      }
      
      this.setState('connecting');
      
      // Initialize WebSocket connection
      this.whisperWs = new VoiceWebSocket(this.config.whisperWsUrl);
      await this.setupWebSocketHandlers();
      
      // Initialize audio recorder
      this.recorder = new AudioRecorder();
      const recorderInitialized = await this.recorder.initialize();
      if (!recorderInitialized) {
        throw new Error('Failed to initialize audio recorder');
      }
      
      this.setupRecorderHandlers();
      
      // Initialize audio player
      this.player = new AudioPlayer();
      this.setupPlayerHandlers();
      
      // Connect to WhisperLive
      this.whisperWs.connect(this.sessionId);
      
      this.isInitialized = true;
      return true;
      
    } catch (error) {
      this.handleError(error instanceof Error ? error.message : 'Initialization failed');
      return false;
    }
  }
  
  /**
   * Setup WebSocket event handlers
   */
  private async setupWebSocketHandlers(): Promise<void> {
    if (!this.whisperWs) return;
    
    this.whisperWs.onConnect(() => {
      this.setState('connected');
    });
    
    this.whisperWs.onError((error) => {
      this.handleError(`WhisperLive: ${error}`);
    });
    
    this.whisperWs.onDisconnect(() => {
      if (this.state !== 'error') {
        this.setState('idle');
      }
    });
    
    this.whisperWs.onTranscript((transcript) => {
      this.emit('transcript-received', transcript);
    });
    
    this.whisperWs.onSentence(async (sentence) => {
      await this.processSentence(sentence);
    });
  }
  
  /**
   * Setup audio recorder event handlers
   */
  private setupRecorderHandlers(): void {
    if (!this.recorder) return;
    
    // Voice activity detection for barge-in
    this.recorder.onAudioLevel((level) => {
      this.emit('audio-level', level);
      
      const isVoiceActive = this.recorder!.isVoiceActive();
      this.voiceActivityRef = isVoiceActive;
      
      // Immediate barge-in detection
      const hasPendingAudio = this.audioQueue.length > 0;
      if (isVoiceActive && (this.state === 'playing' || hasPendingAudio) && this.state !== 'recording') {
        this.handleBargeIn();
      }
    });
  }
  
  /**
   * Setup audio player event handlers
   */
  private setupPlayerHandlers(): void {
    if (!this.player) return;
    
    this.player.onPlaybackStart(() => {
      this.setState('playing');
    });
    
    this.player.onPlaybackEnd(() => {
      if (this.audioQueue.length === 0) {
        this.setState('connected');
      }
    });
  }
  
  /**
   * State machine with validation
   */
  private setState(newState: VoicePipelineState): void {
    if (this.isDestroyed) return;
    
    // Validate state transitions
    if (!this.isValidStateTransition(this.state, newState)) {
      console.warn(`Invalid state transition: ${this.state} -> ${newState}`);
      return;
    }
    
    const oldState = this.state;
    this.state = newState;
    
    console.log(`State transition: ${oldState} -> ${newState}`);
    this.emit('state-changed', newState);
  }
  
  /**
   * Validate state transitions
   */
  private isValidStateTransition(from: VoicePipelineState, to: VoicePipelineState): boolean {
    const validTransitions: Record<VoicePipelineState, VoicePipelineState[]> = {
      'idle': ['connecting', 'error'],
      'connecting': ['connected', 'error'],
      'connected': ['recording', 'processing', 'playing', 'error', 'idle'],
      'recording': ['processing', 'connected', 'error'],
      'processing': ['playing', 'connected', 'error'],
      'playing': ['connected', 'recording', 'error'],
      'error': ['idle', 'connecting']
    };
    
    return validTransitions[from]?.includes(to) ?? false;
  }
  
  /**
   * Process a complete sentence
   */
  private async processSentence(sentence: string): Promise<void> {
    if (this.state === 'playing') {
      console.log('🚨 BLOCKING NEW TTS: Audio is currently playing');
      return;
    }
    
    this.setState('processing');
    
    // Abort any ongoing stream
    this.abortCurrentStream();
    
    try {
      const streamController = new AbortController();
      this.currentStreamController = streamController;
      
      const response = await fetch(`${this.config.orchestratorUrl}/process-transcript`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Use-Streaming': 'true'
        },
        body: JSON.stringify({
          transcript: sentence,
          session_id: this.sessionId,
          speaker_id: this.speakerId,
          speaker_recognized: this.speakerRecognized,
          embedding_attempts: this.embeddingAttempts,
        }),
        signal: streamController.signal
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      
      await this.processStreamingResponse(response);
      
    } catch (error: any) {
      if (error.name === 'AbortError') {
        console.log('TTS request aborted due to barge-in');
      } else {
        this.handleError('Failed to process sentence');
      }
    } finally {
      if (this.currentStreamController) {
        this.currentStreamController = null;
      }
    }
  }
  
  /**
   * Process streaming response from orchestrator
   */
  private async processStreamingResponse(response: Response): Promise<void> {
    if (!response.body) {
      throw new Error('No response stream available');
    }
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    
    // Reset audio queue
    this.audioQueue = [];
    this.nextToPlay = 1;
    
    while (true) {
      const { done, value } = await reader.read();
      
      if (done) break;
      
      if (this.currentStreamController?.signal.aborted) {
        reader.cancel();
        break;
      }
      
      if (value) {
        buffer += decoder.decode(value, { stream: true });
        
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.type === 'sentence_audio') {
                this.audioQueue.push({
                  sequence: data.sequence,
                  audioData: data.audio_data,
                  text: data.text
                });
                
                // Start playing if this is the next expected sentence
                if (data.sequence === this.nextToPlay) {
                  this.playNextInQueue();
                }
              }
            } catch (e) {
              console.error('Failed to parse streaming data:', e);
            }
          }
        }
      }
    }
  }
  
  /**
   * Play next item in audio queue
   */
  private async playNextInQueue(): Promise<void> {
    if (this.state === 'playing') return;
    
    const nextItem = this.audioQueue.find(item => item.sequence === this.nextToPlay);
    if (!nextItem) return;
    
    this.setState('playing');
    
    try {
      const audioBytes = Uint8Array.from(atob(nextItem.audioData), c => c.charCodeAt(0));
      
      if (this.player) {
        await this.player.play(audioBytes.buffer);
      }
      
      // Remove played item
      const index = this.audioQueue.findIndex(item => item.sequence === this.nextToPlay);
      if (index >= 0) {
        this.audioQueue.splice(index, 1);
      }
      this.nextToPlay++;
      
      // Check if more audio is queued
      if (this.audioQueue.length === 0) {
        this.setState('connected');
      } else {
        // Small delay between sentences
        setTimeout(() => this.playNextInQueue(), 50);
      }
      
    } catch (error) {
      console.error('Failed to play audio:', error);
      this.nextToPlay++;
      if (this.audioQueue.length === 0) {
        this.setState('connected');
      } else {
        setTimeout(() => this.playNextInQueue(), 50);
      }
    }
  }
  
  /**
   * Handle barge-in (user interruption)
   */
  private async handleBargeIn(): Promise<void> {
    console.log('🚨 BARGE-IN: User interrupt detected');
    
    // Stop all audio immediately
    this.abortCurrentStream();
    if (this.player) {
      this.player.stopAll();
    }
    
    // Clear audio queue
    this.audioQueue = [];
    this.nextToPlay = 1;
    
    // Start recording if not already recording
    if (this.state !== 'recording' && this.state !== 'error') {
      this.startRecording();
    }
  }
  
  /**
   * Abort current streaming
   */
  private abortCurrentStream(): void {
    if (this.currentStreamController) {
      this.currentStreamController.abort();
      this.currentStreamController = null;
    }
  }
  
  /**
   * Start recording
   */
  async startRecording(): Promise<void> {
    if (!this.whisperWs?.isConnected() || !this.recorder) {
      this.handleError('Services not ready');
      return;
    }
    
    try {
      this.setState('recording');
      
      this.recorder.start((audioData: Float32Array) => {
        if (this.whisperWs?.isConnected()) {
          this.whisperWs.sendAudio(audioData.buffer);
        }
      });
      
    } catch (error) {
      this.handleError('Failed to start recording');
    }
  }
  
  /**
   * Stop recording
   */
  async stopRecording(): Promise<void> {
    if (!this.recorder || !this.whisperWs) return;
    
    try {
      this.recorder.stop();
      this.whisperWs.sendEndOfAudio();
      
    } catch (error) {
      this.handleError('Failed to stop recording');
    }
  }
  
  /**
   * Toggle recording state
   */
  async toggleRecording(): Promise<void> {
    if (this.state === 'recording') {
      await this.stopRecording();
    } else if (this.state === 'connected') {
      await this.startRecording();
    }
  }
  
  /**
   * Handle errors
   */
  private handleError(message: string): void {
    console.error('Voice Pipeline Error:', message);
    this.setState('error');
    this.emit('error', message);
  }
  
  /**
   * Event emitter
   */
  private emit<K extends keyof VoicePipelineEvents>(event: K, ...args: Parameters<VoicePipelineEvents[K]>): void {
    const listeners = this.eventListeners.get(event);
    if (listeners) {
      listeners.forEach(listener => {
        try {
          (listener as any)(...args);
        } catch (error) {
          console.error(`Error in event listener for ${event}:`, error);
        }
      });
    }
  }
  
  /**
   * Subscribe to events
   */
  on<K extends keyof VoicePipelineEvents>(event: K, listener: VoicePipelineEvents[K]): void {
    const listeners = this.eventListeners.get(event);
    if (listeners) {
      listeners.add(listener);
    }
  }
  
  /**
   * Unsubscribe from events
   */
  off<K extends keyof VoicePipelineEvents>(event: K, listener: VoicePipelineEvents[K]): void {
    const listeners = this.eventListeners.get(event);
    if (listeners) {
      listeners.delete(listener);
    }
  }
  
  /**
   * Get current state
   */
  getState(): VoicePipelineState {
    return this.state;
  }
  
  /**
   * Check if service is ready
   */
  isReady(): boolean {
    return this.isInitialized && !this.isDestroyed && this.state !== 'error';
  }
  
  /**
   * Cleanup and destroy service
   */
  async destroy(): Promise<void> {
    if (this.isDestroyed) return;
    
    this.isDestroyed = true;
    
    // Cleanup timeouts
    if (this.bargeInTimeoutRef) {
      clearTimeout(this.bargeInTimeoutRef);
      this.bargeInTimeoutRef = null;
    }
    
    // Abort any ongoing streams
    this.abortCurrentStream();
    
    // Cleanup audio components
    if (this.whisperWs) {
      this.whisperWs.disconnect();
      this.whisperWs = null;
    }
    
    if (this.recorder) {
      this.recorder.cleanup();
      this.recorder = null;
    }
    
    if (this.player) {
      this.player.cleanup();
      this.player = null;
    }
    
    // Clear event listeners
    this.eventListeners.clear();
    
    console.log('Voice Pipeline Service destroyed');
  }
}