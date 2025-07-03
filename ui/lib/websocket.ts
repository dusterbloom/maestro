export class VoiceWebSocket {
  private ws: WebSocket | null = null;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 3;
  private reconnectDelay = 2000;
  
  // Sentence detection for transcript buffering
  private transcriptBuffer = '';
  private sentenceTimer: NodeJS.Timeout | null = null;
  private readonly sentenceTimeoutMs = 2000; // 2 seconds timeout for sentence completion
  private lastProcessedSentence = ''; // Track the last processed sentence to prevent duplicates
  
  private onConnectCallback?: () => void;
  private onAudioCallback?: (audio: ArrayBuffer) => void;
  private onTranscriptCallback?: (transcript: string) => void;
  private onSentenceCallback?: (sentence: string) => void;
  private onErrorCallback?: (error: string) => void;
  private onDisconnectCallback?: () => void;
  
  constructor(private url: string) {}
  
  connect(sessionId?: string) {
    try {
      const wsUrl = sessionId ? `${this.url}?session_id=${sessionId}` : this.url;
      console.log('Connecting to WebSocket URL:', wsUrl);
      this.ws = new WebSocket(wsUrl);
      this.ws.binaryType = 'arraybuffer';
      
      this.ws.onopen = () => {
        console.log('WebSocket connected to WhisperLive');
        this.reconnectAttempts = 0;
        
        // Send initial config exactly like WhisperLive expects (must be first message)
        const config = {
          uid: sessionId || `session_${Date.now()}`,
          language: "en",
          task: "transcribe", 
          model: "tiny",
          use_vad: true,
          max_clients: 4,
          max_connection_time: 600,
          send_last_n_segments: 10,
          no_speech_thresh: 0.45,
          clip_audio: false,
          same_output_threshold: 10
        };
        this.ws?.send(JSON.stringify(config));
        console.log('Sent config to WhisperLive:', config);
        
        // Don't call onConnectCallback here - wait for SERVER_READY
      };
      
      this.ws.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) {
          // Audio response from TTS (not expected from WhisperLive)
          this.onAudioCallback?.(event.data);
        } else if (typeof event.data === 'string') {
          console.log('WhisperLive message:', event.data);
          try {
            const message = JSON.parse(event.data);
            if (message.message === 'SERVER_READY') {
              console.log('WhisperLive SERVER_READY received');
              // Stop any pending reconnection timers
              if (this.reconnectTimer) {
                clearTimeout(this.reconnectTimer);
                this.reconnectTimer = null;
              }
              this.reconnectAttempts = 0;
              this.onConnectCallback?.();
            } else if (message.segments && message.segments.length > 0) {
              // Transcription from WhisperLive
              const transcript = message.segments.map((seg: any) => seg.text || '').join(' ');
              const isCompleted = message.segments.some((seg: any) => seg.completed === true);
              
              if (transcript.trim()) {
                console.log('Transcript received:', transcript);
                this.onTranscriptCallback?.(transcript);
                // Only process transcript when WhisperLive indicates it's completed
                this.processTranscript(transcript, isCompleted);
              }
            }
          } catch (e) {
            console.warn('Non-JSON message:', event.data);
          }
        }
      };
      
      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.onErrorCallback?.('WebSocket connection error');
      };
      
      this.ws.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason);
        this.onDisconnectCallback?.();
        // No automatic reconnection
      };
    } catch (error) {
      console.error('Failed to connect:', error);
      this.onErrorCallback?.('Failed to connect to voice service');
      // No automatic reconnection
    }
  }
  
  private scheduleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      this.onErrorCallback?.('Maximum reconnection attempts reached');
      return;
    }
    
    this.reconnectTimer = setTimeout(() => {
      this.reconnectAttempts++;
      console.log(`Reconnection attempt ${this.reconnectAttempts}`);
      this.connect();
    }, this.reconnectDelay * this.reconnectAttempts);
  }
  
  sendAudio(audioData: ArrayBuffer) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(audioData);
    } else {
      console.warn('WebSocket not connected, cannot send audio');
    }
  }
  
  sendEndOfAudio() {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send("END_OF_AUDIO");
    }
  }
  
  private processTranscript(transcript: string, isCompleted: boolean = false) {
    const trimmedTranscript = transcript.trim();
    
    // Skip if this is the same transcript we already have in buffer
    if (trimmedTranscript === this.transcriptBuffer) {
      return;
    }
    
    // Update buffer with latest transcript
    this.transcriptBuffer = trimmedTranscript;
    
    // Clear existing sentence timer
    if (this.sentenceTimer) {
      clearTimeout(this.sentenceTimer);
    }
    
    // If WhisperLive indicates transcription is completed, process immediately
    if (isCompleted) {
      console.log('WhisperLive transcription completed:', this.transcriptBuffer);
      this.emitSentence(this.transcriptBuffer);
      this.transcriptBuffer = '';
      return;
    }
    
    // For incomplete transcripts, only use timeout as backup - don't process on punctuation
    // This prevents processing the same sentence multiple times while WhisperLive is still transcribing
    this.sentenceTimer = setTimeout(() => {
      if (this.transcriptBuffer.trim()) {
        // Only process if we have a sentence-like structure or significant content
        const sentenceEndPattern = /[.!?]\s*$/;
        const hasMinimumWords = this.transcriptBuffer.trim().split(/\s+/).length >= 3;
        
        if (sentenceEndPattern.test(this.transcriptBuffer) && hasMinimumWords) {
          console.log('Sentence timeout - emitting:', this.transcriptBuffer);
          this.emitSentence(this.transcriptBuffer);
          this.transcriptBuffer = '';
        } else {
          console.log('Incomplete sentence on timeout, waiting longer:', this.transcriptBuffer);
          // Extend timeout for incomplete sentences
          this.sentenceTimer = setTimeout(() => {
            if (this.transcriptBuffer.trim()) {
              console.log('Extended timeout - emitting:', this.transcriptBuffer);
              this.emitSentence(this.transcriptBuffer);
              this.transcriptBuffer = '';
            }
          }, this.sentenceTimeoutMs);
        }
      }
    }, this.sentenceTimeoutMs);
  }
  
  private emitSentence(sentence: string) {
    const trimmedSentence = sentence.trim();
    
    // Prevent duplicate processing of the same sentence
    if (trimmedSentence === this.lastProcessedSentence) {
      console.log('Duplicate sentence detected, skipping:', trimmedSentence);
      return;
    }
    
    // Prevent processing sentences that are just extensions of the previous one
    // This catches cases like "Hello there." → "Hello there, can you tell me?"
    if (this.lastProcessedSentence && trimmedSentence.startsWith(this.lastProcessedSentence.replace(/[.!?]+$/, '').trim())) {
      console.log('Sentence extension detected, skipping:', trimmedSentence);
      return;
    }
    
    this.lastProcessedSentence = trimmedSentence;
    console.log('Complete sentence received:', trimmedSentence);
    this.onSentenceCallback?.(trimmedSentence);
  }

  disconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.sentenceTimer) {
      clearTimeout(this.sentenceTimer);
      this.sentenceTimer = null;
    }
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    this.resetSentenceTracking();
  }
  
  resetSentenceTracking() {
    this.transcriptBuffer = '';
    this.lastProcessedSentence = '';
    if (this.sentenceTimer) {
      clearTimeout(this.sentenceTimer);
      this.sentenceTimer = null;
    }
  }
  
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
  
  onConnect(callback: () => void) {
    this.onConnectCallback = callback;
  }
  
  onAudio(callback: (audio: ArrayBuffer) => void) {
    this.onAudioCallback = callback;
  }
  
  onTranscript(callback: (transcript: string) => void) {
    this.onTranscriptCallback = callback;
  }
  
  onSentence(callback: (sentence: string) => void) {
    this.onSentenceCallback = callback;
  }
  
  onError(callback: (error: string) => void) {
    this.onErrorCallback = callback;
  }
  
  onDisconnect(callback: () => void) {
    this.onDisconnectCallback = callback;
  }
}