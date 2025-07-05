export class VoiceWebSocket {
  private ws: WebSocket | null = null;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 3;
  private reconnectDelay = 2000;
  
  // Simple deduplication for completed segments
  private processedSegments = new Set<string>();
  
  private onConnectCallback?: () => void;
  private onAudioCallback?: (audio: ArrayBuffer) => void;
  private onTranscriptCallback?: (transcript: string) => void;
  private onSentenceCallback?: (sentence: string) => void;
  private onErrorCallback?: (error: string) => void;
  private onDisconnectCallback?: () => void;
  private onInterruptionAckCallback?: (success: boolean, message: string) => void;
  
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
        // Based on actual WhisperLive Client.on_open implementation
        const config = {
          uid: sessionId || `session_${Date.now()}`,
          language: "en",
          task: "transcribe", 
          model: "tiny",
          use_vad: true,
          max_clients: 4,
          max_connection_time: 600,
          send_last_n_segments: 10,
          no_speech_thresh: 0.45,   // Default from WhisperLive
          clip_audio: false,
          same_output_threshold: 10  // Default from WhisperLive
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
              // Process each completed segment individually - KISS approach
              message.segments.forEach((segment: any) => {
                if (segment.completed && segment.text && segment.text.trim()) {
                  const text = segment.text.trim();
                  
                  // Simple deduplication - process each completed segment once
                  if (!this.processedSegments.has(text)) {
                    this.processedSegments.add(text);
                    console.log('Processing completed segment:', text);
                    this.onSentenceCallback?.(text);
                  }
                }
              });
              
              // Only show current/incomplete segments in transcript (not old completed ones)
              const incompleteSegments = message.segments.filter((seg: any) => !seg.completed);
              const transcript = incompleteSegments.map((seg: any) => seg.text || '').join(' ');
              if (transcript.trim()) {
                this.onTranscriptCallback?.(transcript);
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
  
  async sendInterruptTts(sessionId: string): Promise<{ success: boolean; message: string }> {
    /**
     * Send TTS interruption request to orchestrator backend
     * This is sent via HTTP API rather than WebSocket for reliability
     */
    try {
      console.log(`🛑 Sending TTS interruption request for session ${sessionId}`);
      
      const response = await fetch('/api/interrupt-tts', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      
      const result = await response.json();
      
      console.log(`✅ TTS interruption response:`, result);
      
      const success = result.status === 'interrupted';
      const message = result.message || 'TTS interruption completed';
      
      // Notify via callback if set
      this.onInterruptionAckCallback?.(success, message);
      
      return { success, message };
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error(`❌ TTS interruption failed:`, errorMessage);
      
      // Notify error via callback
      this.onInterruptionAckCallback?.(false, errorMessage);
      
      return { success: false, message: errorMessage };
    }
  }
  

  disconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    this.processedSegments.clear();
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
  
  onInterruptionAck(callback: (success: boolean, message: string) => void) {
    this.onInterruptionAckCallback = callback;
  }
}