export class VoiceWebSocket {
  private async reconnect(): Promise<void> {
    if (this.ws) {
      this.ws.close();
    }
    await this.connect();
    console.log('🔌 WebSocket reconnected after interruption');
  }

  private ws: WebSocket | null = null;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private keepaliveInterval: NodeJS.Timeout | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 2000;
  private reconnectDelayMultiplier = 2;
  
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
      console.log(`🔌 WebSocket: Connecting to ${wsUrl}`);
      this.ws = new WebSocket(wsUrl);
      this.ws.binaryType = 'arraybuffer';
      
      this.ws.onopen = () => {
        console.log('✅ WebSocket: Connected to WhisperLive');
        this.reconnectAttempts = 0;
        
        // Setup ping/pong keepalive with more aggressive monitoring
        this.keepaliveInterval = setInterval(() => {
          if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({type: "ping"}));
          } else if (this.ws?.readyState === WebSocket.CLOSED) {
            console.log('🔍 WebSocket: Detected closed connection, attempting reconnect...');
            this.scheduleReconnect();
          }
        }, 10000); // More frequent keepalive - every 10 seconds
        
        // Send initial config exactly like WhisperLive expects (must be first message)
        // Based on actual WhisperLive Client.on_open implementation
        const config = {
          uid: sessionId || `session_${Date.now()}`,
          language: "en",
          task: "transcribe", 
          model: "tiny",
          use_vad: true,
          max_clients: 4,
          max_connection_time: 3600, // 1 hour - prevent premature timeout
          send_last_n_segments: 10,
          no_speech_thresh: 0.45,   // Default from WhisperLive
          clip_audio: false,
          same_output_threshold: 10  // Default from WhisperLive
        };
        this.ws?.send(JSON.stringify(config));
        console.log('📤 WebSocket: Sent config to WhisperLive:', config);
        
        // Don't call onConnectCallback here - wait for SERVER_READY
      };
      
      this.ws.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) {
          // Audio response from TTS (not expected from WhisperLive)
          console.log('📦 WebSocket: Received audio data:', event.data.byteLength, 'bytes');
          this.onAudioCallback?.(event.data);
        } else if (typeof event.data === 'string') {
          console.log('📨 WebSocket: Received message:', event.data.slice(0, 200) + (event.data.length > 200 ? '...' : ''));
          try {
            const message = JSON.parse(event.data);
            if (message.message === 'SERVER_READY') {
              console.log('✅ WebSocket: WhisperLive SERVER_READY received');
              // Stop any pending reconnection timers
              if (this.reconnectTimer) {
                clearTimeout(this.reconnectTimer);
                this.reconnectTimer = null;
              }
              this.reconnectAttempts = 0;
              this.onConnectCallback?.();
            } else if (message.segments && message.segments.length > 0) {
              console.log(`📋 WebSocket: Processing ${message.segments.length} segments`);
              // Process each completed segment individually - KISS approach
              message.segments.forEach((segment: any) => {
                if (segment.completed && segment.text && segment.text.trim()) {
                  const text = segment.text.trim();
                  
                  // Simple deduplication - process each completed segment once
                  if (!this.processedSegments.has(text)) {
                    this.processedSegments.add(text);
                    console.log('🎯 WebSocket: Processing completed segment:', text);
                    this.onSentenceCallback?.(text);
                  } else {
                    console.log('⏭️ WebSocket: Skipping duplicate segment:', text);
                  }
                }
              });
              
              // Only show current/incomplete segments in transcript (not old completed ones)
              const incompleteSegments = message.segments.filter((seg: any) => !seg.completed);
              const transcript = incompleteSegments.map((seg: any) => seg.text || '').join(' ');
              if (transcript.trim()) {
                console.log('📝 WebSocket: Incomplete transcript:', transcript);
                this.onTranscriptCallback?.(transcript);
              }
            }
          } catch (e) {
            console.warn('⚠️ WebSocket: Non-JSON message:', event.data);
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
        
        // Clear keepalive interval
        if (this.keepaliveInterval) {
          clearInterval(this.keepaliveInterval);
          this.keepaliveInterval = null;
        }
        
        // Only reconnect on unexpected close codes (not normal/going away)
        // 1000 = Normal closure, 1001 = Going away, 1006 = Abnormal closure
        if (event.code !== 1000 && event.code !== 1001 && event.reason !== 'Client disconnect') {
          console.log(`Auto-reconnecting after unexpected close (${event.code}): ${event.reason}`);
          this.scheduleReconnect();
        } else {
          console.log('WebSocket closed normally, not reconnecting');
        }
      };
    } catch (error) {
      console.error('Failed to connect:', error);
      this.onErrorCallback?.('Failed to connect to voice service');
      // No automatic reconnection
    }
  }
  
  private scheduleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      this.onErrorCallback?.('Maximum reconnection attempts reached. Please refresh the page.');
      return;
    }
    
    // Clear any existing timer
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    
    const delay = this.reconnectDelay * Math.pow(this.reconnectDelayMultiplier, this.reconnectAttempts);
    
    this.reconnectTimer = setTimeout(() => {
      this.reconnectAttempts++;
      console.log(`Reconnection attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
      
      try {
        this.connect();
      } catch (error) {
        console.error('Reconnection attempt failed:', error);
        // Try again if we haven't exceeded max attempts
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
          this.scheduleReconnect();
        } else {
          this.onErrorCallback?.('Failed to reconnect to voice service. Please refresh the page.');
        }
      }
    }, delay);
  }
  
  sendAudio(audioData: ArrayBuffer) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      // Convert Float32Array to the exact format WhisperLive expects
      // WhisperLive server expects np.float32 bytes, same as Python client sends
      const float32Array = new Float32Array(audioData);
      
      // Create the bytes exactly as Python's audio_array.tobytes() would
      const bytes = new Uint8Array(float32Array.buffer);
      
      this.ws.send(bytes);
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
     * Send TTS interruption request to orchestrator backend with improved error handling
     * This is sent via HTTP API rather than WebSocket for reliability
     */
    try {
      console.log(`🛑 Sending TTS interruption request for session ${sessionId}`);
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
      
      const response = await fetch('/api/interrupt-tts', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId
        }),
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const result = await response.json();
      
      console.log(`✅ TTS interruption response:`, result);
      
      const success = result.status === 'interrupted' || result.status === 'no_active_session';
      const message = result.message || 'TTS interruption completed';
      
      // Notify via callback if set
      this.onInterruptionAckCallback?.(success, message);
      
      // Reconnect WebSocket if interruption was successful and connection is lost
      if (success && (!this.ws || this.ws.readyState !== WebSocket.OPEN)) {
        console.log('🔌 Reconnecting WebSocket after successful interruption');
        await this.reconnect();
      }
      
      return { success, message };
      
    } catch (error) {
      let errorMessage = 'Unknown error';
      
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          errorMessage = 'Interruption request timed out';
        } else {
          errorMessage = error.message;
        }
      }
      
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