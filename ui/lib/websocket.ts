export class VoiceWebSocket {
  private ws: WebSocket | null = null;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  
  private onConnectCallback?: () => void;
  private onAudioCallback?: (audio: ArrayBuffer) => void;
  private onErrorCallback?: (error: string) => void;
  private onDisconnectCallback?: () => void;
  
  constructor(private url: string) {}
  
  connect(sessionId?: string) {
    try {
      const wsUrl = sessionId ? `${this.url}?session_id=${sessionId}` : this.url;
      this.ws = new WebSocket(wsUrl);
      this.ws.binaryType = 'arraybuffer';
      
      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        this.onConnectCallback?.();
      };
      
      this.ws.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) {
          this.onAudioCallback?.(event.data);
        }
      };
      
      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.onErrorCallback?.('WebSocket connection error');
      };
      
      this.ws.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason);
        this.onDisconnectCallback?.();
        if (!event.wasClean) {
          this.scheduleReconnect();
        }
      };
    } catch (error) {
      console.error('Failed to connect:', error);
      this.onErrorCallback?.('Failed to connect to voice service');
      this.scheduleReconnect();
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
      this.ws.send(new TextEncoder().encode("END_OF_AUDIO"));
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
  
  onError(callback: (error: string) => void) {
    this.onErrorCallback = callback;
  }
  
  onDisconnect(callback: () => void) {
    this.onDisconnectCallback = callback;
  }
}