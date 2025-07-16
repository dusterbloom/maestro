import { useVoiceStore } from '@/stores/voice';
import { EventEmitter } from 'events';

export interface WebSocketMessage {
  type: string;
  data?: any;
  sessionId?: string;
  timestamp?: number;
}

export interface AudioDataMessage extends WebSocketMessage {
  type: 'audio_data';
  data: {
    audio: string; // base64 encoded audio
    sampleRate: number;
    channels: number;
  };
}

export interface TranscriptMessage extends WebSocketMessage {
  type: 'transcript';
  data: {
    text: string;
    isFinal: boolean;
    confidence?: number;
  };
}

export interface ResponseMessage extends WebSocketMessage {
  type: 'response';
  data: {
    text: string;
    audio?: string; // base64 encoded audio response
  };
}

export interface ErrorMessage extends WebSocketMessage {
  type: 'error';
  data: {
    message: string;
    code?: string;
  };
}

export interface SessionMessage extends WebSocketMessage {
  type: 'session_started' | 'session_ended';
  data: {
    sessionId: string;
  };
}

export class OrchestratorWebSocket extends EventEmitter {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private isReconnecting = false;

  constructor(url: string) {
    super();
    this.url = url;
  }

  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      this.ws = new WebSocket(this.url);
      
      this.ws.onopen = () => {
        console.log('Connected to orchestrator WebSocket');
        useVoiceStore.getState().setConnected(true);
        useVoiceStore.getState().setError(null);
        this.reconnectAttempts = 0;
        this.isReconnecting = false;
        this.startHeartbeat();
        this.emit('connected');
      };

      this.ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          this.handleMessage(message);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      this.ws.onclose = (event) => {
        console.log('WebSocket connection closed:', event.code, event.reason);
        useVoiceStore.getState().setConnected(false);
        this.stopHeartbeat();
        this.emit('disconnected', event);

        if (!event.wasClean && !this.isReconnecting) {
          this.attemptReconnect();
        }
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        useVoiceStore.getState().setError('WebSocket connection failed');
        this.emit('error', error);
      };
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      useVoiceStore.getState().setError('Failed to create WebSocket connection');
    }
  }

  disconnect(): void {
    this.isReconnecting = false;
    
    if (this.ws) {
      this.ws.close(1000, 'Manual disconnect');
      this.ws = null;
    }
    
    this.stopHeartbeat();
    useVoiceStore.getState().setConnected(false);
  }

  send(message: WebSocketMessage): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.error('WebSocket is not connected');
      return false;
    }

    try {
      const messageWithTimestamp = {
        ...message,
        timestamp: Date.now(),
      };
      
      this.ws.send(JSON.stringify(messageWithTimestamp));
      return true;
    } catch (error) {
      console.error('Failed to send WebSocket message:', error);
      return false;
    }
  }

  sendAudioData(audioData: ArrayBuffer, sampleRate: number, channels: number): boolean {
    const base64Audio = btoa(
      new Uint8Array(audioData).reduce(
        (data, byte) => data + String.fromCharCode(byte),
        ''
      )
    );

    return this.send({
      type: 'audio_data',
      data: {
        audio: base64Audio,
        sampleRate,
        channels,
      },
    });
  }

  startSession(): boolean {
    return this.send({
      type: 'start_session',
    });
  }

  endSession(): boolean {
    return this.send({
      type: 'end_session',
    });
  }

  private handleMessage(message: WebSocketMessage): void {
    const { setTranscript, setResponse, setSessionId, setError } = useVoiceStore.getState();

    switch (message.type) {
      case 'transcript':
        const transcriptMsg = message as TranscriptMessage;
        setTranscript(transcriptMsg.data.text);
        this.emit('transcript', transcriptMsg.data);
        break;

      case 'response':
        const responseMsg = message as ResponseMessage;
        setResponse(responseMsg.data.text);
        this.emit('response', responseMsg.data);
        break;

      case 'session_started':
        const sessionStartMsg = message as SessionMessage;
        setSessionId(sessionStartMsg.data.sessionId);
        this.emit('sessionStarted', sessionStartMsg.data.sessionId);
        break;

      case 'session_ended':
        const sessionEndMsg = message as SessionMessage;
        setSessionId(null);
        this.emit('sessionEnded', sessionEndMsg.data.sessionId);
        break;

      case 'error':
        const errorMsg = message as ErrorMessage;
        setError(errorMsg.data.message);
        this.emit('error', errorMsg.data);
        break;

      default:
        console.warn('Unknown message type:', message.type);
        this.emit('message', message);
    }
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      useVoiceStore.getState().setError('Failed to reconnect to orchestrator');
      return;
    }

    this.isReconnecting = true;
    this.reconnectAttempts++;
    
    console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
    
    setTimeout(() => {
      if (this.isReconnecting) {
        this.connect();
      }
    }, this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1));
  }

  private startHeartbeat(): void {
    this.heartbeatInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.send({ type: 'ping' });
      }
    }, 30000);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

// Singleton instance
let orchestratorWS: OrchestratorWebSocket | null = null;

export const getOrchestratorWS = (): OrchestratorWebSocket => {
  if (!orchestratorWS) {
    const wsUrl = process.env.NEXT_PUBLIC_ORCHESTRATOR_WS_URL || 'ws://localhost:8000/ws';
    orchestratorWS = new OrchestratorWebSocket(wsUrl);
  }
  return orchestratorWS;
};

export const resetOrchestratorWS = (): void => {
  if (orchestratorWS) {
    orchestratorWS.disconnect();
    orchestratorWS = null;
  }
};