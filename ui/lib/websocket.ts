export class VoiceWebSocket {
  private ws: WebSocket | null = null;
  private sessionId: string;

  private onConnectCallback?: () => void;
  private onDisconnectCallback?: () => void;
  private onErrorCallback?: (error: string) => void;
  private onSpeakerIdentifiedCallback?: (data: { user_id: string; name: string; status?: string }) => void;
  private onSpeakerRenamedCallback?: (data: { user_id: string; new_name: string }) => void;
  private onAssistantSpeakCallback?: (data: { text: string; audio_data?: string }) => void;
  private onTranscriptCallback?: (data: { text: string }) => void;

  constructor(private url: string) {
    this.sessionId = `session_${Date.now()}`;
  }

  connect() {
    try {
      const wsUrl = `${this.url}/${this.sessionId}`;
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log("WebSocket connected to Orchestrator");
        this.onConnectCallback?.();
      };

      this.ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        switch (message.event) {
          case "speaker.identified":
            this.onSpeakerIdentifiedCallback?.(message.data);
            break;
          case "speaker.renamed":
            this.onSpeakerRenamedCallback?.(message.data);
            break;
          case "assistant.speak":
            this.onAssistantSpeakCallback?.(message.data);
            break;
          case "transcript.update":
            this.onTranscriptCallback?.(message.data);
            break;
          case "error":
            this.onErrorCallback?.(message.data.message);
            break;
        }
      };

      this.ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        this.onErrorCallback?.("WebSocket connection error");
      };

      this.ws.onclose = () => {
        console.log("WebSocket closed");
        this.onDisconnectCallback?.();
      };
    } catch (error) {
      console.error("Failed to connect:", error);
      this.onErrorCallback?.("Failed to connect to voice service");
    }
  }

  sendAudio(audioData: ArrayBuffer) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      const audioBase64 = this.arrayBufferToBase64(audioData);
      this.ws.send(JSON.stringify({ event: "audio_stream", audio_data: audioBase64 }));
    } else {
      console.warn("WebSocket not connected, cannot send audio");
    }
  }

  claimSpeakerName(userId: string, newName: string) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ event: "speaker.claim", user_id: userId, new_name: newName }));
    } else {
      console.warn("WebSocket not connected, cannot claim speaker name");
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close(1000, "Client disconnect");
      this.ws = null;
    }
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  onConnect(callback: () => void) {
    this.onConnectCallback = callback;
  }

  onDisconnect(callback: () => void) {
    this.onDisconnectCallback = callback;
  }

  onError(callback: (error: string) => void) {
    this.onErrorCallback = callback;
  }

  onSpeakerIdentified(callback: (data: { user_id: string; name: string; status?: string }) => void) {
    this.onSpeakerIdentifiedCallback = callback;
  }

  onSpeakerRenamed(callback: (data: { user_id: string; new_name: string }) => void) {
    this.onSpeakerRenamedCallback = callback;
  }

  onAssistantSpeak(callback: (data: { text: string; audio_data?: string }) => void) {
    this.onAssistantSpeakCallback = callback;
  }

  onTranscript(callback: (data: { text: string }) => void) {
    this.onTranscriptCallback = callback;
  }

  private arrayBufferToBase64(buffer: ArrayBuffer): string {
    let binary = "";
    const bytes = new Uint8Array(buffer);
    const len = bytes.byteLength;
    for (let i = 0; i < len; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return window.btoa(binary);
  }
}
