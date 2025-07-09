
export class MaestroWebSocket {
    private ws: WebSocket | null = null;
    private sessionId: string;

    private onConnectCallback?: () => void;
    private onDisconnectCallback?: () => void;
    private onErrorCallback?: (error: string) => void;
    private onEventCallback?: (event: any) => void;

    constructor(sessionId: string) {
        this.sessionId = sessionId;
    }

    public connect() {
        const wsUrl = `${process.env.NEXT_PUBLIC_ORCHESTRATOR_WS_URL}/${this.sessionId}`;
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            this.onConnectCallback?.();
        };

        this.ws.onmessage = (event) => {
            console.log("[WebSocket] Message received:", event.data, "State:", this.ws?.readyState);
            let parsedEvent;
            try {
                parsedEvent = JSON.parse(event.data);
            } catch (err) {
                console.error("[WebSocket] Failed to parse message:", event.data, err);
                return;
            }
            if (parsedEvent.type === "ping") {
                console.log("[WebSocket] Received ping from backend, sending pong.");
                try {
                    this.send({ type: "pong" });
                    console.log("[WebSocket] Pong sent to backend.");
                } catch (err) {
                    console.error("[WebSocket] Failed to send pong:", err);
                }
            } else {
                this.onEventCallback?.(parsedEvent);
            }
        };

        this.ws.onerror = (event) => {
            this.onErrorCallback?.("WebSocket error");
        };

        this.ws.onclose = () => {
            this.onDisconnectCallback?.();
        };
    }

    public send(event: any) {
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(event));
        }
    }

    public disconnect() {
        this.ws?.close();
    }

    public onConnect(callback: () => void) {
        this.onConnectCallback = callback;
    }

    public onDisconnect(callback: () => void) {
        this.onDisconnectCallback = callback;
    }

    public onError(callback: (error: string) => void) {
        this.onErrorCallback = callback;
    }

    public onEvent(callback: (event: any) => void) {
        this.onEventCallback = callback;
    }
}
