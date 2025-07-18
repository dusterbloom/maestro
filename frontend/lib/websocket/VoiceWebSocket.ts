import { EventBusManager } from './EventBusManager'
import { OrchestratorMessage } from './types'

export interface VoiceWebSocketConfig {
  url?: string
  reconnectDelay?: number
  maxReconnectAttempts?: number
}

export class VoiceWebSocket extends EventTarget {
  private ws: WebSocket | null = null
  private eventBus: EventBusManager
  private config: Required<VoiceWebSocketConfig>
  private reconnectAttempts = 0
  private reconnectTimer: NodeJS.Timeout | null = null
  private sessionId: string | null = null
  private isIntentionalClose = false
  
  constructor(config: VoiceWebSocketConfig = {}) {
    super()
    this.eventBus = new EventBusManager()
    this.config = {
      url: config.url || process.env.NEXT_PUBLIC_ORCHESTRATOR_WS_URL || 'ws://localhost:8000',
      reconnectDelay: config.reconnectDelay || 1000,
      maxReconnectAttempts: config.maxReconnectAttempts || 5
    }
  }
  
  async connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        resolve()
        return
      }
      
      this.isIntentionalClose = false
      const wsUrl = `${this.config.url}/ws/voice`
      
      try {
        this.ws = new WebSocket(wsUrl)
        
        this.ws.onopen = () => {
          console.log('🔗 WebSocket connected to orchestrator:', wsUrl)
          this.reconnectAttempts = 0
          this.dispatchEvent(new Event('connected'))
          resolve()
        }
        
        this.ws.onmessage = async (event) => {
          if (typeof event.data === 'string') {
            try {
              const message = JSON.parse(event.data) as OrchestratorMessage
              await this.handleMessage(message)
            } catch (error) {
              console.error('Failed to parse orchestrator message:', error)
            }
          }
        }
        
        this.ws.onclose = (event) => {
          console.log('WebSocket closed:', event.code, event.reason)
          this.dispatchEvent(new CustomEvent('disconnected', { detail: event }))
          
          if (!this.isIntentionalClose && this.reconnectAttempts < this.config.maxReconnectAttempts) {
            this.scheduleReconnect()
          }
        }
        
        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error)
          this.dispatchEvent(new CustomEvent('error', { detail: error }))
          reject(error)
        }
        
        // Add connection timeout
        setTimeout(() => {
          if (this.ws?.readyState !== WebSocket.OPEN) {
            reject(new Error('Connection timeout'))
          }
        }, 10000)
        
      } catch (error) {
        reject(error)
      }
    })
  }
  
  private async handleMessage(message: OrchestratorMessage): Promise<void> {
    // Handle session ID from ready message
    if (message.type === 'ready') {
      this.sessionId = message.session_id
    }
    
    // Handle interrupt acknowledgment
    if (message.type === 'interrupted') {
      console.log('✅ Interrupt acknowledged by orchestrator')
    }
    
    // Emit to event bus for fire-and-forget handling
    await this.eventBus.emit(message)
    
    // Also dispatch as DOM event for compatibility
    this.dispatchEvent(new CustomEvent('message', { detail: message }))
  }
  
  private scheduleReconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
    }
    
    this.reconnectAttempts++
    const delay = this.config.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1)
    
    console.log(`Scheduling reconnect attempt ${this.reconnectAttempts} in ${delay}ms`)
    
    this.reconnectTimer = setTimeout(() => {
      this.connect().catch(error => {
        console.error('Reconnection failed:', error)
      })
    }, delay)
  }
  
  sendAudio(audioData: Float32Array): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      // Ensure buffer is properly aligned
      if (audioData.buffer.byteLength % 4 !== 0) {
        const alignedLength = Math.floor(audioData.buffer.byteLength / 4) * 4
        const alignedBuffer = audioData.buffer.slice(0, alignedLength)
        this.ws.send(alignedBuffer)
      } else {
        this.ws.send(audioData.buffer)
      }
    } else {
      console.warn('Cannot send audio: WebSocket not connected')
    }
  }
  
  sendMessage(message: object): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      const jsonMessage = JSON.stringify(message)
      console.log('📤 Sending WebSocket message:', jsonMessage)
      this.ws.send(jsonMessage)
    } else {
      console.warn('Cannot send message: WebSocket not connected', message)
    }
  }
  
  interrupt(): void {
    console.log('🛑 Sending interrupt signal to orchestrator')
    this.sendMessage({ type: 'interrupt' })
  }
  
  endAudio(): void {
    console.log('📡 Sending end_audio message to orchestrator')
    this.sendMessage({ type: 'end_audio' })
  }
  
  sendUltraFastText(text: string): void {
    this.sendMessage({ type: 'ultra_fast_text', text })
  }
  
  disconnect(): void {
    this.isIntentionalClose = true
    
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
    
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect')
      this.ws = null
    }
    
    this.eventBus.clear()
    this.sessionId = null
  }
  
  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN
  }
  
  get currentSessionId(): string | null {
    return this.sessionId
  }
  
  get bus(): EventBusManager {
    return this.eventBus
  }
}