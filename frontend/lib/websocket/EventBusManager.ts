import { OrchestratorMessage } from './types'

export type EventHandler<T = any> = (data: T) => void | Promise<void>

export class EventBusManager extends EventTarget {
  private handlers = new Map<string, Set<EventHandler>>()
  
  on<T extends OrchestratorMessage>(
    eventType: T['type'], 
    handler: EventHandler<T>
  ): void {
    if (!this.handlers.has(eventType)) {
      this.handlers.set(eventType, new Set())
    }
    this.handlers.get(eventType)!.add(handler)
  }
  
  off<T extends OrchestratorMessage>(
    eventType: T['type'], 
    handler: EventHandler<T>
  ): void {
    const handlers = this.handlers.get(eventType)
    if (handlers) {
      handlers.delete(handler)
      if (handlers.size === 0) {
        this.handlers.delete(eventType)
      }
    }
  }
  
  async emit<T extends OrchestratorMessage>(message: T): Promise<void> {
    // Simplified emit - let orchestrator handle complex event coordination
    // Frontend only handles UI-specific events
    const handlers = this.handlers.get(message.type)
    if (handlers) {
      for (const handler of handlers) {
        try {
          const result = handler(message)
          if (result instanceof Promise) {
            // Fire and forget for async handlers
            result.catch(error => {
              console.error(`Error in event handler for ${message.type}:`, error)
            })
          }
        } catch (error) {
          console.error(`Error in event handler for ${message.type}:`, error)
        }
      }
    }
    
    // Emit as DOM event for compatibility
    this.dispatchEvent(new CustomEvent(message.type, { detail: message }))
  }
  
  clear(): void {
    this.handlers.clear()
  }
}