export interface WebSocketMessage {
  type: string
  [key: string]: any
}

export interface ReadyMessage extends WebSocketMessage {
  type: 'ready'
  session_id: string
  mode?: 'ultra_fast'
}

export interface LiveTranscriptMessage extends WebSocketMessage {
  type: 'live_transcript'
  text: string
}

export interface ProcessingStartedMessage extends WebSocketMessage {
  type: 'processing_started'
  text: string
}

export interface SentenceAudioMessage extends WebSocketMessage {
  type: 'sentence_audio'
  sequence: number
  text: string
  audio_data: string // base64 encoded
  size_bytes: number
}

export interface ProcessingCompleteMessage extends WebSocketMessage {
  type: 'processing_complete'
}

export interface InterruptedMessage extends WebSocketMessage {
  type: 'interrupted'
}

export interface ErrorMessage extends WebSocketMessage {
  type: 'error'
  message: string
}

export interface SegmentsMessage extends WebSocketMessage {
  type: 'segments'
  segments: Array<{
    text: string
    completed: boolean
    start?: number
    end?: number
  }>
}

// Client-side message types (sent to orchestrator)
export interface InterruptRequestMessage extends WebSocketMessage {
  type: 'interrupt'
}

export interface EndAudioMessage extends WebSocketMessage {
  type: 'end_audio'
}

export interface UltraFastTextMessage extends WebSocketMessage {
  type: 'ultra_fast_text'
  text: string
}

export type OrchestratorMessage = 
  | ReadyMessage
  | LiveTranscriptMessage
  | ProcessingStartedMessage
  | SentenceAudioMessage
  | ProcessingCompleteMessage
  | InterruptedMessage
  | ErrorMessage
  | SegmentsMessage

export type ClientMessage = 
  | InterruptRequestMessage
  | EndAudioMessage
  | UltraFastTextMessage