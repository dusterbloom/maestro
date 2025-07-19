import { atom } from 'jotai'

// Recording state
export const isRecordingAtom = atom(false)
export const audioLevelAtom = atom(0)

// Represents when the assistant's speech is paused for a user interruption
export const isPausedAtom = atom(false)


// Transcription state
export const liveTranscriptAtom = atom('')
export const finalTranscriptAtom = atom('')
export const transcriptSegmentsAtom = atom<Array<{
  text: string
  completed: boolean
  timestamp?: number
}>>([])

// Processing state
export const isProcessingAtom = atom(false)
export const processingTextAtom = atom('')

// Audio playback state
export const isPlayingAtom = atom(false)
export const audioQueueLengthAtom = atom(0)

// Error state
export const errorAtom = atom<string | null>(null)

// Derived atoms
export const isActiveAtom = atom(
  get => get(isRecordingAtom) || get(isProcessingAtom) || get(isPlayingAtom)
)

export const hasTranscriptAtom = atom(
  get => get(liveTranscriptAtom).length > 0 || get(finalTranscriptAtom).length > 0
)