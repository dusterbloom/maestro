'use client'

import { useAtomValue } from 'jotai'
import { cn } from '@/lib/utils/cn'
import {
  liveTranscriptAtom,
  transcriptSegmentsAtom,
  isProcessingAtom,
  processingTextAtom
} from '@/atoms/voice.atoms'
import { uiPreferencesAtom } from '@/atoms/settings.atoms'

export function TranscriptDisplay() {
  const liveTranscript = useAtomValue(liveTranscriptAtom)
  const segments = useAtomValue(transcriptSegmentsAtom)
  const isProcessing = useAtomValue(isProcessingAtom)
  const processingText = useAtomValue(processingTextAtom)
  const { showTranscriptTimestamps } = useAtomValue(uiPreferencesAtom)
  
  if (!liveTranscript && segments.length === 0 && !processingText) {
    return null
  }
  
  return (
    <div className="w-full max-w-2xl mx-auto p-4 space-y-4">
      {/* Processing indicator */}
      {isProcessing && processingText && (
        <div className="bg-primary/10 rounded-lg p-3 border border-primary/20">
          <div className="flex items-center gap-2 text-sm text-primary">
            <span className="inline-block w-2 h-2 bg-primary rounded-full animate-pulse" />
            Processing: {processingText}
          </div>
        </div>
      )}
      
      {/* Completed segments */}
      {segments.length > 0 && (
        <div className="space-y-2">
          {segments.map((segment, index) => (
            <div
              key={index}
              className={cn(
                'p-3 rounded-lg transition-all',
                segment.completed
                  ? 'bg-secondary/50 text-foreground'
                  : 'bg-muted/50 text-muted-foreground'
              )}
            >
              <p className="text-sm leading-relaxed">{segment.text}</p>
              {showTranscriptTimestamps && segment.timestamp && (
                <time className="text-xs text-muted-foreground mt-1 block">
                  {new Date(segment.timestamp).toLocaleTimeString()}
                </time>
              )}
            </div>
          ))}
        </div>
      )}
      
      {/* Live transcript */}
      {liveTranscript && (
        <div className="relative">
          <div className="absolute -left-2 top-0 bottom-0 w-1 bg-primary rounded-full animate-pulse" />
          <div className="pl-4">
            <p className="text-sm text-muted-foreground italic">
              {liveTranscript}
              <span className="inline-block w-1 h-4 bg-muted-foreground/50 ml-1 animate-pulse" />
            </p>
          </div>
        </div>
      )}
    </div>
  )
}