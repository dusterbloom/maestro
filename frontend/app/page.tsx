'use client'

import { useVoicePipeline } from '@/lib/hooks/useVoicePipeline'
import { useAutoInterrupt } from '@/lib/hooks/useAutoInterrupt'
import { VoiceButton } from '@/components/voice/VoiceButton'
import { TranscriptDisplay } from '@/components/voice/TranscriptDisplay'
import { AudioWaveform } from '@/components/voice/AudioWaveform'
import { StatusIndicator } from '@/components/voice/StatusIndicator'
import { useAtomValue } from 'jotai'
import { errorAtom } from '@/atoms/voice.atoms'
import { connectionStatusAtom } from '@/atoms/session.atoms'
import { uiPreferencesAtom } from '@/atoms/settings.atoms'

export default function HomePage() {
  const { isConnected, sessionId } = useVoicePipeline()
  useAutoInterrupt() // Re-enabled with lightweight architecture
  const error = useAtomValue(errorAtom)
  const connectionStatus = useAtomValue(connectionStatusAtom)
  const { showWaveform } = useAtomValue(uiPreferencesAtom)
  
  return (
    <main className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <header className="text-center mb-12">
          <h1 className="text-4xl font-bold mb-2">Maestro Voice Assistant</h1>
          <p className="text-muted-foreground">Ultra-low latency voice interactions</p>
        </header>
        
        {/* Connection Status */}
        <div className="flex justify-center mb-8">
          <div className="flex items-center gap-2 text-sm">
            <div className={`w-2 h-2 rounded-full ${
              connectionStatus === 'connected' ? 'bg-green-500' :
              connectionStatus === 'connecting' ? 'bg-yellow-500 animate-pulse' :
              connectionStatus === 'error' ? 'bg-red-500' :
              'bg-gray-500'
            }`} />
            <span className="text-muted-foreground">
              {connectionStatus === 'connected' ? 'Connected' :
               connectionStatus === 'connecting' ? 'Connecting...' :
               connectionStatus === 'error' ? 'Connection Error' :
               'Disconnected'}
            </span>
            {sessionId && (
              <span className="text-xs text-muted-foreground ml-2">
                Session: {sessionId.slice(-8)}
              </span>
            )}
          </div>
        </div>
        
        {/* Error Display */}
        {error && (
          <div className="max-w-md mx-auto mb-8">
            <div className="bg-destructive/10 text-destructive px-4 py-2 rounded-lg text-sm">
              {error}
            </div>
          </div>
        )}
        
        {/* Voice Button */}
        <div className="flex justify-center mb-8">
          <VoiceButton />
        </div>
        
        {/* Audio Waveform */}
        {showWaveform && (
          <div className="max-w-2xl mx-auto mb-8">
            <AudioWaveform />
          </div>
        )}
        
        {/* Transcript Display */}
        <TranscriptDisplay />
        
        {/* Instructions */}
        <div className="text-center mt-12 text-sm text-muted-foreground">
          <p>Click to start/stop recording • Click while processing to interrupt</p>
          <p className="mt-1">Press Space key to toggle recording</p>
        </div>
      </div>
      
      {/* Status Indicator */}
      <StatusIndicator />
    </main>
  )
}