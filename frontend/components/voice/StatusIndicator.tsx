'use client'

import { useAtomValue } from 'jotai'
import { Activity, Wifi, WifiOff } from 'lucide-react'
import { cn } from '@/lib/utils/cn'
import { 
  isConnectedAtom, 
  connectionStatusAtom,
  sessionDurationAtom,
  sessionMetricsAtom
} from '@/atoms/session.atoms'
import {
  isRecordingAtom,
  isProcessingAtom,
  isPlayingAtom
} from '@/atoms/voice.atoms'

export function StatusIndicator() {
  const isConnected = useAtomValue(isConnectedAtom)
  const connectionStatus = useAtomValue(connectionStatusAtom)
  const sessionDuration = useAtomValue(sessionDurationAtom)
  const sessionMetrics = useAtomValue(sessionMetricsAtom)
  
  const isRecording = useAtomValue(isRecordingAtom)
  const isProcessing = useAtomValue(isProcessingAtom)
  const isPlaying = useAtomValue(isPlayingAtom)
  
  const formatDuration = (ms: number) => {
    const seconds = Math.floor(ms / 1000)
    const minutes = Math.floor(seconds / 60)
    const hours = Math.floor(minutes / 60)
    
    if (hours > 0) {
      return `${hours}h ${minutes % 60}m`
    } else if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`
    } else {
      return `${seconds}s`
    }
  }
  
  const getActivityStatus = () => {
    if (isRecording) return 'Recording'
    if (isProcessing) return 'Processing'
    if (isPlaying) return 'Playing'
    return 'Idle'
  }
  
  const getActivityColor = () => {
    if (isRecording) return 'text-red-500'
    if (isProcessing) return 'text-yellow-500'
    if (isPlaying) return 'text-green-500'
    return 'text-muted-foreground'
  }
  
  return (
    <div className="fixed bottom-4 right-4 bg-card rounded-lg shadow-lg p-4 space-y-3 min-w-[200px]">
      {/* Connection Status */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {isConnected ? (
            <Wifi className="w-4 h-4 text-green-500" />
          ) : (
            <WifiOff className="w-4 h-4 text-muted-foreground" />
          )}
          <span className="text-sm font-medium">
            {connectionStatus === 'connected' ? 'Connected' :
             connectionStatus === 'connecting' ? 'Connecting' :
             connectionStatus === 'error' ? 'Error' :
             'Disconnected'}
          </span>
        </div>
        {sessionDuration > 0 && (
          <span className="text-xs text-muted-foreground">
            {formatDuration(sessionDuration)}
          </span>
        )}
      </div>
      
      {/* Activity Status */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className={cn('w-4 h-4', getActivityColor())} />
          <span className="text-sm">{getActivityStatus()}</span>
        </div>
      </div>
      
      {/* Session Metrics */}
      {sessionMetrics.messageCount > 0 && (
        <div className="pt-2 border-t space-y-1">
          <div className="flex justify-between text-xs">
            <span className="text-muted-foreground">Messages</span>
            <span>{sessionMetrics.messageCount}</span>
          </div>
          <div className="flex justify-between text-xs">
            <span className="text-muted-foreground">Audio Sent</span>
            <span>{(sessionMetrics.audioBytesSent / 1024).toFixed(1)} KB</span>
          </div>
        </div>
      )}
    </div>
  )
}