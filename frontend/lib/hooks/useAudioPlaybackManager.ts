'use client'

import { useEffect } from 'react'
import { useAtomValue } from 'jotai'
import { isPausedAtom } from '@/atoms/voice.atoms'
import { audioPlayerAtom } from '@/atoms/session.atoms'

/**
 * This hook manages the audio player's state (pause/resume)
 * based on the global `isPausedAtom`.
 */
export function useAudioPlaybackManager() {
  const isPaused = useAtomValue(isPausedAtom)
  const audioPlayer = useAtomValue(audioPlayerAtom)

  useEffect(() => {
    if (!audioPlayer) return

    if (isPaused) {
      audioPlayer.pause()
    } else {
      audioPlayer.resume()
    }
  }, [isPaused, audioPlayer])
}