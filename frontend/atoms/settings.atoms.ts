import { atom } from 'jotai'

// Audio settings
export const audioSettingsAtom = atom({
  echoCancellation: true,
  noiseSuppression: true,
  autoGainControl: true,
  sampleRate: 16000
})

// UI preferences
export const uiPreferencesAtom = atom({
  theme: 'system' as 'light' | 'dark' | 'system',
  showWaveform: true,
  showTranscriptTimestamps: false,
  compactMode: false
})

// Feature flags
export const featureFlagsAtom = atom({
  ultraFastMode: true,
  debugMode: false,
  experimentalFeatures: false
})