// components/VoiceButton.tsx - Updated with feature flag support
'use client';

import dynamic from 'next/dynamic';
import { isFeatureEnabled } from '@/lib/feature-flags';

// Import both components
import VoiceButtonOld from './VoiceButton-old';
import VoiceButtonNew from './VoiceButton-new';

// Re-export the interface for backward compatibility
export interface VoiceButtonProps {
  onStatusChange?: (status: 'idle' | 'connecting' | 'connected' | 'recording' | 'processing' | 'error') => void;
  onTranscript?: (transcript: string) => void;
  onError?: (error: string) => void;
}

// Create the feature-flagged component
export default function VoiceButton(props: VoiceButtonProps) {
  const useNewArchitecture = isFeatureEnabled('useNewVoiceButton');
  
  if (useNewArchitecture) {
    return <VoiceButtonNew {...props} />;
  }
  
  return <VoiceButtonOld {...props} />;
}

// Export both components for testing
export { VoiceButtonOld, VoiceButtonNew };