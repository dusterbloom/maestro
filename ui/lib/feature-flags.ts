// Feature flags for toggling between old and new architecture
export interface FeatureFlags {
  useNewVoiceButton: boolean;
}

export const featureFlags: FeatureFlags = {
  // Toggle between old and new VoiceButton architecture
  useNewVoiceButton: process.env.NEXT_PUBLIC_USE_NEW_VOICE_BUTTON === 'true',
};

// Helper function to check if a feature is enabled
export const isFeatureEnabled = (flag: keyof FeatureFlags): boolean => {
  return featureFlags[flag];
};

// Runtime feature flag updates
export const setFeatureFlag = (flag: keyof FeatureFlags, enabled: boolean): void => {
  featureFlags[flag] = enabled;
};