// Test component to verify VoiceButton integration
'use client';

import { useState } from 'react';
import VoiceButton from './VoiceButton';

export default function VoiceButtonTest() {
  const [status, setStatus] = useState<string>('idle');
  const [transcript, setTranscript] = useState<string>('');
  const [error, setError] = useState<string>('');

  return (
    <div className="p-8 space-y-4">
      <h2 className="text-2xl font-bold">Voice Button Test</h2>
      
      <div className="space-y-2">
        <p><strong>Status:</strong> {status}</p>
        <p><strong>Transcript:</strong> {transcript}</p>
        <p><strong>Error:</strong> {error}</p>
      </div>

      <VoiceButton
        onStatusChange={setStatus}
        onTranscript={setTranscript}
        onError={setError}
      />
    </div>
  );
}