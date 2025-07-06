'use client';

import { useState } from 'react';
import { DESIGN_TOKENS } from '../design-system';
import VoiceButton from '@/components/VoiceButton';
import StatusIndicator from '@/components/StatusIndicator';
import Waveform from '@/components/Waveform';

export default function Home() {
  const [status, setStatus] = useState<'idle' | 'connecting' | 'connected' | 'recording' | 'processing' | 'error'>('idle');
  const [error, setError] = useState<string>('');
  const [transcript, setTranscript] = useState<string>('');

  const handleTranscript = (newTranscript: string) => {
    setTranscript(newTranscript);
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex flex-col items-center justify-center p-6">
      <div className="max-w-md w-full text-center space-y-8">
        <div className="space-y-2">
          <h1 className={DESIGN_TOKENS.heading}>Voice Assistant 2.0</h1>
          <p className={DESIGN_TOKENS.body}>With Magical Speaker Identification</p>
        </div>

        <div className="flex justify-center">
          <StatusIndicator status={status} error={error} />
        </div>

        <div className="relative flex justify-center py-8">
          <VoiceButton
            onStatusChange={setStatus}
            onTranscript={handleTranscript}
            onError={setError}
          />
          <Waveform isRecording={status === 'recording'} audioLevel={0.5} />
        </div>

        {transcript && (
          <div className="bg-white/70 rounded-lg p-4 backdrop-blur-sm border border-blue-200">
            <p className={DESIGN_TOKENS.accent}>You said:</p>
            <p className="text-gray-700 italic">"{transcript}"</p>
          </div>
        )}

      </div>
    </main>
  );
}