'use client';

import { useState, useEffect, useRef } from 'react';
import { DESIGN_TOKENS } from '../design-system';
import VoiceButton from '@/components/VoiceButton';
import StatusIndicator from '@/components/StatusIndicator';
import Waveform from '@/components/Waveform';
import { VoiceWebSocket } from '@/lib/websocket';

export default function Home() {
  const [status, setStatus] = useState<'idle' | 'connecting' | 'connected' | 'recording' | 'processing' | 'error'>('idle');
  const [error, setError] = useState<string>('');
  const [lastTranscript, setLastTranscript] = useState<string>('');
  const [speaker, setSpeaker] = useState<{ userId: string; name: string; status?: string } | null>(null);
  const [assistantResponse, setAssistantResponse] = useState<string>('');
  const ws = useRef<VoiceWebSocket | null>(null);

  useEffect(() => {
    ws.current = new VoiceWebSocket("ws://localhost:8000/ws/v1/voice");

    ws.current.onConnect(() => setStatus('connected'));
    ws.current.onDisconnect(() => setStatus('idle'));
    ws.current.onError((errorMessage) => {
      setStatus('error');
      setError(errorMessage);
    });

    ws.current.onSpeakerIdentified((data) => {
      setSpeaker({ userId: data.user_id, name: data.name, status: data.status });
    });

    ws.current.onSpeakerRenamed((data) => {
      setSpeaker((prev) => (prev ? { ...prev, name: data.new_name, status: 'active' } : null));
    });

    ws.current.onAssistantSpeak((data) => {
      setAssistantResponse(data.text);
    });

    ws.current.connect();

    return () => {
      ws.current?.disconnect();
    };
  }, []);

  const handleClaimName = (newName: string) => {
    if (speaker && speaker.status === 'unclaimed') {
      ws.current?.claimSpeakerName(speaker.userId, newName);
    }
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
            onAudioData={(audioData) => ws.current?.sendAudio(audioData)}
            onError={setError}
          />
          <Waveform isRecording={status === 'recording'} audioLevel={0.5} />
        </div>

        {speaker && (
          <div className="bg-white/70 rounded-lg p-4 backdrop-blur-sm border-blue-200">
            <p className={DESIGN_TOKENS.accent}>Speaker:</p>
            <p className="text-gray-700 italic">{speaker.name}</p>
            {speaker.status === 'unclaimed' && (
              <div className="mt-2">
                <input
                  type="text"
                  placeholder="What should I call you?"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      handleClaimName(e.currentTarget.value);
                      e.currentTarget.value = '';
                    }
                  }}
                  className="border rounded p-1"
                />
              </div>
            )}
          </div>
        )}

        {assistantResponse && (
          <div className="bg-white/70 rounded-lg p-4 backdrop-blur-sm border-blue-200">
            <p className={DESIGN_TOKENS.accent}>Assistant:</p>
            <p className="text-gray-700 italic">"{assistantResponse}"</p>
          </div>
        )}

      </div>
    </main>
  );
}
