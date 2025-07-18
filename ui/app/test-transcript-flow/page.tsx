'use client';

import { useState } from 'react';
import WorkingHandsFreeInterface from '@/components/WorkingHandsFreeInterface';

export default function TestTranscriptFlowPage() {
  const [showDebug, setShowDebug] = useState(false);

  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50 p-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-4">
            Working Voice Pipeline Demo
          </h1>
          <p className="text-lg text-gray-600">
            Complete hands-free voice-to-text pipeline with TTS responses
          </p>
        </div>

        <div className="mb-4 flex justify-center">
          <button
            onClick={() => setShowDebug(!showDebug)}
            className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
          >
            {showDebug ? 'Hide' : 'Show'} Debug Info
          </button>
        </div>

        <WorkingHandsFreeInterface />

        {showDebug && (
          <div className="mt-8 bg-white rounded-lg p-6 shadow-lg max-w-2xl mx-auto">
            <h2 className="text-xl font-semibold mb-4">Debug Information</h2>
            <div className="space-y-2 text-sm text-gray-700">
              <p>• Backend: ws://localhost:8000/ws/voice</p>
              <p>• Audio Format: 16-bit PCM, 16kHz, mono</p>
              <p>• Pipeline: WhisperLive → LLM → TTS</p>
              <p>• Messages: live_transcript, sentence_audio, processing_*</p>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}