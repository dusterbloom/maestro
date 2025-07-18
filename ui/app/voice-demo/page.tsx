'use client';

import VoiceInterface from '@/components/VoiceInterface';

export default function VoiceDemoPage() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50 p-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-4">
            Voice Interface Demo
          </h1>
          <p className="text-lg text-gray-600">
            Complete voice-to-text pipeline with real-time transcription
          </p>
        </div>

        <VoiceInterface 
          className="w-full max-w-2xl mx-auto"
          showDebug={true}
          onTranscriptComplete={(transcript) => {
            console.log('🎯 Transcript completed:', transcript);
          }}
        />

        <div className="mt-8 bg-white rounded-lg p-6 shadow-lg max-w-2xl mx-auto">
          <h2 className="text-xl font-semibold mb-4">How to Use</h2>
          <ol className="space-y-2 text-gray-700">
            <li>1. Click "Start Recording" to begin</li>
            <li>2. Speak naturally - you'll see live transcription</li>
            <li>3. Click "Stop Recording" when done</li>
            <li>4. The system will process your speech and provide responses</li>
            <li>5. Use "Interrupt" to stop current processing</li>
          </ol>
        </div>
      </div>
    </main>
  );
}