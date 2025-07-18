'use client';

import { useState } from 'react';
import EnhancedTranscriptDisplay from '@/components/EnhancedTranscriptDisplay';
import { useVoiceStore } from '@/stores/voice';
import { getVoicePipeline } from '@/lib/voice-pipeline';

export default function TestEnhancedTranscript() {
  const [testMode, setTestMode] = useState<'manual' | 'live'>('manual');
  const transcript = useVoiceStore(state => state.transcript);
  
  // Manual test data
  const [manualTranscripts, setManualTranscripts] = useState([
    "Hello, this is a test transcript.",
    "The EnhancedTranscriptDisplay component is working correctly.",
    "This is the third transcript in the history."
  ]);

  const addManualTranscript = () => {
    const newTranscript = `Manual test at ${new Date().toLocaleTimeString()}`;
    setManualTranscripts(prev => [...prev, newTranscript]);
    useVoiceStore.getState().setTranscript(newTranscript);
  };

  const startLiveTest = async () => {
    setTestMode('live');
    try {
      const pipeline = getVoicePipeline();
      await pipeline.initialize();
      console.log('Live test started - speak to see transcriptions');
    } catch (error) {
      console.error('Failed to start live test:', error);
    }
  };

  const clearTranscripts = () => {
    useVoiceStore.getState().setTranscript('');
    setManualTranscripts([]);
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-purple-50 to-pink-50 p-8">
      <div className="max-w-4xl mx-auto space-y-8">
        <div className="text-center">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">
            Enhanced Transcript Display Test
          </h1>
          <p className="text-gray-600">
            Testing the EnhancedTranscriptDisplay component with live and manual data
          </p>
        </div>

        {/* Test Controls */}
        <div className="bg-white rounded-lg p-6 shadow-lg">
          <h2 className="text-xl font-semibold mb-4">Test Controls</h2>
          <div className="flex gap-4 flex-wrap">
            <button
              onClick={addManualTranscript}
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              Add Manual Transcript
            </button>
            <button
              onClick={startLiveTest}
              className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
            >
              Start Live Test
            </button>
            <button
              onClick={clearTranscripts}
              className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
            >
              Clear All
            </button>
          </div>
          <p className="text-sm text-gray-600 mt-2">
            Current mode: <span className="font-semibold">{testMode}</span>
          </p>
        </div>

        {/* Enhanced Transcript Display */}
        <div className="bg-white rounded-lg p-6 shadow-lg">
          <h2 className="text-xl font-semibold mb-4">Enhanced Transcript Display</h2>
          <EnhancedTranscriptDisplay className="border border-gray-200 rounded-lg p-4" />
        </div>

        {/* Debug Info */}
        <div className="bg-gray-100 rounded-lg p-4">
          <h3 className="font-semibold mb-2">Debug Info</h3>
          <pre className="text-xs bg-gray-800 text-white p-3 rounded overflow-auto">
{JSON.stringify({
  currentTranscript: transcript,
  timestamp: new Date().toISOString()
}, null, 2)}
          </pre>
        </div>

        {/* Instructions */}
        <div className="bg-blue-50 rounded-lg p-4">
          <h3 className="font-semibold text-blue-800 mb-2">How to Test</h3>
          <ol className="text-sm text-blue-700 space-y-1 list-decimal list-inside">
            <li>Click "Add Manual Transcript" to add test data</li>
            <li>Click "Start Live Test" to connect to the voice pipeline</li>
            <li>After live test starts, use the VoiceButton to record speech</li>
            <li>Watch the EnhancedTranscriptDisplay update with live transcriptions</li>
            <li>Use browser dev tools to check console logs for debugging</li>
          </ol>
        </div>
      </div>
    </main>
  );
}