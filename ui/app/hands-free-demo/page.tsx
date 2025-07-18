'use client';

import HandsFreeVoiceInterface from '@/components/HandsFreeVoiceInterface';

export default function HandsFreeDemoPage() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50 p-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-4">
            Hands-Free Voice Assistant
          </h1>
          <p className="text-lg text-gray-600">
            Talk naturally and get spoken responses - no buttons needed!
          </p>
        </div>

        <HandsFreeVoiceInterface />

        <div className="mt-8 bg-white rounded-lg p-6 shadow-lg max-w-2xl mx-auto">
          <h2 className="text-xl font-semibold mb-4">How It Works</h2>
          <ol className="space-y-2 text-gray-700">
            <li>1. Click "Start Listening" once</li>
            <li>2. Speak naturally - the system listens continuously</li>
            <li>3. When you pause, it processes and responds with TTS</li>
            <li>4. To interrupt, just start speaking again</li>
            <li>5. The system automatically resumes listening after responses</li>
          </ol>
          
          <div className="mt-4 p-3 bg-blue-50 rounded">
            <p className="text-sm text-blue-700">
              <strong>Tip:</strong> You can interrupt the AI's response by simply starting to speak again!
            </p>
          </div>
        </div>
      </div>
    </main>
  );
}