import { NextRequest, NextResponse } from 'next/server';

const ORCHESTRATOR_URL = process.env.ORCHESTRATOR_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    console.log('🛑 API: Forwarding TTS interruption request to orchestrator:', body);
    
    // Forward the interruption request to the orchestrator
    const response = await fetch(`${ORCHESTRATOR_URL}/interrupt-tts`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });
    
    if (!response.ok) {
      throw new Error(`Orchestrator responded with status ${response.status}`);
    }
    
    const result = await response.json();
    
    console.log('✅ API: TTS interruption response from orchestrator:', result);
    
    return NextResponse.json(result);
    
  } catch (error) {
    console.error('❌ API: TTS interruption error:', error);
    
    return NextResponse.json(
      { error: 'Failed to interrupt TTS', details: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
}