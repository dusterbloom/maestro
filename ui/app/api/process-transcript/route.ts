import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    // Forward request to orchestrator ultra-fast endpoint
    const orchestratorUrl = process.env.ORCHESTRATOR_URL || 'http://localhost:8000';
    const response = await fetch(`${orchestratorUrl}/ultra-fast`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });
    
    if (!response.ok) {
      throw new Error(`Orchestrator error: ${response.status}`);
    }
    
    const data = await response.json();
    
    // Transform the response to match expected format
    const transformedData = {
      ...data,
      type: 'wav_audio',
      data: data.audio_data
    };
    
    return NextResponse.json(transformedData);
    
  } catch (error) {
    console.error('API error:', error);
    return NextResponse.json(
      { error: 'Failed to process transcript' },
      { status: 500 }
    );
  }
}